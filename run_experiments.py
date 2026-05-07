"""
Experiment runner.

    1. MLPSarsaAgent — replay buffer (50k), Q-learning target, target network
    2. Distance-based reward shaping — small bonus for moving toward food
    3. max_steps_factor=3 (1,200 steps max)
    4. Larger tile hash table (262,144)
    5. Epsilon decay over 80% of training
    6. Default 20,000 episodes (MLP needs more steps to converge with replay)

Usage:
    python run_experiments.py                          # all configs
    python run_experiments.py --quick                  # 2k eps, 2 seeds
    python run_experiments.py --algo mlp --rep compact # single config
    python run_experiments.py --resume                 # skip existing results
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.representations.features import (
    CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation,
)
from snake_rl.agents.linear_sarsa import LinearSarsaAgent
from snake_rl.agents.tile_sarsa import TileCodingSarsaAgent
from snake_rl.agents.train import train_sarsa
from snake_rl.utils.experiment import (
    ExperimentConfig, ExperimentResult, save_results, load_results,
)
from snake_rl.utils.save_load import save_agent

try:
    from snake_rl.agents.mlp_sarsa import MLPSarsaAgent
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. MLP experiments will be skipped.")


# ============================================================
# Reward shaping wrapper
# ============================================================

class DistanceShapingWrapper:
    """
    Adds potential-based reward shaping: F(s, s') = shaping_factor * (d(s) - d(s'))
    where d is the Manhattan distance from the snake's head to the food.

    Moving closer to food gives a small positive bonus; moving away gives a
    small penalty. This doesn't change the optimal policy (Ng et al., 1999)
    but provides a denser reward signal that speeds up learning.

    The bonus is not applied when food is eaten (the food position resets,
    so the distance comparison would be meaningless).
    """

    def __init__(self, env: SnakeEnv, shaping_factor: float = 0.5):
        self._env = env
        self.shaping_factor = shaping_factor
        self._prev_dist: float = 0.0

    def reset(self, seed=None):
        obs, info = self._env.reset(seed=seed)
        self._prev_dist = self._dist(obs)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not terminated:
            curr_dist = self._dist(obs)
            ate_food = reward >= 5.0   # food reward is +10; step reward is -0.1
            if not ate_food:
                reward += self.shaping_factor * (self._prev_dist - curr_dist)
            self._prev_dist = curr_dist
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _dist(obs: dict) -> float:
        hx, hy = obs["snake"][0]
        fx, fy = obs["food"]
        return float(abs(fx - hx) + abs(fy - hy))

    def __getattr__(self, name: str):
        return getattr(self._env, name)


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = "results"
WEIGHTS_DIR = "weights"
GRID_SIZE = 20
MAX_STEPS_FACTOR = 3   # 3 × 400 = 1,200 steps max

REPRESENTATIONS = {
    "compact":  CompactRepresentation,
    "local":    LocalNeighborhoodRepresentation,
    "extended": ExtendedRepresentation,
}

ALGO_CONFIGS = {
    "linear": {
        "alpha": 0.01,
        "algo_params": {},
    },
    "tile": {
        "alpha": 0.05,
        "algo_params": {
            # n_tilings per rep: compact is low-dim so 8 tilings works well;
            # local/extended are 109+/126+ dims — 8 tilings hurts speed with
            # no quality gain given the 262k hash table size.
            "n_tilings": {"compact": 8, "local": 4, "extended": 4},
            "n_tiles_per_dim": 4,
            "max_size": 262_144,
        },
    },
    "mlp": {
        "alpha": 0.001,
        "algo_params": {
            "hidden_dims": (256, 128),
            "buffer_capacity": 50_000,
            "batch_size": 64,
            "target_update_freq": 100,
        },
    },
}


def make_env(seed: int, use_shaping: bool = True) -> SnakeEnv:
    env = SnakeEnv(
        grid_size=GRID_SIZE,
        max_steps_factor=MAX_STEPS_FACTOR,
        seed=seed,
    )
    if use_shaping:
        return DistanceShapingWrapper(env, shaping_factor=0.5)
    return env


def make_config(algo: str, rep: str, n_episodes: int, n_seeds: int) -> ExperimentConfig:
    ac = ALGO_CONFIGS[algo]
    return ExperimentConfig(
        algorithm=f"{algo}_sarsa",
        representation=rep,
        n_episodes=n_episodes,
        n_seeds=n_seeds,
        grid_size=GRID_SIZE,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_fraction=0.8,
        alpha=ac["alpha"],
        algo_params=ac["algo_params"],
    )


def make_agent(algo: str, rep_instance, config: ExperimentConfig, seed: int, env):
    if algo == "linear":
        return LinearSarsaAgent(
            representation=rep_instance,
            alpha=config.alpha, gamma=config.gamma, seed=seed,
        )
    elif algo == "tile":
        p = config.algo_params
        n_tilings_cfg = p.get("n_tilings", 8)
        rep_name = config.representation
        n_tilings = (
            n_tilings_cfg[rep_name]
            if isinstance(n_tilings_cfg, dict)
            else n_tilings_cfg
        )
        agent = TileCodingSarsaAgent(
            base_representation=rep_instance,
            n_tilings=n_tilings,
            n_tiles_per_dim=p.get("n_tiles_per_dim", 4),
            max_size=p.get("max_size", 262_144),
            alpha=config.alpha, gamma=config.gamma, seed=seed,
        )
        agent.initialize(env)
        return agent
    elif algo == "mlp":
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MLP agent")
        p = config.algo_params
        return MLPSarsaAgent(
            representation=rep_instance,
            hidden_dims=p.get("hidden_dims", (256, 128)),
            alpha=config.alpha,
            gamma=config.gamma,
            buffer_capacity=p.get("buffer_capacity", 50_000),
            batch_size=p.get("batch_size", 64),
            target_update_freq=p.get("target_update_freq", 100),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def result_path(algo: str, rep: str) -> str:
    return os.path.join(RESULTS_DIR, f"{algo}_sarsa__{rep}.json")


# ============================================================
# Parallel worker  (must be module-level for macOS spawn)
# ============================================================

def _seed_worker(args: tuple) -> tuple:
    """Train one seed in a subprocess. Returns (logger_dict, seed, summary)."""
    algo, rep_name, config_dict, seed, use_shaping = args

    from snake_rl.utils.experiment import ExperimentConfig, RunLogger
    from snake_rl.agents.train import train_sarsa

    config = ExperimentConfig.from_dict(config_dict)
    rep_instance = REPRESENTATIONS[rep_name]()
    env = make_env(seed + 1000, use_shaping=use_shaping)
    agent = make_agent(algo, rep_instance, config, seed, env)

    logger = train_sarsa(agent, env, config, print_every=0)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    save_agent(agent, f"{algo}_sarsa__{rep_name}_seed{seed}", WEIGHTS_DIR)

    return logger.to_dict(), seed, logger.summary()


# ============================================================
# Experiment execution
# ============================================================

def run_single(
    algo: str,
    rep: str,
    n_episodes: int,
    n_seeds: int,
    seeds: list[int],
    print_every: int = 500,
    parallel: bool = False,
) -> ExperimentResult:
    config = make_config(algo, rep, n_episodes, n_seeds)
    result = ExperimentResult(config)
    use_shaping = (algo == "mlp")

    print(f"\n{'#'*70}")
    print(f"# {algo.upper()} SARSA × {rep.upper()}")
    print(f"# {n_episodes} episodes × {len(seeds)} seeds  |  shaping={use_shaping}  |  parallel={parallel}")
    print(f"{'#'*70}")

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if parallel and len(seeds) > 1:
        n_workers = min(len(seeds), os.cpu_count() or 4)
        print(f"\n  Spawning {n_workers} worker processes...")
        args_list = [
            (algo, rep, config.to_dict(), seed, use_shaping)
            for seed in seeds
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_seed_worker, a): a[3] for a in args_list}
            for future in as_completed(futures):
                logger_dict, seed, summary = future.result()
                from snake_rl.utils.experiment import RunLogger
                logger = RunLogger.from_dict(logger_dict)
                result.add_run(logger, seed)
                print(
                    f"  seed={seed} done | "
                    f"final={summary['mean_score_final']:.2f} ± {summary['std_score_final']:.2f} | "
                    f"max={summary['max_score']} | "
                    f"weights: {algo}_sarsa__{rep}_seed{seed}"
                )
    else:
        for i, seed in enumerate(seeds):
            print(f"\n--- Seed {i+1}/{len(seeds)} (seed={seed}) ---")
            rep_instance = REPRESENTATIONS[rep]()
            env = make_env(seed + 1000, use_shaping=use_shaping)
            agent = make_agent(algo, rep_instance, config, seed, env)

            start = time.time()
            logger = train_sarsa(agent, env, config, print_every=print_every)
            elapsed = time.time() - start

            result.add_run(logger, seed)
            summary = logger.summary()
            print(
                f"  Done in {elapsed:.0f}s | "
                f"Final: {summary['mean_score_final']:.2f} ± {summary['std_score_final']:.2f} | "
                f"Max: {summary['max_score']}"
            )
            save_agent(agent, f"{algo}_sarsa__{rep}_seed{seed}", WEIGHTS_DIR)

    save_results(result, result_path(algo, rep))
    perf = result.final_performance()
    print(f"\n  >> {algo}_sarsa × {rep}: {perf['mean_score']:.2f} ± {perf['std_score']:.2f}")
    print(f"  Weights saved: {WEIGHTS_DIR}/{algo}_sarsa__{rep}_seed{{0..{len(seeds)-1}}}")
    return result


def run_all(
    algos: list[str],
    reps: list[str],
    n_episodes: int,
    n_seeds: int,
    resume: bool = False,
    parallel: bool = False,
) -> None:
    seeds = list(range(n_seeds))
    all_results = {}
    total = len(algos) * len(reps)
    completed = 0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    overall_start = time.time()

    print("=" * 70)
    print("SNAKE RL EXPERIMENTS")
    print(f"  max_steps_factor : {MAX_STEPS_FACTOR}  ({GRID_SIZE**2 * MAX_STEPS_FACTOR} steps max)")
    print(f"  tile hash size   : {ALGO_CONFIGS['tile']['algo_params']['max_size']:,}")
    print(f"  MLP architecture : {ALGO_CONFIGS['mlp']['algo_params']['hidden_dims']}")
    print(f"  replay buffer    : {ALGO_CONFIGS['mlp']['algo_params']['buffer_capacity']:,}")
    print(f"  episodes         : {n_episodes}  |  seeds: {n_seeds}")
    print("=" * 70)

    for algo in algos:
        if algo == "mlp" and not HAS_TORCH:
            print(f"\nSKIPPING {algo} (PyTorch not installed)")
            completed += len(reps)
            continue

        for rep in reps:
            completed += 1
            fp = result_path(algo, rep)

            if resume and os.path.exists(fp):
                print(f"\n[{completed}/{total}] {algo} × {rep} — SKIPPING (results exist)")
                try:
                    all_results[(algo, rep)] = load_results(fp)
                except Exception:
                    pass
                continue

            print(f"\n[{completed}/{total}] {algo} × {rep}")
            result = run_single(
                algo, rep, n_episodes, n_seeds, seeds,
                print_every=max(n_episodes // 10, 100),
                parallel=parallel,
            )
            all_results[(algo, rep)] = result

    total_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time / 60:.1f} minutes\n")
    print(f"{'Algorithm':<20} {'Representation':<15} {'Mean Score':>12} {'± Std':>8} {'Max':>6}")
    print("-" * 64)

    for algo in algos:
        for rep in reps:
            key = (algo, rep)
            if key in all_results:
                perf = all_results[key].final_performance()
                max_s = max(max(lg.scores) for lg in all_results[key].run_loggers)
                print(
                    f"{algo+'_sarsa':<20} {rep:<15} "
                    f"{perf['mean_score']:>12.2f} {perf['std_score']:>8.2f} {max_s:>6d}"
                )

    print(f"\nResults saved to: {os.path.abspath(RESULTS_DIR)}/")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Snake RL Experiments")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["linear", "tile", "mlp"])
    parser.add_argument("--rep", type=str, default=None,
                        choices=["compact", "local", "extended"])
    parser.add_argument("--episodes", type=int, default=20_000,
                        help="Episodes per seed (default: 20000)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 2000 episodes, 2 seeds")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs with existing results")
    parser.add_argument("--parallel", action="store_true",
                        help="Run seeds in parallel (one process per seed)")
    args = parser.parse_args()

    if args.quick:
        args.episodes = 2_000
        args.seeds = 2

    algos = [args.algo] if args.algo else ["linear", "tile", "mlp"]
    reps  = [args.rep]  if args.rep  else ["compact", "local", "extended"]

    run_all(algos, reps, args.episodes, args.seeds, resume=args.resume, parallel=args.parallel)


if __name__ == "__main__":
    main()
