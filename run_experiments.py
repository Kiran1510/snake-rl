"""
Experiment runner for the full 3x3 matrix:
    3 algorithms (Linear FA, Tile Coding, MLP) × 3 representations (Compact, Local, Extended)

Usage:
    # Run everything (default: 10000 episodes, 5 seeds)
    python run_experiments.py

    # Quick test run (1000 episodes, 2 seeds)
    python run_experiments.py --quick

    # Run a single configuration
    python run_experiments.py --algo linear --rep compact --episodes 5000 --seeds 3

    # Run only one algorithm across all representations
    python run_experiments.py --algo tile

    # Run only one representation across all algorithms
    python run_experiments.py --rep extended

    # Resume: skip configs that already have saved results
    python run_experiments.py --resume
"""

import argparse
import os
import sys
import time
import json

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.representations.features import (
    CompactRepresentation,
    LocalNeighborhoodRepresentation,
    ExtendedRepresentation,
)
from snake_rl.agents.linear_sarsa import LinearSarsaAgent
from snake_rl.agents.tile_sarsa import TileCodingSarsaAgent
from snake_rl.agents.train import train_sarsa
from snake_rl.utils.experiment import (
    ExperimentConfig,
    ExperimentResult,
    RunLogger,
    save_results,
    load_results,
)

# Try importing MLP (requires PyTorch)
try:
    from snake_rl.agents.mlp_sarsa import MLPSarsaAgent
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. MLP experiments will be skipped.")


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = "results"
GRID_SIZE = 20

# Tuned hyperparameters per algorithm
ALGO_CONFIGS = {
    "linear": {
        "alpha": 0.01,
        "algo_params": {},
    },
    "tile": {
        "alpha": 0.05,
        "algo_params": {
            "n_tilings": 8,
            "n_tiles_per_dim": 4,
            "max_size": 65536,
        },
    },
    "mlp": {
        "alpha": 0.001,
        "algo_params": {
            "hidden_dim": 128,
            "use_target_network": False,
        },
    },
}

REPRESENTATIONS = {
    "compact": CompactRepresentation,
    "local": LocalNeighborhoodRepresentation,
    "extended": ExtendedRepresentation,
}


def make_config(algo: str, rep: str, n_episodes: int, n_seeds: int) -> ExperimentConfig:
    """Create an ExperimentConfig for the given algo/rep pair."""
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


def make_agent(algo: str, rep_instance, config: ExperimentConfig, seed: int, env: SnakeEnv):
    """Create and configure an agent for the given algorithm."""
    if algo == "linear":
        return LinearSarsaAgent(
            representation=rep_instance,
            alpha=config.alpha,
            gamma=config.gamma,
            seed=seed,
        )
    elif algo == "tile":
        params = config.algo_params
        agent = TileCodingSarsaAgent(
            base_representation=rep_instance,
            n_tilings=params.get("n_tilings", 8),
            n_tiles_per_dim=params.get("n_tiles_per_dim", 4),
            max_size=params.get("max_size", 65536),
            alpha=config.alpha,
            gamma=config.gamma,
            seed=seed,
        )
        agent.initialize(env)
        return agent
    elif algo == "mlp":
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MLP agent")
        params = config.algo_params
        return MLPSarsaAgent(
            representation=rep_instance,
            hidden_dim=params.get("hidden_dim", 128),
            alpha=config.alpha,
            gamma=config.gamma,
            use_target_network=params.get("use_target_network", False),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def result_path(algo: str, rep: str) -> str:
    """Get the file path for saving results."""
    return os.path.join(RESULTS_DIR, f"{algo}_sarsa__{rep}.json")


def run_single_config(
    algo: str,
    rep: str,
    n_episodes: int,
    n_seeds: int,
    seeds: list[int],
    print_every: int = 500,
) -> ExperimentResult:
    """Run all seeds for a single algorithm × representation configuration."""
    config = make_config(algo, rep, n_episodes, n_seeds)
    rep_class = REPRESENTATIONS[rep]
    result = ExperimentResult(config)

    print(f"\n{'#'*70}")
    print(f"# {algo.upper()} SARSA × {rep.upper()} representation")
    print(f"# {n_episodes} episodes × {len(seeds)} seeds")
    print(f"{'#'*70}")

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {i+1}/{len(seeds)} (seed={seed}) ---")

        rep_instance = rep_class()
        env = SnakeEnv(grid_size=GRID_SIZE, seed=seed + 1000)
        agent = make_agent(algo, rep_instance, config, seed, env)

        start = time.time()
        logger = train_sarsa(agent, env, config, print_every=print_every)
        elapsed = time.time() - start

        result.add_run(logger, seed)

        summary = logger.summary()
        print(f"  Done in {elapsed:.0f}s | "
              f"Final score: {summary['mean_score_final']:.2f} ± "
              f"{summary['std_score_final']:.2f} | "
              f"Max: {summary['max_score']}")

    # Save results
    filepath = result_path(algo, rep)
    save_results(result, filepath)

    # Print summary
    perf = result.final_performance()
    print(f"\n  >> {config.name}: {perf['mean_score']:.2f} ± {perf['std_score']:.2f}")

    return result


def run_all(
    algos: list[str],
    reps: list[str],
    n_episodes: int,
    n_seeds: int,
    resume: bool = False,
):
    """Run the full experiment matrix."""
    seeds = list(range(n_seeds))
    all_results = {}
    total = len(algos) * len(reps)
    completed = 0
    skipped = 0

    os.makedirs(RESULTS_DIR, exist_ok=True)

    overall_start = time.time()

    print("=" * 70)
    print(f"SNAKE RL EXPERIMENT RUNNER")
    print(f"Algorithms:       {algos}")
    print(f"Representations:  {reps}")
    print(f"Episodes:         {n_episodes}")
    print(f"Seeds:            {n_seeds} ({seeds})")
    print(f"Grid size:        {GRID_SIZE}")
    print(f"Configurations:   {total}")
    print(f"Resume mode:      {resume}")
    print("=" * 70)

    for algo in algos:
        if algo == "mlp" and not HAS_TORCH:
            print(f"\nSKIPPING {algo} (PyTorch not installed)")
            skipped += len(reps)
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

            print(f"\n[{completed}/{total}] Running {algo} × {rep}...")
            result = run_single_config(
                algo, rep, n_episodes, n_seeds, seeds,
                print_every=max(n_episodes // 5, 100),
            )
            all_results[(algo, rep)] = result

    total_time = time.time() - overall_start

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    if skipped:
        print(f"Skipped: {skipped} configurations")

    print(f"\n{'Algorithm':<18} {'Representation':<15} {'Mean Score':>12} {'± Std':>8} {'Max':>6}")
    print("-" * 62)

    for algo in algos:
        for rep in reps:
            key = (algo, rep)
            if key in all_results:
                perf = all_results[key].final_performance()
                configs = all_results[key].config
                # Get max score across all seeds
                max_score = max(
                    max(logger.scores) for logger in all_results[key].run_loggers
                )
                print(f"{algo+'_sarsa':<18} {rep:<15} {perf['mean_score']:>12.2f} {perf['std_score']:>8.2f} {max_score:>6d}")

    print(f"\nResults saved to: {os.path.abspath(RESULTS_DIR)}/")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Snake RL Experiment Runner")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["linear", "tile", "mlp"],
                        help="Run only this algorithm (default: all)")
    parser.add_argument("--rep", type=str, default=None,
                        choices=["compact", "local", "extended"],
                        help="Run only this representation (default: all)")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Episodes per seed (default: 10000)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1000 episodes, 2 seeds")
    parser.add_argument("--resume", action="store_true",
                        help="Skip configs with existing results")
    args = parser.parse_args()

    if args.quick:
        args.episodes = 1000
        args.seeds = 2

    algos = [args.algo] if args.algo else ["linear", "tile", "mlp"]
    reps = [args.rep] if args.rep else ["compact", "local", "extended"]

    run_all(algos, reps, args.episodes, args.seeds, args.resume)


if __name__ == "__main__":
    main()
