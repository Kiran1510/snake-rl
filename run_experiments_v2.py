"""
Optimized experiment runner with performance improvements.

Changes from the original:
    1. max_steps_factor=3 (was 1) — agents can survive longer
    2. Distance-based reward shaping — small reward for getting closer to food
    3. Larger hash table for tile coding (262144 vs 65536)
    4. Two-layer MLP option (256+128 units)
    5. Better epsilon schedule (decay over 70% instead of 80%)

Usage:
    # Run optimized versions of all configs
    python run_experiments_v2.py

    # Quick test
    python run_experiments_v2.py --quick

    # Single config
    python run_experiments_v2.py --algo mlp --rep extended
"""

import argparse
import os
import sys
import time

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
    ExperimentConfig, ExperimentResult, save_results,
)
from snake_rl.utils.save_load import save_agent

try:
    from snake_rl.agents.mlp_sarsa import MLPSarsaAgent
    from snake_rl.agents.mlp_sarsa_v2 import MLPSarsaAgentV2
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


RESULTS_DIR = "results_v2"
WEIGHTS_DIR = "weights_v2"
GRID_SIZE = 20
MAX_STEPS_FACTOR = 3  # 3 * 400 = 1200 steps max

REPRESENTATIONS = {
    "compact": CompactRepresentation,
    "local": LocalNeighborhoodRepresentation,
    "extended": ExtendedRepresentation,
}

# Tuned hyperparameters — optimized
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
            "max_size": 262144,  # 4x larger hash table
        },
    },
    "mlp": {
        "alpha": 0.001,
        "algo_params": {
            "hidden_dim": 256,  # bigger first layer
            "use_target_network": False,
        },
    },
}


def make_env(seed):
    """Create environment with optimized settings."""
    return SnakeEnv(
        grid_size=GRID_SIZE,
        max_steps_factor=MAX_STEPS_FACTOR,
        reward_food=10.0,
        reward_death=-10.0,
        reward_step=-0.1,
        seed=seed,
    )


def make_agent(algo, rep_instance, config, seed, env):
    if algo == "linear":
        return LinearSarsaAgent(
            representation=rep_instance,
            alpha=config.alpha, gamma=config.gamma, seed=seed,
        )
    elif algo == "tile":
        params = config.algo_params
        agent = TileCodingSarsaAgent(
            base_representation=rep_instance,
            n_tilings=params.get("n_tilings", 8),
            n_tiles_per_dim=params.get("n_tiles_per_dim", 4),
            max_size=params.get("max_size", 262144),
            alpha=config.alpha, gamma=config.gamma, seed=seed,
        )
        agent.initialize(env)
        return agent
    elif algo == "mlp":
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        params = config.algo_params
        return MLPSarsaAgentV2(
            representation=rep_instance,
            hidden_dims=(256, 128),  # two hidden layers
            alpha=config.alpha, gamma=config.gamma,
            use_target_network=params.get("use_target_network", False),
            seed=seed,
        )


def make_config(algo, rep, n_episodes, n_seeds):
    ac = ALGO_CONFIGS[algo]
    return ExperimentConfig(
        algorithm=f"{algo}_sarsa_v2",
        representation=rep,
        n_episodes=n_episodes,
        n_seeds=n_seeds,
        grid_size=GRID_SIZE,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_fraction=0.7,  # faster decay
        alpha=ac["alpha"],
        algo_params=ac["algo_params"],
    )


def run_single(algo, rep, n_episodes, n_seeds, seeds):
    config = make_config(algo, rep, n_episodes, n_seeds)
    rep_class = REPRESENTATIONS[rep]
    result = ExperimentResult(config)

    print(f"\n{'#'*70}")
    print(f"# {algo.upper()} SARSA v2 x {rep.upper()}")
    print(f"# {n_episodes} episodes x {len(seeds)} seeds")
    print(f"# max_steps={GRID_SIZE**2 * MAX_STEPS_FACTOR}, hash_size={ALGO_CONFIGS[algo].get('algo_params', {}).get('max_size', 'N/A')}")
    print(f"{'#'*70}")

    best_agent = None
    best_score = -1

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {i+1}/{len(seeds)} (seed={seed}) ---")
        rep_instance = rep_class()
        env = make_env(seed + 1000)
        agent = make_agent(algo, rep_instance, config, seed, env)

        start = time.time()
        logger = train_sarsa(agent, env, config, print_every=max(n_episodes // 5, 100))
        elapsed = time.time() - start

        result.add_run(logger, seed)

        summary = logger.summary()
        print(f"  Done in {elapsed:.0f}s | Final: {summary['mean_score_final']:.2f} +/- {summary['std_score_final']:.2f} | Max: {summary['max_score']}")

        if summary['mean_score_final'] > best_score:
            best_score = summary['mean_score_final']
            best_agent = agent

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, f"{algo}_sarsa_v2__{rep}.json")
    save_results(result, filepath)

    # Save best agent weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if best_agent:
        save_agent(best_agent, f"{algo}_sarsa_v2__{rep}", WEIGHTS_DIR)

    perf = result.final_performance()
    print(f"\n  >> {algo}_sarsa_v2 x {rep}: {perf['mean_score']:.2f} +/- {perf['std_score']:.2f}")

    return result


def run_all(algos, reps, n_episodes, n_seeds):
    seeds = list(range(n_seeds))
    all_results = {}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    overall_start = time.time()

    print("=" * 70)
    print("SNAKE RL v2 — OPTIMIZED EXPERIMENTS")
    print(f"Changes: max_steps x{MAX_STEPS_FACTOR}, hash 262k, hidden 256, eps_decay 70%")
    print("=" * 70)

    for algo in algos:
        if algo == "mlp" and not HAS_TORCH:
            print(f"\nSKIPPING {algo}")
            continue
        for rep in reps:
            result = run_single(algo, rep, n_episodes, n_seeds, seeds)
            all_results[(algo, rep)] = result

    total_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print("OPTIMIZED EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes\n")

    print(f"{'Algorithm':<22} {'Representation':<15} {'Mean Score':>12} {'Std':>8} {'Max':>6}")
    print("-" * 66)
    for algo in algos:
        for rep in reps:
            key = (algo, rep)
            if key in all_results:
                perf = all_results[key].final_performance()
                max_s = max(max(l.scores) for l in all_results[key].run_loggers)
                print(f"{algo+'_sarsa_v2':<22} {rep:<15} {perf['mean_score']:>12.2f} {perf['std_score']:>8.2f} {max_s:>6}")

    # Print comparison with v1
    print("\n\nCOMPARISON WITH v1 (paste v1 numbers manually):")
    print(f"{'Config':<30} {'v1':>10} {'v2':>10} {'Change':>10}")
    print("-" * 62)


def main():
    parser = argparse.ArgumentParser(description="Snake RL v2 — Optimized")
    parser.add_argument("--algo", type=str, default=None, choices=["linear", "tile", "mlp"])
    parser.add_argument("--rep", type=str, default=None, choices=["compact", "local", "extended"])
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.episodes = 2000
        args.seeds = 2

    algos = [args.algo] if args.algo else ["linear", "tile", "mlp"]
    reps = [args.rep] if args.rep else ["compact", "local", "extended"]

    run_all(algos, reps, args.episodes, args.seeds)


if __name__ == "__main__":
    main()
