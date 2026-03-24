"""
Tests and evaluation for baseline agents.

Run with: python tests/test_baselines.py
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.agents.baselines import (
    RandomAgent, GreedyHeuristicAgent, evaluate_agent,
)


class TestRandomAgent:

    def test_act_returns_valid_action(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        agent = RandomAgent(seed=42)
        for _ in range(100):
            action = agent.act(obs)
            assert action in [0, 1, 2]

    def test_reproducible_with_seed(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        a1 = RandomAgent(seed=123)
        a2 = RandomAgent(seed=123)
        for _ in range(50):
            assert a1.act(obs) == a2.act(obs)

    def test_runs_full_episode(self):
        env = SnakeEnv(grid_size=10, seed=42)
        agent = RandomAgent(seed=42)
        obs, _ = env.reset()
        steps = 0
        while not env.done and steps < 10000:
            action = agent.act(obs)
            obs, _, _, _, _ = env.step(action)
            steps += 1
        assert env.done


class TestGreedyHeuristicAgent:

    def test_act_returns_valid_action(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        agent = GreedyHeuristicAgent()
        for _ in range(100):
            if env.done:
                obs, _ = env.reset()
            action = agent.act(obs)
            assert action in [0, 1, 2]
            obs, _, _, _, _ = env.step(action)

    def test_avoids_immediate_wall(self):
        """Agent near wall should not walk into it."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()
        # Place snake near right wall, facing right
        env.snake.clear()
        env.snake.extend([(9, 5), (8, 5), (7, 5)])
        env.direction = (1, 0)  # RIGHT
        env.food = (5, 5)
        obs = env._get_obs()
        agent = GreedyHeuristicAgent()
        action = agent.act(obs)
        # Should NOT go straight (into wall)
        assert action != 0, "Agent should avoid walking into wall"

    def test_moves_toward_food(self):
        """With no danger, agent should move toward food."""
        env = SnakeEnv(grid_size=20, seed=42)
        env.reset()
        # Place snake in center, food to the right
        env.snake.clear()
        env.snake.extend([(10, 10), (9, 10), (8, 10)])
        env.direction = (1, 0)  # RIGHT
        env.food = (15, 10)  # directly right
        obs = env._get_obs()
        agent = GreedyHeuristicAgent()
        action = agent.act(obs)
        # Going straight moves toward food (same row)
        assert action == 0, "Should go straight toward food"

    def test_beats_random_agent(self):
        """Greedy heuristic should significantly outperform random."""
        env_r = SnakeEnv(grid_size=10, seed=42)
        env_g = SnakeEnv(grid_size=10, seed=42)
        random_agent = RandomAgent(seed=42)
        greedy_agent = GreedyHeuristicAgent()

        r_results = evaluate_agent(random_agent, env_r, n_episodes=50)
        g_results = evaluate_agent(greedy_agent, env_g, n_episodes=50)

        assert g_results["mean_score"] > r_results["mean_score"], \
            f"Greedy ({g_results['mean_score']:.1f}) should beat random ({r_results['mean_score']:.1f})"

    def test_runs_full_episode(self):
        env = SnakeEnv(grid_size=10, seed=42)
        agent = GreedyHeuristicAgent()
        obs, _ = env.reset()
        while not env.done:
            action = agent.act(obs)
            obs, _, _, _, _ = env.step(action)
        assert env.done


class TestEvaluateAgent:

    def test_returns_correct_keys(self):
        env = SnakeEnv(grid_size=10, seed=42)
        agent = RandomAgent(seed=42)
        results = evaluate_agent(agent, env, n_episodes=5)
        expected_keys = {"scores", "lengths", "steps", "mean_score",
                         "std_score", "max_score", "mean_steps", "causes"}
        assert set(results.keys()) == expected_keys

    def test_correct_episode_count(self):
        env = SnakeEnv(grid_size=10, seed=42)
        agent = RandomAgent(seed=42)
        results = evaluate_agent(agent, env, n_episodes=10)
        assert len(results["scores"]) == 10
        assert len(results["lengths"]) == 10
        assert len(results["steps"]) == 10

    def test_causes_sum_to_episodes(self):
        env = SnakeEnv(grid_size=10, seed=42)
        agent = RandomAgent(seed=42)
        results = evaluate_agent(agent, env, n_episodes=20)
        total_causes = sum(results["causes"].values())
        assert total_causes == 20


# ============================================================
# Runner
# ============================================================

def run_all_tests():
    test_module = sys.modules[__name__]
    test_classes = [
        obj for name, obj in vars(test_module).items()
        if isinstance(obj, type) and name.startswith("Test")
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("BASELINE AGENTS TEST SUITE")
    print("=" * 70)

    start_time = time.time()

    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        methods = [m for m in dir(cls) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            instance = cls()
            method = getattr(instance, method_name)

            try:
                method()
                passed += 1
                print(f"  ✓ {method_name}")
            except AssertionError as e:
                failed += 1
                errors.append((cls.__name__, method_name, e, traceback.format_exc()))
                print(f"  ✗ {method_name} — ASSERTION FAILED")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e, traceback.format_exc()))
                print(f"  ✗ {method_name} — ERROR: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {total} total ({elapsed:.2f}s)")
    print("=" * 70)

    if errors:
        print("\n--- FAILURES ---\n")
        for cls_name, method_name, err, tb in errors:
            print(f"{cls_name}.{method_name}:")
            print(f"  {err}")
            print()

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()

    if success:
        # Run the actual baseline evaluation
        print("\n\n" + "=" * 70)
        print("BASELINE EVALUATION (200 episodes each, grid_size=20)")
        print("=" * 70)

        env = SnakeEnv(grid_size=20, seed=0)

        print("\n>> Random Agent:")
        random_results = evaluate_agent(RandomAgent(seed=0), env, n_episodes=200, verbose=True)
        print(f"   Mean score: {random_results['mean_score']:.2f} ± {random_results['std_score']:.2f}")
        print(f"   Max score:  {random_results['max_score']}")
        print(f"   Mean steps: {random_results['mean_steps']:.1f}")
        print(f"   Causes:     {random_results['causes']}")

        print("\n>> Greedy Heuristic Agent:")
        greedy_results = evaluate_agent(GreedyHeuristicAgent(), env, n_episodes=200, verbose=True)
        print(f"   Mean score: {greedy_results['mean_score']:.2f} ± {greedy_results['std_score']:.2f}")
        print(f"   Max score:  {greedy_results['max_score']}")
        print(f"   Mean steps: {greedy_results['mean_steps']:.1f}")
        print(f"   Causes:     {greedy_results['causes']}")

    sys.exit(0 if success else 1)
