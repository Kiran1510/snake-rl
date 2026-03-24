"""
Tests for experiment infrastructure (logging, results, persistence, plotting).

Run with: python tests/test_experiment.py
"""

import sys
import os
import time
import traceback
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from snake_rl.utils.experiment import (
    RunLogger,
    ExperimentConfig,
    ExperimentResult,
    save_results,
    load_results,
    get_epsilon,
)


# ============================================================
# RunLogger Tests
# ============================================================

class TestRunLogger:

    def test_log_episode(self):
        logger = RunLogger()
        logger.log_episode(score=5, reward=10.0, steps=100, epsilon=0.5, cause="wall")
        assert len(logger.scores) == 1
        assert logger.scores[0] == 5
        assert logger.rewards[0] == 10.0
        assert logger.steps[0] == 100
        assert logger.epsilons[0] == 0.5
        assert logger.causes[0] == "wall"
        assert len(logger.wall_times) == 1

    def test_multiple_episodes(self):
        logger = RunLogger()
        for i in range(50):
            logger.log_episode(score=i, reward=float(i), steps=10)
        assert len(logger.scores) == 50
        assert logger.scores[-1] == 49

    def test_smoothed_scores(self):
        logger = RunLogger()
        for i in range(200):
            logger.log_episode(score=i, reward=0.0, steps=1)
        smoothed = logger.get_smoothed_scores(window=50)
        assert len(smoothed) == 200 - 50 + 1
        # Smoothed values should be increasing (since raw scores are)
        assert smoothed[-1] > smoothed[0]

    def test_smoothed_scores_small_window(self):
        logger = RunLogger()
        for i in range(10):
            logger.log_episode(score=i, reward=0.0, steps=1)
        smoothed = logger.get_smoothed_scores(window=5)
        assert len(smoothed) > 0

    def test_smoothed_scores_fewer_episodes_than_window(self):
        logger = RunLogger()
        for i in range(5):
            logger.log_episode(score=i, reward=0.0, steps=1)
        smoothed = logger.get_smoothed_scores(window=100)
        assert len(smoothed) > 0  # should adapt window

    def test_summary(self):
        logger = RunLogger()
        for i in range(200):
            logger.log_episode(score=i % 10, reward=float(i), steps=5)
        s = logger.summary(last_n=50)
        assert s["total_episodes"] == 200
        assert s["total_steps"] == 200 * 5
        assert "mean_score_final" in s
        assert "std_score_final" in s
        assert "wall_time_seconds" in s

    def test_to_dict_and_back(self):
        logger = RunLogger()
        for i in range(20):
            logger.log_episode(score=i, reward=float(i * 2), steps=10,
                               epsilon=0.5, cause="wall")
        data = logger.to_dict()
        restored = RunLogger.from_dict(data)
        assert restored.scores == logger.scores
        assert restored.rewards == logger.rewards
        assert restored.steps == logger.steps
        assert restored.epsilons == logger.epsilons
        assert restored.causes == logger.causes


# ============================================================
# ExperimentConfig Tests
# ============================================================

class TestExperimentConfig:

    def test_default_values(self):
        config = ExperimentConfig(algorithm="linear_sarsa", representation="compact")
        assert config.n_episodes == 10000
        assert config.n_seeds == 5
        assert config.gamma == 0.95
        assert config.grid_size == 20

    def test_name_property(self):
        config = ExperimentConfig(algorithm="mlp_sarsa", representation="extended")
        assert config.name == "mlp_sarsa__extended"

    def test_to_dict_and_back(self):
        config = ExperimentConfig(
            algorithm="tile_sarsa",
            representation="local",
            n_episodes=5000,
            alpha=0.01,
            algo_params={"n_tilings": 8},
        )
        data = config.to_dict()
        restored = ExperimentConfig.from_dict(data)
        assert restored.algorithm == config.algorithm
        assert restored.representation == config.representation
        assert restored.n_episodes == config.n_episodes
        assert restored.alpha == config.alpha
        assert restored.algo_params == config.algo_params

    def test_custom_params(self):
        config = ExperimentConfig(
            algorithm="mlp_sarsa",
            representation="compact",
            algo_params={"hidden_size": 128, "lr": 0.001},
        )
        assert config.algo_params["hidden_size"] == 128


# ============================================================
# ExperimentResult Tests
# ============================================================

class TestExperimentResult:

    def _make_result(self, n_seeds=3, n_episodes=500):
        """Helper to create a fake ExperimentResult."""
        config = ExperimentConfig(algorithm="test_algo", representation="test_rep",
                                  n_episodes=n_episodes)
        result = ExperimentResult(config)
        for seed in range(n_seeds):
            logger = RunLogger()
            rng = np.random.default_rng(seed)
            for ep in range(n_episodes):
                # Simulate improving performance
                mean_score = min(20, ep / 50)
                score = int(max(0, rng.normal(mean_score, 2)))
                reward = float(score * 10 - 5)
                logger.log_episode(score=score, reward=reward, steps=50)
            result.add_run(logger, seed)
        return result

    def test_add_run(self):
        result = self._make_result(n_seeds=3)
        assert result.n_runs == 3
        assert len(result.seeds) == 3

    def test_mean_learning_curve(self):
        result = self._make_result(n_seeds=3, n_episodes=500)
        mean, std = result.mean_learning_curve(window=50)
        assert len(mean) == len(std)
        assert len(mean) > 0
        # Performance should generally improve
        assert mean[-1] > mean[0]

    def test_mean_reward_curve(self):
        result = self._make_result(n_seeds=3, n_episodes=500)
        mean, std = result.mean_reward_curve(window=50)
        assert len(mean) == len(std)
        assert len(mean) > 0

    def test_final_performance(self):
        result = self._make_result(n_seeds=3, n_episodes=500)
        perf = result.final_performance(last_n=100)
        assert "mean_score" in perf
        assert "std_score" in perf
        assert perf["n_seeds"] == 3
        assert len(perf["per_seed_means"]) == 3
        assert perf["mean_score"] > 0

    def test_convergence_episode(self):
        result = self._make_result(n_seeds=3, n_episodes=500)
        conv = result.convergence_episode(threshold_fraction=0.9, window=50)
        assert "per_seed" in conv
        assert "mean" in conv
        assert len(conv["per_seed"]) == 3

    def test_summary(self):
        result = self._make_result(n_seeds=2, n_episodes=200)
        s = result.summary()
        assert "config" in s
        assert "final_performance" in s
        assert "convergence" in s
        assert s["n_runs"] == 2

    def test_serialization_roundtrip(self):
        result = self._make_result(n_seeds=2, n_episodes=100)
        data = result.to_dict()
        restored = ExperimentResult.from_dict(data)
        assert restored.n_runs == result.n_runs
        assert restored.seeds == result.seeds
        assert restored.config.algorithm == result.config.algorithm
        assert len(restored.run_loggers[0].scores) == len(result.run_loggers[0].scores)


# ============================================================
# Persistence Tests
# ============================================================

class TestPersistence:

    def test_save_and_load(self):
        config = ExperimentConfig(algorithm="test", representation="test")
        result = ExperimentResult(config)
        logger = RunLogger()
        for i in range(50):
            logger.log_episode(score=i, reward=float(i), steps=10)
        result.add_run(logger, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            save_results(result, filepath)
            loaded = load_results(filepath)
            assert loaded.n_runs == 1
            assert loaded.seeds == [42]
            assert loaded.run_loggers[0].scores == logger.scores
            assert loaded.config.algorithm == "test"
        finally:
            os.unlink(filepath)

    def test_save_creates_directory(self):
        config = ExperimentConfig(algorithm="test", representation="test")
        result = ExperimentResult(config)
        logger = RunLogger()
        logger.log_episode(score=1, reward=1.0, steps=1)
        result.add_run(logger, seed=0)

        dirpath = tempfile.mkdtemp()
        filepath = os.path.join(dirpath, "subdir", "results.json")

        try:
            save_results(result, filepath)
            assert os.path.exists(filepath)
            loaded = load_results(filepath)
            assert loaded.n_runs == 1
        finally:
            import shutil
            shutil.rmtree(dirpath)


# ============================================================
# Epsilon Schedule Tests
# ============================================================

class TestEpsilonSchedule:

    def test_epsilon_start(self):
        config = ExperimentConfig(algorithm="t", representation="t",
                                  epsilon_start=1.0, epsilon_end=0.01)
        eps = get_epsilon(0, config)
        assert eps == 1.0

    def test_epsilon_end(self):
        config = ExperimentConfig(algorithm="t", representation="t",
                                  n_episodes=1000, epsilon_start=1.0,
                                  epsilon_end=0.01, epsilon_decay_fraction=0.8)
        # After decay period (episode 800+)
        eps = get_epsilon(900, config)
        assert eps == 0.01

    def test_epsilon_midpoint(self):
        config = ExperimentConfig(algorithm="t", representation="t",
                                  n_episodes=1000, epsilon_start=1.0,
                                  epsilon_end=0.0, epsilon_decay_fraction=1.0)
        eps = get_epsilon(500, config)
        assert abs(eps - 0.5) < 0.01

    def test_epsilon_monotonically_decreasing(self):
        config = ExperimentConfig(algorithm="t", representation="t",
                                  n_episodes=1000, epsilon_start=1.0,
                                  epsilon_end=0.01, epsilon_decay_fraction=0.8)
        prev = get_epsilon(0, config)
        for ep in range(1, 1000):
            curr = get_epsilon(ep, config)
            assert curr <= prev + 1e-10, f"Epsilon increased at episode {ep}"
            prev = curr

    def test_epsilon_never_below_end(self):
        config = ExperimentConfig(algorithm="t", representation="t",
                                  n_episodes=100, epsilon_start=1.0,
                                  epsilon_end=0.05, epsilon_decay_fraction=0.5)
        for ep in range(200):  # go past n_episodes
            eps = get_epsilon(ep, config)
            assert eps >= config.epsilon_end - 1e-10


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
    print("EXPERIMENT INFRASTRUCTURE TEST SUITE")
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
    sys.exit(0 if success else 1)
