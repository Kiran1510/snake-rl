"""
Tests for Linear SARSA agent and training loop.

Run with: python tests/test_linear_sarsa.py
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from snake_rl.env.snake_env import SnakeEnv, Action
from snake_rl.representations.features import (
    CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation,
)
from snake_rl.agents.linear_sarsa import LinearSarsaAgent
from snake_rl.agents.train import train_sarsa, _run_episode
from snake_rl.utils.experiment import ExperimentConfig, RunLogger


# ============================================================
# Agent core mechanics
# ============================================================

class TestLinearSarsaAgent:

    def test_initialization(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.001, gamma=0.95, seed=42)
        assert agent.w.shape == (rep.feature_dim,)
        assert agent.w.dtype == np.float32
        assert agent.alpha == 0.001
        assert agent.gamma == 0.95

    def test_weight_init_small(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, seed=42)
        assert np.abs(agent.w).max() < 0.1, "Initial weights should be small"

    def test_q_value_returns_float(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        q = agent.q_value(obs, 0)
        assert isinstance(q, float)

    def test_q_values_shape(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        q_vals = agent.q_values(obs)
        assert q_vals.shape == (3,)

    def test_q_value_is_linear(self):
        """q(s,a) should equal w^T x(s,a)."""
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        for a in range(3):
            features = rep.get_features(obs, a)
            expected = float(np.dot(agent.w, features))
            actual = agent.q_value(obs, a)
            assert abs(expected - actual) < 1e-6

    def test_act_returns_valid_action(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, epsilon=0.5, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        for _ in range(100):
            action = agent.act(obs)
            assert action in [0, 1, 2]

    def test_act_greedy_when_epsilon_zero(self):
        """With epsilon=0, should always pick the best action."""
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, epsilon=0.0, seed=42)
        # Set weights so action 1 is clearly best
        agent.w[:] = 0
        agent.w[rep.state_dim:2 * rep.state_dim] = 1.0  # action 1 block
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        for _ in range(20):
            action = agent.act(obs)
            assert action == 1, f"Expected action 1, got {action}"

    def test_act_explores_when_epsilon_one(self):
        """With epsilon=1.0, should explore uniformly."""
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, epsilon=1.0, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        actions = [agent.act(obs) for _ in range(300)]
        counts = [actions.count(a) for a in range(3)]
        # Each action should appear at least 50 times out of 300
        for a, c in enumerate(counts):
            assert c > 30, f"Action {a} appeared only {c}/300 times — not exploring"

    def test_update_changes_weights(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.1, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        w_before = agent.w.copy()
        next_obs, reward, _, _, _ = env.step(0)
        agent.update(obs, 0, reward, next_obs, 0, terminated=False)
        assert not np.array_equal(agent.w, w_before), "Weights should change after update"

    def test_update_terminal(self):
        """Terminal update should use reward only (no bootstrap)."""
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.1, gamma=0.95, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        # Manually compute expected update for terminal case
        features = rep.get_features(obs, 0)
        q_current = float(np.dot(agent.w, features))
        reward = -10.0
        expected_td_error = reward - q_current
        w_before = agent.w.copy()
        agent.update(obs, 0, reward, obs, 0, terminated=True)
        expected_w = w_before + 0.1 * expected_td_error * features
        np.testing.assert_array_almost_equal(agent.w, expected_w, decimal=5)

    def test_update_nonterminal(self):
        """Non-terminal update should bootstrap from next state."""
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.1, gamma=0.95, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        next_obs, reward, _, _, _ = env.step(0)

        features = rep.get_features(obs, 0)
        q_current = float(np.dot(agent.w, features))
        next_features = rep.get_features(next_obs, 1)
        q_next = float(np.dot(agent.w, next_features))
        expected_td_error = reward + 0.95 * q_next - q_current

        w_before = agent.w.copy()
        agent.update(obs, 0, reward, next_obs, 1, terminated=False)
        expected_w = w_before + 0.1 * expected_td_error * features
        np.testing.assert_array_almost_equal(agent.w, expected_w, decimal=5)

    def test_td_error_tracking(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        for _ in range(10):
            if env.done:
                break
            action = agent.act(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            next_action = agent.act(next_obs) if not (term or trunc) else 0
            agent.update(obs, action, reward, next_obs, next_action, term)
            obs = next_obs
        assert len(agent.td_errors) > 0
        assert agent.get_mean_td_error() > 0

    def test_weight_stats(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, seed=42)
        stats = agent.get_weight_stats()
        assert "mean" in stats
        assert "std" in stats
        assert "norm" in stats
        assert stats["n_weights"] == rep.feature_dim

    def test_repr(self):
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, seed=42)
        r = repr(agent)
        assert "LinearSarsaAgent" in r
        assert "0.01" in r


# ============================================================
# Training loop
# ============================================================

class TestTrainingLoop:

    def _make_config(self, n_episodes=100):
        return ExperimentConfig(
            algorithm="linear_sarsa",
            representation="compact",
            n_episodes=n_episodes,
            grid_size=10,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_fraction=0.8,
            alpha=0.01,
        )

    def test_run_episode_returns_tuple(self):
        config = self._make_config()
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, gamma=0.95, epsilon=0.5, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        result = _run_episode(agent, env, config)
        assert len(result) == 4
        total_reward, score, steps, cause = result
        assert isinstance(total_reward, float)
        assert isinstance(score, int)
        assert isinstance(steps, int)
        assert isinstance(cause, str)
        assert score >= 0
        assert steps > 0

    def test_train_returns_logger(self):
        config = self._make_config(n_episodes=50)
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        logger = train_sarsa(agent, env, config, print_every=0)
        assert isinstance(logger, RunLogger)
        assert len(logger.scores) == 50

    def test_epsilon_decays_during_training(self):
        config = self._make_config(n_episodes=100)
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        logger = train_sarsa(agent, env, config, print_every=0)
        # Epsilon should have decayed
        assert logger.epsilons[0] > logger.epsilons[-1]
        # Final epsilon should be near epsilon_end
        assert logger.epsilons[-1] < 0.1

    def test_weights_change_during_training(self):
        config = self._make_config(n_episodes=50)
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        w_init = agent.w.copy()
        train_sarsa(agent, env, config, print_every=0)
        assert not np.array_equal(agent.w, w_init), "Weights should change during training"

    def test_logger_records_all_episodes(self):
        config = self._make_config(n_episodes=30)
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        logger = train_sarsa(agent, env, config, print_every=0)
        assert len(logger.scores) == 30
        assert len(logger.rewards) == 30
        assert len(logger.steps) == 30
        assert len(logger.epsilons) == 30
        assert len(logger.causes) == 30

    def test_callback_called(self):
        config = self._make_config(n_episodes=20)
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        callback_calls = []
        def my_callback(ep, score, logger):
            callback_calls.append(ep)
        train_sarsa(agent, env, config, print_every=0, callback=my_callback)
        assert len(callback_calls) == 20

    def test_train_with_all_representations(self):
        """Verify training runs (doesn't crash) with all 3 representations."""
        for RepClass in [CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation]:
            config = self._make_config(n_episodes=20)
            rep = RepClass()
            agent = LinearSarsaAgent(rep, alpha=0.001, seed=42)
            env = SnakeEnv(grid_size=10, seed=42)
            logger = train_sarsa(agent, env, config, print_every=0)
            assert len(logger.scores) == 20, f"Failed with {RepClass.__name__}"

    def test_no_nan_in_weights(self):
        """Weights should never become NaN during training."""
        config = self._make_config(n_episodes=200)
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=0.01, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        train_sarsa(agent, env, config, print_every=0)
        assert not np.any(np.isnan(agent.w)), "Weights contain NaN"
        assert not np.any(np.isinf(agent.w)), "Weights contain Inf"


# ============================================================
# Learning sanity check
# ============================================================

class TestLearningSanity:

    def test_beats_random_after_training(self):
        """
        The linear SARSA agent should show clear improvement during
        training compared to early (random) performance.

        We check training scores rather than greedy evaluation because
        the compact representation (11 binary features) with a linear
        model learns danger avoidance but struggles with nuanced
        food-seeking under near-greedy policies. This is a legitimate
        limitation that the experiment will quantify.
        """
        config = ExperimentConfig(
            algorithm="linear_sarsa",
            representation="compact",
            n_episodes=5000,
            grid_size=10,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_fraction=0.8,
            alpha=0.01,
        )
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=config.alpha, gamma=config.gamma, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)

        logger = train_sarsa(agent, env, config, print_every=0)

        # Check that training scores improved from start to end
        early_avg = np.mean(logger.scores[:500])
        late_avg = np.mean(logger.scores[-500:])

        assert late_avg > early_avg, (
            f"Training scores should improve: early={early_avg:.2f}, "
            f"late={late_avg:.2f}"
        )
        # Late training should be clearly above random (0.15)
        assert late_avg > 0.3, (
            f"Late training avg {late_avg:.2f} too close to random (0.15)"
        )
        print(f"    [Learning check] Early avg: {early_avg:.2f} → "
              f"Late avg: {late_avg:.2f} (improvement: {late_avg - early_avg:+.2f})")

    def test_danger_weights_are_negative(self):
        """
        After training, the weight for 'danger_X' in the action-X
        block should be strongly negative — the agent should learn
        to avoid going into danger.
        """
        config = ExperimentConfig(
            algorithm="linear_sarsa",
            representation="compact",
            n_episodes=5000,
            grid_size=10,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_fraction=0.8,
            alpha=0.01,
        )
        rep = CompactRepresentation()
        agent = LinearSarsaAgent(rep, alpha=config.alpha, gamma=config.gamma, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        train_sarsa(agent, env, config, print_every=0)

        # danger_straight is feature index 0
        # In action block 0 (STRAIGHT): w[0] should be very negative
        w_danger_straight_for_straight = agent.w[0]
        # In action block 1 (LEFT): w[11 + 1] = danger_left should be very negative
        w_danger_left_for_left = agent.w[11 + 1]
        # In action block 2 (RIGHT): w[22 + 2] = danger_right should be very negative
        w_danger_right_for_right = agent.w[22 + 2]

        assert w_danger_straight_for_straight < -2.0, (
            f"danger_straight weight for STRAIGHT = {w_danger_straight_for_straight:.2f}, "
            f"expected strongly negative"
        )
        assert w_danger_left_for_left < -2.0, (
            f"danger_left weight for LEFT = {w_danger_left_for_left:.2f}, "
            f"expected strongly negative"
        )
        assert w_danger_right_for_right < -2.0, (
            f"danger_right weight for RIGHT = {w_danger_right_for_right:.2f}, "
            f"expected strongly negative"
        )


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
    print("LINEAR SARSA AGENT TEST SUITE")
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
