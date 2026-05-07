"""
Tests for MLP SARSA agent.

Requires PyTorch. Run with: python tests/test_mlp_sarsa.py

If PyTorch is not installed, all tests are skipped gracefully.
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from snake_rl.agents.mlp_sarsa import MLPSarsaAgent, QNetwork
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.representations.features import (
    CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation,
)
from snake_rl.agents.train import train_sarsa
from snake_rl.utils.experiment import ExperimentConfig


class TestQNetwork:

    def test_output_shape(self):
        net = QNetwork(input_dim=11, hidden_dims=(64,), n_actions=3)
        x = torch.randn(1, 11)
        out = net(x)
        assert out.shape == (1, 3)

    def test_batch_forward(self):
        net = QNetwork(input_dim=11, hidden_dims=(128,), n_actions=3)
        x = torch.randn(32, 11)
        out = net(x)
        assert out.shape == (32, 3)

    def test_different_hidden_dims(self):
        for h in [32, 64, 128, 256]:
            net = QNetwork(input_dim=109, hidden_dims=(h,), n_actions=3)
            x = torch.randn(1, 109)
            out = net(x)
            assert out.shape == (1, 3)

    def test_param_count(self):
        net = QNetwork(input_dim=11, hidden_dims=(128,), n_actions=3)
        n_params = sum(p.numel() for p in net.parameters())
        # 11*128 + 128 + 128*3 + 3 = 1408 + 128 + 384 + 3 = 1923
        assert n_params == 1923


class TestMLPSarsaAgent:

    def test_initialization(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, hidden_dims=(64,), seed=42)
        assert agent.hidden_dims == (64,)
        assert agent.gamma == 0.95
        stats = agent.get_weight_stats()
        assert stats["n_params"] > 0

    def test_q_values_shape(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        q = agent.q_values(obs)
        assert q.shape == (3,)

    def test_q_value_single(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        q_all = agent.q_values(obs)
        for a in range(3):
            q_single = agent.q_value(obs, a)
            assert abs(q_single - q_all[a]) < 1e-5

    def test_act_returns_valid(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, epsilon=0.5, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        for _ in range(50):
            assert agent.act(obs) in [0, 1, 2]

    def test_act_greedy(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, epsilon=0.0, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        q = agent.q_values(obs)
        best = int(np.argmax(q))
        # Should consistently pick the best action
        for _ in range(10):
            assert agent.act(obs) == best

    def test_update_changes_params(self):
        rep = CompactRepresentation()
        # min_buffer_size=1, batch_size=1 so the very first update triggers a gradient step
        agent = MLPSarsaAgent(rep, alpha=0.01, min_buffer_size=1, batch_size=1, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        params_before = [p.data.clone() for p in agent.q_net.parameters()]
        next_obs, reward, _, _, _ = env.step(0)
        agent.update(obs, 0, reward, next_obs, 0, terminated=False)
        params_after = [p.data for p in agent.q_net.parameters()]
        changed = any(
            not torch.equal(b, a)
            for b, a in zip(params_before, params_after)
        )
        assert changed, "Parameters should change after update"

    def test_update_terminal(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, alpha=0.01, min_buffer_size=1, batch_size=1, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        # Terminal update should not crash
        td_err = agent.update(obs, 0, -10.0, obs, 0, terminated=True)
        assert isinstance(td_err, float)

    def test_td_error_tracking(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, alpha=0.001, min_buffer_size=1, batch_size=1, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        for _ in range(5):
            if env.done:
                break
            action = agent.act(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            next_action = agent.act(next_obs) if not (term or trunc) else 0
            agent.update(obs, action, reward, next_obs, next_action, term)
            obs = next_obs
        assert len(agent.td_errors) > 0
        assert agent.get_mean_td_error() > 0

    def test_target_network_init(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, use_target_network=True, seed=42)
        assert agent.target_net is not None
        # Target should have same params as q_net initially
        for p1, p2 in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.equal(p1.data, p2.data)

    def test_target_network_update(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(
            rep, use_target_network=True, target_update_freq=5,
            alpha=0.01, min_buffer_size=1, batch_size=1, seed=42,
        )
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        # Do some updates to change q_net params
        for _ in range(3):
            if env.done:
                obs, _ = env.reset()
            next_obs, reward, term, trunc, _ = env.step(0)
            agent.update(obs, 0, reward, next_obs, 0, term or trunc)
            obs = next_obs if not (term or trunc) else env.reset()[0]

        # Target should still be different from q_net
        same = all(
            torch.equal(p1.data, p2.data)
            for p1, p2 in zip(agent.q_net.parameters(), agent.target_net.parameters())
        )
        assert not same, "Target net should differ from q_net after updates"

        # Trigger target update
        for _ in range(5):
            agent.on_episode_end()
        # Now they should match
        for p1, p2 in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.equal(p1.data, p2.data), "Target should sync after update freq"

    def test_train_compact(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, hidden_dims=(64,), alpha=0.001,
                              min_buffer_size=32, batch_size=32, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        config = ExperimentConfig(
            algorithm="mlp_sarsa", representation="compact",
            n_episodes=50, grid_size=10, alpha=0.001,
        )
        logger = train_sarsa(agent, env, config, print_every=0)
        assert len(logger.scores) == 50

    def test_train_all_representations(self):
        for RepClass in [CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation]:
            rep = RepClass()
            agent = MLPSarsaAgent(rep, hidden_dims=(32,), alpha=0.001,
                                  min_buffer_size=32, batch_size=32, seed=42)
            env = SnakeEnv(grid_size=10, seed=42)
            config = ExperimentConfig(
                algorithm="mlp_sarsa", representation="test",
                n_episodes=20, grid_size=10, alpha=0.001,
            )
            logger = train_sarsa(agent, env, config, print_every=0)
            assert len(logger.scores) == 20, f"Failed with {RepClass.__name__}"

    def test_no_nan_in_params(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, alpha=0.001, min_buffer_size=64, batch_size=64, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        config = ExperimentConfig(
            algorithm="mlp_sarsa", representation="compact",
            n_episodes=200, grid_size=10, alpha=0.001,
        )
        train_sarsa(agent, env, config, print_every=0)
        for name, p in agent.q_net.named_parameters():
            assert not torch.any(torch.isnan(p.data)), f"NaN in {name}"
            assert not torch.any(torch.isinf(p.data)), f"Inf in {name}"

    def test_repr(self):
        rep = CompactRepresentation()
        agent = MLPSarsaAgent(rep, hidden_dims=(128,), seed=42)
        r = repr(agent)
        assert "MLPSarsaAgent" in r
        assert "hidden=(128,)" in r


class TestMLPLearningSanity:

    def test_training_improves(self):
        rep = CompactRepresentation()
        # Smaller min_buffer_size so learning starts within the 3000-episode budget
        agent = MLPSarsaAgent(rep, hidden_dims=(64,), alpha=0.001,
                              gamma=0.95, min_buffer_size=200, batch_size=32, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        config = ExperimentConfig(
            algorithm="mlp_sarsa", representation="compact",
            n_episodes=3000, grid_size=10, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay_fraction=0.8, alpha=0.001,
        )
        logger = train_sarsa(agent, env, config, print_every=0)
        early = np.mean(logger.scores[:300])
        late = np.mean(logger.scores[-300:])
        assert late > early, f"Scores should improve: early={early:.2f}, late={late:.2f}"
        print(f"    [MLP] Early: {early:.2f} → Late: {late:.2f}")


# ============================================================
# Runner
# ============================================================

def run_all_tests():
    if not HAS_TORCH:
        print("=" * 70)
        print("MLP SARSA TEST SUITE — SKIPPED (PyTorch not installed)")
        print("Install with: pip install torch")
        print("=" * 70)
        return True

    test_module = sys.modules[__name__]
    test_classes = [
        obj for name, obj in vars(test_module).items()
        if isinstance(obj, type) and name.startswith("Test")
    ]
    total = passed = failed = 0
    errors = []
    print("=" * 70)
    print("MLP SARSA TEST SUITE")
    print("=" * 70)
    start_time = time.time()
    for cls in test_classes:
        print(f"\n--- {cls.__name__} ---")
        for method_name in [m for m in dir(cls) if m.startswith("test_")]:
            total += 1
            try:
                getattr(cls(), method_name)()
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
    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed, {total} total ({elapsed:.2f}s)")
    print(f"{'='*70}")
    if errors:
        print("\n--- FAILURES ---\n")
        for cn, mn, e, tb in errors:
            print(f"{cn}.{mn}:\n  {e}\n")
    return failed == 0

if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
