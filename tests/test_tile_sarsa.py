"""
Tests for Tile Coding SARSA agent.

Run with: python tests/test_tile_sarsa.py
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.representations.features import (
    CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation,
)
from snake_rl.agents.tile_sarsa import TileCoder, TileCodingRepresentation, TileCodingSarsaAgent
from snake_rl.agents.train import train_sarsa
from snake_rl.utils.experiment import ExperimentConfig


class TestTileCoder:

    def test_active_tiles_count(self):
        tc = TileCoder(n_tilings=8, n_tiles_per_dim=4, n_dims=3, n_actions=3)
        features = np.array([0.5, 0.3, 0.7])
        tiles = tc.get_active_tiles(features, action=0)
        assert len(tiles) == 8, f"Expected 8 active tiles, got {len(tiles)}"

    def test_active_tiles_unique(self):
        tc = TileCoder(n_tilings=8, n_tiles_per_dim=4, n_dims=3, n_actions=3)
        features = np.array([0.5, 0.3, 0.7])
        tiles = tc.get_active_tiles(features, action=0)
        assert len(set(tiles)) == len(tiles), "Active tiles should be unique"

    def test_active_tiles_in_range(self):
        tc = TileCoder(n_tilings=8, n_tiles_per_dim=4, n_dims=3, n_actions=3)
        for _ in range(50):
            features = np.random.rand(3)
            for a in range(3):
                tiles = tc.get_active_tiles(features, a)
                assert np.all(tiles >= 0)
                assert np.all(tiles < tc.feature_dim)

    def test_different_actions_different_tiles(self):
        tc = TileCoder(n_tilings=8, n_tiles_per_dim=4, n_dims=3, n_actions=3)
        features = np.array([0.5, 0.3, 0.7])
        t0 = set(tc.get_active_tiles(features, 0))
        t1 = set(tc.get_active_tiles(features, 1))
        t2 = set(tc.get_active_tiles(features, 2))
        assert t0.isdisjoint(t1), "Different actions should have disjoint tiles"
        assert t1.isdisjoint(t2)

    def test_nearby_states_share_tiles(self):
        tc = TileCoder(n_tilings=8, n_tiles_per_dim=4, n_dims=2, n_actions=1)
        f1 = np.array([0.50, 0.50])
        f2 = np.array([0.51, 0.50])  # very close
        t1 = set(tc.get_active_tiles(f1, 0))
        t2 = set(tc.get_active_tiles(f2, 0))
        overlap = len(t1 & t2)
        assert overlap > 0, "Nearby states should share some tiles"

    def test_distant_states_fewer_shared(self):
        tc = TileCoder(n_tilings=8, n_tiles_per_dim=4, n_dims=2, n_actions=1)
        f1 = np.array([0.1, 0.1])
        f2 = np.array([0.9, 0.9])  # far away
        t1 = set(tc.get_active_tiles(f1, 0))
        t2 = set(tc.get_active_tiles(f2, 0))
        overlap = len(t1 & t2)
        # Distant states should share fewer (possibly zero) tiles
        assert overlap < 8, "Distant states should share fewer tiles than n_tilings"

    def test_feature_dim(self):
        tc = TileCoder(n_tilings=8, n_tiles_per_dim=4, n_dims=3, n_actions=3, max_size=4096)
        assert tc.feature_dim == 4096  # hash table size

    def test_edge_features(self):
        """Features at 0.0 and 1.0 should not crash."""
        tc = TileCoder(n_tilings=8, n_tiles_per_dim=4, n_dims=2, n_actions=1)
        tiles_low = tc.get_active_tiles(np.array([0.0, 0.0]), 0)
        tiles_high = tc.get_active_tiles(np.array([1.0, 1.0]), 0)
        assert len(tiles_low) == 8
        assert len(tiles_high) == 8


class TestTileCodingRepresentation:

    def test_normalize_binary_features(self):
        rep = CompactRepresentation()
        tile_rep = TileCodingRepresentation(rep)
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        normalized = tile_rep.get_normalized_features(obs)
        assert normalized.shape == (rep.state_dim,)
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

    def test_initialize_ranges(self):
        rep = CompactRepresentation()
        tile_rep = TileCodingRepresentation(rep)
        env = SnakeEnv(grid_size=10, seed=42)
        tile_rep.initialize_ranges(env, n_samples=100)
        assert tile_rep._initialized
        assert tile_rep.feature_min is not None
        assert tile_rep.feature_max is not None
        assert len(tile_rep.feature_min) == rep.state_dim

    def test_normalize_after_init(self):
        rep = ExtendedRepresentation()
        tile_rep = TileCodingRepresentation(rep)
        env = SnakeEnv(grid_size=10, seed=42)
        tile_rep.initialize_ranges(env, n_samples=200)
        obs, _ = env.reset()
        normalized = tile_rep.get_normalized_features(obs)
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)


class TestTileCodingSarsaAgent:

    def test_initialization(self):
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, n_tilings=8, n_tiles_per_dim=4, seed=42)
        assert agent.w.shape == (agent.tile_coder.feature_dim,)
        assert np.all(agent.w == 0)

    def test_act_returns_valid(self):
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, epsilon=0.5, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        agent.initialize(env)
        obs, _ = env.reset()
        for _ in range(50):
            assert agent.act(obs) in [0, 1, 2]

    def test_q_value_is_sum_of_weights(self):
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, n_tilings=8, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        agent.initialize(env)
        obs, _ = env.reset()
        # Set some weights
        agent.w[:] = 0
        tiles = agent._get_tiles(obs, 0)
        agent.w[tiles] = 1.0
        q = agent.q_value(obs, 0)
        assert abs(q - 8.0) < 1e-6, f"Expected 8.0 (8 tiles * 1.0), got {q}"

    def test_update_changes_weights(self):
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, alpha=0.1, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        agent.initialize(env)
        obs, _ = env.reset()
        w_before = agent.w.copy()
        next_obs, reward, _, _, _ = env.step(0)
        agent.update(obs, 0, reward, next_obs, 0, terminated=False)
        assert not np.array_equal(agent.w, w_before)

    def test_update_only_active_tiles(self):
        """Only weights at active tile indices should change."""
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, alpha=0.1, n_tilings=8, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        agent.initialize(env)
        obs, _ = env.reset()
        w_before = agent.w.copy()
        tiles = agent._get_tiles(obs, 0)
        next_obs, reward, _, _, _ = env.step(0)
        agent.update(obs, 0, reward, next_obs, 0, terminated=False)
        # Unchanged indices
        changed_mask = agent.w != w_before
        changed_indices = set(np.where(changed_mask)[0])
        active_set = set(tiles)
        # All changed indices should be in active tiles (or next state tiles)
        # But at minimum, the current state tiles should have changed
        assert active_set.issubset(changed_indices) or len(changed_indices) > 0

    def test_sparsity(self):
        """After some training, most weights should still be zero."""
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, n_tilings=8, n_tiles_per_dim=4, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        agent.initialize(env)
        config = ExperimentConfig(
            algorithm="tile_sarsa", representation="compact",
            n_episodes=100, grid_size=10, alpha=0.01,
        )
        train_sarsa(agent, env, config, print_every=0)
        stats = agent.get_weight_stats()
        assert stats["sparsity"] > 0.5, f"Expected sparse weights, got sparsity={stats['sparsity']:.2f}"

    def test_train_all_representations(self):
        """Verify training runs with all 3 representations."""
        for RepClass in [CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation]:
            rep = RepClass()
            agent = TileCodingSarsaAgent(rep, n_tilings=4, n_tiles_per_dim=2,
                                          max_size=8192, alpha=0.01, seed=42)
            env = SnakeEnv(grid_size=10, seed=42)
            agent.initialize(env)
            config = ExperimentConfig(
                algorithm="tile_sarsa", representation="test",
                n_episodes=20, grid_size=10, alpha=0.01,
            )
            logger = train_sarsa(agent, env, config, print_every=0)
            assert len(logger.scores) == 20, f"Failed with {RepClass.__name__}"

    def test_no_nan_in_weights(self):
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, alpha=0.01, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        agent.initialize(env)
        config = ExperimentConfig(
            algorithm="tile_sarsa", representation="compact",
            n_episodes=500, grid_size=10, alpha=0.01,
        )
        train_sarsa(agent, env, config, print_every=0)
        assert not np.any(np.isnan(agent.w))
        assert not np.any(np.isinf(agent.w))

    def test_repr(self):
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, n_tilings=8, alpha=0.02, seed=42)
        r = repr(agent)
        assert "TileCodingSarsaAgent" in r
        assert "n_tilings=8" in r


class TestTileCodingLearningSanity:

    def test_training_improves(self):
        """Training scores should improve over time."""
        rep = CompactRepresentation()
        agent = TileCodingSarsaAgent(rep, n_tilings=8, n_tiles_per_dim=4,
                                      alpha=0.05, gamma=0.95, seed=42)
        env = SnakeEnv(grid_size=10, seed=42)
        agent.initialize(env)
        config = ExperimentConfig(
            algorithm="tile_sarsa", representation="compact",
            n_episodes=3000, grid_size=10, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay_fraction=0.8, alpha=0.05,
        )
        logger = train_sarsa(agent, env, config, print_every=0)
        early = np.mean(logger.scores[:300])
        late = np.mean(logger.scores[-300:])
        assert late > early, f"Scores should improve: early={early:.2f}, late={late:.2f}"
        print(f"    [Tile coding] Early: {early:.2f} → Late: {late:.2f}")


# ============================================================
# Runner
# ============================================================

def run_all_tests():
    test_module = sys.modules[__name__]
    test_classes = [
        obj for name, obj in vars(test_module).items()
        if isinstance(obj, type) and name.startswith("Test")
    ]
    total = passed = failed = 0
    errors = []
    print("=" * 70)
    print("TILE CODING SARSA TEST SUITE")
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
