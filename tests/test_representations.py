"""
Tests for state representation extractors.

Run with: python tests/test_representations.py
"""

import sys
import os
import time
import traceback
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from snake_rl.env.snake_env import SnakeEnv, Action, UP, DOWN, LEFT_DIR, RIGHT_DIR
from snake_rl.representations.features import (
    CompactRepresentation,
    LocalNeighborhoodRepresentation,
    ExtendedRepresentation,
    get_representation,
)


# ============================================================
# Compact Representation Tests
# ============================================================

class TestCompactRepresentation:

    def test_state_dim(self):
        rep = CompactRepresentation()
        assert rep.state_dim == 11

    def test_feature_dim(self):
        rep = CompactRepresentation()
        assert rep.feature_dim == 33  # 11 * 3 actions

    def test_output_shape(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = CompactRepresentation()
        feats = rep.get_state_features(obs)
        assert feats.shape == (11,)
        assert feats.dtype == np.float32

    def test_sa_feature_shape(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = CompactRepresentation()
        for a in range(3):
            sa = rep.get_features(obs, a)
            assert sa.shape == (33,)

    def test_sa_features_one_hot_structure(self):
        """Only one action block should be nonzero."""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = CompactRepresentation()
        state_feats = rep.get_state_features(obs)

        for a in range(3):
            sa = rep.get_features(obs, a)
            # Check the active block matches state features
            start = a * 11
            np.testing.assert_array_equal(sa[start:start + 11], state_feats)
            # Check other blocks are zero
            for other_a in range(3):
                if other_a != a:
                    other_start = other_a * 11
                    assert np.all(sa[other_start:other_start + 11] == 0)

    def test_features_are_binary(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = CompactRepresentation()
        feats = rep.get_state_features(obs)
        for val in feats:
            assert val in (0.0, 1.0), f"Expected binary, got {val}"

    def test_direction_one_hot(self):
        """Exactly one direction bit should be active."""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = CompactRepresentation()
        feats = rep.get_state_features(obs)
        dir_bits = feats[3:7]  # dir_up, dir_down, dir_left, dir_right
        assert np.sum(dir_bits) == 1.0, "Exactly one direction should be active"

    def test_initial_direction_is_right(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = CompactRepresentation()
        feats = rep.get_state_features(obs)
        # dir_up=3, dir_down=4, dir_left=5, dir_right=6
        assert feats[6] == 1.0, "Initial direction should be right"
        assert feats[3] == 0.0 and feats[4] == 0.0 and feats[5] == 0.0

    def test_danger_detected_near_wall(self):
        """Snake near top wall facing up should have danger straight."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()
        # Manually position snake near top wall facing up
        env.snake.clear()
        env.snake.extend([(5, 0), (5, 1), (5, 2)])
        env.direction = UP
        env.food = (8, 8)  # far away
        obs = env._get_obs()
        rep = CompactRepresentation()
        feats = rep.get_state_features(obs)
        assert feats[0] == 1.0, "Should detect danger straight (wall above)"

    def test_food_direction_signals(self):
        """Food to the upper-right should activate food_up and food_right."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()
        env.snake.clear()
        env.snake.extend([(5, 5), (4, 5), (3, 5)])
        env.direction = RIGHT_DIR
        env.food = (8, 2)  # upper-right
        obs = env._get_obs()
        rep = CompactRepresentation()
        feats = rep.get_state_features(obs)
        # food_up=7, food_down=8, food_left=9, food_right=10
        assert feats[7] == 1.0, "Food is above"
        assert feats[8] == 0.0, "Food is not below"
        assert feats[9] == 0.0, "Food is not left"
        assert feats[10] == 1.0, "Food is right"

    def test_no_danger_in_open_space(self):
        """Snake in the middle with open space should have no danger."""
        env = SnakeEnv(grid_size=20, seed=42)
        env.reset()
        # Default start: center of 20x20, facing right, plenty of room
        obs = env._get_obs()
        rep = CompactRepresentation()
        feats = rep.get_state_features(obs)
        assert feats[0] == 0.0, "No danger straight"
        assert feats[1] == 0.0, "No danger left"
        assert feats[2] == 0.0, "No danger right"


# ============================================================
# Local Neighborhood Representation Tests
# ============================================================

class TestLocalNeighborhoodRepresentation:

    def test_state_dim(self):
        rep = LocalNeighborhoodRepresentation()
        assert rep.state_dim == 109  # 5*5*4 + 4 + 5

    def test_feature_dim(self):
        rep = LocalNeighborhoodRepresentation()
        assert rep.feature_dim == 109 * 3

    def test_output_shape(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = LocalNeighborhoodRepresentation()
        feats = rep.get_state_features(obs)
        assert feats.shape == (109,)
        assert feats.dtype == np.float32

    def test_window_cells_one_hot(self):
        """Each cell in the window should have exactly one category active."""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = LocalNeighborhoodRepresentation()
        feats = rep.get_state_features(obs)
        # First 100 features are 25 cells × 4 categories
        for i in range(25):
            cell = feats[i * 4:(i + 1) * 4]
            assert np.sum(cell) == 1.0, f"Cell {i} should have exactly one active category, got {cell}"

    def test_center_cell_is_empty(self):
        """Center of window is the head — classified as empty (since head is excluded)."""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = LocalNeighborhoodRepresentation()
        feats = rep.get_state_features(obs)
        # Center cell is index 12 (row 2, col 2 of 5x5)
        center = feats[12 * 4:(12 + 1) * 4]  # [empty, food, wall, body]
        assert center[0] == 1.0, "Center cell (head) should be classified as empty"

    def test_wall_detection_near_edge(self):
        """Snake near corner should detect walls in the window."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()
        env.snake.clear()
        env.snake.extend([(0, 0), (1, 0), (2, 0)])
        env.direction = LEFT_DIR
        env.food = (5, 5)
        obs = env._get_obs()
        rep = LocalNeighborhoodRepresentation()
        feats = rep.get_state_features(obs)
        # Top-left corner of window at head (0,0): cells at (-2,-2) to (2,2)
        # Cell (0,0) in window = (-2,-2) in grid → should be wall
        first_cell = feats[0:4]
        assert first_cell[2] == 1.0, "Off-grid cell should be wall"

    def test_length_bucket_initial(self):
        """Initial snake of length 3 should be in first bucket."""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = LocalNeighborhoodRepresentation()
        feats = rep.get_state_features(obs)
        # Length buckets are the last 5 features
        buckets = feats[-5:]
        assert buckets[0] == 1.0, "Length 3 should be in bucket [1-5]"
        assert np.sum(buckets) == 1.0, "Only one bucket should be active"

    def test_custom_window_size(self):
        rep = LocalNeighborhoodRepresentation(window_size=3)
        expected = 3 * 3 * 4 + 4 + 5  # 36 + 4 + 5 = 45
        assert rep.state_dim == expected


# ============================================================
# Extended Representation Tests
# ============================================================

class TestExtendedRepresentation:

    def test_state_dim_without_interactions(self):
        rep = ExtendedRepresentation(include_interactions=False)
        assert rep.state_dim == 126

    def test_state_dim_with_interactions(self):
        rep = ExtendedRepresentation(include_interactions=True)
        assert rep.state_dim == 138  # 126 + 12

    def test_output_shape(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = ExtendedRepresentation()
        feats = rep.get_state_features(obs)
        assert feats.shape == (126,)

    def test_output_shape_with_interactions(self):
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = ExtendedRepresentation(include_interactions=True)
        feats = rep.get_state_features(obs)
        assert feats.shape == (138,)

    def test_continuous_features_normalized(self):
        """Manhattan distance and wall distances should be in [0, 1]."""
        env = SnakeEnv(grid_size=10, seed=42)
        rep = ExtendedRepresentation()
        for _ in range(20):
            obs, _ = env.reset()
            feats = rep.get_state_features(obs)
            # Manhattan distance is at index 109 (after window + food_dir + length)
            manhattan = feats[109]
            assert 0.0 <= manhattan <= 1.0, f"Manhattan distance {manhattan} not normalized"
            # Wall distances at indices 114-117
            for i in range(114, 118):
                assert 0.0 <= feats[i] <= 1.0, f"Wall distance at index {i}: {feats[i]} not normalized"
            # Normalized length at index 118
            assert 0.0 <= feats[118] <= 1.0

    def test_wall_distances_correct(self):
        """Verify wall distance calculations for a known position."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()
        env.snake.clear()
        env.snake.extend([(3, 2), (2, 2), (1, 2)])
        env.direction = RIGHT_DIR
        env.food = (8, 8)
        obs = env._get_obs()
        rep = ExtendedRepresentation()
        feats = rep.get_state_features(obs)
        # Wall distances: up=2/9, down=7/9, left=3/9, right=6/9
        # Indices 114-117
        assert abs(feats[114] - 2.0 / 9.0) < 1e-5, f"Wall up: expected {2/9:.4f}, got {feats[114]:.4f}"
        assert abs(feats[115] - 7.0 / 9.0) < 1e-5, f"Wall down: expected {7/9:.4f}, got {feats[115]:.4f}"
        assert abs(feats[116] - 3.0 / 9.0) < 1e-5, f"Wall left: expected {3/9:.4f}, got {feats[116]:.4f}"
        assert abs(feats[117] - 6.0 / 9.0) < 1e-5, f"Wall right: expected {6/9:.4f}, got {feats[117]:.4f}"

    def test_body_distance_detection(self):
        """Snake with body to the left should detect it."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()
        # Create a long snake with body to the left of the head
        env.snake.clear()
        env.snake.extend([(5, 5), (5, 6), (5, 7), (4, 7), (3, 7), (3, 6), (3, 5)])
        env.direction = UP
        env.food = (8, 1)
        obs = env._get_obs()
        rep = ExtendedRepresentation()
        feats = rep.get_state_features(obs)
        # Body scan left (dx=-1, dy=0): should find body at (3,5), distance=2
        # Body distances are at indices 110-113: up, down, left, right
        left_body_dist = feats[112]  # left direction scan
        assert left_body_dist > 0.0, "Should detect body to the left"

    def test_danger_signals_present(self):
        """Danger signals should match compact representation logic."""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep_ext = ExtendedRepresentation()
        rep_compact = CompactRepresentation()

        ext_feats = rep_ext.get_state_features(obs)
        compact_feats = rep_compact.get_state_features(obs)

        # Danger signals in extended: last 3 before interactions (indices 123-125)
        ext_dangers = ext_feats[123:126]
        compact_dangers = compact_feats[0:3]

        np.testing.assert_array_equal(
            ext_dangers, compact_dangers,
            err_msg="Danger signals should match between representations"
        )

    def test_interactions_are_products(self):
        """Interaction features should be danger × food products."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()
        env.snake.clear()
        env.snake.extend([(5, 0), (5, 1), (5, 2)])  # near top wall
        env.direction = UP
        env.food = (8, 5)
        obs = env._get_obs()

        rep = ExtendedRepresentation(include_interactions=True)
        feats = rep.get_state_features(obs)

        # Danger: straight=1 (wall), left=?, right=?
        # Food: up=0, down=1, left=0, right=1
        danger_straight = feats[123]
        food_down = feats[101]
        food_right = feats[103]

        # First interaction: danger_straight × food_up (index 126)
        # Second: danger_straight × food_down (index 127)
        assert feats[126] == danger_straight * feats[100]  # × food_up
        assert feats[127] == danger_straight * food_down


# ============================================================
# Cross-representation Tests
# ============================================================

class TestCrossRepresentation:

    def test_all_representations_run(self):
        """All representations should produce valid output for a normal game state."""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        for name in ["compact", "local", "extended"]:
            rep = get_representation(name)
            feats = rep.get_state_features(obs)
            assert feats.ndim == 1
            assert len(feats) == rep.state_dim
            assert feats.dtype == np.float32
            assert not np.any(np.isnan(feats)), f"{name} has NaN features"
            assert not np.any(np.isinf(feats)), f"{name} has Inf features"

    def test_representations_stable_over_episode(self):
        """Run a full episode and verify no NaN/Inf for any representation."""
        env = SnakeEnv(grid_size=10, seed=42)
        reps = [
            CompactRepresentation(),
            LocalNeighborhoodRepresentation(),
            ExtendedRepresentation(),
        ]
        obs, _ = env.reset()
        steps = 0
        while not env.done and steps < 500:
            for rep in reps:
                feats = rep.get_state_features(obs)
                assert not np.any(np.isnan(feats)), f"NaN at step {steps}"
                assert not np.any(np.isinf(feats)), f"Inf at step {steps}"
            action = env.rng.integers(3)
            obs, _, _, _, _ = env.step(action)
            steps += 1

    def test_sa_features_different_per_action(self):
        """State-action features should differ for different actions."""
        env = SnakeEnv(grid_size=10, seed=42)
        obs, _ = env.reset()
        rep = CompactRepresentation()
        f0 = rep.get_features(obs, 0)
        f1 = rep.get_features(obs, 1)
        f2 = rep.get_features(obs, 2)
        # All should differ (different active blocks)
        assert not np.array_equal(f0, f1)
        assert not np.array_equal(f1, f2)
        assert not np.array_equal(f0, f2)

    def test_get_representation_registry(self):
        for name in ["compact", "local", "extended", "extended_interactions"]:
            rep = get_representation(name)
            assert rep.state_dim > 0
            assert rep.feature_dim > 0

    def test_invalid_representation_name(self):
        try:
            get_representation("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_performance_benchmark(self):
        """Feature extraction should be fast enough for training."""
        env = SnakeEnv(grid_size=20, seed=42)
        obs, _ = env.reset()
        reps = {
            "compact": CompactRepresentation(),
            "local": LocalNeighborhoodRepresentation(),
            "extended": ExtendedRepresentation(),
        }
        n_calls = 5000
        for name, rep in reps.items():
            start = time.time()
            for _ in range(n_calls):
                rep.get_state_features(obs)
            elapsed = time.time() - start
            per_call_us = (elapsed / n_calls) * 1e6
            print(f"    [{name}] {per_call_us:.1f} µs/call ({n_calls} calls in {elapsed:.3f}s)")
            # Should be well under 1ms per call for training to be feasible
            assert per_call_us < 1000, f"{name} too slow: {per_call_us:.0f} µs/call"


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
    print("STATE REPRESENTATION TEST SUITE")
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
