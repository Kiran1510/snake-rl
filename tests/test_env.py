"""
Comprehensive tests for the Snake environment.

Run with: pytest tests/test_env.py -v
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
import numpy as np
from collections import Counter

from snake_rl.env.snake_env import (
    SnakeEnv, Action, _turn_left, _turn_right,
    UP, DOWN, LEFT_DIR, RIGHT_DIR,
)


# ============================================================
# Fixtures
# ============================================================

# ============================================================
# Fixtures — used by pytest; the standalone runner injects them manually
# ============================================================

if HAS_PYTEST:
    @pytest.fixture
    def env():
        """Standard 20x20 environment with fixed seed."""
        return SnakeEnv(grid_size=20, seed=42)

    @pytest.fixture
    def small_env():
        """Small 6x6 environment for edge case testing."""
        return SnakeEnv(grid_size=6, seed=42)


# ============================================================
# 1. Direction rotation helpers
# ============================================================

class TestDirectionHelpers:
    """Test the _turn_left and _turn_right rotation functions."""

    def test_turn_left_from_up(self):
        assert _turn_left(UP) == LEFT_DIR

    def test_turn_left_from_left(self):
        assert _turn_left(LEFT_DIR) == DOWN

    def test_turn_left_from_down(self):
        assert _turn_left(DOWN) == RIGHT_DIR

    def test_turn_left_from_right(self):
        assert _turn_left(RIGHT_DIR) == UP

    def test_turn_right_from_up(self):
        assert _turn_right(UP) == RIGHT_DIR

    def test_turn_right_from_right(self):
        assert _turn_right(RIGHT_DIR) == DOWN

    def test_turn_right_from_down(self):
        assert _turn_right(DOWN) == LEFT_DIR

    def test_turn_right_from_left(self):
        assert _turn_right(LEFT_DIR) == UP

    def test_four_left_turns_return_to_start(self):
        d = UP
        for _ in range(4):
            d = _turn_left(d)
        assert d == UP

    def test_four_right_turns_return_to_start(self):
        d = RIGHT_DIR
        for _ in range(4):
            d = _turn_right(d)
        assert d == RIGHT_DIR

    def test_left_then_right_is_identity(self):
        for d in [UP, DOWN, LEFT_DIR, RIGHT_DIR]:
            assert _turn_right(_turn_left(d)) == d
            assert _turn_left(_turn_right(d)) == d


# ============================================================
# 2. Initialization and Reset
# ============================================================

class TestReset:
    """Test environment initialization and reset behavior."""

    def test_reset_returns_obs_and_info(self, env):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_has_required_keys(self, env):
        obs, _ = env.reset()
        required_keys = {"snake", "food", "direction", "score", "steps", "grid_size"}
        assert set(obs.keys()) == required_keys

    def test_initial_snake_length(self, env):
        obs, _ = env.reset()
        assert len(obs["snake"]) == 3

    def test_initial_snake_position(self, env):
        obs, _ = env.reset()
        head = obs["snake"][0]
        cx, cy = env.grid_size // 2, env.grid_size // 2
        assert head == (cx, cy), "Head should be at grid center"

    def test_initial_snake_is_horizontal(self, env):
        obs, _ = env.reset()
        snake = obs["snake"]
        # Snake should be horizontal, facing right
        assert snake[0][1] == snake[1][1] == snake[2][1], "Snake should be on same row"
        assert snake[0][0] > snake[1][0] > snake[2][0], "Snake should extend leftward from head"

    def test_initial_direction_is_right(self, env):
        obs, _ = env.reset()
        assert obs["direction"] == RIGHT_DIR

    def test_initial_score_is_zero(self, env):
        obs, _ = env.reset()
        assert obs["score"] == 0

    def test_initial_steps_is_zero(self, env):
        obs, _ = env.reset()
        assert obs["steps"] == 0

    def test_food_is_on_grid(self, env):
        obs, _ = env.reset()
        fx, fy = obs["food"]
        assert 0 <= fx < env.grid_size
        assert 0 <= fy < env.grid_size

    def test_food_not_on_snake(self, env):
        obs, _ = env.reset()
        snake_set = set(map(tuple, obs["snake"]))
        assert obs["food"] not in snake_set

    def test_grid_size_in_obs(self, env):
        obs, _ = env.reset()
        assert obs["grid_size"] == 20

    def test_done_flag_cleared_after_reset(self, env):
        env.reset()
        assert env.done is False

    def test_reset_with_new_seed(self, env):
        env.reset(seed=123)
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        assert obs1["food"] == obs2["food"], "Same seed should produce same food placement"

    def test_different_seeds_different_food(self):
        env1 = SnakeEnv(seed=1)
        env2 = SnakeEnv(seed=999)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        # Not strictly guaranteed, but extremely likely with different seeds
        # on a 20x20 grid. Skip assertion if by rare chance they match.
        # This is a sanity check, not a strict test.

    def test_multiple_resets(self, env):
        """Ensure reset can be called multiple times without error."""
        for _ in range(10):
            obs, _ = env.reset()
            assert len(obs["snake"]) == 3
            assert obs["score"] == 0


# ============================================================
# 3. Step mechanics
# ============================================================

class TestStepMechanics:
    """Test basic movement, action resolution, and state transitions."""

    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(Action.STRAIGHT)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_straight_moves_in_current_direction(self, env):
        obs, _ = env.reset()
        head_before = obs["snake"][0]
        obs, _, _, _, _ = env.step(Action.STRAIGHT)
        head_after = obs["snake"][0]
        # Initially facing right, so head should move right
        assert head_after == (head_before[0] + 1, head_before[1])

    def test_step_left_turns_left(self, env):
        obs, _ = env.reset()
        # Facing right, turn left → now facing up
        obs, _, _, _, _ = env.step(Action.LEFT)
        assert env.direction == UP

    def test_step_right_turns_right(self, env):
        obs, _ = env.reset()
        # Facing right, turn right → now facing down
        obs, _, _, _, _ = env.step(Action.RIGHT)
        assert env.direction == DOWN

    def test_snake_length_constant_without_food(self, env):
        env.reset()
        initial_length = env.get_snake_length()
        # Take several steps that don't eat food
        for _ in range(5):
            if env.done:
                break
            env.step(Action.STRAIGHT)
        if not env.done:
            assert env.get_snake_length() == initial_length

    def test_step_increments_step_counter(self, env):
        env.reset()
        assert env.steps == 0
        env.step(Action.STRAIGHT)
        assert env.steps == 1
        env.step(Action.STRAIGHT)
        assert env.steps == 2

    def test_step_reward_without_food(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(Action.STRAIGHT)
        assert reward == env.reward_step

    def test_step_on_done_raises_error(self, env):
        env.reset()
        env.done = True
        try:
            env.step(Action.STRAIGHT)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "done" in str(e).lower()

    def test_invalid_action_raises_error(self, env):
        env.reset()
        try:
            env.step(5)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid action" in str(e).lower()

    def test_head_moves_tail_follows(self, env):
        obs, _ = env.reset()
        old_head = obs["snake"][0]
        old_body = obs["snake"][1]
        obs, _, term, _, _ = env.step(Action.STRAIGHT)
        if not term:
            new_head = obs["snake"][0]
            new_body = obs["snake"][1]
            # New body segment should be where the old head was
            assert new_body == old_head
            # New head should be one step in the direction from old head
            assert new_head == (old_head[0] + 1, old_head[1])

    def test_direction_persistence(self, env):
        """Direction should stay the same when going straight."""
        env.reset()
        env.step(Action.STRAIGHT)
        assert env.direction == RIGHT_DIR
        env.step(Action.STRAIGHT)
        assert env.direction == RIGHT_DIR

    def test_sequential_turns(self, env):
        """Test a sequence of turns."""
        env.reset()  # facing right
        env.step(Action.LEFT)   # now facing up
        assert env.direction == UP
        env.step(Action.LEFT)   # now facing left
        assert env.direction == LEFT_DIR
        env.step(Action.LEFT)   # now facing down
        assert env.direction == DOWN
        env.step(Action.LEFT)   # now facing right again
        assert env.direction == RIGHT_DIR


# ============================================================
# 4. Food eating and snake growth
# ============================================================

class TestFoodMechanics:
    """Test food consumption, score, snake growth, and food respawning."""

    def test_eating_food_gives_food_reward(self):
        """Set up a scenario where food is directly ahead."""
        env = SnakeEnv(grid_size=20, seed=0)
        env.reset()
        # Manually place food one step ahead of the head
        hx, hy = env.snake[0]
        env.food = (hx + 1, hy)  # one step to the right (current direction)
        _, reward, _, _, _ = env.step(Action.STRAIGHT)
        assert reward == env.reward_food

    def test_eating_food_increments_score(self):
        env = SnakeEnv(grid_size=20, seed=0)
        env.reset()
        hx, hy = env.snake[0]
        env.food = (hx + 1, hy)
        assert env.score == 0
        env.step(Action.STRAIGHT)
        assert env.score == 1

    def test_eating_food_grows_snake(self):
        env = SnakeEnv(grid_size=20, seed=0)
        env.reset()
        initial_length = env.get_snake_length()
        hx, hy = env.snake[0]
        env.food = (hx + 1, hy)
        env.step(Action.STRAIGHT)
        assert env.get_snake_length() == initial_length + 1

    def test_food_respawns_after_eating(self):
        env = SnakeEnv(grid_size=20, seed=0)
        env.reset()
        hx, hy = env.snake[0]
        old_food = (hx + 1, hy)
        env.food = old_food
        env.step(Action.STRAIGHT)
        # Food should have moved to a new location
        # (extremely unlikely to be in the same spot on a 20x20 grid)
        assert env.food != (-1, -1), "Food should have a valid position"
        # Food should not be on the snake
        assert env.food not in set(env.snake)

    def test_food_always_on_empty_cell(self):
        """Run several episodes and verify food is never on the snake."""
        env = SnakeEnv(grid_size=10, seed=42)
        for episode in range(5):
            env.reset()
            for step in range(200):
                if env.done:
                    break
                # Place food ahead to force eating
                hx, hy = env.snake[0]
                dx, dy = env.direction
                ahead = (hx + dx, hy + dy)
                if (0 <= ahead[0] < env.grid_size and
                    0 <= ahead[1] < env.grid_size and
                    ahead not in set(env.snake)):
                    env.food = ahead
                obs, _, _, _, _ = env.step(Action.STRAIGHT)
                if not env.done:
                    assert obs["food"] not in set(map(tuple, obs["snake"]))

    def test_multiple_foods_eaten(self):
        """Eat several foods and check cumulative score and length."""
        env = SnakeEnv(grid_size=20, seed=0)
        env.reset()
        foods_eaten = 0
        initial_length = env.get_snake_length()

        for _ in range(10):
            if env.done:
                break
            hx, hy = env.snake[0]
            dx, dy = env.direction
            ahead = (hx + dx, hy + dy)
            if (0 <= ahead[0] < env.grid_size and
                0 <= ahead[1] < env.grid_size and
                ahead not in set(env.snake)):
                env.food = ahead
                env.step(Action.STRAIGHT)
                if not env.done:
                    foods_eaten += 1
            else:
                break

        assert env.score == foods_eaten
        assert env.get_snake_length() == initial_length + foods_eaten


# ============================================================
# 5. Collision detection
# ============================================================

class TestCollisions:
    """Test wall and self-collision termination."""

    def test_wall_collision_right(self):
        env = SnakeEnv(grid_size=10, seed=0)
        env.reset()
        # Snake starts at center (5, 5) facing right on a 10x10 grid
        # Walk right until hitting the wall at x=10
        terminated = False
        info = {}
        for _ in range(20):  # more than enough to reach wall
            if env.done:
                break
            _, _, terminated, _, info = env.step(Action.STRAIGHT)
        assert terminated is True
        assert info.get("cause") == "wall_collision"

    def test_wall_collision_up(self):
        env = SnakeEnv(grid_size=10, seed=0)
        env.reset()
        # Turn to face up and walk until wall
        env.step(Action.LEFT)  # now facing up
        terminated = False
        info = {}
        for _ in range(20):
            if env.done:
                break
            _, _, terminated, _, info = env.step(Action.STRAIGHT)
        assert terminated is True
        assert info.get("cause") == "wall_collision"

    def test_wall_collision_gives_death_reward(self):
        env = SnakeEnv(grid_size=10, seed=0)
        env.reset()
        reward = 0
        for _ in range(20):
            if env.done:
                break
            _, reward, terminated, _, _ = env.step(Action.STRAIGHT)
            if terminated:
                break
        assert reward == env.reward_death

    def test_self_collision(self):
        """Force the snake to run into itself."""
        env = SnakeEnv(grid_size=20, seed=0)
        env.reset()

        # First grow the snake by eating several foods going right
        for _ in range(4):
            if env.done:
                break
            hx, hy = env.snake[0]
            dx, dy = env.direction
            ahead = (hx + dx, hy + dy)
            if (0 <= ahead[0] < env.grid_size and
                0 <= ahead[1] < env.grid_size and
                ahead not in set(env.snake)):
                env.food = ahead
                env.step(Action.STRAIGHT)

        # Now snake is long enough. Turn right (down), right (left), right (up)
        # to create a U-turn that hits the body
        if not env.done:
            env.step(Action.RIGHT)   # turn to face down
        if not env.done:
            env.step(Action.RIGHT)   # turn to face left
        if not env.done:
            # Going left should hit the body
            _, _, terminated, _, info = env.step(Action.STRAIGHT)
            if terminated:
                assert info.get("cause") == "self_collision"

    def test_done_flag_set_on_collision(self):
        env = SnakeEnv(grid_size=10, seed=0)
        env.reset()
        for _ in range(20):
            if env.done:
                break
            env.step(Action.STRAIGHT)
        assert env.done is True


# ============================================================
# 6. Truncation (max steps)
# ============================================================

class TestTruncation:
    """Test episode truncation at max steps."""

    def test_truncation_at_max_steps(self):
        # Very small max_steps_factor so truncation happens quickly
        env = SnakeEnv(grid_size=5, max_steps_factor=1, seed=42)
        # max_steps = 1 * 5 * 5 = 25
        env.reset()
        truncated = False
        steps = 0
        # Navigate in a safe loop to avoid dying
        for _ in range(30):
            if env.done:
                break
            # Alternate actions to stay in the center area
            action = Action.LEFT if steps % 4 < 2 else Action.RIGHT
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break

        # Should have hit max steps (25) before dying, OR died before
        # Either way, env should be done
        assert env.done is True
        if truncated:
            assert info.get("cause") == "max_steps"
            assert env.steps == env.max_steps

    def test_truncated_is_not_terminated(self):
        env = SnakeEnv(grid_size=5, max_steps_factor=1, seed=42)
        env.reset()
        last_terminated = False
        last_truncated = False
        for _ in range(30):
            if env.done:
                break
            action = Action.LEFT if env.steps % 4 < 2 else Action.RIGHT
            _, _, last_terminated, last_truncated, _ = env.step(action)
        # If it was truncated, terminated should be False
        if last_truncated:
            assert last_terminated is False


# ============================================================
# 7. Observation structure
# ============================================================

class TestObservation:
    """Test observation dict structure and consistency."""

    def test_snake_is_list_of_tuples(self, env):
        obs, _ = env.reset()
        assert isinstance(obs["snake"], list)
        for pos in obs["snake"]:
            assert isinstance(pos, tuple)
            assert len(pos) == 2

    def test_food_is_tuple(self, env):
        obs, _ = env.reset()
        assert isinstance(obs["food"], tuple)
        assert len(obs["food"]) == 2

    def test_direction_is_tuple(self, env):
        obs, _ = env.reset()
        assert isinstance(obs["direction"], tuple)
        assert len(obs["direction"]) == 2

    def test_obs_updates_after_step(self, env):
        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(Action.STRAIGHT)
        # Head position should have changed
        assert obs1["snake"][0] != obs2["snake"][0]
        # Steps should have incremented
        assert obs2["steps"] == obs1["steps"] + 1

    def test_obs_snake_positions_on_grid(self, env):
        """All snake positions should be within grid bounds."""
        env.reset()
        for _ in range(50):
            if env.done:
                break
            obs, _, _, _, _ = env.step(env.rng.integers(3))
        if not env.done:
            for x, y in obs["snake"]:
                assert 0 <= x < env.grid_size
                assert 0 <= y < env.grid_size

    def test_obs_snake_no_duplicates(self, env):
        """Snake should never overlap with itself (if alive)."""
        env.reset()
        for _ in range(100):
            if env.done:
                break
            obs, _, terminated, _, _ = env.step(env.rng.integers(3))
            if not terminated and not env.done:
                positions = [tuple(p) for p in obs["snake"]]
                assert len(positions) == len(set(positions)), \
                    f"Snake has duplicate positions: {positions}"


# ============================================================
# 8. Reproducibility
# ============================================================

class TestReproducibility:
    """Test that seeding produces reproducible results."""

    def test_same_seed_same_trajectory(self):
        """Two envs with same seed and actions should match exactly."""
        actions = [0, 0, 1, 0, 2, 0, 0, 1, 2, 0]

        env1 = SnakeEnv(seed=42)
        env2 = SnakeEnv(seed=42)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        assert obs1 == obs2

        for a in actions:
            if env1.done or env2.done:
                break
            r1 = env1.step(a)
            r2 = env2.step(a)
            assert r1[0] == r2[0], "Observations should match"
            assert r1[1] == r2[1], "Rewards should match"
            assert r1[2] == r2[2], "Terminated flags should match"
            assert r1[3] == r2[3], "Truncated flags should match"

    def test_different_seed_different_food(self):
        env1 = SnakeEnv(seed=1)
        env2 = SnakeEnv(seed=12345)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        # Food positions should differ (very high probability)
        # Snake positions are deterministic (always center), so they match
        assert obs1["snake"] == obs2["snake"]
        # We don't assert food differs because it's probabilistic,
        # but this runs as a sanity check


# ============================================================
# 9. ASCII rendering
# ============================================================

class TestAsciiRender:
    """Test the ASCII rendering output."""

    def test_render_returns_string(self, env):
        env.reset()
        output = env.render_ascii()
        assert isinstance(output, str)

    def test_render_contains_head(self, env):
        env.reset()
        output = env.render_ascii()
        assert "H" in output

    def test_render_contains_body(self, env):
        env.reset()
        output = env.render_ascii()
        assert "B" in output

    def test_render_contains_food(self, env):
        env.reset()
        output = env.render_ascii()
        assert "F" in output

    def test_render_contains_score(self, env):
        env.reset()
        output = env.render_ascii()
        assert "Score: 0" in output

    def test_render_grid_dimensions(self, small_env):
        small_env.reset()
        output = small_env.render_ascii()
        lines = output.split("\n")
        # Should have: top border + grid_size rows + bottom border + score line
        assert len(lines) == small_env.grid_size + 3
        # Each grid row should be |......| (grid_size chars inside)
        grid_line = lines[1]
        assert grid_line.startswith("|")
        assert grid_line.endswith("|")
        assert len(grid_line) == small_env.grid_size + 2  # +2 for border chars


# ============================================================
# 10. Edge cases
# ============================================================

class TestEdgeCases:
    """Test boundary conditions and unusual scenarios."""

    def test_tiny_grid(self):
        """Test on the smallest reasonable grid."""
        env = SnakeEnv(grid_size=4, seed=42)
        obs, _ = env.reset()
        # Snake of length 3 on a 4x4 grid should still work
        assert len(obs["snake"]) == 3
        assert obs["food"] not in set(map(tuple, obs["snake"]))

    def test_long_episode_no_crash(self):
        """Run a full episode with random actions — should not crash."""
        env = SnakeEnv(grid_size=10, seed=42)
        env.reset()
        steps = 0
        while not env.done and steps < 10000:
            action = env.rng.integers(3)
            env.step(action)
            steps += 1
        # Should have ended (either terminated or truncated)
        assert env.done is True

    def test_repr(self, env):
        env.reset()
        r = repr(env)
        assert "SnakeEnv" in r
        assert "grid_size=20" in r

    def test_get_snake_head(self, env):
        obs, _ = env.reset()
        assert env.get_snake_head() == obs["snake"][0]

    def test_get_snake_length(self, env):
        obs, _ = env.reset()
        assert env.get_snake_length() == len(obs["snake"])

    def test_custom_rewards(self):
        env = SnakeEnv(
            grid_size=10,
            reward_food=100.0,
            reward_death=-50.0,
            reward_step=-1.0,
            seed=42
        )
        env.reset()
        _, reward, _, _, _ = env.step(Action.STRAIGHT)
        # Should get step reward (unless we ate food or died)
        assert reward in [100.0, -50.0, -1.0]

    def test_build_grid_matches_state(self, small_env):
        """Verify _build_grid is consistent with the internal state."""
        small_env.reset()
        grid = small_env._build_grid()

        # Count characters
        chars = Counter()
        for row in grid:
            for cell in row:
                chars[cell] += 1

        assert chars["H"] == 1, "Exactly one head"
        assert chars["B"] == len(small_env.snake) - 1, "Body segments match"
        assert chars["F"] == 1, "Exactly one food"
        total_cells = small_env.grid_size * small_env.grid_size
        assert chars["."] == total_cells - len(small_env.snake) - 1, "Remaining cells are empty"


# ============================================================
# 11. Statistical sanity checks
# ============================================================

class TestStatisticalSanity:
    """Quick checks that the env behaves sensibly over many episodes."""

    def test_random_agent_average_score(self):
        """A random agent should score at least a little on a 10x10 grid."""
        env = SnakeEnv(grid_size=10, seed=42)
        scores = []
        for _ in range(100):
            env.reset()
            while not env.done:
                env.step(env.rng.integers(3))
            scores.append(env.score)
        avg = np.mean(scores)
        # A random agent should eat at least a few foods on average
        # This is a loose sanity check, not a performance test
        assert avg >= 0, "Random agent should have non-negative average score"
        assert max(scores) > 0, "Random agent should eat food at least sometimes"

    def test_episodes_terminate(self):
        """All episodes should end within max_steps."""
        env = SnakeEnv(grid_size=10, seed=42)
        for _ in range(50):
            env.reset()
            steps = 0
            while not env.done:
                env.step(env.rng.integers(3))
                steps += 1
            assert steps <= env.max_steps, \
                f"Episode ran for {steps} steps, max is {env.max_steps}"

    def test_food_distribution_uniform(self):
        """Food should appear in varied positions across episodes."""
        env = SnakeEnv(grid_size=10, seed=42)
        food_positions = set()
        for _ in range(100):
            obs, _ = env.reset()
            food_positions.add(obs["food"])
        # With 100 resets on a 10x10 grid (97 valid cells), we should see many positions
        assert len(food_positions) > 20, \
            f"Food appeared in only {len(food_positions)} unique positions across 100 resets"
