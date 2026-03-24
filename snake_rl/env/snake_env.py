"""
Snake environment with Gymnasium-style interface.

The environment manages pure game logic and exposes raw state as a dictionary.
State representation extraction is handled by separate modules.

Grid coordinates: (0, 0) is top-left, x increases rightward, y increases downward.
"""

from collections import deque
from enum import IntEnum
from typing import Optional

import numpy as np


class Action(IntEnum):
    """Relative actions with respect to the snake's current heading."""
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


# Absolute direction vectors: (dx, dy)
UP = (0, -1)
DOWN = (0, 1)
LEFT_DIR = (-1, 0)
RIGHT_DIR = (1, 0)

# All four absolute directions, useful for initialization
DIRECTIONS = [UP, DOWN, LEFT_DIR, RIGHT_DIR]


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    """Rotate direction 90 degrees counterclockwise."""
    dx, dy = direction
    return (dy, -dx)


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    """Rotate direction 90 degrees clockwise."""
    dx, dy = direction
    return (-dy, dx)


class SnakeEnv:
    """
    Snake game environment following a Gymnasium-style interface.

    Parameters
    ----------
    grid_size : int
        Size of the square grid (default: 20).
    max_steps_factor : int
        Maximum steps per episode = max_steps_factor * grid_size^2.
        Prevents infinite loops where the agent avoids food indefinitely.
    reward_food : float
        Reward for eating food (default: +10).
    reward_death : float
        Reward for dying (default: -10).
    reward_step : float
        Reward per step (default: -0.1).
    seed : int or None
        Random seed for reproducibility.
    """

    metadata = {"render_modes": ["ascii", "pygame"]}

    def __init__(
        self,
        grid_size: int = 20,
        max_steps_factor: int = 1,
        reward_food: float = 10.0,
        reward_death: float = -10.0,
        reward_step: float = -0.1,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps_factor * grid_size * grid_size
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step

        self.rng = np.random.default_rng(seed)

        # Game state (initialized in reset)
        self.snake: deque[tuple[int, int]] = deque()
        self.direction: tuple[int, int] = RIGHT_DIR
        self.food: tuple[int, int] = (0, 0)
        self.score: int = 0
        self.steps: int = 0
        self.done: bool = True

        # Action space / observation space metadata
        self.n_actions = 3  # STRAIGHT, LEFT, RIGHT

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        """
        Reset the environment to a new episode.

        Parameters
        ----------
        seed : int or None
            Optional new random seed.

        Returns
        -------
        observation : dict
            Raw game state dictionary.
        info : dict
            Additional information (empty on reset).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Place snake in the center, facing right, length 3
        cx, cy = self.grid_size // 2, self.grid_size // 2
        self.snake = deque([
            (cx, cy),       # head
            (cx - 1, cy),   # body
            (cx - 2, cy),   # tail
        ])
        self.direction = RIGHT_DIR
        self.score = 0
        self.steps = 0
        self.done = False

        self._place_food()

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int
            0 = STRAIGHT, 1 = LEFT, 2 = RIGHT (relative to current heading).

        Returns
        -------
        observation : dict
            Raw game state dictionary.
        reward : float
            Reward for this step.
        terminated : bool
            True if the snake died (hit wall or self).
        truncated : bool
            True if max steps reached without dying.
        info : dict
            Additional information (cause of episode end, etc.).
        """
        if self.done:
            raise RuntimeError(
                "Environment is done. Call reset() before step()."
            )

        # Resolve relative action to absolute direction
        self.direction = self._resolve_action(action)

        # Compute new head position
        hx, hy = self.snake[0]
        dx, dy = self.direction
        new_head = (hx + dx, hy + dy)

        # Check for wall collision
        nx, ny = new_head
        if nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size:
            self.done = True
            return (
                self._get_obs(),
                self.reward_death,
                True,   # terminated
                False,  # truncated
                {"cause": "wall_collision"},
            )

        # Check for self collision (check against current body, before moving)
        # Note: we check against all segments including the tail because
        # the tail hasn't moved yet at this point
        if new_head in self.snake:
            self.done = True
            return (
                self._get_obs(),
                self.reward_death,
                True,   # terminated
                False,  # truncated
                {"cause": "self_collision"},
            )

        # Move: add new head
        self.snake.appendleft(new_head)

        # Check for food
        ate_food = (new_head == self.food)
        if ate_food:
            self.score += 1
            self._place_food()
            reward = self.reward_food
        else:
            # Remove tail (snake doesn't grow)
            self.snake.pop()
            reward = self.reward_step

        self.steps += 1

        # Check for max steps (truncation)
        if self.steps >= self.max_steps:
            self.done = True
            return (
                self._get_obs(),
                reward,
                False,  # terminated
                True,   # truncated
                {"cause": "max_steps"},
            )

        return self._get_obs(), reward, False, False, {}

    def _resolve_action(self, action: int) -> tuple[int, int]:
        """Convert a relative action to an absolute direction."""
        if action == Action.STRAIGHT:
            return self.direction
        elif action == Action.LEFT:
            return _turn_left(self.direction)
        elif action == Action.RIGHT:
            return _turn_right(self.direction)
        else:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2.")

    def _place_food(self) -> None:
        """Place food on a random empty cell."""
        snake_set = set(self.snake)
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in snake_set
        ]
        if not empty_cells:
            # Snake fills the entire grid — game is effectively won
            # Place food at an impossible location; next step will end
            self.food = (-1, -1)
            return
        idx = self.rng.integers(len(empty_cells))
        self.food = empty_cells[idx]

    def _get_obs(self) -> dict:
        """
        Return the raw game state as a dictionary.

        This dict contains all information any state representation
        extractor could need. The representation modules (built later)
        will convert this into feature vectors.
        """
        return {
            "snake": list(self.snake),
            "food": self.food,
            "direction": self.direction,
            "score": self.score,
            "steps": self.steps,
            "grid_size": self.grid_size,
        }

    def get_snake_head(self) -> tuple[int, int]:
        """Return the current head position."""
        return self.snake[0]

    def get_snake_length(self) -> int:
        """Return the current snake length."""
        return len(self.snake)

    def _build_grid(self) -> list[list[str]]:
        """
        Build a 2D character grid for rendering.

        Returns a grid_size x grid_size list of lists where:
        - '.' = empty
        - 'H' = snake head
        - 'B' = snake body
        - 'F' = food
        """
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Place snake body (tail first so head overwrites)
        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                grid[y][x] = "H"
            else:
                grid[y][x] = "B"

        # Place food
        fx, fy = self.food
        if 0 <= fx < self.grid_size and 0 <= fy < self.grid_size:
            grid[fy][fx] = "F"

        return grid

    def render_ascii(self) -> str:
        """
        Return a string representation of the current game state.

        Useful for debugging and terminal visualization.
        """
        grid = self._build_grid()
        border = "+" + "-" * self.grid_size + "+"
        lines = [border]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append(border)
        lines.append(f"Score: {self.score}  Steps: {self.steps}  Length: {len(self.snake)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SnakeEnv(grid_size={self.grid_size}, "
            f"score={self.score}, steps={self.steps}, "
            f"length={len(self.snake)}, done={self.done})"
        )
