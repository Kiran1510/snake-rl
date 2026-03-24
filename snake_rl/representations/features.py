"""
State representation extractors for the Snake environment.

Each extractor converts the raw observation dict from SnakeEnv into a
numerical feature vector. The three representations increase in complexity:

1. CompactRepresentation  (~11 features) — binary danger/direction/food signals
2. LocalNeighborhoodRepresentation (~100+ features) — 5x5 grid window + metadata
3. ExtendedRepresentation (~150+ features) — local window + continuous distances

All extractors expose:
    - get_state_features(obs) -> np.ndarray  (state-only, for neural net)
    - get_features(obs, action) -> np.ndarray (state-action, for linear methods)
    - state_dim: int  (dimensionality of state features)
    - feature_dim: int (dimensionality of state-action features)
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseRepresentation(ABC):
    """Abstract base class for state representations."""

    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions

    @abstractmethod
    def get_state_features(self, obs: dict) -> np.ndarray:
        """
        Extract state features from a raw observation.

        Parameters
        ----------
        obs : dict
            Raw observation from SnakeEnv (snake, food, direction, etc.)

        Returns
        -------
        np.ndarray of shape (state_dim,)
            Feature vector representing the state.
        """
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimensionality of state feature vector."""
        pass

    @property
    def feature_dim(self) -> int:
        """Dimensionality of state-action feature vector (for linear methods)."""
        return self.state_dim * self.n_actions

    def get_features(self, obs: dict, action: int) -> np.ndarray:
        """
        Extract state-action features using one-hot action stacking.

        The feature vector is constructed by placing the state features
        in the block corresponding to the given action, with zeros elsewhere.
        This gives each action its own weight vector in a linear model:

            x(s, a=0) = [phi(s), 0, 0]
            x(s, a=1) = [0, phi(s), 0]
            x(s, a=2) = [0, 0, phi(s)]

        Parameters
        ----------
        obs : dict
            Raw observation from SnakeEnv.
        action : int
            Action index (0=STRAIGHT, 1=LEFT, 2=RIGHT).

        Returns
        -------
        np.ndarray of shape (feature_dim,)
            State-action feature vector.
        """
        state_feats = self.get_state_features(obs)
        sa_feats = np.zeros(self.feature_dim, dtype=np.float32)
        start = action * self.state_dim
        sa_feats[start:start + self.state_dim] = state_feats
        return sa_feats


# ============================================================
# Representation 1: Compact Features (~11 dimensions)
# ============================================================

class CompactRepresentation(BaseRepresentation):
    """
    Compact binary/categorical feature representation.

    Features (11 total):
        - danger_straight, danger_left, danger_right  (3 binary)
        - direction: up, down, left, right             (4 binary, one-hot)
        - food: up, down, left, right                  (4 binary, can have 2 active)

    This matches the standard 11-feature design used in most Snake RL
    tutorials (Loeber 2020, Zhou 2023).
    """

    @property
    def state_dim(self) -> int:
        return 11

    def get_state_features(self, obs: dict) -> np.ndarray:
        snake = obs["snake"]
        food = obs["food"]
        direction = obs["direction"]
        grid_size = obs["grid_size"]

        head_x, head_y = snake[0]
        dx, dy = direction
        snake_set = set(map(tuple, snake))

        # Compute the three relative directions
        # Straight: continue in current direction
        straight = (head_x + dx, head_y + dy)
        # Left: turn left from current direction
        left_dx, left_dy = dy, -dx  # _turn_left
        left_pos = (head_x + left_dx, head_y + left_dy)
        # Right: turn right from current direction
        right_dx, right_dy = -dy, dx  # _turn_right
        right_pos = (head_x + right_dx, head_y + right_dy)

        def is_danger(pos):
            """Check if a position is a wall or snake body."""
            px, py = pos
            if px < 0 or px >= grid_size or py < 0 or py >= grid_size:
                return 1.0
            if pos in snake_set:
                return 1.0
            return 0.0

        # Danger signals
        danger_straight = is_danger(straight)
        danger_left = is_danger(left_pos)
        danger_right = is_danger(right_pos)

        # Direction one-hot
        dir_up = 1.0 if direction == (0, -1) else 0.0
        dir_down = 1.0 if direction == (0, 1) else 0.0
        dir_left = 1.0 if direction == (-1, 0) else 0.0
        dir_right = 1.0 if direction == (1, 0) else 0.0

        # Food direction (relative to head, not to heading)
        food_x, food_y = food
        food_up = 1.0 if food_y < head_y else 0.0
        food_down = 1.0 if food_y > head_y else 0.0
        food_left = 1.0 if food_x < head_x else 0.0
        food_right = 1.0 if food_x > head_x else 0.0

        features = np.array([
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right,
        ], dtype=np.float32)

        return features


# ============================================================
# Representation 2: Local Neighborhood (~109 dimensions)
# ============================================================

class LocalNeighborhoodRepresentation(BaseRepresentation):
    """
    Local grid window centered on the snake's head.

    Features:
        - 5x5 window, each cell one-hot encoded as [empty, food, wall, body]
          → 5 * 5 * 4 = 100 features
        - Food direction (4 binary): up, down, left, right
        - Snake length bucket (5 binary): one-hot for length ranges
          [1-5, 6-10, 11-20, 21-50, 51+]

    Total: 100 + 4 + 5 = 109 features.

    The window is oriented in absolute coordinates (not relative to heading)
    so that spatial patterns are consistent regardless of direction.
    """

    def __init__(self, window_size: int = 5, n_actions: int = 3):
        super().__init__(n_actions)
        self.window_size = window_size
        self.half_w = window_size // 2
        # 4 categories per cell + 4 food direction + 5 length buckets
        self._state_dim = window_size * window_size * 4 + 4 + 5

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def get_state_features(self, obs: dict) -> np.ndarray:
        snake = obs["snake"]
        food = obs["food"]
        grid_size = obs["grid_size"]

        head_x, head_y = snake[0]
        snake_set = set(map(tuple, snake))

        features = []

        # 5x5 window centered on head
        for wy in range(-self.half_w, self.half_w + 1):
            for wx in range(-self.half_w, self.half_w + 1):
                cell_x = head_x + wx
                cell_y = head_y + wy

                # Determine cell type
                is_empty = 0.0
                is_food = 0.0
                is_wall = 0.0
                is_body = 0.0

                if cell_x < 0 or cell_x >= grid_size or cell_y < 0 or cell_y >= grid_size:
                    is_wall = 1.0
                elif (cell_x, cell_y) == food:
                    is_food = 1.0
                elif (cell_x, cell_y) in snake_set and (cell_x, cell_y) != (head_x, head_y):
                    is_body = 1.0
                else:
                    is_empty = 1.0

                features.extend([is_empty, is_food, is_wall, is_body])

        # Food direction
        food_x, food_y = food
        features.append(1.0 if food_y < head_y else 0.0)  # up
        features.append(1.0 if food_y > head_y else 0.0)  # down
        features.append(1.0 if food_x < head_x else 0.0)  # left
        features.append(1.0 if food_x > head_x else 0.0)  # right

        # Snake length bucket (one-hot)
        length = len(snake)
        length_buckets = [
            1.0 if length <= 5 else 0.0,
            1.0 if 6 <= length <= 10 else 0.0,
            1.0 if 11 <= length <= 20 else 0.0,
            1.0 if 21 <= length <= 50 else 0.0,
            1.0 if length > 50 else 0.0,
        ]
        features.extend(length_buckets)

        return np.array(features, dtype=np.float32)


# ============================================================
# Representation 3: Extended Features (~163 dimensions)
# ============================================================

class ExtendedRepresentation(BaseRepresentation):
    """
    Rich feature set combining local neighborhood with continuous features.

    Features:
        - Local neighborhood (5x5 one-hot): 100 features
        - Food direction (4 binary): 4 features
        - Snake length bucket (5 one-hot): 5 features
        - Normalized Manhattan distance to food: 1 feature
        - Distance to nearest body segment in each of 4 cardinal
          directions (normalized, 0 if none): 4 features
        - Distance to nearest wall in each of 4 cardinal directions
          (normalized): 4 features
        - Normalized snake length: 1 feature
        - Current direction (one-hot): 4 features
        - Danger signals (straight, left, right): 3 features
          (redundant with window, but explicit for linear model)

    Total: 100 + 4 + 5 + 1 + 4 + 4 + 1 + 4 + 3 = 126 features

    Additionally, if polynomial interactions are desired, they can be
    added via the `include_interactions` flag, which adds degree-2
    interaction terms between the danger signals and food direction
    (up to 12 additional features).

    Total with interactions: 126 + 12 = 138 features.
    """

    def __init__(
        self,
        window_size: int = 5,
        n_actions: int = 3,
        include_interactions: bool = False,
    ):
        super().__init__(n_actions)
        self.window_size = window_size
        self.half_w = window_size // 2
        self.include_interactions = include_interactions

        base_dim = (window_size * window_size * 4) + 4 + 5 + 1 + 4 + 4 + 1 + 4 + 3
        interaction_dim = 12 if include_interactions else 0
        self._state_dim = base_dim + interaction_dim

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def get_state_features(self, obs: dict) -> np.ndarray:
        snake = obs["snake"]
        food = obs["food"]
        direction = obs["direction"]
        grid_size = obs["grid_size"]

        head_x, head_y = snake[0]
        dx, dy = direction
        snake_set = set(map(tuple, snake))
        max_dist = grid_size * 2  # max Manhattan distance for normalization

        features = []

        # --- Local neighborhood (same as LocalNeighborhoodRepresentation) ---
        for wy in range(-self.half_w, self.half_w + 1):
            for wx in range(-self.half_w, self.half_w + 1):
                cell_x = head_x + wx
                cell_y = head_y + wy

                is_empty = 0.0
                is_food = 0.0
                is_wall = 0.0
                is_body = 0.0

                if cell_x < 0 or cell_x >= grid_size or cell_y < 0 or cell_y >= grid_size:
                    is_wall = 1.0
                elif (cell_x, cell_y) == food:
                    is_food = 1.0
                elif (cell_x, cell_y) in snake_set and (cell_x, cell_y) != (head_x, head_y):
                    is_body = 1.0
                else:
                    is_empty = 1.0

                features.extend([is_empty, is_food, is_wall, is_body])

        # --- Food direction (4 binary) ---
        food_x, food_y = food
        food_up = 1.0 if food_y < head_y else 0.0
        food_down = 1.0 if food_y > head_y else 0.0
        food_left = 1.0 if food_x < head_x else 0.0
        food_right = 1.0 if food_x > head_x else 0.0
        features.extend([food_up, food_down, food_left, food_right])

        # --- Snake length bucket (5 one-hot) ---
        length = len(snake)
        features.extend([
            1.0 if length <= 5 else 0.0,
            1.0 if 6 <= length <= 10 else 0.0,
            1.0 if 11 <= length <= 20 else 0.0,
            1.0 if 21 <= length <= 50 else 0.0,
            1.0 if length > 50 else 0.0,
        ])

        # --- Normalized Manhattan distance to food ---
        manhattan = abs(food_x - head_x) + abs(food_y - head_y)
        features.append(manhattan / max_dist)

        # --- Distance to nearest body in each cardinal direction ---
        for scan_dx, scan_dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            dist = self._scan_for_body(
                head_x, head_y, scan_dx, scan_dy, snake_set, grid_size
            )
            features.append(dist / grid_size if dist > 0 else 0.0)

        # --- Distance to nearest wall in each cardinal direction ---
        # Up: head_y steps to y=0
        features.append(head_y / (grid_size - 1))
        # Down: (grid_size - 1 - head_y) steps to y=grid_size-1
        features.append((grid_size - 1 - head_y) / (grid_size - 1))
        # Left: head_x steps to x=0
        features.append(head_x / (grid_size - 1))
        # Right: (grid_size - 1 - head_x) steps to x=grid_size-1
        features.append((grid_size - 1 - head_x) / (grid_size - 1))

        # --- Normalized snake length ---
        features.append(length / (grid_size * grid_size))

        # --- Current direction (one-hot) ---
        features.append(1.0 if direction == (0, -1) else 0.0)  # up
        features.append(1.0 if direction == (0, 1) else 0.0)   # down
        features.append(1.0 if direction == (-1, 0) else 0.0)  # left
        features.append(1.0 if direction == (1, 0) else 0.0)   # right

        # --- Danger signals (straight, left, right) ---
        straight_pos = (head_x + dx, head_y + dy)
        left_dx, left_dy = dy, -dx
        left_pos = (head_x + left_dx, head_y + left_dy)
        right_dx, right_dy = -dy, dx
        right_pos = (head_x + right_dx, head_y + right_dy)

        danger_straight = self._is_danger(straight_pos, snake_set, grid_size)
        danger_left = self._is_danger(left_pos, snake_set, grid_size)
        danger_right = self._is_danger(right_pos, snake_set, grid_size)
        features.extend([danger_straight, danger_left, danger_right])

        # --- Optional polynomial interactions ---
        if self.include_interactions:
            # Danger × food direction interactions (3 × 4 = 12 terms)
            dangers = [danger_straight, danger_left, danger_right]
            foods = [food_up, food_down, food_left, food_right]
            for d in dangers:
                for f in foods:
                    features.append(d * f)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _scan_for_body(
        hx: int, hy: int, dx: int, dy: int,
        snake_set: set, grid_size: int
    ) -> int:
        """
        Scan from head in a direction until hitting body or going off-grid.

        Returns the distance to the nearest body segment, or 0 if none found.
        """
        x, y = hx + dx, hy + dy
        dist = 1
        while 0 <= x < grid_size and 0 <= y < grid_size:
            if (x, y) in snake_set:
                return dist
            x += dx
            y += dy
            dist += 1
        return 0  # no body found in this direction

    @staticmethod
    def _is_danger(pos: tuple, snake_set: set, grid_size: int) -> float:
        px, py = pos
        if px < 0 or px >= grid_size or py < 0 or py >= grid_size:
            return 1.0
        if pos in snake_set:
            return 1.0
        return 0.0


# ============================================================
# Registry for easy access
# ============================================================

REPRESENTATIONS = {
    "compact": CompactRepresentation,
    "local": LocalNeighborhoodRepresentation,
    "extended": ExtendedRepresentation,
    "extended_interactions": lambda n_actions=3: ExtendedRepresentation(
        n_actions=n_actions, include_interactions=True
    ),
}


def get_representation(name: str, **kwargs) -> BaseRepresentation:
    """
    Get a representation by name.

    Parameters
    ----------
    name : str
        One of 'compact', 'local', 'extended', 'extended_interactions'.

    Returns
    -------
    BaseRepresentation
        Instantiated representation extractor.
    """
    if name not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation: {name}. "
            f"Choose from: {list(REPRESENTATIONS.keys())}"
        )
    return REPRESENTATIONS[name](**kwargs)
