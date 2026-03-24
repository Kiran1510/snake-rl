"""
Semi-gradient SARSA with tile coding function approximation.

Tile coding automatically constructs binary feature vectors from
continuous or discrete state variables using overlapping tilings.
The value function remains linear in these features:

    q_hat(s, a; w) = sum of w[i] for each active tile i

Each tiling partitions the state space into tiles. Multiple tilings
are offset from each other so that nearby states share most (but not
all) active tiles, providing smooth generalization.

The update rule is identical to linear FA:
    w[active_tiles] += alpha * td_error

With the standard convention that alpha is divided by the number
of tilings (Sutton & Barto, 2018, §9.5.4).

Reference:
    Sutton & Barto (2018), Section 9.5.4
    Sherstov & Stone (2005), "Function Approximation via Tile Coding"
"""

import numpy as np
from typing import Optional
from snake_rl.representations.features import BaseRepresentation


class TileCoder:
    """
    Tile coding with hashing for converting features into
    sparse binary feature vectors.

    Uses hashing to handle high-dimensional inputs that would
    create astronomically large tile indices. The hash maps
    (tiling, tile_coords, action) to a fixed-size weight table.

    Parameters
    ----------
    n_tilings : int
        Number of overlapping tilings (typically 8 or 16).
    n_tiles_per_dim : int
        Number of tiles per dimension in each tiling.
    n_dims : int
        Number of input dimensions.
    n_actions : int
        Number of actions (tiles are per-action).
    max_size : int
        Maximum hash table size (total weight vector length).
        Larger = fewer collisions, more memory.
    """

    def __init__(
        self,
        n_tilings: int = 8,
        n_tiles_per_dim: int = 4,
        n_dims: int = 11,
        n_actions: int = 3,
        max_size: int = 65536,
    ):
        self.n_tilings = n_tilings
        self.n_tiles_per_dim = n_tiles_per_dim
        self.n_actions = n_actions
        self.n_dims = n_dims
        self.max_size = max_size

        # Generate offsets for each tiling (deterministic)
        self.offsets = np.zeros((self.n_tilings, self.n_dims))
        for t in range(self.n_tilings):
            for d in range(self.n_dims):
                self.offsets[t, d] = t / self.n_tilings / self.n_tiles_per_dim

    @property
    def feature_dim(self) -> int:
        """Total weight vector size (hash table size)."""
        return self.max_size

    def get_active_tiles(self, features: np.ndarray, action: int) -> np.ndarray:
        """
        Get indices of active tiles for a (state, action) pair.

        Uses a hash function to map (tiling_id, tile_coordinates, action)
        to the fixed-size weight table.

        Parameters
        ----------
        features : np.ndarray
            State features normalized to [0, 1]. Shape: (n_dims,).
        action : int
            Action index.

        Returns
        -------
        np.ndarray of int
            Indices of active tiles (one per tiling).
        """
        active = np.zeros(self.n_tilings, dtype=np.int64)

        for t in range(self.n_tilings):
            # Compute integer tile coordinates for this tiling
            coords = []
            for d in range(self.n_dims):
                shifted = features[d] + self.offsets[t, d]
                shifted = min(max(shifted, 0.0), 0.9999)
                tile_d = int(shifted * self.n_tiles_per_dim)
                coords.append(tile_d)

            # Hash (tiling_id, action, tile_coords) to table index
            # Using a simple but effective hash
            h = hash((t, action) + tuple(coords)) % self.max_size
            active[t] = h

        return active


class TileCodingRepresentation:
    """
    Wrapper that converts raw observations into normalized features
    suitable for tile coding.

    Takes the state features from any BaseRepresentation and normalizes
    them to [0, 1] for the tile coder.
    """

    def __init__(self, base_rep: BaseRepresentation, n_actions: int = 3):
        self.base_rep = base_rep
        self.n_actions = n_actions
        # Track feature ranges for normalization
        self.feature_min = None
        self.feature_max = None
        self._initialized = False

    def initialize_ranges(self, env, n_samples: int = 1000):
        """
        Estimate feature ranges by sampling random states.

        Should be called before training to set up normalization.
        """
        samples = []
        for _ in range(n_samples):
            obs, _ = env.reset()
            samples.append(self.base_rep.get_state_features(obs))
            # Also step randomly to see diverse states
            for _ in range(10):
                if env.done:
                    break
                action = env.rng.integers(3)
                obs, _, _, _, _ = env.step(action)
                samples.append(self.base_rep.get_state_features(obs))

        samples = np.array(samples)
        self.feature_min = samples.min(axis=0) - 0.01
        self.feature_max = samples.max(axis=0) + 0.01
        # Avoid division by zero for constant features
        ranges = self.feature_max - self.feature_min
        ranges[ranges < 1e-6] = 1.0
        self.feature_max = self.feature_min + ranges
        self._initialized = True

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1]."""
        if not self._initialized:
            # Fallback: assume binary features in [0, 1]
            return np.clip(features, 0.0, 1.0)
        normalized = (features - self.feature_min) / (self.feature_max - self.feature_min)
        return np.clip(normalized, 0.0, 1.0)

    def get_normalized_features(self, obs: dict) -> np.ndarray:
        """Extract and normalize state features."""
        raw = self.base_rep.get_state_features(obs)
        return self.normalize(raw)


class TileCodingSarsaAgent:
    """
    Semi-gradient SARSA agent with tile coding function approximation.

    Parameters
    ----------
    base_representation : BaseRepresentation
        Underlying feature extractor.
    n_tilings : int
        Number of overlapping tilings.
    n_tiles_per_dim : int
        Tiles per dimension per tiling.
    max_size : int
        Hash table size (weight vector length).
    alpha : float
        Learning rate (will be divided by n_tilings internally).
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration rate.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        base_representation: BaseRepresentation,
        n_tilings: int = 8,
        n_tiles_per_dim: int = 4,
        max_size: int = 65536,
        alpha: float = 0.01,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.base_rep = base_representation
        self.n_tilings = n_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Alpha divided by n_tilings (standard convention)
        self.alpha = alpha / n_tilings
        self._raw_alpha = alpha

        # Tile coding setup
        self.tile_rep = TileCodingRepresentation(base_representation)
        self.tile_coder = TileCoder(
            n_tilings=n_tilings,
            n_tiles_per_dim=n_tiles_per_dim,
            n_dims=base_representation.state_dim,
            n_actions=3,
            max_size=max_size,
        )

        # Weight vector (one weight per tile)
        self.w = np.zeros(self.tile_coder.feature_dim, dtype=np.float32)

        # Diagnostics
        self.td_errors: list[float] = []

    def initialize(self, env):
        """Initialize feature normalization ranges. Call before training."""
        self.tile_rep.initialize_ranges(env)

    def _get_tiles(self, obs: dict, action: int) -> np.ndarray:
        """Get active tile indices for a (state, action) pair."""
        normalized = self.tile_rep.get_normalized_features(obs)
        return self.tile_coder.get_active_tiles(normalized, action)

    def q_value(self, obs: dict, action: int) -> float:
        """Compute q_hat(s, a) = sum of weights at active tiles."""
        tiles = self._get_tiles(obs, action)
        return float(np.sum(self.w[tiles]))

    def q_values(self, obs: dict) -> np.ndarray:
        """Compute q_hat(s, a) for all actions."""
        return np.array([self.q_value(obs, a) for a in range(3)])

    def act(self, obs: dict) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(3))

        q_vals = self.q_values(obs)
        max_q = np.max(q_vals)
        best = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
        return int(self.rng.choice(best))

    def update(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        next_action: int,
        terminated: bool,
    ) -> float:
        """
        Semi-gradient SARSA update for tile coding.

        Only the weights at active tile indices are updated.
        """
        tiles = self._get_tiles(obs, action)
        q_current = float(np.sum(self.w[tiles]))

        if terminated:
            td_target = reward
        else:
            q_next = self.q_value(next_obs, next_action)
            td_target = reward + self.gamma * q_next

        td_error = td_target - q_current

        # Update only active tiles
        self.w[tiles] += self.alpha * td_error

        self.td_errors.append(abs(td_error))
        return td_error

    def get_weight_stats(self) -> dict:
        """Weight vector statistics."""
        nonzero = self.w[self.w != 0]
        return {
            "mean": float(np.mean(self.w)),
            "std": float(np.std(self.w)),
            "min": float(np.min(self.w)),
            "max": float(np.max(self.w)),
            "norm": float(np.linalg.norm(self.w)),
            "n_weights": len(self.w),
            "n_nonzero": len(nonzero),
            "sparsity": 1.0 - len(nonzero) / len(self.w),
        }

    def get_mean_td_error(self, last_n: int = 1000) -> float:
        """Mean absolute TD error over recent updates."""
        if not self.td_errors:
            return 0.0
        n = min(last_n, len(self.td_errors))
        return float(np.mean(self.td_errors[-n:]))

    def __repr__(self) -> str:
        return (
            f"TileCodingSarsaAgent(n_tilings={self.n_tilings}, "
            f"alpha={self._raw_alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.3f}, "
            f"total_tiles={self.tile_coder.feature_dim})"
        )
