"""
Semi-gradient SARSA with linear function approximation.

The action-value function is approximated as:
    q_hat(s, a; w) = w^T x(s, a)

where x(s, a) is a state-action feature vector constructed by the
representation module, and w is a learned weight vector.

The semi-gradient SARSA update rule:
    w <- w + alpha * [R + gamma * q_hat(S', A'; w) - q_hat(S, A; w)] * x(S, A)

For terminal transitions (S' is terminal):
    w <- w + alpha * [R - q_hat(S, A; w)] * x(S, A)

Reference: Sutton & Barto (2018), Section 10.1
"""

import numpy as np
from snake_rl.representations.features import BaseRepresentation


class LinearSarsaAgent:
    """
    Semi-gradient SARSA agent with linear function approximation.

    Parameters
    ----------
    representation : BaseRepresentation
        Feature extractor (compact, local, or extended).
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration rate (can be updated externally).
    seed : int or None
        Random seed for action selection.
    """

    def __init__(
        self,
        representation: BaseRepresentation,
        alpha: float = 0.01,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        seed: int = None,
    ):
        self.rep = representation
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Initialize weights to small random values (breaks symmetry)
        self.w = self.rng.normal(0, 0.01, size=self.rep.feature_dim).astype(np.float32)

        # Track for diagnostics
        self.td_errors: list[float] = []

    def q_value(self, obs: dict, action: int) -> float:
        """Compute q_hat(s, a) = w^T x(s, a)."""
        features = self.rep.get_features(obs, action)
        return float(np.dot(self.w, features))

    def q_values(self, obs: dict) -> np.ndarray:
        """Compute q_hat(s, a) for all actions."""
        return np.array([self.q_value(obs, a) for a in range(self.rep.n_actions)])

    def act(self, obs: dict) -> int:
        """
        Select action using epsilon-greedy policy.

        With probability epsilon: random action.
        Otherwise: action with highest q_hat(s, a).
        Ties are broken randomly.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.rep.n_actions))

        q_vals = self.q_values(obs)
        # Break ties randomly
        max_q = np.max(q_vals)
        best_actions = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
        return int(self.rng.choice(best_actions))

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
        Perform one semi-gradient SARSA update.

        Parameters
        ----------
        obs : dict
            Current state observation.
        action : int
            Action taken in current state.
        reward : float
            Reward received.
        next_obs : dict
            Next state observation.
        next_action : int
            Action selected in next state (SARSA is on-policy).
        terminated : bool
            Whether the episode ended (no bootstrap if True).

        Returns
        -------
        float
            The TD error magnitude (for diagnostics).
        """
        features = self.rep.get_features(obs, action)
        q_current = float(np.dot(self.w, features))

        if terminated:
            td_target = reward
        else:
            q_next = self.q_value(next_obs, next_action)
            td_target = reward + self.gamma * q_next

        td_error = td_target - q_current

        # Semi-gradient update: w += alpha * td_error * gradient
        # For linear FA, gradient of q_hat w.r.t. w is just x(s, a)
        self.w += self.alpha * td_error * features

        self.td_errors.append(abs(td_error))
        return td_error

    def get_weight_stats(self) -> dict:
        """Return statistics about the weight vector (for analysis)."""
        return {
            "mean": float(np.mean(self.w)),
            "std": float(np.std(self.w)),
            "min": float(np.min(self.w)),
            "max": float(np.max(self.w)),
            "norm": float(np.linalg.norm(self.w)),
            "n_weights": len(self.w),
        }

    def get_mean_td_error(self, last_n: int = 1000) -> float:
        """Return mean absolute TD error over recent updates."""
        if not self.td_errors:
            return 0.0
        n = min(last_n, len(self.td_errors))
        return float(np.mean(self.td_errors[-n:]))

    def __repr__(self) -> str:
        return (
            f"LinearSarsaAgent(alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.3f}, feature_dim={self.rep.feature_dim})"
        )
