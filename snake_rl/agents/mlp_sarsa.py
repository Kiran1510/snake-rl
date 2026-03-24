"""
Semi-gradient SARSA with a neural network (MLP) function approximation.

The action-value function is approximated by a small MLP:
    q_hat(s; theta) = W2 * ReLU(W1 * x(s) + b1) + b2

The network takes state features as input and outputs Q-values for
all 3 actions simultaneously. The semi-gradient SARSA update uses
backpropagation to compute the gradient:

    theta <- theta + alpha * [R + gamma * q(S', A') - q(S, A)] * grad_theta q(S, A)

This is deliberately kept simple (one hidden layer, no target network,
no experience replay by default) to isolate the effect of nonlinear
function approximation. Stabilization can be added if needed.

Reference: Sutton & Barto (2018), Section 9.7
"""

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from snake_rl.representations.features import BaseRepresentation


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for MLPSarsaAgent. "
            "Install with: pip install torch"
        )


if HAS_TORCH:
    class QNetwork(nn.Module):
        """
        Simple MLP that maps state features to Q-values for all actions.

        Architecture: input → Linear → ReLU → Linear → output (3 values)
        """

        def __init__(self, input_dim: int, hidden_dim: int = 128, n_actions: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    class MLPSarsaAgent:
        """
        Semi-gradient SARSA agent with neural network function approximation.

        Parameters
        ----------
        representation : BaseRepresentation
            Feature extractor (compact, local, or extended).
        hidden_dim : int
            Number of units in the hidden layer.
        alpha : float
            Learning rate for Adam optimizer.
        gamma : float
            Discount factor.
        epsilon : float
            Initial exploration rate.
        use_target_network : bool
            If True, use a target network updated every `target_update_freq`
            episodes to stabilize training.
        target_update_freq : int
            Episodes between target network updates.
        seed : int or None
            Random seed.
        """

        def __init__(
            self,
            representation: BaseRepresentation,
            hidden_dim: int = 128,
            alpha: float = 0.001,
            gamma: float = 0.95,
            epsilon: float = 1.0,
            use_target_network: bool = False,
            target_update_freq: int = 100,
            seed: Optional[int] = None,
        ):
            _check_torch()

            self.rep = representation
            self.hidden_dim = hidden_dim
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.use_target_network = use_target_network
            self.target_update_freq = target_update_freq
            self.rng = np.random.default_rng(seed)

            # Set PyTorch seed
            if seed is not None:
                torch.manual_seed(seed)

            # Device (CPU for this project — Snake is not GPU-bound)
            self.device = torch.device("cpu")

            # Q-network
            self.q_net = QNetwork(
                input_dim=representation.state_dim,
                hidden_dim=hidden_dim,
                n_actions=3,
            ).to(self.device)

            # Target network (optional)
            if use_target_network:
                self.target_net = QNetwork(
                    input_dim=representation.state_dim,
                    hidden_dim=hidden_dim,
                    n_actions=3,
                ).to(self.device)
                self.target_net.load_state_dict(self.q_net.state_dict())
                self.target_net.eval()
            else:
                self.target_net = None

            # Optimizer
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)

            # Episode counter (for target network updates)
            self._episode_count = 0

            # Diagnostics
            self.td_errors: list[float] = []

        def _state_to_tensor(self, obs: dict) -> torch.Tensor:
            """Convert observation to a feature tensor."""
            features = self.rep.get_state_features(obs)
            return torch.FloatTensor(features).unsqueeze(0).to(self.device)

        def q_values(self, obs: dict) -> np.ndarray:
            """Compute Q-values for all actions (no gradient)."""
            with torch.no_grad():
                x = self._state_to_tensor(obs)
                q = self.q_net(x).squeeze(0).numpy()
            return q

        def q_value(self, obs: dict, action: int) -> float:
            """Compute Q-value for a specific action."""
            return float(self.q_values(obs)[action])

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
            Semi-gradient SARSA update via backpropagation.

            Computes: loss = 0.5 * (td_target - q(S, A))^2
            Then backpropagates through q(S, A) only (semi-gradient).
            """
            # Compute TD target (no gradient through target)
            with torch.no_grad():
                if terminated:
                    td_target = reward
                else:
                    if self.target_net is not None:
                        x_next = self._state_to_tensor(next_obs)
                        q_next = self.target_net(x_next).squeeze(0)[next_action].item()
                    else:
                        q_next = self.q_value(next_obs, next_action)
                    td_target = reward + self.gamma * q_next

            # Compute q(S, A) with gradient
            x = self._state_to_tensor(obs)
            q_all = self.q_net(x).squeeze(0)
            q_current = q_all[action]

            # Semi-gradient: only differentiate through q_current, not td_target
            td_error = td_target - q_current.item()
            loss = 0.5 * (torch.tensor(td_target) - q_current) ** 2

            # Backprop and update
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
            self.optimizer.step()

            self.td_errors.append(abs(td_error))
            return td_error

        def on_episode_end(self):
            """
            Called at the end of each episode.
            Updates target network if applicable.
            """
            self._episode_count += 1
            if (self.target_net is not None and
                    self._episode_count % self.target_update_freq == 0):
                self.target_net.load_state_dict(self.q_net.state_dict())

        def get_weight_stats(self) -> dict:
            """Network parameter statistics."""
            all_params = []
            for p in self.q_net.parameters():
                all_params.append(p.data.cpu().numpy().flatten())
            all_params = np.concatenate(all_params)
            return {
                "mean": float(np.mean(all_params)),
                "std": float(np.std(all_params)),
                "min": float(np.min(all_params)),
                "max": float(np.max(all_params)),
                "norm": float(np.linalg.norm(all_params)),
                "n_params": len(all_params),
            }

        def get_mean_td_error(self, last_n: int = 1000) -> float:
            """Mean absolute TD error over recent updates."""
            if not self.td_errors:
                return 0.0
            n = min(last_n, len(self.td_errors))
            return float(np.mean(self.td_errors[-n:]))

        def __repr__(self) -> str:
            n_params = sum(p.numel() for p in self.q_net.parameters())
            return (
                f"MLPSarsaAgent(hidden={self.hidden_dim}, "
                f"alpha={self.alpha}, gamma={self.gamma}, "
                f"epsilon={self.epsilon:.3f}, "
                f"params={n_params}, "
                f"target_net={self.use_target_network})"
            )
