"""
Enhanced MLP SARSA agent with configurable depth.

Supports 1 or 2 hidden layers. Drop-in replacement for the original.
"""

import os
import sys
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from snake_rl.representations.features import BaseRepresentation

if HAS_TORCH:
    class QNetworkV2(nn.Module):
        """MLP with configurable depth (1 or 2 hidden layers)."""

        def __init__(self, input_dim, hidden_dims=(256, 128), n_actions=3):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.ReLU())
                prev_dim = h
            layers.append(nn.Linear(prev_dim, n_actions))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class MLPSarsaAgentV2:
        """
        Enhanced semi-gradient SARSA with deeper MLP.

        Changes from v1:
            - Configurable hidden layer sizes (default: 256, 128)
            - Gradient clipping tightened to 5.0
            - Optional learning rate scheduling
        """

        def __init__(
            self,
            representation: BaseRepresentation,
            hidden_dims: tuple = (256, 128),
            alpha: float = 0.001,
            gamma: float = 0.95,
            epsilon: float = 1.0,
            use_target_network: bool = False,
            target_update_freq: int = 100,
            seed: Optional[int] = None,
        ):
            self.rep = representation
            self.hidden_dims = hidden_dims
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.use_target_network = use_target_network
            self.target_update_freq = target_update_freq
            self.rng = np.random.default_rng(seed)

            if seed is not None:
                torch.manual_seed(seed)

            self.device = torch.device("cpu")

            self.q_net = QNetworkV2(
                input_dim=representation.state_dim,
                hidden_dims=hidden_dims,
                n_actions=3,
            ).to(self.device)

            if use_target_network:
                self.target_net = QNetworkV2(
                    input_dim=representation.state_dim,
                    hidden_dims=hidden_dims,
                    n_actions=3,
                ).to(self.device)
                self.target_net.load_state_dict(self.q_net.state_dict())
                self.target_net.eval()
            else:
                self.target_net = None

            self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)
            self._episode_count = 0
            self.td_errors = []

        def _state_to_tensor(self, obs):
            features = self.rep.get_state_features(obs)
            return torch.FloatTensor(features).unsqueeze(0).to(self.device)

        def q_values(self, obs):
            with torch.no_grad():
                x = self._state_to_tensor(obs)
                q = self.q_net(x).squeeze(0).numpy()
            return q

        def q_value(self, obs, action):
            return float(self.q_values(obs)[action])

        def act(self, obs):
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(3))
            q_vals = self.q_values(obs)
            max_q = np.max(q_vals)
            best = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
            return int(self.rng.choice(best))

        def update(self, obs, action, reward, next_obs, next_action, terminated):
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

            x = self._state_to_tensor(obs)
            q_all = self.q_net(x).squeeze(0)
            q_current = q_all[action]

            td_error = td_target - q_current.item()
            loss = 0.5 * (torch.tensor(td_target, dtype=torch.float32, device=self.device) - q_current) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
            self.optimizer.step()

            self.td_errors.append(abs(td_error))
            return td_error

        def on_episode_end(self):
            self._episode_count += 1
            if (self.target_net is not None and
                    self._episode_count % self.target_update_freq == 0):
                self.target_net.load_state_dict(self.q_net.state_dict())

        def get_weight_stats(self):
            all_params = []
            for p in self.q_net.parameters():
                all_params.append(p.data.cpu().numpy().flatten())
            all_params = np.concatenate(all_params)
            return {
                "mean": float(np.mean(all_params)),
                "std": float(np.std(all_params)),
                "norm": float(np.linalg.norm(all_params)),
                "n_params": len(all_params),
            }

        def get_mean_td_error(self, last_n=1000):
            if not self.td_errors:
                return 0.0
            n = min(last_n, len(self.td_errors))
            return float(np.mean(self.td_errors[-n:]))

        def __repr__(self):
            n_params = sum(p.numel() for p in self.q_net.parameters())
            return (
                f"MLPSarsaAgentV2(hidden={self.hidden_dims}, "
                f"params={n_params}, alpha={self.alpha})"
            )
