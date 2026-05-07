"""
MLP agent with experience replay and target network.

Replaces single-step semi-gradient SARSA with mini-batch Q-learning from a
replay buffer. Keeps the same training-loop interface (.act, .update,
.on_episode_end, .epsilon) so it drops into train_sarsa unchanged.

    - Replay buffer (default 50k): transitions stored each step;
      a random mini-batch sampled for every gradient update.
    - Q-learning TD target: max_a Q_target(s', a).
    - Target network: hard-copied every target_update_freq episodes.
    - Configurable depth (default: 256 → 128).
    - Gradient clipping at 5.0.

Note: experience replay breaks on-policy sampling, eliminating the
Tsitsiklis & Van Roy (1997) convergence guarantee. The empirical stability
gains from replay outweigh the theoretical cost for neural-net approximators.
"""

from collections import deque
import random
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from snake_rl.representations.features import BaseRepresentation


class ReplayBuffer:
    """
    Circular buffer of (state_feat, action, reward, next_state_feat, terminated).

    Features are pre-extracted at push time so sampling involves no env interaction.
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, float(terminated)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


if HAS_TORCH:
    class QNetwork(nn.Module):
        """MLP with configurable depth (1 or 2 hidden layers)."""

        def __init__(self, input_dim: int, hidden_dims: tuple = (256, 128), n_actions: int = 3):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, n_actions))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    class MLPSarsaAgent:
        """
        Q-learning agent with replay buffer and target network.

        Parameters
        ----------
        representation : BaseRepresentation
        hidden_dims : tuple
            Hidden layer sizes, e.g. (256, 128) or (256,).
        alpha : float
            Adam learning rate.
        gamma : float
            Discount factor.
        epsilon : float
            Initial ε for ε-greedy exploration (updated externally by train loop).
        buffer_capacity : int
            Maximum replay buffer size.
        batch_size : int
            Mini-batch size for each gradient update.
        min_buffer_size : int
            Transitions required before learning starts.
        target_update_freq : int
            Episodes between hard target-network copies.
        seed : int or None
        """

        def __init__(
            self,
            representation: BaseRepresentation,
            hidden_dims: tuple = (256, 128),
            alpha: float = 0.001,
            gamma: float = 0.95,
            epsilon: float = 1.0,
            buffer_capacity: int = 50_000,
            batch_size: int = 64,
            min_buffer_size: int = 1_000,
            target_update_freq: int = 100,
            use_target_network: bool = True,   # kept for API compat, always on
            seed: Optional[int] = None,
        ):
            self.rep = representation
            self.hidden_dims = hidden_dims
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = epsilon
            self.batch_size = batch_size
            self.min_buffer_size = min_buffer_size
            self.target_update_freq = target_update_freq
            self.rng = np.random.default_rng(seed)

            if seed is not None:
                torch.manual_seed(seed)
                random.seed(int(seed))

            self.device = torch.device("cpu")

            self.q_net = QNetwork(
                input_dim=representation.state_dim,
                hidden_dims=hidden_dims,
                n_actions=3,
            ).to(self.device)

            self.target_net = QNetwork(
                input_dim=representation.state_dim,
                hidden_dims=hidden_dims,
                n_actions=3,
            ).to(self.device)
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)
            self.replay_buffer = ReplayBuffer(buffer_capacity)

            self._episode_count: int = 0
            self.td_errors: list[float] = []

        # ------------------------------------------------------------------
        # Inference
        # ------------------------------------------------------------------

        def _feat(self, obs: dict) -> np.ndarray:
            return self.rep.get_state_features(obs)

        def _to_tensor(self, feat: np.ndarray) -> "torch.Tensor":
            return torch.FloatTensor(feat).unsqueeze(0).to(self.device)

        def q_values(self, obs: dict) -> np.ndarray:
            with torch.no_grad():
                return self.q_net(self._to_tensor(self._feat(obs))).squeeze(0).numpy()

        def q_value(self, obs: dict, action: int) -> float:
            return float(self.q_values(obs)[action])

        def act(self, obs: dict) -> int:
            if self.rng.random() < self.epsilon:
                return int(self.rng.integers(3))
            q_vals = self.q_values(obs)
            max_q = np.max(q_vals)
            best = np.where(np.abs(q_vals - max_q) < 1e-8)[0]
            return int(self.rng.choice(best))

        # ------------------------------------------------------------------
        # Learning
        # ------------------------------------------------------------------

        def update(
            self,
            obs: dict,
            action: int,
            reward: float,
            next_obs: dict,
            next_action: int,   # unused — we use max-Q target
            terminated: bool,
        ) -> float:
            """Store transition; run one batch update if buffer is ready."""
            self.replay_buffer.push(
                self._feat(obs), action, reward, self._feat(next_obs), terminated
            )

            if len(self.replay_buffer) < self.min_buffer_size:
                return 0.0

            return self._batch_update()

        def _batch_update(self) -> float:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)

            states_t      = torch.FloatTensor(states)
            actions_t     = torch.LongTensor(actions)
            rewards_t     = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            dones_t       = torch.FloatTensor(dones)

            with torch.no_grad():
                next_q  = self.target_net(next_states_t).max(dim=1)[0]
                targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

            q_current = self.q_net(states_t).gather(
                1, actions_t.unsqueeze(1)
            ).squeeze(1)

            loss = F.mse_loss(q_current, targets)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
            self.optimizer.step()

            td_err = float((targets - q_current.detach()).abs().mean())
            self.td_errors.append(td_err)
            return td_err

        def on_episode_end(self) -> None:
            self._episode_count += 1
            if self._episode_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

        # ------------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------------

        def get_mean_td_error(self, last_n: int = 1000) -> float:
            if not self.td_errors:
                return 0.0
            n = min(last_n, len(self.td_errors))
            return float(np.mean(self.td_errors[-n:]))

        def get_weight_stats(self) -> dict:
            params = np.concatenate(
                [p.data.cpu().numpy().flatten() for p in self.q_net.parameters()]
            )
            return {
                "mean":        float(np.mean(params)),
                "std":         float(np.std(params)),
                "norm":        float(np.linalg.norm(params)),
                "n_params":    len(params),
                "buffer_size": len(self.replay_buffer),
            }

        def __repr__(self) -> str:
            n = sum(p.numel() for p in self.q_net.parameters())
            return (
                f"MLPSarsaAgent(hidden={self.hidden_dims}, params={n}, "
                f"alpha={self.alpha}, "
                f"buffer={len(self.replay_buffer)}/"
                f"{self.replay_buffer.buffer.maxlen})"
            )
