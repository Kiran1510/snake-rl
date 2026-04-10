"""
Tabular Q-learning demonstration.

Shows that tabular methods work on the compact representation but fail
on richer representations due to state space explosion. This motivates
the need for function approximation.

Usage:
    python run_tabular_demo.py
"""

import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.representations.features import (
    CompactRepresentation,
    LocalNeighborhoodRepresentation,
    ExtendedRepresentation,
)


class TabularQLearningAgent:
    """
    Simple tabular Q-learning agent.

    States are discretized by converting the feature vector to a
    hashable tuple. This works when the state space is small (compact)
    but fails when features are continuous or high-dimensional.
    """

    def __init__(self, n_actions=3, alpha=0.1, gamma=0.95, epsilon=1.0, seed=None):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table = {}  # (state_tuple, action) -> q_value
        self.state_count = set()

    def _discretize(self, features):
        """Convert feature vector to hashable state key."""
        return tuple(np.round(features, 2))

    def q_value(self, state_key, action):
        return self.q_table.get((state_key, action), 0.0)

    def act(self, state_key):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q_vals = [self.q_value(state_key, a) for a in range(self.n_actions)]
        max_q = max(q_vals)
        best = [a for a, q in enumerate(q_vals) if abs(q - max_q) < 1e-8]
        return int(self.rng.choice(best))

    def update(self, state_key, action, reward, next_state_key, terminated):
        current_q = self.q_value(state_key, action)
        if terminated:
            target = reward
        else:
            next_max = max(self.q_value(next_state_key, a) for a in range(self.n_actions))
            target = reward + self.gamma * next_max
        self.q_table[(state_key, action)] = current_q + self.alpha * (target - current_q)
        self.state_count.add(state_key)


def run_tabular_demo(rep_class, rep_name, n_episodes=3000, grid_size=10):
    """Run tabular Q-learning with a given representation."""
    rep = rep_class()
    env = SnakeEnv(grid_size=grid_size, seed=42)
    agent = TabularQLearningAgent(alpha=0.1, gamma=0.95, seed=42)

    scores = []
    start = time.time()

    for ep in range(n_episodes):
        agent.epsilon = max(0.05, 1.0 - ep / (n_episodes * 0.8))
        obs, _ = env.reset()
        state_key = agent._discretize(rep.get_state_features(obs))
        done = False

        while not done:
            action = agent.act(state_key)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_key = agent._discretize(rep.get_state_features(next_obs))
            agent.update(state_key, action, reward, next_key, terminated)
            state_key = next_key

        scores.append(env.score)

    elapsed = time.time() - start
    early = np.mean(scores[:300])
    late = np.mean(scores[-300:])
    unique_states = len(agent.state_count)
    q_entries = len(agent.q_table)

    print(f"\n  {rep_name} representation ({rep.state_dim}d features):")
    print(f"    Unique states visited:  {unique_states:,}")
    print(f"    Q-table entries:        {q_entries:,}")
    print(f"    Early avg score:        {early:.2f}")
    print(f"    Final avg score:        {late:.2f}")
    print(f"    Max score:              {max(scores)}")
    print(f"    Time:                   {elapsed:.1f}s")

    return {
        "rep": rep_name,
        "dims": rep.state_dim,
        "unique_states": unique_states,
        "q_entries": q_entries,
        "early_score": early,
        "final_score": late,
        "max_score": max(scores),
        "time": elapsed,
    }


def main():
    print("=" * 70)
    print("TABULAR Q-LEARNING DEMONSTRATION")
    print("Where do tabular methods break down?")
    print("=" * 70)
    print(f"\nGrid: 10×10, Episodes: 3000, Alpha: 0.1, Gamma: 0.95")

    results = []
    for rep_class, name in [
        (CompactRepresentation, "Compact"),
        (LocalNeighborhoodRepresentation, "Local Neighborhood"),
        (ExtendedRepresentation, "Extended"),
    ]:
        r = run_tabular_demo(rep_class, name)
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Representation':<22} {'Dims':>5} {'States':>10} {'Q-entries':>11} {'Score':>8} {'Max':>5}")
    print("-" * 65)
    for r in results:
        print(f"{r['rep']:<22} {r['dims']:>5} {r['unique_states']:>10,} {r['q_entries']:>11,} {r['final_score']:>8.2f} {r['max_score']:>5}")

    print("\nConclusion: Tabular Q-learning works on compact features but the")
    print("state space explodes with richer representations; most states are")
    print("visited only once, so the Q-table cannot generalize. This shows the")
    print("need for function approximation.")


if __name__ == "__main__":
    main()
