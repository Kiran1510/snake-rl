"""
Save and load trained agent weights.

Supports all three agent types: Linear FA, Tile Coding, MLP.
"""

import os
import json
import numpy as np

WEIGHTS_DIR = "weights"


def save_agent(agent, name: str, directory: str = WEIGHTS_DIR):
    """Save agent weights to disk."""
    os.makedirs(directory, exist_ok=True)
    agent_type = type(agent).__name__

    if agent_type == "LinearSarsaAgent":
        filepath = os.path.join(directory, f"{name}.npz")
        np.savez(filepath, w=agent.w, alpha=agent.alpha, gamma=agent.gamma)
        print(f"Saved LinearSarsaAgent weights to {filepath}")

    elif agent_type == "TileCodingSarsaAgent":
        filepath = os.path.join(directory, f"{name}.npz")
        np.savez(filepath,
                 w=agent.w,
                 alpha=np.array([agent._raw_alpha]),
                 gamma=np.array([agent.gamma]),
                 n_tilings=np.array([agent.n_tilings]),
                 feature_min=agent.tile_rep.feature_min if agent.tile_rep._initialized else np.array([]),
                 feature_max=agent.tile_rep.feature_max if agent.tile_rep._initialized else np.array([]),
                 )
        print(f"Saved TileCodingSarsaAgent weights to {filepath}")

    elif agent_type == "MLPSarsaAgent":
        try:
            import torch
            filepath = os.path.join(directory, f"{name}.pt")
            torch.save({
                "q_net": agent.q_net.state_dict(),
                "alpha": agent.alpha,
                "gamma": agent.gamma,
                "hidden_dims": agent.hidden_dims,
            }, filepath)
            print(f"Saved {agent_type} weights to {filepath}")
        except ImportError:
            print("PyTorch required to save MLP weights")

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def load_agent_weights(agent, name: str, directory: str = WEIGHTS_DIR):
    """Load saved weights into an existing agent."""
    agent_type = type(agent).__name__

    if agent_type == "LinearSarsaAgent":
        filepath = os.path.join(directory, f"{name}.npz")
        data = np.load(filepath)
        agent.w = data["w"]
        print(f"Loaded LinearSarsaAgent weights from {filepath}")

    elif agent_type == "TileCodingSarsaAgent":
        filepath = os.path.join(directory, f"{name}.npz")
        data = np.load(filepath)
        agent.w = data["w"]
        if len(data["feature_min"]) > 0:
            agent.tile_rep.feature_min = data["feature_min"]
            agent.tile_rep.feature_max = data["feature_max"]
            agent.tile_rep._initialized = True
        print(f"Loaded TileCodingSarsaAgent weights from {filepath}")

    elif agent_type == "MLPSarsaAgent":
        try:
            import torch
            filepath = os.path.join(directory, f"{name}.pt")
            checkpoint = torch.load(filepath, weights_only=False)
            agent.q_net.load_state_dict(checkpoint["q_net"])
            print(f"Loaded {agent_type} weights from {filepath}")
        except ImportError:
            print("PyTorch required to load MLP weights")

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
