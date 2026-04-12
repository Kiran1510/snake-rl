"""
Record GIFs for all 18 configurations (9 v1 + 9 v2).

Usage:
    python record_all_gifs.py
    python record_all_gifs.py --v1-only
    python record_all_gifs.py --v2-only
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("ERROR: pygame required. pip install pygame")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("ERROR: Pillow required. pip install Pillow")
    sys.exit(1)

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.representations.features import (
    CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation,
)
from snake_rl.agents.linear_sarsa import LinearSarsaAgent
from snake_rl.agents.tile_sarsa import TileCodingSarsaAgent
from snake_rl.utils.save_load import load_agent_weights
from record_gameplay import render_frame

try:
    from snake_rl.agents.mlp_sarsa import MLPSarsaAgent
    from snake_rl.agents.mlp_sarsa_v2 import MLPSarsaAgentV2
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. MLP GIFs will be skipped.")

GRID_SIZE = 20
CELL_SIZE = 30

REPS = {
    "compact": CompactRepresentation,
    "local": LocalNeighborhoodRepresentation,
    "extended": ExtendedRepresentation,
}


def make_agent(algo, rep_name, version):
    """Create a fresh agent matching the version used during training."""
    rep = REPS[rep_name]()

    if algo == "linear":
        return LinearSarsaAgent(rep, alpha=0.01, gamma=0.95, seed=42)

    elif algo == "tile":
        ms = 262144 if version == "v2" else 65536
        return TileCodingSarsaAgent(
            rep, n_tilings=8, n_tiles_per_dim=4,
            max_size=ms, alpha=0.05, gamma=0.95, seed=42,
        )

    elif algo == "mlp":
        if not HAS_TORCH:
            return None
        if version == "v2":
            return MLPSarsaAgentV2(
                rep, hidden_dims=(256, 128),
                alpha=0.001, gamma=0.95, seed=42,
            )
        else:
            return MLPSarsaAgent(
                rep, hidden_dim=128,
                alpha=0.001, gamma=0.95, seed=42,
            )


def weight_path_exists(algo, rep, version):
    """Check if weights exist for this config."""
    suffix = "_v2" if version == "v2" else ""
    name = f"{algo}_sarsa{suffix}__{rep}"
    wdir = "weights_v2" if version == "v2" else "weights"

    npz = os.path.join(wdir, f"{name}.npz")
    pt = os.path.join(wdir, f"{name}.pt")
    return os.path.exists(npz) or os.path.exists(pt)


def load_agent(algo, rep_name, version):
    """Create agent and load saved weights."""
    agent = make_agent(algo, rep_name, version)
    if agent is None:
        return None

    suffix = "_v2" if version == "v2" else ""
    name = f"{algo}_sarsa{suffix}__{rep_name}"
    wdir = "weights_v2" if version == "v2" else "weights"

    try:
        load_agent_weights(agent, name, wdir)
        agent.epsilon = 0.01
        return agent
    except Exception as e:
        print(f"  Failed to load weights: {e}")
        return None


def record_one_gif(algo, rep_name, version, n_episodes=3, max_steps_per_ep=300, fps=10):
    """Record a GIF for one agent configuration."""
    agent = load_agent(algo, rep_name, version)
    if agent is None:
        print(f"  SKIP: no agent for {algo} {rep_name} {version}")
        return False

    suffix = "_v2" if version == "v2" else "_v1"
    out_dir = f"recordings{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    max_factor = 10 if version == "v2" else 3
    env = SnakeEnv(grid_size=GRID_SIZE, max_steps_factor=max_factor, seed=999)

    grid_w = GRID_SIZE * CELL_SIZE
    panel_h = 60

    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((grid_w, grid_w + panel_h))

    label = f"{algo}_sarsa {version} + {rep_name}"
    frames = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_ep:
            q_vals = agent.q_values(obs) if hasattr(agent, "q_values") else None
            action = agent.act(obs)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc
            steps += 1

            render_frame(surface, env, agent_name=label, q_vals=q_vals)
            frame_data = pygame.image.tostring(surface, "RGB")
            frame_img = Image.frombytes("RGB", (grid_w, grid_w + panel_h), frame_data)
            frames.append(frame_img)

    pygame.quit()

    if frames:
        filename = f"{algo}_sarsa__{rep_name}.gif"
        filepath = os.path.join(out_dir, filename)
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0,
            optimize=True,
        )
        print(f"  Saved: {filepath} ({len(frames)} frames)")
        return True
    return False


def record_all(versions):
    """Record GIFs for all configurations."""
    algos = ["linear", "tile", "mlp"]
    reps = ["compact", "local", "extended"]

    total = 0
    saved = 0
    skipped = 0

    for version in versions:
        print(f"\n{'='*60}")
        print(f"RECORDING {version.upper()} AGENTS")
        print(f"{'='*60}")

        for algo in algos:
            for rep in reps:
                total += 1
                tag = f"{algo} x {rep} ({version})"

                if not weight_path_exists(algo, rep, version):
                    print(f"\n[{tag}] No weights found — skipping")
                    skipped += 1
                    continue

                print(f"\n[{tag}] Recording...")
                success = record_one_gif(algo, rep, version)
                if success:
                    saved += 1
                else:
                    skipped += 1

    print(f"\n{'='*60}")
    print(f"DONE: {saved} GIFs saved, {skipped} skipped, {total} total")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Record GIFs for all agents")
    parser.add_argument("--v1-only", action="store_true", help="Only v1 agents")
    parser.add_argument("--v2-only", action="store_true", help="Only v2 agents")
    args = parser.parse_args()

    if args.v1_only:
        versions = ["v1"]
    elif args.v2_only:
        versions = ["v2"]
    else:
        versions = ["v1", "v2"]

    record_all(versions)


if __name__ == "__main__":
    main()
