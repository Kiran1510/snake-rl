"""
Record GIFs for all 9 configurations (3 algorithms × 3 representations).

Usage:
    python record_all_gifs.py
    python record_all_gifs.py --algo mlp
    python record_all_gifs.py --rep compact
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    print("ERROR: pygame required. pip install pygame")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    print("ERROR: Pillow required. pip install Pillow")
    sys.exit(1)

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.utils.save_load import load_agent_weights
from record_gameplay import render_frame, make_agent, weight_name

try:
    from snake_rl.agents.double_dqn import DoubleDQNAgent
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. MLP GIFs will be skipped.")

GRID_SIZE = 20
CELL_SIZE = 30
WEIGHTS_DIR = "weights"
RECORDINGS_DIR = "recordings"


def record_one_gif(algo, rep_name, n_episodes=3, max_steps_per_ep=300, fps=10, seed=0):
    """Record a GIF for one agent configuration."""
    name = weight_name(algo, rep_name, seed=seed)
    ext = ".pt" if algo == "mlp" else ".npz"
    weight_file = os.path.join(WEIGHTS_DIR, f"{name}{ext}")

    if not os.path.exists(weight_file):
        print(f"  SKIP: no weights at {weight_file}")
        return False

    try:
        agent = make_agent(algo, rep_name, weights_dir=WEIGHTS_DIR, name=name)
        load_agent_weights(agent, name, WEIGHTS_DIR)
        agent.epsilon = 0.0
    except Exception as e:
        print(f"  SKIP: failed to load agent — {e}")
        return False

    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    env = SnakeEnv(grid_size=GRID_SIZE, max_steps_factor=3, seed=999)

    grid_w = GRID_SIZE * CELL_SIZE
    panel_h = 60

    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((grid_w, grid_w + panel_h))

    label = f"{algo}_sarsa + {rep_name}"
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
            frames.append(Image.frombytes("RGB", (grid_w, grid_w + panel_h), frame_data))

    pygame.quit()

    if frames:
        algo_label = "double_dqn" if algo == "mlp" else f"{algo}_sarsa"
        filepath = os.path.join(RECORDINGS_DIR, f"{algo_label}__{rep_name}.gif")
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


def main():
    parser = argparse.ArgumentParser(description="Record GIFs for all trained agents")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["linear", "tile", "mlp"])
    parser.add_argument("--rep", type=str, default=None,
                        choices=["compact", "local", "extended"])
    parser.add_argument("--seed", type=int, default=0,
                        help="Which seed's weights to load (default: 0)")
    args = parser.parse_args()

    algos = [args.algo] if args.algo else (
        ["linear", "tile", "mlp"] if HAS_TORCH else ["linear", "tile"]
    )
    reps = [args.rep] if args.rep else ["compact", "local", "extended"]

    saved = 0
    skipped = 0

    for algo in algos:
        for rep in reps:
            print(f"\n[{algo} × {rep}] Recording...")
            success = record_one_gif(algo, rep, seed=args.seed)
            if success:
                saved += 1
            else:
                skipped += 1

    print(f"\nDone: {saved} GIFs saved, {skipped} skipped")
    print(f"Output: {os.path.abspath(RECORDINGS_DIR)}/")


if __name__ == "__main__":
    main()
