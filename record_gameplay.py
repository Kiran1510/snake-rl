"""
Record and visualize trained agents playing Snake.

Features:
    - Record gameplay as GIF animations
    - Live Pygame playback of trained agents
    - Side-by-side comparison of multiple agents
    - Overlay Q-values and agent info on screen

Usage:
    # Train all agents and save weights
    python record_gameplay.py --train

    # Record GIFs for all saved agents
    python record_gameplay.py --record

    # Watch a specific agent play live
    python record_gameplay.py --watch tile compact

    # Record a specific agent
    python record_gameplay.py --record --algo tile --rep compact
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.representations.features import (
    CompactRepresentation, LocalNeighborhoodRepresentation, ExtendedRepresentation,
)
from snake_rl.agents.linear_sarsa import LinearSarsaAgent
from snake_rl.agents.tile_sarsa import TileCodingSarsaAgent
from snake_rl.agents.train import train_sarsa
from snake_rl.utils.experiment import ExperimentConfig
from snake_rl.utils.save_load import save_agent, load_agent_weights

try:
    from snake_rl.agents.mlp_sarsa import MLPSarsaAgent
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


GRID_SIZE = 20
CELL_SIZE = 30
WEIGHTS_DIR = "weights"
RECORDINGS_DIR = "recordings"

REPRESENTATIONS = {
    "compact": CompactRepresentation,
    "local": LocalNeighborhoodRepresentation,
    "extended": ExtendedRepresentation,
}

ALGO_CONFIGS = {
    "linear": {"alpha": 0.01, "n_episodes": 10000},
    "tile":   {"alpha": 0.05, "n_episodes": 10000},
    "mlp":    {"alpha": 0.001, "n_episodes": 10000},
}

# Colors
COLOR_BG = (20, 20, 20)
COLOR_GRID = (40, 40, 40)
COLOR_SNAKE_HEAD = (0, 200, 80)
COLOR_SNAKE_BODY = (0, 150, 60)
COLOR_FOOD = (220, 50, 50)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 140)
COLOR_PANEL = (30, 30, 30)


def make_agent(algo, rep_name):
    """Create a fresh agent for the given algo/rep."""
    rep = REPRESENTATIONS[rep_name]()

    if algo == "linear":
        return LinearSarsaAgent(rep, alpha=0.01, gamma=0.95, seed=42)
    elif algo == "tile":
        agent = TileCodingSarsaAgent(rep, n_tilings=8, n_tiles_per_dim=4,
                                      max_size=65536, alpha=0.05, gamma=0.95, seed=42)
        return agent
    elif algo == "mlp":
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MLP")
        return MLPSarsaAgent(rep, hidden_dim=128, alpha=0.001, gamma=0.95, seed=42)


def weight_name(algo, rep):
    return f"{algo}_sarsa__{rep}"


def train_and_save(algo, rep_name):
    """Train an agent and save its weights."""
    print(f"\nTraining {algo} on {rep_name}...")
    config = ExperimentConfig(
        algorithm=f"{algo}_sarsa", representation=rep_name,
        n_episodes=ALGO_CONFIGS[algo]["n_episodes"],
        grid_size=GRID_SIZE, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.01,
        epsilon_decay_fraction=0.8,
        alpha=ALGO_CONFIGS[algo]["alpha"],
    )

    agent = make_agent(algo, rep_name)
    env = SnakeEnv(grid_size=GRID_SIZE, seed=42)

    if algo == "tile":
        agent.initialize(env)

    train_sarsa(agent, env, config, print_every=2000)
    save_agent(agent, weight_name(algo, rep_name), WEIGHTS_DIR)
    return agent


def load_trained_agent(algo, rep_name):
    """Load a trained agent from saved weights."""
    agent = make_agent(algo, rep_name)
    name = weight_name(algo, rep_name)
    load_agent_weights(agent, name, WEIGHTS_DIR)
    agent.epsilon = 0.01  # near-greedy for evaluation
    return agent


def render_frame(surface, env, agent_name="", score_info="", q_vals=None):
    """Render one frame of the game onto a pygame surface."""
    cs = CELL_SIZE
    grid_w = env.grid_size * cs
    panel_h = 60

    surface.fill(COLOR_BG)

    # Grid lines
    for x in range(env.grid_size + 1):
        pygame.draw.line(surface, COLOR_GRID, (x * cs, 0), (x * cs, grid_w))
    for y in range(env.grid_size + 1):
        pygame.draw.line(surface, COLOR_GRID, (0, y * cs), (grid_w, y * cs))

    # Snake
    for i in range(len(env.snake) - 1, -1, -1):
        sx, sy = env.snake[i]
        color = COLOR_SNAKE_HEAD if i == 0 else COLOR_SNAKE_BODY
        rect = pygame.Rect(sx * cs + 1, sy * cs + 1, cs - 2, cs - 2)
        pygame.draw.rect(surface, color, rect)

        if i == 0:
            # Eyes
            cx, cy = sx * cs + cs // 2, sy * cs + cs // 2
            dx, dy = env.direction
            perp_x, perp_y = -dy, dx
            offset = cs // 5
            fwd = cs // 6
            eye_r = max(2, cs // 10)
            e1 = (cx + perp_x * offset + dx * fwd, cy + perp_y * offset + dy * fwd)
            e2 = (cx - perp_x * offset + dx * fwd, cy - perp_y * offset + dy * fwd)
            pygame.draw.circle(surface, (255, 255, 255), e1, eye_r)
            pygame.draw.circle(surface, (255, 255, 255), e2, eye_r)

    # Food
    fx, fy = env.food
    if 0 <= fx < env.grid_size and 0 <= fy < env.grid_size:
        food_rect = pygame.Rect(fx * cs + 2, fy * cs + 2, cs - 4, cs - 4)
        pygame.draw.rect(surface, COLOR_FOOD, food_rect, border_radius=cs // 4)

    # Info panel
    panel_y = grid_w + 5
    font = pygame.font.SysFont("monospace", 16)
    font_small = pygame.font.SysFont("monospace", 13)

    # Agent name
    if agent_name:
        name_surf = font.render(agent_name, True, COLOR_TEXT)
        surface.blit(name_surf, (8, panel_y))

    # Score and steps
    info_text = f"Score: {env.score}   Steps: {env.steps}   Length: {len(env.snake)}"
    info_surf = font_small.render(info_text, True, COLOR_TEXT_DIM)
    surface.blit(info_surf, (8, panel_y + 22))

    # Q-values
    if q_vals is not None:
        q_text = f"Q: [{q_vals[0]:+.1f}, {q_vals[1]:+.1f}, {q_vals[2]:+.1f}]"
        q_surf = font_small.render(q_text, True, COLOR_TEXT_DIM)
        surface.blit(q_surf, (8, panel_y + 40))


def record_gif(algo, rep_name, n_episodes=3, max_steps_per_ep=200, fps=10):
    """Record gameplay as a GIF."""
    if not HAS_PYGAME:
        print("pygame required for recording")
        return
    if not HAS_PIL:
        print("Pillow required for GIF recording. Install: pip install Pillow")
        return

    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    agent = load_trained_agent(algo, rep_name)
    env = SnakeEnv(grid_size=GRID_SIZE, seed=999)

    grid_w = GRID_SIZE * CELL_SIZE
    panel_h = 60
    surface = pygame.Surface((grid_w, grid_w + panel_h))

    name = f"{algo}_sarsa + {rep_name}"
    frames = []

    pygame.init()
    pygame.font.init()

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

            render_frame(surface, env, agent_name=name, q_vals=q_vals)
            frame_data = pygame.image.tostring(surface, "RGB")
            frame_img = Image.frombytes("RGB", (grid_w, grid_w + panel_h), frame_data)
            frames.append(frame_img)

    pygame.quit()

    if frames:
        filepath = os.path.join(RECORDINGS_DIR, f"{algo}_sarsa__{rep_name}.gif")
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0,
            optimize=True,
        )
        print(f"Saved GIF ({len(frames)} frames): {filepath}")


def watch_live(algo, rep_name, fps=10):
    """Watch a trained agent play live with Pygame."""
    if not HAS_PYGAME:
        print("pygame required for live playback")
        return

    pygame.init()
    agent = load_trained_agent(algo, rep_name)
    env = SnakeEnv(grid_size=GRID_SIZE, seed=int(time.time()) % 10000)

    grid_w = GRID_SIZE * CELL_SIZE
    panel_h = 60
    screen = pygame.display.set_mode((grid_w, grid_w + panel_h))
    pygame.display.set_caption(f"Snake RL — {algo}_sarsa + {rep_name}")
    clock = pygame.time.Clock()

    name = f"{algo}_sarsa + {rep_name}"
    obs, _ = env.reset()
    running = True
    total_episodes = 0
    total_score = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                elif event.key == pygame.K_UP:
                    fps = min(60, fps + 2)
                elif event.key == pygame.K_DOWN:
                    fps = max(2, fps - 2)

        q_vals = agent.q_values(obs) if hasattr(agent, "q_values") else None
        action = agent.act(obs)
        obs, _, term, trunc, info = env.step(action)

        render_frame(screen, env, agent_name=name, q_vals=q_vals)
        pygame.display.flip()
        clock.tick(fps)

        if term or trunc:
            total_episodes += 1
            total_score += env.score
            avg = total_score / total_episodes
            pygame.display.set_caption(
                f"Snake RL — {name} | Ep {total_episodes} | Avg: {avg:.1f} | "
                f"Last: {env.score} | {info.get('cause', '')}"
            )
            time.sleep(0.5)
            obs, _ = env.reset()

    pygame.quit()


def train_all_and_save():
    """Train all 9 configurations and save weights."""
    algos = ["linear", "tile", "mlp"] if HAS_TORCH else ["linear", "tile"]
    reps = ["compact", "local", "extended"]

    for algo in algos:
        for rep in reps:
            try:
                train_and_save(algo, rep)
            except Exception as e:
                print(f"FAILED: {algo} x {rep}: {e}")


def record_all():
    """Record GIFs for all saved agents."""
    algos = ["linear", "tile", "mlp"] if HAS_TORCH else ["linear", "tile"]
    reps = ["compact", "local", "extended"]

    for algo in algos:
        for rep in reps:
            name = weight_name(algo, rep)
            weight_file = os.path.join(WEIGHTS_DIR, f"{name}.npz")
            pt_file = os.path.join(WEIGHTS_DIR, f"{name}.pt")
            if os.path.exists(weight_file) or os.path.exists(pt_file):
                try:
                    print(f"\nRecording {algo} x {rep}...")
                    record_gif(algo, rep, n_episodes=3, fps=10)
                except Exception as e:
                    print(f"FAILED: {algo} x {rep}: {e}")
            else:
                print(f"No weights for {algo} x {rep} — skipping")


def main():
    parser = argparse.ArgumentParser(description="Snake RL Gameplay Recording")
    parser.add_argument("--train", action="store_true",
                        help="Train all agents and save weights")
    parser.add_argument("--record", action="store_true",
                        help="Record GIFs for all saved agents")
    parser.add_argument("--watch", nargs=2, metavar=("ALGO", "REP"),
                        help="Watch a trained agent play live (e.g., --watch tile compact)")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["linear", "tile", "mlp"])
    parser.add_argument("--rep", type=str, default=None,
                        choices=["compact", "local", "extended"])
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if args.train:
        train_all_and_save()
    elif args.watch:
        algo, rep = args.watch
        watch_live(algo, rep, fps=args.fps)
    elif args.record:
        if args.algo and args.rep:
            record_gif(args.algo, args.rep, fps=args.fps)
        else:
            record_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
