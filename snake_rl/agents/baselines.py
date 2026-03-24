"""
Baseline agents for the Snake environment.

These provide reference points for evaluating learned policies:
    - RandomAgent: uniform random actions (lower bound)
    - GreedyHeuristicAgent: hand-coded "go toward food, avoid danger" (upper reference)
"""

import numpy as np
from snake_rl.env.snake_env import Action, _turn_left, _turn_right


class RandomAgent:
    """
    Uniform random action selection.

    This is the lower bound on performance — any reasonable
    learned policy should beat this.
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> int:
        """Select a random action."""
        return int(self.rng.integers(3))

    def __repr__(self):
        return "RandomAgent()"


class GreedyHeuristicAgent:
    """
    Hand-coded greedy heuristic that moves toward food while avoiding
    immediate danger (wall or own body one step away).

    Strategy:
        1. Compute which of the 3 relative actions are safe (no immediate collision).
        2. Among safe actions, prefer the one that moves closest to food
           (by Manhattan distance).
        3. If no action is safe (trapped), go straight as a last resort.

    This is a strong baseline that represents what a simple rule-based
    agent can achieve without any learning.
    """

    def act(self, obs: dict) -> int:
        """Select the best heuristic action."""
        snake = obs["snake"]
        food = obs["food"]
        direction = obs["direction"]
        grid_size = obs["grid_size"]

        head_x, head_y = snake[0]
        snake_set = set(map(tuple, snake))

        # Compute the new head position for each relative action
        candidates = []
        for action in [Action.STRAIGHT, Action.LEFT, Action.RIGHT]:
            new_dir = self._resolve_action(direction, action)
            dx, dy = new_dir
            new_head = (head_x + dx, head_y + dy)

            # Check safety
            nx, ny = new_head
            is_safe = (
                0 <= nx < grid_size and
                0 <= ny < grid_size and
                new_head not in snake_set
            )

            # Manhattan distance to food from new position
            dist = abs(nx - food[0]) + abs(ny - food[1])

            candidates.append((action, new_head, is_safe, dist))

        # Filter to safe actions
        safe = [(a, h, d) for a, h, safe, d in candidates if safe]

        if not safe:
            # No safe action — just go straight (we're dead anyway)
            return Action.STRAIGHT

        # Among safe actions, pick the one closest to food
        safe.sort(key=lambda x: x[2])  # sort by distance
        return safe[0][0]

    @staticmethod
    def _resolve_action(direction: tuple, action: int) -> tuple:
        """Convert relative action to absolute direction."""
        if action == Action.STRAIGHT:
            return direction
        elif action == Action.LEFT:
            return _turn_left(direction)
        elif action == Action.RIGHT:
            return _turn_right(direction)
        return direction

    def __repr__(self):
        return "GreedyHeuristicAgent()"


def evaluate_agent(agent, env, n_episodes: int = 100, verbose: bool = False) -> dict:
    """
    Evaluate an agent over multiple episodes.

    Parameters
    ----------
    agent : object with .act(obs) -> int
        The agent to evaluate.
    env : SnakeEnv
        The environment.
    n_episodes : int
        Number of evaluation episodes.
    verbose : bool
        Whether to print per-episode results.

    Returns
    -------
    dict with keys:
        - scores: list of scores per episode
        - lengths: list of snake lengths at episode end
        - steps: list of steps per episode
        - mean_score: float
        - std_score: float
        - max_score: int
        - mean_steps: float
        - causes: dict mapping cause -> count
    """
    scores = []
    lengths = []
    steps_list = []
    causes = {}

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_done = False

        while not episode_done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_done = terminated or truncated

        scores.append(env.score)
        lengths.append(env.get_snake_length())
        steps_list.append(env.steps)

        cause = info.get("cause", "unknown")
        causes[cause] = causes.get(cause, 0) + 1

        if verbose and (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: "
                  f"score={env.score}, steps={env.steps}, cause={cause}")

    scores_arr = np.array(scores)
    return {
        "scores": scores,
        "lengths": lengths,
        "steps": steps_list,
        "mean_score": float(np.mean(scores_arr)),
        "std_score": float(np.std(scores_arr)),
        "max_score": int(np.max(scores_arr)),
        "mean_steps": float(np.mean(steps_list)),
        "causes": causes,
    }
