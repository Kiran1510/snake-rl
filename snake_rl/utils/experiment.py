"""
Experiment infrastructure for tracking, saving, and analyzing training runs.

Core components:
    - RunLogger: Tracks per-episode metrics during a single training run.
    - ExperimentResult: Stores and analyzes results across multiple seeds.
    - save_results / load_results: JSON-based persistence.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


class RunLogger:
    """
    Tracks per-episode metrics during a single training run.

    Usage
    -----
    >>> logger = RunLogger()
    >>> for episode in range(n_episodes):
    ...     # ... run episode ...
    ...     logger.log_episode(score=score, reward=total_reward,
    ...                        steps=steps, epsilon=eps)
    >>> logger.summary()
    """

    def __init__(self):
        self.scores: list[int] = []
        self.rewards: list[float] = []
        self.steps: list[int] = []
        self.epsilons: list[float] = []
        self.causes: list[str] = []
        self.wall_time_start: float = time.time()
        self.wall_times: list[float] = []  # cumulative wall time per episode

    def log_episode(
        self,
        score: int,
        reward: float,
        steps: int,
        epsilon: float = 0.0,
        cause: str = "",
    ) -> None:
        """Log metrics for one completed episode."""
        self.scores.append(score)
        self.rewards.append(reward)
        self.steps.append(steps)
        self.epsilons.append(epsilon)
        self.causes.append(cause)
        self.wall_times.append(time.time() - self.wall_time_start)

    def get_smoothed_scores(self, window: int = 100) -> np.ndarray:
        """Return scores smoothed with a moving average."""
        if len(self.scores) < window:
            window = max(1, len(self.scores))
        return np.convolve(
            self.scores, np.ones(window) / window, mode="valid"
        )

    def get_smoothed_rewards(self, window: int = 100) -> np.ndarray:
        """Return rewards smoothed with a moving average."""
        if len(self.rewards) < window:
            window = max(1, len(self.rewards))
        return np.convolve(
            self.rewards, np.ones(window) / window, mode="valid"
        )

    def summary(self, last_n: int = 100) -> dict:
        """
        Return summary statistics.

        Parameters
        ----------
        last_n : int
            Number of final episodes to compute "final performance" over.
        """
        scores = np.array(self.scores)
        rewards = np.array(self.rewards)
        n = min(last_n, len(scores))

        return {
            "total_episodes": len(self.scores),
            "total_steps": sum(self.steps),
            "wall_time_seconds": self.wall_times[-1] if self.wall_times else 0,
            "mean_score_all": float(np.mean(scores)),
            "std_score_all": float(np.std(scores)),
            "max_score": int(np.max(scores)) if len(scores) > 0 else 0,
            "mean_score_final": float(np.mean(scores[-n:])),
            "std_score_final": float(np.std(scores[-n:])),
            "mean_reward_final": float(np.mean(rewards[-n:])),
            "mean_steps_final": float(np.mean(self.steps[-n:])),
        }

    def to_dict(self) -> dict:
        """Serialize all logged data to a dictionary."""
        return {
            "scores": self.scores,
            "rewards": self.rewards,
            "steps": self.steps,
            "epsilons": self.epsilons,
            "causes": self.causes,
            "wall_times": self.wall_times,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunLogger":
        """Reconstruct a RunLogger from a dictionary."""
        logger = cls()
        logger.scores = data["scores"]
        logger.rewards = data["rewards"]
        logger.steps = data["steps"]
        logger.epsilons = data.get("epsilons", [])
        logger.causes = data.get("causes", [])
        logger.wall_times = data.get("wall_times", [])
        return logger


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment (one algorithm × one representation)."""
    algorithm: str          # e.g., "linear_sarsa", "tile_sarsa", "mlp_sarsa"
    representation: str     # e.g., "compact", "local", "extended"
    n_episodes: int = 10000
    n_seeds: int = 5
    grid_size: int = 20
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_fraction: float = 0.8  # decay over this fraction of episodes
    alpha: float = 0.001
    # Algorithm-specific params stored as a dict
    algo_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(**data)

    @property
    def name(self) -> str:
        """Short identifier for this experiment."""
        return f"{self.algorithm}__{self.representation}"


class ExperimentResult:
    """
    Stores results from multiple seeds of the same experiment configuration.

    Provides methods for computing mean/std learning curves and
    convergence statistics across seeds.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.run_loggers: list[RunLogger] = []
        self.seeds: list[int] = []

    def add_run(self, logger: RunLogger, seed: int) -> None:
        """Add results from one seed."""
        self.run_loggers.append(logger)
        self.seeds.append(seed)

    @property
    def n_runs(self) -> int:
        return len(self.run_loggers)

    def mean_learning_curve(self, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and std of smoothed score curves across seeds.

        Returns
        -------
        mean : np.ndarray
        std : np.ndarray
        """
        curves = []
        min_len = min(
            len(logger.get_smoothed_scores(window))
            for logger in self.run_loggers
        )
        for logger in self.run_loggers:
            curve = logger.get_smoothed_scores(window)[:min_len]
            curves.append(curve)

        curves = np.array(curves)
        return np.mean(curves, axis=0), np.std(curves, axis=0)

    def mean_reward_curve(self, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and std of smoothed reward curves across seeds."""
        curves = []
        min_len = min(
            len(logger.get_smoothed_rewards(window))
            for logger in self.run_loggers
        )
        for logger in self.run_loggers:
            curve = logger.get_smoothed_rewards(window)[:min_len]
            curves.append(curve)

        curves = np.array(curves)
        return np.mean(curves, axis=0), np.std(curves, axis=0)

    def final_performance(self, last_n: int = 1000) -> dict:
        """
        Aggregate final performance across all seeds.

        Returns mean and std of the per-seed final scores.
        """
        seed_means = []
        for logger in self.run_loggers:
            scores = np.array(logger.scores)
            n = min(last_n, len(scores))
            seed_means.append(float(np.mean(scores[-n:])))

        seed_means = np.array(seed_means)
        return {
            "mean_score": float(np.mean(seed_means)),
            "std_score": float(np.std(seed_means)),
            "per_seed_means": seed_means.tolist(),
            "n_seeds": self.n_runs,
        }

    def convergence_episode(self, threshold_fraction: float = 0.9, window: int = 100) -> dict:
        """
        Estimate the episode at which each seed reaches threshold_fraction
        of its final performance.

        Returns per-seed and mean convergence episodes.
        """
        convergence_eps = []
        for logger in self.run_loggers:
            smoothed = logger.get_smoothed_scores(window)
            if len(smoothed) == 0:
                convergence_eps.append(float("inf"))
                continue
            final_perf = np.mean(smoothed[-100:]) if len(smoothed) >= 100 else np.mean(smoothed)
            threshold = threshold_fraction * final_perf
            reached = np.where(smoothed >= threshold)[0]
            if len(reached) > 0:
                convergence_eps.append(int(reached[0]) + window)
            else:
                convergence_eps.append(len(logger.scores))

        return {
            "per_seed": convergence_eps,
            "mean": float(np.mean(convergence_eps)),
            "std": float(np.std(convergence_eps)),
        }

    def summary(self) -> dict:
        """Full summary including config, final performance, and convergence."""
        return {
            "config": self.config.to_dict(),
            "n_runs": self.n_runs,
            "seeds": self.seeds,
            "final_performance": self.final_performance(),
            "convergence": self.convergence_episode(),
            "per_seed_summaries": [
                logger.summary() for logger in self.run_loggers
            ],
        }

    def to_dict(self) -> dict:
        """Serialize everything for saving."""
        return {
            "config": self.config.to_dict(),
            "seeds": self.seeds,
            "runs": [logger.to_dict() for logger in self.run_loggers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        """Reconstruct from a dictionary."""
        config = ExperimentConfig.from_dict(data["config"])
        result = cls(config)
        result.seeds = data["seeds"]
        result.run_loggers = [
            RunLogger.from_dict(run) for run in data["runs"]
        ]
        return result


# ============================================================
# Persistence
# ============================================================

def save_results(result: ExperimentResult, filepath: str) -> None:
    """Save experiment results to a JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> ExperimentResult:
    """Load experiment results from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return ExperimentResult.from_dict(data)


# ============================================================
# Epsilon schedule
# ============================================================

def get_epsilon(episode: int, config: ExperimentConfig) -> float:
    """
    Compute epsilon for a given episode using linear decay.

    Decays from epsilon_start to epsilon_end over the first
    epsilon_decay_fraction of total episodes, then stays at epsilon_end.
    """
    decay_episodes = int(config.n_episodes * config.epsilon_decay_fraction)
    if episode >= decay_episodes:
        return config.epsilon_end
    fraction = episode / decay_episodes
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)
