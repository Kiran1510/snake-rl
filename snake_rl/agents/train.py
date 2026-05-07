"""
Training loop for SARSA-based agents.

This is a generic SARSA training loop that works with any agent
implementing .act(obs), .update(...), and .epsilon. It handles:
    - Episode management (reset, step, terminal detection)
    - Epsilon scheduling (linear decay)
    - Logging (per-episode metrics via RunLogger)
    - Progress reporting

The same loop is used for linear FA, tile coding, and DQN agents.
Linear FA and tile coding agents do a semi-gradient SARSA update on
each step; the DQN agent stores the transition and runs a mini-batch
Double-Q update internally. Only the agent's `.update()` method differs
— the surrounding loop interface is identical.
"""

import time
from typing import Optional, Callable

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.utils.experiment import (
    RunLogger,
    ExperimentConfig,
    ExperimentResult,
    get_epsilon,
)


def train_sarsa(
    agent,
    env: SnakeEnv,
    config: ExperimentConfig,
    logger: Optional[RunLogger] = None,
    print_every: int = 500,
    callback: Optional[Callable] = None,
) -> RunLogger:
    """
    Train a SARSA agent on the Snake environment.

    Parameters
    ----------
    agent : object
        Agent with .act(obs), .update(obs, action, reward, next_obs,
        next_action, terminated), and .epsilon attribute.
    env : SnakeEnv
        The Snake environment.
    config : ExperimentConfig
        Training configuration (n_episodes, epsilon schedule, etc.).
    logger : RunLogger or None
        Logger to use. Created if None.
    print_every : int
        Print progress every N episodes. Set to 0 to disable.
    callback : callable or None
        Optional function called after each episode with
        (episode, score, logger) arguments.

    Returns
    -------
    RunLogger
        Logger with all episode metrics.
    """
    if logger is None:
        logger = RunLogger()

    for episode in range(config.n_episodes):
        # Update epsilon
        agent.epsilon = get_epsilon(episode, config)

        # Run one episode
        episode_reward, score, steps, cause = _run_episode(agent, env, config)

        # Log
        logger.log_episode(
            score=score,
            reward=episode_reward,
            steps=steps,
            epsilon=agent.epsilon,
            cause=cause,
        )

        # Agent end-of-episode hook (e.g., target network update)
        if hasattr(agent, "on_episode_end"):
            agent.on_episode_end()

        # Progress reporting
        if print_every > 0 and (episode + 1) % print_every == 0:
            recent_scores = logger.scores[-print_every:]
            avg = sum(recent_scores) / len(recent_scores)
            max_recent = max(recent_scores)
            td_err = agent.get_mean_td_error(1000) if hasattr(agent, "get_mean_td_error") else 0
            elapsed = logger.wall_times[-1] if logger.wall_times else 0
            print(
                f"  Episode {episode + 1:>6d}/{config.n_episodes} | "
                f"Avg score: {avg:>6.2f} | "
                f"Max: {max_recent:>3d} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"TD err: {td_err:.3f} | "
                f"Time: {elapsed:.0f}s"
            )

        # Optional callback
        if callback is not None:
            callback(episode, score, logger)

    return logger


def _run_episode(agent, env: SnakeEnv, config: ExperimentConfig) -> tuple:
    """
    Run a single SARSA episode.

    SARSA requires knowing the next action before updating (on-policy),
    so the loop follows the pattern:
        1. Observe S, choose A
        2. Take A, observe R, S'
        3. Choose A' (using current policy)
        4. Update with (S, A, R, S', A')
        5. S <- S', A <- A'

    Returns
    -------
    tuple of (total_reward, score, steps, cause)
    """
    obs, _ = env.reset()
    action = agent.act(obs)

    total_reward = 0.0
    done = False

    while not done:
        # Take action, observe outcome
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if done:
            # Terminal update: no next action needed
            agent.update(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                next_action=0,  # placeholder, not used when terminated=True
                terminated=True,
            )
        else:
            # Choose next action under current policy (SARSA is on-policy)
            next_action = agent.act(next_obs)

            # Update
            agent.update(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                next_action=next_action,
                terminated=False,
            )

            # Advance
            obs = next_obs
            action = next_action

    cause = info.get("cause", "unknown")
    return total_reward, env.score, env.steps, cause


def run_experiment(
    agent_factory: Callable,
    config: ExperimentConfig,
    seeds: Optional[list[int]] = None,
    print_every: int = 500,
) -> ExperimentResult:
    """
    Run a full experiment: multiple seeds of the same configuration.

    Parameters
    ----------
    agent_factory : callable
        Function that takes (seed: int) and returns a configured agent.
        Example: lambda seed: LinearSarsaAgent(rep, alpha=0.001, seed=seed)
    config : ExperimentConfig
        Experiment configuration.
    seeds : list of int or None
        Random seeds to use. Defaults to [0, 1, ..., n_seeds-1].
    print_every : int
        Print progress every N episodes per seed.

    Returns
    -------
    ExperimentResult
        Aggregated results across all seeds.
    """
    if seeds is None:
        seeds = list(range(config.n_seeds))

    result = ExperimentResult(config)

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Seed {i + 1}/{len(seeds)} (seed={seed})")
        print(f"{'='*60}")

        # Create fresh agent and environment for this seed
        agent = agent_factory(seed)
        env = SnakeEnv(
            grid_size=config.grid_size,
            seed=seed + 1000,  # offset so env seed != agent seed
        )

        logger = train_sarsa(
            agent=agent,
            env=env,
            config=config,
            print_every=print_every,
        )

        result.add_run(logger, seed)

        # Print seed summary
        summary = logger.summary()
        print(f"\n  Seed {seed} complete: "
              f"final_score={summary['mean_score_final']:.2f} ± "
              f"{summary['std_score_final']:.2f}, "
              f"time={summary['wall_time_seconds']:.0f}s")

    # Print overall summary
    perf = result.final_performance()
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {config.name}")
    print(f"  Mean final score: {perf['mean_score']:.2f} ± {perf['std_score']:.2f}")
    print(f"  Per-seed means:   {perf['per_seed_means']}")
    print(f"{'='*60}")

    return result
