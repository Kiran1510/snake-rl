from snake_rl.agents.baselines import RandomAgent, GreedyHeuristicAgent, evaluate_agent
from snake_rl.agents.linear_sarsa import LinearSarsaAgent
from snake_rl.agents.tile_sarsa import TileCodingSarsaAgent
from snake_rl.agents.train import train_sarsa, run_experiment

try:
    from snake_rl.agents.double_dqn import DoubleDQNAgent
except ImportError:
    pass
