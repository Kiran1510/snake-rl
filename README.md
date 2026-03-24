# Snake RL: Comparing Function Approximation Methods and State Representations

**CS 5180: Reinforcement Learning and Sequential Decision Making — Spring 2026**
**Northeastern University**

Kiran Sairam Bethi Balagangadaran (NEU ID: 002031062)

---

## Overview

This project investigates how different **state representations** interact with different **function approximation methods** in reinforcement learning, using the classic game Snake as a testbed.

The central research question:

> *Given state representations of varying complexity, how do hand-crafted linear features, automatically constructed tile-coded features, and neural network-learned features compare in their ability to learn effective control policies via on-policy temporal-difference learning?*

The project compares **three function approximation approaches** — linear function approximation with hand-crafted features, tile coding, and a small MLP — across **three state representations** of increasing complexity, all using semi-gradient SARSA as the underlying control algorithm. This isolates the function approximation method as the primary experimental variable.

### Why This Matters

Most existing Snake RL work uses off-policy DQN with a single state representation. No prior work has systematically compared function approximation methods across multiple representations under a common on-policy TD framework. This project fills that gap, with theoretical grounding in the convergence guarantees of linear TD methods (Tsitsiklis & Van Roy, 1997) and the deadly triad framework (Sutton & Barto, 2018).

---

## Project Structure

```
snake_RL/
├── snake_rl/                       # Main package
│   ├── __init__.py
│   ├── env/                        # Game environment
│   │   ├── __init__.py
│   │   ├── snake_env.py            # Core Snake environment (Gymnasium-style)
│   │   └── renderer.py             # ASCII + Pygame rendering, human play mode
│   ├── representations/            # State feature extractors
│   │   ├── __init__.py
│   │   └── features.py             # Compact, Local Neighborhood, Extended representations
│   ├── agents/                     # RL agents and baselines
│   │   ├── __init__.py
│   │   └── baselines.py            # Random agent, Greedy heuristic, evaluation utility
│   └── utils/                      # Experiment infrastructure
│       ├── __init__.py
│       ├── experiment.py           # RunLogger, ExperimentConfig, ExperimentResult, persistence
│       └── plotting.py             # Learning curves, comparisons, final performance charts
├── tests/                          # Test suite (145 tests)
│   ├── run_tests.py                # Standalone test runner (no pytest dependency)
│   ├── test_env.py                 # 75 environment tests
│   ├── test_representations.py     # 34 representation tests
│   ├── test_baselines.py           # 11 baseline agent tests
│   └── test_experiment.py          # 25 experiment infrastructure tests
├── revised_project_proposal.tex    # LaTeX project proposal (approved)
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

---

## Environment

### Snake Game (`snake_rl/env/snake_env.py`)

A clean, from-scratch implementation of Snake following the **Gymnasium-style interface** (`reset()`, `step()` returning the standard 5-tuple). The environment manages pure game logic and exposes raw state as a dictionary — state representation extraction is handled by separate modules.

#### MDP Formulation

| Component | Details |
|---|---|
| **Grid** | 20 × 20 (configurable) |
| **Actions** | `STRAIGHT (0)`, `LEFT (1)`, `RIGHT (2)` — relative to current heading |
| **Rewards** | +10 (food), -10 (death), -0.1 (per step) |
| **Discount** | γ = 0.95 |
| **Termination** | Wall collision, self-collision, or max steps exceeded |
| **Episode** | Starts with snake of length 3 at grid center, facing right |

The action space uses **relative actions** (straight/left/right) rather than absolute directions (up/down/left/right). This avoids the "reverse into yourself" problem entirely — the environment converts relative actions to absolute directions using rotation.

#### Observation Dictionary

`reset()` and `step()` return a raw observation dict containing everything any representation extractor could need:

```python
{
    "snake": [(x, y), ...],   # list of positions, head first
    "food": (x, y),           # current food position
    "direction": (dx, dy),    # current heading as a unit vector
    "score": int,             # food eaten this episode
    "steps": int,             # steps taken this episode
    "grid_size": int          # grid dimension
}
```

#### Key Design Decisions

- **Relative actions over absolute:** Reduces action space from 4 to 3 and eliminates invalid actions (reversing). Direction resolution uses vector rotation (`_turn_left`, `_turn_right`).
- **Max step truncation:** Episodes are capped at `max_steps_factor × grid_size²` to prevent infinite loops where the agent avoids food indefinitely. Truncated episodes set `truncated=True` (distinct from `terminated=True` for deaths).
- **Food placement:** Food spawns uniformly at random on empty cells. When the snake fills the entire grid, food is placed at an impossible location.
- **Configurable rewards:** `reward_food`, `reward_death`, and `reward_step` are constructor parameters for reward shaping experiments.
- **Reproducibility:** Full seeding support via `numpy.random.default_rng`. Same seed + same actions = identical trajectory.

### Rendering (`snake_rl/env/renderer.py`)

Two rendering backends:

- **`AsciiRenderer`**: Terminal-based, no dependencies. Outputs a character grid where `H` = head, `B` = body, `F` = food, `.` = empty. Useful for debugging and logging.
- **`PygameRenderer`**: GUI-based with colored cells, snake eyes that follow the heading direction, food with rounded corners, and a score bar. Lazy-imports pygame so the package works without it installed.

#### Human Play Mode

```bash
python -c "from snake_rl.env.renderer import play_human; play_human()"
```

Arrow keys to control. Auto-restarts on death. Close window to quit. Useful for sanity-checking the environment feels correct.

---

## State Representations

### Design Philosophy

The three representations form a spectrum from **minimal hand-crafted features** (where even tabular methods can work) to **rich high-dimensional features** (where function approximation is essential). This allows studying when and why approximation helps.

All representations are implemented as classes in `snake_rl/representations/features.py` with a common interface:

```python
rep = CompactRepresentation()
state_features = rep.get_state_features(obs)         # shape: (state_dim,)
sa_features = rep.get_features(obs, action)           # shape: (feature_dim,)
```

The `get_features(obs, action)` method constructs state-action features using **one-hot action stacking**: the state feature vector is placed in the block corresponding to the given action, with zeros elsewhere. This gives each action its own weight vector in a linear model.

### Representation 1: Compact Features (11 dimensions)

```python
rep = CompactRepresentation()    # state_dim=11, feature_dim=33
```

A small set of binary features encoding the most essential information:

| Features | Count | Description |
|---|---|---|
| Danger signals | 3 | Binary: danger straight, left, right (wall or body one step away) |
| Direction | 4 | One-hot: current heading (up, down, left, right) |
| Food direction | 4 | Binary: food is above, below, left of, right of head |

This matches the **standard 11-feature design** used in most Snake RL tutorials (Loeber 2020, Zhou 2023). It's highly compressed — the agent knows *what's immediately dangerous* and *where food is*, but nothing about the snake's body layout or spatial structure.

### Representation 2: Local Neighborhood (109 dimensions)

```python
rep = LocalNeighborhoodRepresentation()    # state_dim=109, feature_dim=327
```

A 5×5 grid window centered on the snake's head, providing local spatial awareness:

| Features | Count | Description |
|---|---|---|
| Window cells | 100 | 5×5 grid, each cell one-hot as [empty, food, wall, body] |
| Food direction | 4 | Binary: food is above, below, left of, right of head |
| Length bucket | 5 | One-hot: snake length in ranges [1-5, 6-10, 11-20, 21-50, 51+] |

The window uses absolute coordinates (not relative to heading) so spatial patterns are consistent regardless of direction. The window size is configurable.

### Representation 3: Extended Features (126 dimensions)

```python
rep = ExtendedRepresentation()    # state_dim=126, feature_dim=378
```

The richest representation, combining the local neighborhood with continuous-valued distance features:

| Features | Count | Description |
|---|---|---|
| Window cells | 100 | Same 5×5 one-hot grid as Local Neighborhood |
| Food direction | 4 | Binary food direction signals |
| Length bucket | 5 | One-hot length ranges |
| Manhattan distance to food | 1 | Normalized to [0, 1] |
| Distance to body (4 dirs) | 4 | Nearest body segment in each cardinal direction, normalized |
| Distance to wall (4 dirs) | 4 | Steps to wall in each direction, normalized |
| Normalized snake length | 1 | Length / grid_size² |
| Current direction | 4 | One-hot heading |
| Danger signals | 3 | Binary: same as Compact representation |

**Optional polynomial interactions** (enabled via `include_interactions=True`, adds 12 features for 138 total): degree-2 products between the 3 danger signals and 4 food direction signals. These allow the linear model to capture nonlinear relationships like "danger ahead AND food ahead" without a neural network.

### Performance Benchmarks

Feature extraction speed on a 20×20 grid (measured on Apple M-series, single core):

| Representation | Time per call | Feasibility |
|---|---|---|
| Compact | ~2 µs | No bottleneck |
| Local Neighborhood | ~10 µs | No bottleneck |
| Extended | ~18 µs | No bottleneck |

All representations are fast enough that feature extraction will not be the training bottleneck.

---

## Baseline Agents

### Random Agent

Uniform random action selection. Provides the **lower bound** — any learned policy should beat this convincingly.

```python
agent = RandomAgent(seed=42)
action = agent.act(obs)
```

### Greedy Heuristic Agent

A hand-coded policy that:
1. Computes which of the 3 relative actions are **safe** (no immediate wall or body collision).
2. Among safe actions, picks the one that **minimizes Manhattan distance to food**.
3. If no action is safe (trapped), goes straight.

This represents what a simple rule-based agent can achieve **without any learning** and serves as the **upper reference point**.

```python
agent = GreedyHeuristicAgent()
action = agent.act(obs)
```

### Baseline Results (200 episodes, 20×20 grid)

| Agent | Mean Score | Std | Max Score | Mean Steps | Primary Death Cause |
|---|---|---|---|---|---|
| Random | 0.15 | 0.38 | 2 | 66.2 | Wall collision (89%) |
| Greedy Heuristic | 22.89 | 6.00 | 33 | 329.9 | Self-collision (52%) |

Key observations:
- The random agent almost never eats food and dies quickly from wall collisions.
- The greedy heuristic is surprisingly strong, averaging ~23 food before self-trapping. Its primary failure mode is self-collision — it greedily chases food without planning around its own body.
- The gap between these two (0.15 vs 22.89) defines the performance range that learned agents should fall within.

### Agent Evaluation

The `evaluate_agent()` utility runs any agent for multiple episodes and returns comprehensive statistics:

```python
from snake_rl.agents.baselines import evaluate_agent

results = evaluate_agent(agent, env, n_episodes=200, verbose=True)
print(f"Mean score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
print(f"Max score:  {results['max_score']}")
print(f"Causes:     {results['causes']}")
```

---

## Experiment Infrastructure

### Run Logging (`RunLogger`)

Tracks per-episode metrics during a single training run:

```python
from snake_rl.utils.experiment import RunLogger

logger = RunLogger()
for episode in range(n_episodes):
    # ... run episode ...
    logger.log_episode(
        score=env.score,
        reward=total_reward,
        steps=env.steps,
        epsilon=current_epsilon,
        cause=info.get("cause", "")
    )

# Analyze
smoothed = logger.get_smoothed_scores(window=100)
summary = logger.summary(last_n=1000)
```

Logged metrics: scores, cumulative rewards, step counts, epsilon values, death causes, and wall-clock timestamps.

### Experiment Configuration (`ExperimentConfig`)

Dataclass holding all hyperparameters for one algorithm × representation pair:

```python
from snake_rl.utils.experiment import ExperimentConfig

config = ExperimentConfig(
    algorithm="linear_sarsa",
    representation="compact",
    n_episodes=10000,
    n_seeds=5,
    grid_size=20,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_fraction=0.8,
    alpha=0.001,
    algo_params={"include_interactions": False},
)
```

### Multi-Seed Results (`ExperimentResult`)

Aggregates `RunLogger` objects across multiple random seeds and provides statistical analysis:

```python
from snake_rl.utils.experiment import ExperimentResult

result = ExperimentResult(config)
for seed in range(config.n_seeds):
    logger = train(config, seed)    # your training function
    result.add_run(logger, seed)

# Mean learning curve with confidence bands
mean, std = result.mean_learning_curve(window=100)

# Final performance across seeds
perf = result.final_performance(last_n=1000)
print(f"Score: {perf['mean_score']:.2f} ± {perf['std_score']:.2f}")

# Convergence speed
conv = result.convergence_episode(threshold_fraction=0.9)
print(f"Converged at episode: {conv['mean']:.0f}")
```

### Persistence

Results are fully serializable to JSON for archival and reproducibility:

```python
from snake_rl.utils.experiment import save_results, load_results

save_results(result, "results/linear_sarsa__compact.json")
loaded = load_results("results/linear_sarsa__compact.json")
```

### Epsilon Schedule

Linear decay from `epsilon_start` to `epsilon_end` over a configurable fraction of total episodes:

```python
from snake_rl.utils.experiment import get_epsilon

eps = get_epsilon(episode=500, config=config)
```

### Plotting

All plot functions save to file and return the matplotlib figure:

```python
from snake_rl.utils.plotting import (
    plot_learning_curve,           # Single experiment with confidence band
    plot_reward_curve,             # Same for cumulative reward
    plot_comparison,               # Multiple experiments on one plot
    plot_comparison_by_representation,  # 3-panel: one per representation
    plot_final_performance_table,  # Bar chart of final scores
)

plot_learning_curve(result, window=100, save_path="figures/learning_curve.png")
plot_comparison(all_results, save_path="figures/comparison.png")
```

---

## Algorithms (Planned)

All three algorithms use **semi-gradient SARSA** as the underlying control algorithm, differing only in how the action-value function q̂(s, a) is approximated.

### Algorithm 1: Semi-Gradient SARSA with Linear Function Approximation

Hand-crafted feature vectors with a linear value function:

```
q̂(s, a; w) = wᵀ x(s, a)
```

Weight update: `w ← w + α [R + γ q̂(S', A'; w) - q̂(S, A; w)] x(S, A)`

### Algorithm 2: Semi-Gradient SARSA with Tile Coding

Automatic binary feature construction via overlapping tilings. Still linear, but features are automatically constructed rather than hand-engineered.

### Algorithm 3: Semi-Gradient SARSA with Neural Network (MLP)

A single hidden layer MLP (128 units, ReLU) that takes state features as input and outputs Q-values for all 3 actions. Trained with the same semi-gradient TD error via backpropagation.

### Experimental Matrix

| | Compact (11d) | Local (109d) | Extended (126d) |
|---|---|---|---|
| **Linear FA** | Config 1 | Config 2 | Config 3 |
| **Tile Coding** | Config 4 | Config 5 | Config 6 |
| **MLP** | Config 7 | Config 8 | Config 9 |

Each configuration: 10,000 episodes × 5 random seeds = 50,000 episodes per cell.

---

## Setup and Installation

### Requirements

- Python 3.10+
- NumPy
- Pygame (for visual rendering and human play)
- Matplotlib (for plotting)
- PyTorch (for MLP agent — needed later)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/snake-rl.git
cd snake-rl
pip install -r requirements.txt
```

### Quick Start

```bash
# Run all tests (145 tests)
python tests/run_tests.py
python tests/test_representations.py
python tests/test_baselines.py
python tests/test_experiment.py

# Play Snake manually
python -c "from snake_rl.env.renderer import play_human; play_human()"

# Run baseline evaluation
python tests/test_baselines.py  # includes 200-episode evaluation at the end

# Check feature extraction
python -c "
from snake_rl.env import SnakeEnv
from snake_rl.representations import get_representation

env = SnakeEnv(grid_size=20, seed=42)
obs, _ = env.reset()

for name in ['compact', 'local', 'extended']:
    rep = get_representation(name)
    feats = rep.get_state_features(obs)
    print(f'{name}: {feats.shape} features')
"
```

---

## Testing

The project has **145 tests** organized across 4 test files:

| Test File | Count | Covers |
|---|---|---|
| `test_env.py` | 75 | Direction helpers, reset, step mechanics, food, collisions, truncation, observations, reproducibility, rendering, edge cases, statistical sanity |
| `test_representations.py` | 34 | All 3 representations: dimensions, shapes, binary features, one-hot structure, danger detection, food direction, wall distances, body scanning, normalization, cross-representation consistency, performance benchmarks |
| `test_baselines.py` | 11 | Random agent, greedy heuristic (wall avoidance, food seeking, beats random), evaluation utility |
| `test_experiment.py` | 25 | RunLogger, ExperimentConfig, ExperimentResult, serialization, persistence, epsilon schedule |

All tests work with both **pytest** (`pytest tests/ -v`) and the **standalone runner** (`python tests/run_tests.py`) which requires no external testing framework.

---

## References

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Tsitsiklis, J. N. & Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. *IEEE Transactions on Automatic Control*, 42(5), 674–690.
- Bonnici, N. et al. (2022). Exploring reinforcement learning: A case study applied to the popular Snake game. *Springer LNNS*, vol. 382.
- Liang, Y. et al. (2016). State of the art control of Atari games using shallow reinforcement learning. *AAMAS*.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
- Sherstov, A. A. & Stone, P. (2005). Function approximation via tile coding: Automating parameter choice. *SARA*, Springer LNAI 3607.

Full reference list available in `revised_project_proposal.tex`.

---

## License

This project is part of coursework for CS 5180 at Northeastern University. Not intended for redistribution.
