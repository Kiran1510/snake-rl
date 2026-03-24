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
│   ├── agents/                     # RL agents
│   │   ├── __init__.py
│   │   ├── baselines.py            # Random agent, Greedy heuristic, evaluation utility
│   │   ├── linear_sarsa.py         # Semi-gradient SARSA with linear function approximation
│   │   ├── tile_sarsa.py           # Semi-gradient SARSA with tile coding
│   │   ├── mlp_sarsa.py            # Semi-gradient SARSA with neural network (PyTorch)
│   │   └── train.py                # Shared SARSA training loop and experiment runner
│   └── utils/                      # Experiment infrastructure
│       ├── __init__.py
│       ├── experiment.py           # RunLogger, ExperimentConfig, ExperimentResult, persistence
│       └── plotting.py             # Learning curves, comparisons, final performance charts
├── tests/                          # Test suite (~210 tests)
│   ├── run_tests.py                # Standalone test runner (no pytest dependency)
│   ├── test_env.py                 # 75 environment tests
│   ├── test_representations.py     # 34 representation tests
│   ├── test_baselines.py           # 11 baseline agent tests
│   ├── test_experiment.py          # 25 experiment infrastructure tests
│   ├── test_linear_sarsa.py        # 24 linear FA SARSA tests
│   ├── test_tile_sarsa.py          # 21 tile coding SARSA tests
│   └── test_mlp_sarsa.py           # 20+ MLP SARSA tests (requires PyTorch)
├── revised_project_proposal.tex    # LaTeX project proposal (approved by Prof. Platt)
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

Feature extraction speed on a 20×20 grid:

| Representation | Time per call | Feasibility |
|---|---|---|
| Compact | ~2 µs | No bottleneck |
| Local Neighborhood | ~10 µs | No bottleneck |
| Extended | ~18 µs | No bottleneck |

---

## Algorithms

All three algorithms use **semi-gradient SARSA** as the underlying control algorithm, differing only in how the action-value function q̂(s, a) is approximated. They share a common training loop (`snake_rl/agents/train.py`) and conform to the same interface: `.act(obs)`, `.update(obs, action, reward, next_obs, next_action, terminated)`, and `.epsilon`.

### Why On-Policy SARSA

The choice of on-policy SARSA is theoretically motivated. Tsitsiklis and Van Roy (1997) proved that TD learning with linear function approximation converges with probability 1 under on-policy sampling. Off-policy methods with function approximation can diverge even in simple cases (Baird, 1995). This instability arises from the **deadly triad**: the combination of function approximation, bootstrapping, and off-policy learning (Sutton & Barto, 2018). By using on-policy SARSA, the linear methods avoid one leg of the triad entirely, providing a stable baseline against which to measure neural network instability.

### Algorithm 1: Linear FA SARSA (`snake_rl/agents/linear_sarsa.py`)

The action-value function is approximated as a linear combination of features:

```
q̂(s, a; w) = wᵀ x(s, a)
```

The weight update follows the semi-gradient SARSA rule:

```
w ← w + α [R + γ q̂(S', A'; w) - q̂(S, A; w)] x(S, A)
```

For terminal transitions, the bootstrap term is dropped: `w ← w + α [R - q̂(S, A; w)] x(S, A)`.

```python
from snake_rl.agents.linear_sarsa import LinearSarsaAgent
from snake_rl.representations import CompactRepresentation

rep = CompactRepresentation()
agent = LinearSarsaAgent(
    representation=rep,
    alpha=0.01,       # learning rate
    gamma=0.95,       # discount factor
    epsilon=1.0,      # initial exploration rate
    seed=42,
)

# Single step interaction
action = agent.act(obs)
td_error = agent.update(obs, action, reward, next_obs, next_action, terminated)

# Inspect learned weights
stats = agent.get_weight_stats()    # mean, std, norm, etc.
q_vals = agent.q_values(obs)        # Q-values for all 3 actions
```

#### What the Weights Learn

After training on the compact representation (5000 episodes, α=0.01), the learned weights show a clear pattern. The danger signals dominate:

| Feature | STRAIGHT weight | LEFT weight | RIGHT weight |
|---|---|---|---|
| `danger_straight` | **-8.19** | -0.19 | -1.58 |
| `danger_left` | +0.06 | **-8.31** | +0.35 |
| `danger_right` | -0.46 | +0.43 | **-8.05** |

The agent learns strongly negative weights for "danger in the direction of this action" — meaning it avoids walking into walls and its own body. The food-direction weights are smaller and noisier, indicating that the compact representation's limited expressiveness makes it hard for a linear model to learn nuanced food-seeking behavior.

#### Known Limitation

The compact representation with linear FA can create **deterministic action loops** under a pure greedy policy (ε=0). The 11 binary features don't distinguish enough states to break cycles. This is a legitimate finding about the representation's expressiveness — richer representations and nonlinear approximation should address this.

### Algorithm 2: Tile Coding SARSA (`snake_rl/agents/tile_sarsa.py`)

Tile coding automatically constructs binary feature vectors from state features using **multiple overlapping tilings**. The value function remains linear in these tile features:

```
q̂(s, a; w) = Σ w[i]  for each active tile i
```

The implementation uses **hash-based tile coding** to handle high-dimensional inputs (the local and extended representations have 109+ dimensions, making naive tile indexing infeasible). The hash function maps `(tiling_id, tile_coordinates, action)` to a fixed-size weight table.

```python
from snake_rl.agents.tile_sarsa import TileCodingSarsaAgent
from snake_rl.representations import CompactRepresentation

rep = CompactRepresentation()
agent = TileCodingSarsaAgent(
    base_representation=rep,
    n_tilings=8,          # overlapping tilings
    n_tiles_per_dim=4,    # tiles per dimension per tiling
    max_size=65536,       # hash table size
    alpha=0.05,           # learning rate (divided by n_tilings internally)
    gamma=0.95,
    seed=42,
)

# IMPORTANT: initialize feature normalization before training
env = SnakeEnv(grid_size=10, seed=42)
agent.initialize(env)
```

#### Key Design Decisions

- **Alpha divided by n_tilings:** Following the standard convention (Sutton & Barto, 2018, §9.5.4), the learning rate is internally divided by the number of tilings. The user-facing `alpha` parameter is the "effective" rate.
- **Hash-based indexing:** Naive tile coding on 109+ dimensions would require astronomically large index tables. The hash function compresses this to a fixed-size table (default 65536), with rare collisions as a trade-off.
- **Feature normalization:** The `TileCodingRepresentation` wrapper normalizes all input features to [0, 1] before tiling. Ranges are estimated by sampling random states during `agent.initialize(env)`.
- **Sparse weight updates:** Only the weights at active tile indices (one per tiling) are updated each step, making updates very efficient.

#### Tile Coding Components

The tile coding system has three layers:

1. **`TileCoder`**: Low-level hash-based tile coder. Takes normalized [0,1] features and returns active tile indices.
2. **`TileCodingRepresentation`**: Wraps any `BaseRepresentation` and handles normalization. Estimates feature ranges from sampled states.
3. **`TileCodingSarsaAgent`**: The full agent combining tile coding with semi-gradient SARSA.

### Algorithm 3: MLP SARSA (`snake_rl/agents/mlp_sarsa.py`)

The action-value function is approximated by a single hidden layer MLP:

```
q̂(s; θ) = W₂ · ReLU(W₁ · x(s) + b₁) + b₂
```

The network takes state features as input and outputs Q-values for all 3 actions simultaneously. The semi-gradient SARSA update uses backpropagation:

```
θ ← θ + α [R + γ q̂(S')_A' - q̂(S)_A] ∇_θ q̂(S)_A
```

Implemented as MSE loss on the TD error with gradient clipping (max norm 10.0).

```python
from snake_rl.agents.mlp_sarsa import MLPSarsaAgent
from snake_rl.representations import CompactRepresentation

rep = CompactRepresentation()
agent = MLPSarsaAgent(
    representation=rep,
    hidden_dim=128,           # hidden layer units
    alpha=0.001,              # Adam learning rate
    gamma=0.95,
    epsilon=1.0,
    use_target_network=False, # add if training is unstable
    target_update_freq=100,   # episodes between target net syncs
    seed=42,
)
```

#### Key Design Decisions

- **Deliberately simple architecture:** One hidden layer, 128 units, ReLU. No deep networks — the goal is to isolate the effect of nonlinear function approximation, not to maximize performance.
- **Adam optimizer:** Standard choice for neural network training. Learning rate default 0.001.
- **Gradient clipping:** `max_norm=10.0` to prevent gradient explosions during semi-gradient updates.
- **Optional target network:** Disabled by default to maintain closest comparison with the linear methods. If training instability arises (which the deadly triad analysis predicts is possible), enable it with `use_target_network=True`. The target network is updated every `target_update_freq` episodes. *If the MLP requires stabilization techniques that linear methods don't, that's a meaningful finding about the cost of nonlinear approximation.*
- **No experience replay by default:** Keeps the method on-policy (true SARSA). Experience replay would make it off-policy and reintroduce the third leg of the deadly triad.
- **CPU only:** Snake is not compute-intensive enough to benefit from GPU acceleration.

#### Network Architecture

For the compact representation (11 input features):

```
Input (11) → Linear (11 × 128) → ReLU → Linear (128 × 3) → Output (3)
Total parameters: 1,923
```

For the extended representation (126 input features):

```
Input (126) → Linear (126 × 128) → ReLU → Linear (128 × 3) → Output (3)
Total parameters: 16,643
```

---

## Training Loop (`snake_rl/agents/train.py`)

A generic SARSA training loop shared by all three algorithms. Handles the on-policy action selection pattern that SARSA requires.

### SARSA Episode Flow

```
1. Observe S, choose A (ε-greedy)
2. Take A, observe R, S'
3. If terminal: update with (S, A, R, terminated=True), end episode
4. Else: choose A' (ε-greedy), update with (S, A, R, S', A'), set S←S', A←A', go to 2
```

The key difference from Q-learning: step 4 selects A' using the *current policy* (including exploration), not the greedy action. This makes SARSA on-policy — it learns the value of the policy it's actually following.

### Single Training Run

```python
from snake_rl.agents.train import train_sarsa
from snake_rl.utils.experiment import ExperimentConfig

config = ExperimentConfig(
    algorithm="linear_sarsa",
    representation="compact",
    n_episodes=10000,
    grid_size=20,
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_fraction=0.8,
    alpha=0.01,
)

logger = train_sarsa(agent, env, config, print_every=1000)
```

Output during training:

```
Episode   1000/10000 | Avg score:   0.17 | Max:   3 | Eps: 0.763 | TD err: 0.801 | Time: 1s
Episode   2000/10000 | Avg score:   0.24 | Max:   3 | Eps: 0.525 | TD err: 0.612 | Time: 4s
Episode   3000/10000 | Avg score:   0.39 | Max:   5 | Eps: 0.288 | TD err: 0.514 | Time: 7s
...
```

### Multi-Seed Experiment

```python
from snake_rl.agents.train import run_experiment

result = run_experiment(
    agent_factory=lambda seed: LinearSarsaAgent(rep, alpha=0.01, seed=seed),
    config=config,
    seeds=[0, 1, 2, 3, 4],
    print_every=1000,
)

perf = result.final_performance(last_n=1000)
print(f"Mean final score: {perf['mean_score']:.2f} ± {perf['std_score']:.2f}")
```

### Agent End-of-Episode Hook

The training loop calls `agent.on_episode_end()` after each episode if the method exists. The MLP agent uses this for target network updates. Linear and tile coding agents don't need it.

---

## Baseline Agents

### Random Agent

Uniform random action selection. Provides the **lower bound** — any learned policy should beat this convincingly.

### Greedy Heuristic Agent

A hand-coded policy that moves toward food while avoiding immediate danger. Represents what a simple rule-based agent can achieve **without any learning**.

### Baseline Results (200 episodes, 20×20 grid)

| Agent | Mean Score | Std | Max Score | Mean Steps | Primary Death Cause |
|---|---|---|---|---|---|
| Random | 0.15 | 0.38 | 2 | 66.2 | Wall collision (89%) |
| Greedy Heuristic | 22.89 | 6.00 | 33 | 329.9 | Self-collision (52%) |

---

## Early Results

Preliminary training runs on a 10×10 grid already reveal interesting patterns:

| Agent | Compact Representation | Notes |
|---|---|---|
| Random baseline | 0.15 avg | Lower bound |
| Linear FA SARSA (5k ep) | 0.17 → 0.70 | Learns danger avoidance, struggles with food-seeking |
| Tile Coding SARSA (3k ep) | 0.25 → 8.99 | Dramatically better generalization |
| Greedy heuristic | 22.89 avg | Upper reference |

### Key Observations So Far

**Linear FA learns danger avoidance but little else on compact features.** The learned weights show danger signals with weights of -8.x — the agent strongly avoids collisions. But the food-direction weights are small and noisy, and the agent struggles to translate "food is to the right" into effective navigation under a linear model. Under pure greedy evaluation (ε=0), the policy degenerates into deterministic loops.

**Tile coding dramatically outperforms hand-crafted linear features.** Going from 0.70 to 8.99 average score on the same compact representation is a major improvement. Tile coding's overlapping tilings create a richer feature space that enables better generalization between similar states — nearby states share most active tiles, providing smooth value function approximation that hand-crafted binary features can't achieve.

**The full 3×3 experimental matrix has not been run yet.** These early numbers are from initial development and tuning. The richer representations (local neighborhood, extended) and the MLP are expected to show further differences.

---

## Experiment Infrastructure

### Run Logging (`RunLogger`)

Tracks per-episode metrics during a single training run: scores, cumulative rewards, step counts, epsilon values, death causes, and wall-clock timestamps. Supports smoothed curves via moving average.

### Experiment Configuration (`ExperimentConfig`)

Dataclass holding all hyperparameters for one algorithm × representation pair. Fully serializable to/from JSON.

### Multi-Seed Results (`ExperimentResult`)

Aggregates `RunLogger` objects across multiple random seeds. Computes mean/std learning curves, final performance statistics, and convergence episode estimates.

### Persistence

JSON-based save/load for all experiment results:

```python
from snake_rl.utils.experiment import save_results, load_results

save_results(result, "results/linear_sarsa__compact.json")
loaded = load_results("results/linear_sarsa__compact.json")
```

### Epsilon Schedule

Linear decay from `epsilon_start` to `epsilon_end` over a configurable fraction of total episodes.

### Plotting

Five plot functions, all save to file and return the matplotlib figure:

| Function | Description |
|---|---|
| `plot_learning_curve` | Single experiment with confidence band across seeds |
| `plot_reward_curve` | Same for cumulative reward |
| `plot_comparison` | Multiple experiments overlaid on one plot |
| `plot_comparison_by_representation` | 3-panel figure, one per representation |
| `plot_final_performance_table` | Bar chart of final scores across all configurations |

---

## Experimental Matrix (Planned)

| | Compact (11d) | Local (109d) | Extended (126d) |
|---|---|---|---|
| **Linear FA** | Config 1 | Config 2 | Config 3 |
| **Tile Coding** | Config 4 | Config 5 | Config 6 |
| **MLP** | Config 7 | Config 8 | Config 9 |

Each configuration: 10,000 episodes × 5 random seeds. Plus baselines (random, greedy heuristic) and a brief tabular Q-learning demonstration showing where tabular methods break down.

---

## Setup and Installation

### Requirements

- Python 3.10+
- NumPy
- Pygame (for visual rendering and human play)
- Matplotlib (for plotting)
- PyTorch (for MLP agent)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/snake-rl.git
cd snake-rl
pip install -r requirements.txt
```

### Quick Start

```bash
# Run all tests (~210 tests)
python tests/run_tests.py
python tests/test_representations.py
python tests/test_baselines.py
python tests/test_experiment.py
python tests/test_linear_sarsa.py
python tests/test_tile_sarsa.py
python tests/test_mlp_sarsa.py

# Play Snake manually
python -c "from snake_rl.env.renderer import play_human; play_human()"

# Quick training demo (Tile Coding, 3k episodes)
python -c "
from snake_rl.env import SnakeEnv
from snake_rl.representations import CompactRepresentation
from snake_rl.agents.tile_sarsa import TileCodingSarsaAgent
from snake_rl.agents.train import train_sarsa
from snake_rl.utils.experiment import ExperimentConfig

config = ExperimentConfig(
    algorithm='tile_sarsa', representation='compact',
    n_episodes=3000, grid_size=10, alpha=0.05,
)
rep = CompactRepresentation()
agent = TileCodingSarsaAgent(rep, n_tilings=8, alpha=0.05, seed=42)
env = SnakeEnv(grid_size=10, seed=42)
agent.initialize(env)
logger = train_sarsa(agent, env, config, print_every=500)
print(f'Final avg score: {sum(logger.scores[-500:])/500:.2f}')
"
```

---

## Testing

The project has **~210 tests** organized across 7 test files:

| Test File | Count | Covers |
|---|---|---|
| `test_env.py` | 75 | Direction helpers, reset, step mechanics, food, collisions, truncation, observations, reproducibility, rendering, edge cases, statistical sanity |
| `test_representations.py` | 34 | All 3 representations: dimensions, shapes, binary features, one-hot structure, danger detection, food direction, wall distances, body scanning, normalization, cross-representation consistency, performance benchmarks |
| `test_baselines.py` | 11 | Random agent, greedy heuristic (wall avoidance, food seeking, beats random), evaluation utility |
| `test_experiment.py` | 25 | RunLogger, ExperimentConfig, ExperimentResult, serialization, persistence, epsilon schedule |
| `test_linear_sarsa.py` | 24 | Agent mechanics (Q-values, action selection, weight updates, terminal vs non-terminal), training loop (epsilon decay, logging, all representations), learning sanity (score improvement, danger weight analysis) |
| `test_tile_sarsa.py` | 21 | Tile coder (active tiles, uniqueness, hashing, nearby/distant state sharing), normalization, agent mechanics, sparsity, all representations, learning sanity |
| `test_mlp_sarsa.py` | 20+ | QNetwork architecture, agent mechanics, gradient updates, target network init/sync, training with all representations, NaN safety, learning sanity. Requires PyTorch |

All tests work with both **pytest** (`pytest tests/ -v`) and the **standalone runner** (`python tests/run_tests.py`) which requires no external testing framework.

---

## References

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Tsitsiklis, J. N. & Van Roy, B. (1997). An analysis of temporal-difference learning with function approximation. *IEEE Transactions on Automatic Control*, 42(5), 674–690.
- Baird, L. (1995). Residual algorithms: Reinforcement learning with function approximation. *ICML*.
- van Hasselt, H. et al. (2018). Deep reinforcement learning and the deadly triad. *arXiv:1812.02648*.
- Sherstov, A. A. & Stone, P. (2005). Function approximation via tile coding: Automating parameter choice. *SARA*, Springer LNAI 3607.
- Bonnici, N. et al. (2022). Exploring reinforcement learning: A case study applied to the popular Snake game. *Springer LNNS*, vol. 382.
- Liang, Y. et al. (2016). State of the art control of Atari games using shallow reinforcement learning. *AAMAS*.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.
- Tesauro, G. (1995). Temporal difference learning and TD-Gammon. *Communications of the ACM*, 38(3), 58–68.
- Ng, A. Y., Harada, D. & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

Full reference list (25 citations) available in `revised_project_proposal.tex`.

---

## License

This project is part of coursework for CS 5180 at Northeastern University. Not intended for redistribution.