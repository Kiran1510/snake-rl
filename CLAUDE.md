# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A CS 5180 (Northeastern University) research project comparing three function approximation methods (linear FA, tile coding, MLP) × three state representations (compact, local, extended) on the Snake game, all using semi-gradient SARSA. The central codebase is the `snake_rl/` package; the top-level scripts are experiment orchestration.

## Setup

```bash
pip install -r requirements.txt   # numpy, pygame, matplotlib, torch
```

No build step. All scripts add the project root to `sys.path` themselves, so they can be run directly from the repo root.

## Common Commands

```bash
# Run all tests (no pytest required)
python tests/run_tests.py

# Run tests with pytest (single file)
pytest tests/test_env.py -v
pytest tests/test_linear_sarsa.py -v

# Full 3×3 experiment matrix (20k episodes, 5 seeds)
python run_experiments.py

# Quick smoke run (2k episodes, 2 seeds)
python run_experiments.py --quick

# Single configuration
python run_experiments.py --algo linear --rep compact --episodes 5000 --seeds 3

# Parallel seeds (one process per seed, recommended for MLP)
python run_experiments.py --algo mlp --parallel

# Resume, skipping already-saved results
python run_experiments.py --resume

# Generate all plots from saved results
python generate_plots.py                  # reads results/, writes figures/

# Record gameplay GIFs for all saved agents
python record_all_gifs.py

# Watch a specific trained agent live (requires pygame)
python record_gameplay.py --watch tile compact
python record_gameplay.py --watch mlp compact --seed 3

# Tabular Q-learning demo (motivates function approximation)
python run_tabular_demo.py

# Human play mode
python -c "from snake_rl.env.renderer import play_human; play_human()"
```

## Architecture

### Three-layer design

```
SnakeEnv  →  Representation  →  Agent
(raw obs dict)   (feature vec)   (q̂ + update)
```

**`snake_rl/env/snake_env.py`** — pure game logic, Gymnasium-style interface. `reset()` and `step()` return a raw `obs` dict (`snake`, `food`, `direction`, `score`, `steps`, `grid_size`). Actions are **relative** (STRAIGHT/LEFT/RIGHT = 0/1/2), not absolute, which eliminates the "reverse" invalid action.

**`snake_rl/representations/features.py`** — stateless feature extractors. `BaseRepresentation` exposes:
- `get_state_features(obs)` → `(state_dim,)` array (for MLP, which takes state and outputs all Q-values)
- `get_features(obs, action)` → `(feature_dim,)` array using one-hot action stacking (for linear/tile, where Q(s,a) = wᵀx(s,a))
- Registered by name in `REPRESENTATIONS` dict; use `get_representation("compact")` to instantiate

**`snake_rl/agents/`** — all three agents implement the same duck-typed interface: `.act(obs)`, `.update(obs, action, reward, next_obs, next_action, terminated)`, `.epsilon`. `train.py` works with any conforming agent.

### Training loop (`snake_rl/agents/train.py`)

`train_sarsa(agent, env, config)` runs the standard SARSA loop: observe S → choose A → take A → observe R, S' → choose A' → update(S, A, R, S', A') → advance. The terminal case drops the bootstrap term. `run_experiment()` runs multiple seeds and returns an `ExperimentResult`.

### Experiment infrastructure (`snake_rl/utils/experiment.py`)

- `ExperimentConfig` — dataclass holding algorithm name, representation name, hyperparameters
- `RunLogger` — collects per-episode metrics (scores, rewards, steps, epsilon, cause of death)
- `ExperimentResult` — aggregates multiple `RunLogger`s (across seeds), computes mean learning curves and convergence statistics
- Results persist as JSON via `save_results`/`load_results`; file naming convention is `{algo}_sarsa__{rep}.json` under `results/`
- Epsilon schedule: linear decay from `epsilon_start` to `epsilon_end` over `epsilon_decay_fraction` of total episodes

### Weights persistence (`snake_rl/utils/save_load.py`)

- Linear/tile agents: `.npz` files in `weights/`
- MLP agents: `.pt` files in `weights/`; checkpoint includes `hidden_dims` so architecture can be recovered at load time
- Per-seed weights follow the pattern `{algo}_sarsa__{rep}_seed{N}.{ext}`

### MLP agent (`snake_rl/agents/mlp_sarsa.py`)

`MLPSarsaAgent` uses experience replay (50k buffer, batch size 64) and a target network (hard-copied every 100 episodes). The TD target is Q-learning max-Q, not on-policy SARSA. Training uses `DistanceShapingWrapper` in `run_experiments.py` (potential-based, skipped on food eat). Gradient clipping at 5.0, Adam optimizer.

### Key numerical details

| Representation | `state_dim` | `feature_dim` |
|---|---|---|
| Compact | 11 | 33 |
| Local Neighborhood | 109 | 327 |
| Extended | 126 (138 with interactions) | 378 / 414 |

Tile coding uses a hash table (`max_size=262144`) so `feature_dim` is always 262144 regardless of representation. Alpha is divided by `n_tilings` internally (standard convention).

MLP takes `state_dim` as input and outputs Q-values for all 3 actions simultaneously — it does **not** use the one-hot action stacking that linear/tile methods use.

### Test suite

~210 tests across 7 files. Both pytest and the standalone `tests/run_tests.py` runner work. The standalone runner only loads `test_env.py`; pytest discovers all test files. MLP tests require PyTorch.
