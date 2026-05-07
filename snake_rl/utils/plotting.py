"""
Plotting utilities for visualizing training results.

All functions save to file AND return the figure, so they work
both in scripts (save to disk) and notebooks (inline display).

Requires matplotlib: pip install matplotlib
"""

import os
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for saving
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_matplotlib():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )


# ============================================================
# Single experiment plots
# ============================================================

def plot_learning_curve(
    result,
    window: int = 100,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """
    Plot mean learning curve (score) with confidence band across seeds.

    Parameters
    ----------
    result : ExperimentResult
        Results from one experiment configuration.
    window : int
        Smoothing window size.
    title : str or None
        Plot title (auto-generated if None).
    save_path : str or None
        Path to save the figure. If None, not saved.
    """
    _check_matplotlib()

    mean, std = result.mean_learning_curve(window)
    episodes = np.arange(len(mean)) + window

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(episodes, mean, linewidth=1.5, label="Mean score")
    ax.fill_between(
        episodes, mean - std, mean + std,
        alpha=0.2, label="±1 std"
    )

    if title is None:
        title = f"Learning Curve: {result.config.name}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Score (smoothed, window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_reward_curve(
    result,
    window: int = 100,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5),
):
    """Plot mean reward curve with confidence band across seeds."""
    _check_matplotlib()

    mean, std = result.mean_reward_curve(window)
    episodes = np.arange(len(mean)) + window

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(episodes, mean, linewidth=1.5, color="tab:orange", label="Mean reward")
    ax.fill_between(
        episodes, mean - std, mean + std,
        alpha=0.2, color="tab:orange", label="±1 std"
    )

    if title is None:
        title = f"Reward Curve: {result.config.name}"
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Total Reward (smoothed, window={window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# ============================================================
# Comparison plots
# ============================================================

def plot_comparison(
    results: list,
    labels: Optional[list[str]] = None,
    window: int = 100,
    title: str = "Algorithm Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """
    Compare learning curves from multiple experiments on one plot.

    Parameters
    ----------
    results : list of ExperimentResult
        Results to compare.
    labels : list of str or None
        Legend labels (auto-generated from config if None).
    window : int
        Smoothing window.
    title : str
        Plot title.
    save_path : str or None
        Path to save the figure.
    """
    _check_matplotlib()

    if labels is None:
        labels = [r.config.name for r in results]

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    fig, ax = plt.subplots(figsize=figsize)

    for result, label, color in zip(results, labels, colors):
        mean, std = result.mean_learning_curve(window)
        episodes = np.arange(len(mean)) + window
        ax.plot(episodes, mean, linewidth=1.5, color=color, label=label)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Score (smoothed, window={window})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_comparison_by_representation(
    results: list,
    window: int = 100,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 5),
):
    """
    Create a 3-panel figure: one subplot per representation, comparing
    algorithms within each.

    Parameters
    ----------
    results : list of ExperimentResult
        All experiment results (will be grouped by representation).
    """
    _check_matplotlib()

    # Group by representation
    rep_groups: dict[str, list] = {}
    for r in results:
        rep = r.config.representation
        if rep not in rep_groups:
            rep_groups[rep] = []
        rep_groups[rep].append(r)

    ALGO_LABELS = {
        "linear_sarsa": "Linear FA",
        "tile_sarsa":   "Tile Coding",
        "mlp_sarsa":    "MLP",
    }
    REP_TITLES = {
        "compact":  "Compact (11d)",
        "local":    "Local Neighborhood (109d)",
        "extended": "Extended (126d)",
    }

    n_panels = len(rep_groups)
    # sharey=False: each representation has a different score range
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=False)
    if n_panels == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    all_algos = sorted(set(r.config.algorithm for r in results))
    algo_colors = {algo: colors[i] for i, algo in enumerate(all_algos)}

    for ax, (rep_name, rep_results) in zip(axes, sorted(rep_groups.items())):
        for r in rep_results:
            mean, std = r.mean_learning_curve(window)
            episodes = np.arange(len(mean)) + window
            color = algo_colors[r.config.algorithm]
            label = ALGO_LABELS.get(r.config.algorithm, r.config.algorithm)
            ax.plot(episodes, mean, linewidth=1.5, color=color, label=label)
            ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=color)

        title = REP_TITLES.get(rep_name, rep_name)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(f"Score (window={window})")

    fig.suptitle("Algorithm × Representation Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_final_performance_table(
    results: list,
    last_n: int = 1000,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
):
    """Bar chart of final performance grouped by representation, log-scaled."""
    _check_matplotlib()

    ALGO_LABELS = {
        "linear_sarsa": "Linear FA",
        "tile_sarsa":   "Tile Coding",
        "mlp_sarsa":    "MLP",
    }
    REP_ORDER = ["compact", "local", "extended"]
    ALGO_ORDER = ["linear_sarsa", "tile_sarsa", "mlp_sarsa"]
    COLORS = ["#4878d0", "#ee854a", "#6acc65"]

    # Build score lookup
    scores: dict[tuple, tuple] = {}
    for r in results:
        perf = r.final_performance(last_n)
        scores[(r.config.algorithm, r.config.representation)] = (
            perf["mean_score"], perf["std_score"]
        )

    reps = [r for r in REP_ORDER if any((a, r) in scores for a in ALGO_ORDER)]
    algos = [a for a in ALGO_ORDER if any((a, r) in scores for r in REP_ORDER)]

    n_reps = len(reps)
    n_algos = len(algos)
    x = np.arange(n_reps)
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)

    for i, algo in enumerate(algos):
        means = [scores.get((algo, rep), (0, 0))[0] for rep in reps]
        stds  = [scores.get((algo, rep), (0, 0))[1] for rep in reps]
        offset = (i - n_algos / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=ALGO_LABELS.get(algo, algo),
                      color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.4,
                        f"{mean:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Compact (11d)", "Local (109d)", "Extended (126d)"], fontsize=10)
    ax.set_ylabel("Mean Score (final 1k episodes)")
    ax.set_title(f"Final Performance: All Configurations (last {last_n} episodes)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_performance_heatmap(
    results: list,
    last_n: int = 1000,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 4),
):
    """
    3×3 heatmap of final mean scores: algorithms (rows) × representations (cols).
    Clean summary figure for a paper.
    """
    _check_matplotlib()

    ALGO_LABELS = {
        "linear_sarsa": "Linear FA",
        "tile_sarsa":   "Tile Coding",
        "mlp_sarsa":    "MLP",
    }
    REP_LABELS = {
        "compact":  "Compact\n(11d)",
        "local":    "Local\n(109d)",
        "extended": "Extended\n(126d)",
    }
    ALGO_ORDER = ["linear_sarsa", "tile_sarsa", "mlp_sarsa"]
    REP_ORDER  = ["compact", "local", "extended"]

    scores: dict[tuple, float] = {}
    for r in results:
        perf = r.final_performance(last_n)
        scores[(r.config.algorithm, r.config.representation)] = perf["mean_score"]

    algos = [a for a in ALGO_ORDER if any((a, r) in scores for r in REP_ORDER)]
    reps  = [r for r in REP_ORDER  if any((a, r) in scores for a in ALGO_ORDER)]

    matrix = np.array([
        [scores.get((a, r), np.nan) for r in reps]
        for a in algos
    ])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Score")

    ax.set_xticks(range(len(reps)))
    ax.set_yticks(range(len(algos)))
    ax.set_xticklabels([REP_LABELS.get(r, r) for r in reps], fontsize=11)
    ax.set_yticklabels([ALGO_LABELS.get(a, a) for a in algos], fontsize=11)

    for i in range(len(algos)):
        for j in range(len(reps)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "black" if val < matrix[~np.isnan(matrix)].max() * 0.7 else "white"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=text_color)

    ax.set_title(f"Final Mean Score (last {last_n} episodes)", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig
