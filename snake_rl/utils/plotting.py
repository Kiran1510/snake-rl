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

    n_panels = len(rep_groups)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharey=True)
    if n_panels == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Consistent color per algorithm across panels
    all_algos = sorted(set(r.config.algorithm for r in results))
    algo_colors = {algo: colors[i] for i, algo in enumerate(all_algos)}

    for ax, (rep_name, rep_results) in zip(axes, sorted(rep_groups.items())):
        for r in rep_results:
            mean, std = r.mean_learning_curve(window)
            episodes = np.arange(len(mean)) + window
            color = algo_colors[r.config.algorithm]
            ax.plot(episodes, mean, linewidth=1.5, color=color, label=r.config.algorithm)
            ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_title(f"Representation: {rep_name}", fontsize=12)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f"Score (smoothed, window={window})")

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
    figsize: tuple = (10, 4),
):
    """
    Create a bar chart of final performance (mean score ± std) for all experiments.

    Parameters
    ----------
    results : list of ExperimentResult
        All experiment results.
    last_n : int
        Number of final episodes to compute performance over.
    """
    _check_matplotlib()

    names = []
    means = []
    stds = []

    for r in sorted(results, key=lambda x: x.config.name):
        perf = r.final_performance(last_n)
        names.append(r.config.name.replace("__", "\n"))
        means.append(perf["mean_score"])
        stds.append(perf["std_score"])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Mean Score (final episodes)")
    ax.set_title(f"Final Performance Comparison (last {last_n} episodes)")
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.3,
            f"{mean:.1f}", ha="center", va="bottom", fontsize=8
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig
