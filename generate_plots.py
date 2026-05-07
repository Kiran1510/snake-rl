"""
Generate all plots from saved experiment results.

Usage:
    python generate_plots.py              # from results/ directory
    python generate_plots.py --dir path/  # custom results directory
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake_rl.utils.experiment import load_results, ExperimentResult
from snake_rl.utils.plotting import (
    plot_learning_curve,
    plot_comparison,
    plot_comparison_by_representation,
    plot_final_performance_table,
    plot_performance_heatmap,
)

FIGURES_DIR = "figures"


def load_all_results(results_dir: str) -> dict:
    """Load all JSON result files from a directory."""
    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            filepath = os.path.join(results_dir, fname)
            try:
                result = load_results(filepath)
                key = result.config.name
                results[key] = result
                print(f"  Loaded: {fname} ({result.n_runs} seeds)")
            except Exception as e:
                print(f"  FAILED: {fname} — {e}")

    return results


def generate_all_plots(results: dict, figures_dir: str):
    """Generate all standard plots."""
    os.makedirs(figures_dir, exist_ok=True)
    all_results = list(results.values())

    if not all_results:
        print("No results to plot.")
        return

    # 1. Individual learning curves
    print("\n--- Individual learning curves ---")
    for name, result in results.items():
        save_path = os.path.join(figures_dir, f"learning_curve_{name}.png")
        plot_learning_curve(result, window=100, save_path=save_path)

    ALGO_DISPLAY = {
        "linear_sarsa": "Linear FA",
        "tile_sarsa":   "Tile Coding",
        "mlp_sarsa":    "MLP",
    }
    REP_DISPLAY = {
        "compact":  "Compact (11d)",
        "local":    "Local (109d)",
        "extended": "Extended (126d)",
    }

    # 2. Comparison by algorithm (all reps on one plot per algo)
    print("\n--- Algorithm comparison plots ---")
    algos = set()
    reps = set()
    for name in results:
        parts = name.split("__")
        if len(parts) == 2:
            algos.add(parts[0])
            reps.add(parts[1])

    for algo in sorted(algos):
        algo_results = [r for name, r in results.items() if name.startswith(algo)]
        if algo_results:
            labels = [REP_DISPLAY.get(r.config.representation, r.config.representation)
                      for r in algo_results]
            display = ALGO_DISPLAY.get(algo, algo)
            save_path = os.path.join(figures_dir, f"comparison_{algo}.png")
            plot_comparison(
                algo_results, labels=labels,
                title=f"{display} — Score by Representation",
                save_path=save_path,
            )

    # 3. Comparison by representation (all algos on one plot per rep)
    print("\n--- Representation comparison plots ---")
    for rep in sorted(reps):
        rep_results = [r for name, r in results.items() if name.endswith(f"__{rep}")]
        if rep_results:
            labels = [ALGO_DISPLAY.get(r.config.algorithm, r.config.algorithm)
                      for r in rep_results]
            display = REP_DISPLAY.get(rep, rep)
            save_path = os.path.join(figures_dir, f"comparison_{rep}.png")
            plot_comparison(
                rep_results, labels=labels,
                title=f"{display} — Score by Algorithm",
                save_path=save_path,
            )

    # 4. Three-panel comparison by representation
    if len(all_results) >= 3:
        print("\n--- Three-panel comparison ---")
        save_path = os.path.join(figures_dir, "comparison_by_representation.png")
        plot_comparison_by_representation(all_results, save_path=save_path)

    # 5. Final performance bar chart
    print("\n--- Final performance bar chart ---")
    save_path = os.path.join(figures_dir, "final_performance.png")
    plot_final_performance_table(all_results, save_path=save_path)

    # 6. Heatmap (3×3 summary — best single figure for a paper)
    print("\n--- Performance heatmap ---")
    save_path = os.path.join(figures_dir, "heatmap.png")
    plot_performance_heatmap(all_results, save_path=save_path)

    # 7. Print summary table (for copy-paste into report)
    print("\n" + "=" * 70)
    print("RESULTS TABLE (for report)")
    print("=" * 70)
    print(f"\n{'Configuration':<30} {'Mean':>8} {'± Std':>8} {'Max':>6} {'Conv. Ep':>10}")
    print("-" * 65)
    for name in sorted(results.keys()):
        result = results[name]
        perf = result.final_performance()
        conv = result.convergence_episode()
        max_score = max(max(l.scores) for l in result.run_loggers)
        print(f"{name:<30} {perf['mean_score']:>8.2f} {perf['std_score']:>8.2f} {max_score:>6} {conv['mean']:>10.0f}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from experiment results")
    parser.add_argument("--dir", type=str, default="results",
                        help="Results directory (default: results/)")
    parser.add_argument("--figures", type=str, default="figures",
                        help="Output directory for figures (default: figures/)")
    args = parser.parse_args()

    print("=" * 70)
    print("PLOT GENERATION")
    print("=" * 70)
    print(f"\nLoading results from: {os.path.abspath(args.dir)}")

    results = load_all_results(args.dir)
    if not results:
        print("\nNo results found. Run experiments first:")
        print("  python run_experiments.py --quick")
        sys.exit(1)

    print(f"\nLoaded {len(results)} configurations.")
    generate_all_plots(results, args.figures)

    print(f"\nFigures saved to: {os.path.abspath(args.figures)}/")


if __name__ == "__main__":
    main()
