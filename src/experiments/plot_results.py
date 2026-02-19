#!/usr/bin/env python3
"""
Plot the results of the BC hyperparameter experiment.

Reads results from the JSON file and creates a matplotlib plot showing
how Eval_AverageReturn varies with num_agent_train_steps_per_iter.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(results_file: Path) -> dict:
    """Load experiment results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def plot_results(results: dict, output_file: Path):
    """
    Create a plot of Eval_AverageReturn vs num_agent_train_steps_per_iter.

    Args:
        results: Dictionary containing experiment results
        output_file: Path to save the plot
    """
    # Extract data points (skip any with errors)
    data_points = [d for d in results["data"] if "error" not in d]

    if not data_points:
        print("No valid data points to plot!")
        return

    # Sort by num_train_steps for proper line plot
    data_points.sort(key=lambda x: x["num_train_steps"])

    x = [d["num_train_steps"] for d in data_points]
    y = [d["eval_average_return"] for d in data_points]

    # Create the plot with wider figure for more data points
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot line with markers
    ax.plot(x, y, marker='o', markersize=6, linewidth=2, color='#2E86AB', label='BC Agent')

    # Add horizontal reference line for expert performance if available
    expert_return = results.get("fixed_params", {}).get("expert_return")
    if expert_return:
        ax.axhline(y=expert_return, color='#28A745', linestyle='--',
                   linewidth=1.5, label=f'Expert ({expert_return:.0f})')
        ax.legend(loc='lower right')

    # Add labels and title
    ax.set_xlabel("num_agent_train_steps_per_iter", fontsize=12)
    ax.set_ylabel("Eval_AverageReturn", fontsize=12)
    ax.set_title(f"BC Performance vs Training Steps ({results['env']})", fontsize=14)

    # Add grid for readability
    ax.grid(True, alpha=0.3)

    # Use linear scale with rotated x-axis labels for clarity
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x], rotation=45, ha='right')

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Also display if running interactively
    plt.show()


def main():
    """Load results and generate plot."""

    # Paths
    script_dir = Path(__file__).parent
    results_file = script_dir / "results" / "bc_experiment_results.json"
    output_file = script_dir / "results" / "bc_train_steps_experiment.png"

    # Check if results file exists
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Please run run_bc_experiment.py first to generate results.")
        return

    # Load and plot
    results = load_results(results_file)
    plot_results(results, output_file)

    # Print summary statistics
    data_points = [d for d in results["data"] if "error" not in d]
    if data_points:
        returns = [d["eval_average_return"] for d in data_points]
        best_idx = returns.index(max(returns))
        best_point = data_points[best_idx]

        print("\nSummary:")
        print("-" * 40)
        print(f"Best performance: {best_point['eval_average_return']:.2f}")
        print(f"Best num_train_steps: {best_point['num_train_steps']}")


if __name__ == "__main__":
    main()
