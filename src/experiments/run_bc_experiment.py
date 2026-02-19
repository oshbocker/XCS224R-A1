#!/usr/bin/env python3
"""
Experiment script to study how num_agent_train_steps_per_iter affects BC agent performance.

This script runs behavior cloning training with different values of the
num_agent_train_steps_per_iter hyperparameter, saves the results, and generates a plot.
"""

import argparse
import subprocess
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt


# Mapping from environment name to expert policy and data files
ENV_CONFIG = {
    "Ant-v4": {
        "expert_policy_file": "xcs224r/policies/experts/Ant.pkl",
        "expert_data": "xcs224r/expert_data/expert_data_Ant-v4.pkl",
    },
    "Walker2d-v4": {
        "expert_policy_file": "xcs224r/policies/experts/Walker2d.pkl",
        "expert_data": "xcs224r/expert_data/expert_data_Walker2d-v4.pkl",
    },
    "Hopper-v4": {
        "expert_policy_file": "xcs224r/policies/experts/Hopper.pkl",
        "expert_data": "xcs224r/expert_data/expert_data_Hopper-v4.pkl",
    },
    "HalfCheetah-v4": {
        "expert_policy_file": "xcs224r/policies/experts/HalfCheetah.pkl",
        "expert_data": "xcs224r/expert_data/expert_data_HalfCheetah-v4.pkl",
    },
}


def run_bc_experiment(num_train_steps: int, env: str, seed: int = 1) -> dict:
    """
    Run a single BC experiment with the specified number of training steps.

    Args:
        num_train_steps: Number of gradient updates per iteration
        env: Environment name (e.g., "Ant-v4")
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing the experiment results
    """
    if env not in ENV_CONFIG:
        raise ValueError(f"Unknown environment: {env}. Available: {list(ENV_CONFIG.keys())}")

    config = ENV_CONFIG[env]

    # Build the command - run from submission/ directory where xcs224r module lives
    cmd = [
        "python", "run_hw1.py",
        "--expert_policy_file", config["expert_policy_file"],
        "--env_name", env,
        "--n_iter", "1",
        "--expert_data", config["expert_data"],
        "--video_log_freq", "-1",
        "--ep_len", "1000",
        "--eval_batch_size", "5000",
        "--seed", str(seed),
        "--num_agent_train_steps_per_iter", str(num_train_steps),
        "--exp_name", f"bc_steps_{env}_{num_train_steps}",
    ]

    print(f"\n{'='*60}")
    print(f"Running experiment in env {env} with num_agent_train_steps_per_iter = {num_train_steps}")
    print(f"{'='*60}")

    # Run the command and capture output
    # Run from src/submission/ directory where xcs224r module is located
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent / "submission"
    )

    # Print stdout for monitoring
    print(result.stdout)

    if result.returncode != 0:
        print(f"Error running experiment: {result.stderr}")
        return {"num_train_steps": num_train_steps, "error": result.stderr}

    # Parse the output to extract Eval_AverageReturn
    output = result.stdout

    # Look for the pattern "Eval_AverageReturn : <value>"
    match = re.search(r"Eval_AverageReturn\s*:\s*([-\d.]+)", output)

    if match:
        eval_return = float(match.group(1))
        print(f"\nExtracted Eval_AverageReturn: {eval_return}")
        return {
            "num_train_steps": num_train_steps,
            "eval_average_return": eval_return
        }
    else:
        print("Could not find Eval_AverageReturn in output")
        return {"num_train_steps": num_train_steps, "error": "Could not parse output"}


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

    # Close the figure to free memory
    plt.close(fig)

    # Print summary statistics
    returns = [d["eval_average_return"] for d in data_points]
    best_idx = returns.index(max(returns))
    best_point = data_points[best_idx]

    print("\nPlot Summary:")
    print("-" * 40)
    print(f"Best performance: {best_point['eval_average_return']:.2f}")
    print(f"Best num_train_steps: {best_point['num_train_steps']}")


def main():
    """Run experiments for all hyperparameter values and save results."""

    parser = argparse.ArgumentParser(
        description="Run BC hyperparameter experiment varying num_agent_train_steps_per_iter"
    )
    parser.add_argument(
        "--env", "-e",
        type=str,
        default="Ant-v4",
        choices=list(ENV_CONFIG.keys()),
        help="MuJoCo environment to run experiment on (default: Ant-v4)"
    )
    args = parser.parse_args()

    env = args.env

    # Hyperparameter values to test
    train_steps_values = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]

    # Results storage
    results = {
        "experiment": "BC num_agent_train_steps_per_iter study",
        "env": env,
        "fixed_params": {
            "n_iter": 1,
            "ep_len": 1000,
            "eval_batch_size": 5000,
            "seed": 1
        },
        "data": []
    }

    # Run experiments
    for num_steps in train_steps_values:
        result = run_bc_experiment(num_steps, env)
        results["data"].append(result)

    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Save results to JSON
    json_file = results_dir / f"bc_experiment_results_{env}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete! Results saved to: {json_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    print("-" * 40)
    for data_point in results["data"]:
        if "error" in data_point:
            print(f"  Steps: {data_point['num_train_steps']:5d} -> ERROR: {data_point['error'][:50]}")
        else:
            print(f"  Steps: {data_point['num_train_steps']:5d} -> Eval Return: {data_point['eval_average_return']:.2f}")

    # Generate plot
    print("\nGenerating plot...")
    plot_file = results_dir / f"bc_train_steps_experiment_{env}.png"
    plot_results(results, plot_file)


if __name__ == "__main__":
    main()
