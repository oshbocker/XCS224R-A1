#!/usr/bin/env python3
"""
Experiment script to run DAgger and compare with BC and expert performance.

This script runs DAgger training with multiple iterations, captures the learning
curve, and generates a plot comparing DAgger, BC, and expert performance.
"""

import argparse
import subprocess
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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


def run_dagger_experiment(env: str, n_iter: int = 10, seed: int = 1) -> dict:
    """
    Run DAgger experiment with the specified number of iterations.

    Args:
        env: Environment name (e.g., "Ant-v4")
        n_iter: Number of DAgger iterations
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing the experiment results for each iteration
    """
    if env not in ENV_CONFIG:
        raise ValueError(f"Unknown environment: {env}. Available: {list(ENV_CONFIG.keys())}")

    config = ENV_CONFIG[env]

    # Build the command - run from submission/ directory where xcs224r module lives
    cmd = [
        "python", "run_hw1.py",
        "--expert_policy_file", config["expert_policy_file"],
        "--env_name", env,
        "--n_iter", str(n_iter),
        "--expert_data", config["expert_data"],
        "--video_log_freq", "-1",
        "--ep_len", "1000",
        "--eval_batch_size", "5000",
        "--batch_size", "1000",
        "--seed", str(seed),
        "--num_agent_train_steps_per_iter", "1000",
        "--do_dagger",
        "--exp_name", f"dagger_{env}_{n_iter}iter",
    ]

    print(f"\n{'='*60}")
    print(f"Running DAgger experiment in env {env} with {n_iter} iterations")
    print(f"{'='*60}")

    # Run the command and capture output
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
        return {"error": result.stderr}

    # Parse the output to extract Eval_AverageReturn and Eval_StdReturn for each iteration
    output = result.stdout

    # Find all iteration blocks and extract metrics
    iterations_data = []

    # Split by iteration markers
    iter_pattern = r"\*+ Iteration (\d+) \*+"
    iter_matches = list(re.finditer(iter_pattern, output))

    for i, match in enumerate(iter_matches):
        iter_num = int(match.group(1))

        # Get the text for this iteration (until next iteration or end)
        start_pos = match.end()
        end_pos = iter_matches[i + 1].start() if i + 1 < len(iter_matches) else len(output)
        iter_text = output[start_pos:end_pos]

        # Extract Eval_AverageReturn
        avg_match = re.search(r"Eval_AverageReturn\s*:\s*([-\d.]+)", iter_text)
        std_match = re.search(r"Eval_StdReturn\s*:\s*([-\d.]+)", iter_text)

        if avg_match and std_match:
            iterations_data.append({
                "iteration": iter_num,
                "eval_average_return": float(avg_match.group(1)),
                "eval_std_return": float(std_match.group(1))
            })
            print(f"Iteration {iter_num}: Mean={float(avg_match.group(1)):.2f}, Std={float(std_match.group(1)):.2f}")

    # Also extract expert performance (Initial_DataCollection_AverageReturn)
    expert_match = re.search(r"Initial_DataCollection_AverageReturn\s*:\s*([-\d.]+)", output)
    expert_return = float(expert_match.group(1)) if expert_match else None

    return {
        "iterations_data": iterations_data,
        "expert_return": expert_return
    }


def run_bc_baseline(env: str, seed: int = 1) -> dict:
    """
    Run BC baseline (single iteration, no DAgger) for comparison.

    Args:
        env: Environment name
        seed: Random seed

    Returns:
        Dictionary with BC performance metrics
    """
    if env not in ENV_CONFIG:
        raise ValueError(f"Unknown environment: {env}")

    config = ENV_CONFIG[env]

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
        "--num_agent_train_steps_per_iter", "1000",
        "--exp_name", f"bc_baseline_{env}",
    ]

    print(f"\nRunning BC baseline for {env}...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent / "submission"
    )

    if result.returncode != 0:
        print(f"Error running BC baseline: {result.stderr}")
        return {"error": result.stderr}

    output = result.stdout
    avg_match = re.search(r"Eval_AverageReturn\s*:\s*([-\d.]+)", output)
    std_match = re.search(r"Eval_StdReturn\s*:\s*([-\d.]+)", output)

    if avg_match and std_match:
        return {
            "eval_average_return": float(avg_match.group(1)),
            "eval_std_return": float(std_match.group(1))
        }
    return {"error": "Could not parse BC output"}


def plot_dagger_results(results: dict, output_file: Path):
    """
    Create a learning curve plot for DAgger with BC and expert baselines.

    Args:
        results: Dictionary containing DAgger results, BC baseline, and expert performance
        output_file: Path to save the plot
    """
    dagger_data = results["dagger"]["iterations_data"]
    bc_data = results["bc"]
    expert_return = results["dagger"]["expert_return"]
    env = results["env"]

    if not dagger_data:
        print("No valid DAgger data points to plot!")
        return

    # Extract DAgger learning curve data
    iterations = [d["iteration"] for d in dagger_data]
    means = [d["eval_average_return"] for d in dagger_data]
    stds = [d["eval_std_return"] for d in dagger_data]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot DAgger learning curve with error bars
    ax.errorbar(iterations, means, yerr=stds, marker='o', markersize=6,
                linewidth=2, capsize=4, capthick=1.5, color='#2E86AB',
                label='DAgger')

    # Plot BC baseline as horizontal line
    if "eval_average_return" in bc_data:
        bc_mean = bc_data["eval_average_return"]
        bc_std = bc_data["eval_std_return"]
        ax.axhline(y=bc_mean, color='#E94F37', linestyle='--', linewidth=2,
                   label=f'BC (mean={bc_mean:.0f})')
        ax.axhspan(bc_mean - bc_std, bc_mean + bc_std, alpha=0.2, color='#E94F37')

    # Plot expert performance as horizontal line
    if expert_return:
        ax.axhline(y=expert_return, color='#28A745', linestyle='-', linewidth=2,
                   label=f'Expert ({expert_return:.0f})')

    # Add labels and title
    ax.set_xlabel("DAgger Iteration", fontsize=12)
    ax.set_ylabel("Eval_AverageReturn", fontsize=12)
    ax.set_title(f"DAgger Learning Curve ({env})", fontsize=14)

    # Set x-axis to show integer iterations
    ax.set_xticks(iterations)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    # Tight layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

    # Close the figure
    plt.close(fig)


def main():
    """Run DAgger experiment and generate learning curve plot."""

    parser = argparse.ArgumentParser(
        description="Run DAgger experiment and compare with BC baseline"
    )
    parser.add_argument(
        "--env", "-e",
        type=str,
        default="Ant-v4",
        choices=list(ENV_CONFIG.keys()),
        help="MuJoCo environment to run experiment on (default: Ant-v4)"
    )
    parser.add_argument(
        "--n_iter", "-n",
        type=int,
        default=10,
        help="Number of DAgger iterations (default: 10)"
    )
    args = parser.parse_args()

    env = args.env
    n_iter = args.n_iter

    # Results storage
    results = {
        "experiment": "DAgger learning curve",
        "env": env,
        "n_iter": n_iter,
        "fixed_params": {
            "ep_len": 1000,
            "eval_batch_size": 5000,
            "batch_size": 1000,
            "num_agent_train_steps_per_iter": 1000,
            "n_layers": 2,
            "size": 64,
            "learning_rate": 5e-3,
            "seed": 1
        },
        "dagger": None,
        "bc": None
    }

    # Run BC baseline first
    print("\n" + "="*60)
    print("Running BC baseline...")
    print("="*60)
    results["bc"] = run_bc_baseline(env)

    # Run DAgger experiment
    print("\n" + "="*60)
    print(f"Running DAgger with {n_iter} iterations...")
    print("="*60)
    results["dagger"] = run_dagger_experiment(env, n_iter)

    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Save results to JSON
    json_file = results_dir / f"dagger_experiment_results_{env}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete! Results saved to: {json_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    print("-" * 50)
    if results["dagger"].get("expert_return"):
        print(f"  Expert Return: {results['dagger']['expert_return']:.2f}")
    if "eval_average_return" in results["bc"]:
        print(f"  BC Return: {results['bc']['eval_average_return']:.2f} ± {results['bc']['eval_std_return']:.2f}")
    if results["dagger"].get("iterations_data"):
        final = results["dagger"]["iterations_data"][-1]
        print(f"  DAgger Final (iter {final['iteration']}): {final['eval_average_return']:.2f} ± {final['eval_std_return']:.2f}")

    # Generate plot
    print("\nGenerating plot...")
    plot_file = results_dir / f"dagger_learning_curve_{env}.png"
    plot_dagger_results(results, plot_file)


if __name__ == "__main__":
    main()
