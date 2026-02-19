#!/usr/bin/env python3
"""
Experiment script to study how num_agent_train_steps_per_iter affects BC agent performance.

This script runs behavior cloning training with different values of the
num_agent_train_steps_per_iter hyperparameter and saves the results for plotting.
"""

import subprocess
import re
import json
import os
from pathlib import Path


def run_bc_experiment(num_train_steps: int, seed: int = 1) -> dict:
    """
    Run a single BC experiment with the specified number of training steps.

    Args:
        num_train_steps: Number of gradient updates per iteration
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing the experiment results
    """
    # Build the command - run from submission/ directory where xcs224r module lives
    cmd = [
        "python", "run_hw1.py",
        "--expert_policy_file", "xcs224r/policies/experts/Ant.pkl",
        "--env_name", "Ant-v4",
        "--n_iter", "1",
        "--expert_data", "xcs224r/expert_data/expert_data_Ant-v4.pkl",
        "--video_log_freq", "-1",
        "--ep_len", "1000",
        "--eval_batch_size", "5000",
        "--seed", str(seed),
        "--num_agent_train_steps_per_iter", str(num_train_steps),
        "--exp_name", f"bc_steps_{num_train_steps}",
    ]

    print(f"\n{'='*60}")
    print(f"Running experiment with num_agent_train_steps_per_iter = {num_train_steps}")
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


def main():
    """Run experiments for all hyperparameter values and save results."""

    # Hyperparameter values to test
    train_steps_values = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]

    # Results storage
    results = {
        "experiment": "BC num_agent_train_steps_per_iter study",
        "env": "Ant-v4",
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
        result = run_bc_experiment(num_steps)
        results["data"].append(result)

    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Save results to JSON
    output_file = results_dir / "bc_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete! Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSummary:")
    print("-" * 40)
    for data_point in results["data"]:
        if "error" in data_point:
            print(f"  Steps: {data_point['num_train_steps']:5d} -> ERROR: {data_point['error'][:50]}")
        else:
            print(f"  Steps: {data_point['num_train_steps']:5d} -> Eval Return: {data_point['eval_average_return']:.2f}")


if __name__ == "__main__":
    main()
