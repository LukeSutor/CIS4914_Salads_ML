#!/usr/bin/env python3
"""
Hyperparameter search script for MIL-TCN model using Optuna.
Optimizes for Validation F1 score.

Usage:
    uv run tune.py --config ../configs/training.yaml --n-trials 50
"""

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import optuna

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Update config dictionary with flattened parameters from Optuna."""
    new_config = copy.deepcopy(config)
    
    # Model params
    if "model" not in new_config:
        new_config["model"] = {}
    
    # Construct hid_channels dynamically
    if "hidden_dim" in params and "num_layers" in params:
        dim = params["hidden_dim"]
        layers = params["num_layers"]
        # Simple architecture: constant width or doubling?
        # Let's go with a strategy that mimics the original: [128, 128, 256, 256]
        # We'll do half layers at dim, half at dim*2
        half = layers // 2
        rest = layers - half
        channels = [dim] * half + [dim * 2] * rest
        new_config["model"]["hid_channels"] = channels
    
    if "kernel_size" in params:
        new_config["model"]["kernel_size"] = params["kernel_size"]
    if "dropout" in params:
        new_config["model"]["dropout"] = params["dropout"]
    if "mil_hidden" in params:
        new_config["model"]["mil_hidden"] = params["mil_hidden"]
    if "use_noisy_or" in params:
        new_config["model"]["use_noisy_or"] = params["use_noisy_or"]

    # Train params
    if "train" not in new_config:
        new_config["train"] = {}
        
    if "lr" in params:
        new_config["train"]["lr"] = params["lr"]
    if "weight_decay" in params:
        new_config["train"]["weight_decay"] = params["weight_decay"]
    if "batch_size" in params:
        new_config["train"]["batch_size"] = params["batch_size"]
    if "lambda_instance" in params:
        new_config["train"]["lambda_instance"] = params["lambda_instance"]
    if "max_pos_weight_bag" in params:
        new_config["train"]["max_pos_weight_bag"] = params["max_pos_weight_bag"]
        
    return new_config


def run_trial(trial: optuna.Trial, base_config_path: Path, verbose: bool = False) -> float:
    # 1. Suggest Hyperparameters
    params = {
        # Model Architecture
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "mil_hidden": trial.suggest_categorical("mil_hidden", [64, 128, 256]),
        "use_noisy_or": trial.suggest_categorical("use_noisy_or", [True, False]),
        
        # Training
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "lambda_instance": trial.suggest_float("lambda_instance", 0.0, 1.0),
        "max_pos_weight_bag": trial.suggest_float("max_pos_weight_bag", 1.0, 20.0),
    }

    # 2. Prepare Config
    base_config = load_config(base_config_path)
    trial_config = update_config(base_config, params)
    
    # Set run name for tracking
    run_name = f"trial_{trial.number}"
    trial_config["run_name"] = run_name
    
    # Create temp config file
    # We use a temporary directory to avoid cluttering
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(trial_config, tmp, default_flow_style=False)
        tmp_config_path = tmp.name

    try:
        # 3. Run Training
        train_script = Path(__file__).parent / "train.py"
        cmd = [
            sys.executable,
            str(train_script),
            "--config", tmp_config_path,
            "--run-name", run_name
        ]
        
        if verbose:
            print(f"\n[Trial {trial.number}] Running with params: {params}")
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # We handle errors manually
        )

        if result.returncode != 0:
            print(f"\n[Trial {trial.number}] Training failed!")
            print("Stderr:", result.stderr)
            raise optuna.TrialPruned("Training failed")

        # 4. Parse Output
        # Look for: "Training complete! Best val F1: 0.123"
        match = re.search(r"Best val F1: ([-\d\.]+)", result.stdout)
        if match:
            f1_score = float(match.group(1))
            if verbose:
                print(f"[Trial {trial.number}] Result F1: {f1_score}")
            return f1_score
        else:
            print(f"\n[Trial {trial.number}] Could not parse F1 score from output.")
            # print("Stdout:", result.stdout) # Uncomment for debugging
            return 0.0

    finally:
        # Cleanup
        if os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Search")
    parser.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--storage", type=str, default="sqlite:///db.sqlite3", help="Optuna storage URL")
    parser.add_argument("--study-name", type=str, default="mil_tcn_study", help="Study name")
    parser.add_argument("--verbose", action="store_true", help="Print details")
    args = parser.parse_args()

    base_config_path = Path(args.config)
    if not base_config_path.exists():
        print(f"Config file not found: {base_config_path}")
        sys.exit(1)

    print(f"Starting optimization with {args.n_trials} trials...")
    print(f"Base config: {base_config_path}")
    
    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True
    )

    try:
        study.optimize(
            lambda t: run_trial(t, base_config_path, args.verbose),
            n_trials=args.n_trials
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    print("\n" + "="*60)
    print("Optimization Complete")
    print("="*60)
    
    if len(study.trials) == 0:
        print("No trials completed.")
        return

    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value (F1): {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best config
    best_config = update_config(load_config(base_config_path), study.best_params)
    best_config_path = Path("best_config.yaml")
    save_config(best_config, best_config_path)
    print(f"\nSaved best configuration to: {best_config_path.absolute()}")


if __name__ == "__main__":
    main()
