#!/usr/bin/env python3
"""Paired source-state analysis for the locked joint axis holdout."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path("results/submission_revision")
DATASETS = [f"confirmation_{i}" for i in range(1, 6)]
SEEDS = (17, 31, 73)


def state_errors(dataset, model, budget):
    values = []
    for seed in SEEDS:
        path = ROOT / "quantum_confirmation" / dataset / str(budget) / f"{model}__seed{seed}.json"
        report = json.loads(path.read_text())
        values.append(report["metrics"]["test_joint"]["state_mse"])
    errors = np.asarray(values, dtype=float)
    assert errors.shape == (3, 128)
    return errors.mean(axis=0)


def main():
    rng = np.random.default_rng(2026101291)
    output = {"analysis": "Equal-weight mean over three optimizer starts within each dataset; paired by physical source state",
              "conditioning": "State-cluster uncertainty conditional on each dataset's sampled context pool",
              "dataset_count": 5, "bootstrap_replicates": 2000, "budgets": {}}
    for budget in (8, 16, 32, 64):
        rows = []
        for dataset in DATASETS:
            a = state_errors(dataset, "joint_exchange", budget)
            b = state_errors(dataset, "exchange_matched_qnn", budget)
            difference = a - b
            indices = rng.integers(0, len(difference), size=(2000, len(difference)))
            samples = difference[indices].mean(axis=1)
            rows.append({"dataset": dataset, "exchange_mean_mse": float(a.mean()),
                         "matched_qnn_mean_mse": float(b.mean()),
                         "paired_difference_exchange_minus_qnn": float(difference.mean()),
                         "conditional_state_bootstrap_95_percentile_interval":
                             np.quantile(samples, [.025, .975]).tolist()})
        differences = [row["paired_difference_exchange_minus_qnn"] for row in rows]
        output["budgets"][str(budget)] = {"datasets": rows,
            "mean_paired_dataset_difference": statistics.mean(differences),
            "median_paired_dataset_difference": statistics.median(differences),
            "exchange_lower_mse_datasets": sum(value < 0 for value in differences),
            "interpretation": "Five datasets support descriptive paired comparison; state bootstrap does not estimate a new context population"}
    (ROOT / "paired_statistics.json").write_text(json.dumps(output, indent=2) + "\n")
    for budget, details in output["budgets"].items():
        print(budget, details["mean_paired_dataset_difference"], details["exchange_lower_mse_datasets"])


if __name__ == "__main__":
    main()
