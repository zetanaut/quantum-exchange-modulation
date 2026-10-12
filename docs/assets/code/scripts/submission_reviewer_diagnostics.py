#!/usr/bin/env python3
"""Summarize initialization outcomes and single-query risk decomposition."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results/submission_revision"
threshold = 1e-4
initializations = {}
for model in ("joint_exchange", "joint_exchange_neutral"):
    rows = []
    for dataset in range(1, 6):
        for seed in (17, 31, 73):
            path = ROOT / "quantum_confirmation" / f"confirmation_{dataset}" / "64" / f"{model}__seed{seed}.json"
            record = json.loads(path.read_text())
            rows.append({"dataset": dataset, "seed": seed,
                         "validation_mse": record["fit"]["best_validation_mse"],
                         "joint_test_mse": record["metrics"]["test_joint"]["mse"]})
    initializations[model] = {"rows": rows, "validation_success_count": sum(
        row["validation_mse"] < threshold for row in rows),
        "starts": len(rows), "joint_test_median_mse": float(np.median([
            row["joint_test_mse"] for row in rows])),
        "joint_test_mean_mse": float(np.mean([row["joint_test_mse"] for row in rows]))}

risks = {}
for dataset in range(1, 6):
    data = json.loads((ROOT / "copy_risk" / f"confirmation_{dataset}.json").read_text())
    for row in data["rows"]:
        if row["copies_per_query"] == 3840:
            key = "joint_exchange" if row["model"].startswith("joint_exchange__seed") else row["model"]
            if key in ("joint_exchange", "analytic_exchange_direct_K"):
                risks.setdefault(key, []).append({field: row[field] for field in (
                    "expected_mse", "squared_bias", "measurement_variance")})
risk_summary = {name: {field: float(np.mean([row[field] for row in rows]))
                       for field in ("expected_mse", "squared_bias", "measurement_variance")}
                for name, rows in risks.items()}
result = {"validation_success_threshold_mse": threshold,
          "initialization": initializations, "copy_risk_at_3840": risk_summary}
path = ROOT / "reviewer_diagnostics.json"
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"initialization": {k: {key: v[key] for key in (
    "validation_success_count", "joint_test_median_mse", "joint_test_mean_mse")}
    for k, v in initializations.items()}, "risk": risk_summary}, indent=2))
