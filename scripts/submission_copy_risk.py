#!/usr/bin/env python3
"""Copy-matched expected inference risk for every saved 64-state quantum fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qem.pennylane_models import TorchQuantumModel
from train_torch import predict_raw
from finite_shot_benchmark import joint_moments, orthogonal_features, square_variance

BUDGETS = (240, 960, 3840, 15360)


def risk_rows(name, y, ids, heads, correlation, weights, bias):
    prediction, single_shot_variance = joint_moments(heads, correlation, weights, bias)
    bias_sq = (prediction - y) ** 2
    rows = []
    for copies in BUDGETS:
        risk = bias_sq + single_shot_variance / (copies // 2)
        by_state = [float(risk[ids == item].mean()) for item in np.unique(ids)]
        rows.append({"model": name, "copies_per_query": copies,
                     "expected_mse": float(risk.mean()), "squared_bias": float(bias_sq.mean()),
                     "measurement_variance": float(single_shot_variance.mean() / (copies // 2)),
                     "state_risk": by_state, "joint_shots": copies // 2})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/submission_revision/datasets"))
    parser.add_argument("--fits", type=Path, default=Path("results/submission_revision/quantum_confirmation"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/copy_risk"))
    parser.add_argument("--datasets", default="confirmation_1,confirmation_2,confirmation_3,confirmation_4,confirmation_5")
    parser.add_argument("--models", default="joint_exchange,exchange_matched_qnn")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets.split(","):
        path = args.data / f"{dataset}.npz"
        with np.load(path, allow_pickle=False) as archive:
            y = archive["test_joint_y"].copy()
            rho = archive["test_joint_rho"].copy()
            angles = archive["test_joint_angles"].copy()
            context = archive["test_joint_context"].copy()
            P = archive["test_joint_P"].copy()
            ids = archive["test_joint_state_id"].copy()
        batch = {"rho": torch.as_tensor(rho, dtype=torch.complex128),
                 "angles": torch.as_tensor(angles, dtype=torch.float64), "y": torch.as_tensor(y)}
        m = np.einsum("nij,nji->n", rho, P).real
        analytic_heads = np.column_stack((m - y / 2, m + y / 2))
        records = {"dataset": dataset, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "scope": "One fresh state preparation source per query; exact-expectation training; inference shots only",
                   "rows": risk_rows("analytic_exchange_direct_K", y, ids, analytic_heads, m * m,
                                     np.asarray([-1., 1.]), 0.)}
        mu = orthogonal_features(rho, context)
        assert np.max(np.abs(.5 * (mu ** 2).sum(axis=1) - y)) < 1e-12
        for copies in BUDGETS:
            variance = .25 * square_variance(mu, copies // 8).sum(axis=1)
            records["rows"].append({"model": "specified_eight_setting_Pauli", "copies_per_query": copies,
                                    "expected_mse": float(variance.mean()), "squared_bias": 0.,
                                    "measurement_variance": float(variance.mean()),
                                    "state_risk": [float(variance[ids == item].mean()) for item in np.unique(ids)],
                                    "shots_per_setting": copies // 8})
        for name in args.models.split(","):
            for seed in (17, 31, 73):
                fit_path = args.fits / dataset / "64" / f"{name}__seed{seed}.json"
                fit = json.loads(fit_path.read_text())
                assert fit["dataset_sha256"] == records["dataset_sha256"]
                with np.load(fit_path.with_suffix(".npz"), allow_pickle=False) as archive:
                    weights = archive["weights"].copy()
                physical_name = fit["physical_model"]
                model = TorchQuantumModel(physical_name, initial_parameters=weights)
                raw = predict_raw(model, batch, 320)
                summary = fit["fit"]["architecture"]["fitted_summary"]
                rows = risk_rows(f"{name}__seed{seed}", y, ids, raw["heads"], raw["correlation"],
                                 np.asarray(summary["head_weights"]), float(summary["head_bias"]))
                exact_prediction = raw["prediction"]
                independent_prediction = raw["heads"] @ np.asarray(summary["head_weights"]) + summary["head_bias"]
                assert np.max(np.abs(exact_prediction - independent_prediction)) < 1e-12
                assert abs(rows[0]["squared_bias"] - fit["metrics"]["test_joint"]["mse"]) < 1e-12
                records["rows"].extend(rows)
        output = args.output / f"{dataset}.json"
        output.write_text(json.dumps(records, indent=2) + "\n")
        print(dataset, {row["model"]: row["expected_mse"] for row in records["rows"]
                        if row["copies_per_query"] == 3840}, flush=True)


if __name__ == "__main__":
    main()
