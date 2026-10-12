#!/usr/bin/env python3
"""Compare two-copy query shots with reusable unbiased Pauli features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qem.data import fragility, observable, pauli_features
from qem.pennylane_models import TorchQuantumModel
from train_torch import predict_raw
from finite_shot_benchmark import joint_moments


def query_directions(seed, count, threshold):
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        if abs(n[0]) > threshold:
            rows.append(n)
    return np.asarray(rows)


def classical_trials(rho, directions, copies, replicates, seed):
    assert copies % 12 == 0
    shots = copies // 12
    r = pauli_features(rho).reshape(-1, 4, 4)[:, 1:, :]
    rng = np.random.default_rng(seed)
    successes = rng.binomial(shots, (1 + r[None]) / 2, size=(replicates, *r.shape))
    sums = 2 * successes - shots
    mean = sums / shots
    square = (sums ** 2 - shots) / (shots * (shots - 1))
    G = np.einsum("rsib,rsjb->rsij", mean, mean)
    for i in range(3):
        G[:, :, i, i] = square[:, :, i, :].sum(axis=-1)
    trace = np.trace(G, axis1=-2, axis2=-1)
    contraction = np.einsum("rsij,mi,mj->rsm", G, directions, directions)
    return .5 * (trace[:, :, None] - contraction)


def selected_model(fit_root, dataset, name):
    records = [json.loads((fit_root / dataset / "64" / f"{name}__seed{seed}.json").read_text())
               for seed in (17, 31, 73)]
    selected = min(records, key=lambda item: item["fit"]["best_validation_mse"])
    with np.load(fit_root / dataset / "64" / f"{name}__seed{selected['seed']}.npz") as archive:
        weights = archive["weights"].copy()
    return selected, TorchQuantumModel(name, initial_parameters=weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("results/submission_revision/amortization_plan.json"))
    parser.add_argument("--data", type=Path, default=Path("results/submission_revision/datasets"))
    parser.add_argument("--fits", type=Path, default=Path("results/submission_revision/quantum_confirmation"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/amortization"))
    parser.add_argument("--datasets", default="confirmation_1,confirmation_2,confirmation_3,confirmation_4,confirmation_5")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    copies = plan["total_source_copies_per_state"]
    for dataset_index, dataset in enumerate(args.datasets.split(",")):
        with np.load(args.data / f"{dataset}.npz", allow_pickle=False) as archive:
            ids, indices = np.unique(archive["test_joint_state_id"], return_index=True)
            rho = archive["test_joint_rho"][indices].copy()
        directions = query_directions(plan["context_seed"] + dataset_index,
                                      max(plan["contexts_per_state"]), .55)
        result = {"dataset": dataset, "source_states": len(rho), "total_copies_per_state": copies,
                  "query_seed": plan["context_seed"] + dataset_index, "rows": []}
        for count in plan["contexts_per_state"]:
            n = directions[:count]
            flat_rho = np.repeat(rho, count, axis=0)
            flat_n = np.tile(n, (len(rho), 1))
            P = observable(flat_n)
            y = fragility(flat_rho, P).reshape(len(rho), count)
            m = np.einsum("nij,nji->n", flat_rho, P).real.reshape(len(rho), count)
            analytic_risk = (4 * (1 - m ** 2) - 2 * y ** 2) / (copies / count)
            result["rows"].append({"contexts": count, "method": "analytic_exchange_direct_K",
                                   "expected_mse": float(analytic_risk.mean()),
                                   "quantum_copies_per_query": copies // count})
            trials = classical_trials(rho, n, copies, plan["monte_carlo_replicates"],
                                      plan["context_seed"] + 1000 + dataset_index)
            replicate_risks = np.mean((trials - y[None]) ** 2, axis=(1, 2))
            result["rows"].append({"contexts": count, "method": "reusable_12_Pauli_U_statistics",
                                   "expected_mse": float(replicate_risks.mean()),
                                   "monte_carlo_standard_error": float(replicate_risks.std(ddof=1) / np.sqrt(len(replicate_risks))),
                                   "shots_per_setting_once_per_state": copies // 12})
            angles = np.column_stack((np.arccos(flat_n[:, 2]), np.arctan2(flat_n[:, 1], flat_n[:, 0])))
            batch = {"rho": torch.as_tensor(flat_rho, dtype=torch.complex128),
                     "angles": torch.as_tensor(angles, dtype=torch.float64), "y": torch.as_tensor(y.reshape(-1))}
            for model_name in ("joint_exchange", "exchange_matched_qnn"):
                fit, model = selected_model(args.fits, dataset, model_name)
                raw = predict_raw(model, batch, 320)
                summary = fit["fit"]["architecture"]["fitted_summary"]
                prediction, single_variance = joint_moments(raw["heads"], raw["correlation"],
                    np.asarray(summary["head_weights"]), float(summary["head_bias"]))
                squared_bias = (prediction - y.reshape(-1)) ** 2
                shots = copies // (2 * count)
                result["rows"].append({"contexts": count, "method": model_name,
                                       "selected_validation_seed": fit["seed"],
                                       "expected_mse": float(np.mean(squared_bias + single_variance / shots)),
                                       "squared_bias": float(np.mean(squared_bias)),
                                       "measurement_variance": float(np.mean(single_variance / shots)),
                                       "quantum_copies_per_query": 2 * shots})
        (args.output / f"{dataset}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(dataset, [(r["contexts"], r["method"], r["expected_mse"]) for r in result["rows"]], flush=True)


if __name__ == "__main__":
    main()
