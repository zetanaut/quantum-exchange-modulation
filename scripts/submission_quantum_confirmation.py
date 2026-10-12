#!/usr/bin/env python3
"""Run the frozen PennyLane/Torch comparison on fresh QIP datasets."""
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
from train_torch import fit_one, predict_raw
from qem.pennylane_models import TorchQuantumModel


def neutral_parameters(seed, mean):
    rng = np.random.default_rng(seed)
    p = np.empty(25)
    p[[0, 2]] = rng.normal(0, .2, 2)
    p[[1, 3]] = rng.uniform(-np.pi, np.pi, 2)
    p[4:20] = rng.uniform(-np.pi, np.pi, 16)
    p[20] = rng.uniform(.08, np.pi / 2 - .08)
    p[21] = rng.uniform(-np.pi / 2, np.pi / 2)
    p[22:24] = rng.normal(0, .2, 2)
    p[24] = mean
    return p


def metrics(y, prediction, ids):
    y = np.asarray(y).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    assert y.shape == prediction.shape
    residual = y - prediction
    sq = residual ** 2
    state_mse = [float(np.mean(sq[ids == item])) for item in np.unique(ids)]
    mse = float(np.mean(sq))
    variance = float(np.var(y))
    return {"mse": mse, "r2": 1 - mse / variance if variance else None,
            "mean_state_mse": float(np.mean(state_mse)), "state_mse": state_mse,
            "max_abs_error": float(np.max(np.abs(residual)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/submission_revision/datasets"))
    parser.add_argument("--plan", type=Path, default=Path("results/submission_revision/quantum_plan.json"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/quantum_confirmation"))
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument("--budgets", default="all")
    parser.add_argument("--starts", default="all")
    parser.add_argument("--maxiter", type=int, default=None)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    names = plan["models"] if args.models == "all" else args.models.split(",")
    datasets = plan["datasets"] if args.datasets == "all" else args.datasets.split(",")
    budgets = plan["state_budgets"] if args.budgets == "all" else [int(v) for v in args.budgets.split(",")]
    seeds = plan["optimizer_starts"] if args.starts == "all" else [int(v) for v in args.starts.split(",")]
    maxiter = plan["LBFGS_max_iterations"] if args.maxiter is None else args.maxiter
    if maxiter != plan["LBFGS_max_iterations"] and "smoke" not in str(args.output):
        raise ValueError("Nonprotocol iteration count requires a separate smoke output")
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    for dataset in datasets:
        path = args.data / f"{dataset}.npz"
        manifest = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        for budget in budgets:
            subset = manifest["nested_subsets"][str(budget)]
            train_rows = np.asarray(subset["train_rows"], dtype=int)
            axis_rows = np.asarray(subset["axis_only_rows"], dtype=int)
            batches = {}
            for split in ("train", "validation", "test_state", "test_context", "test_joint"):
                rows = train_rows if split == "train" else axis_rows if split == "test_context" else slice(None)
                y = data[f"{split}_y"][rows]
                batches[split] = {"rho": torch.as_tensor(data[f"{split}_rho"][rows], dtype=torch.complex128),
                                  "angles": torch.as_tensor(data[f"{split}_angles"][rows], dtype=torch.float64),
                                  "y": torch.as_tensor(y, dtype=torch.float64), "y_numpy": y.copy(),
                                  "state_id": data[f"{split}_state_id"][rows]}
            for name in names:
                physical_name = "joint_exchange" if name == "joint_exchange_neutral" else name
                for seed in seeds:
                    output = args.output / dataset / str(budget) / f"{name}__seed{seed}.json"
                    if output.exists():
                        print("SKIP", output, flush=True)
                        continue
                    override = neutral_parameters(seed, float(batches["train"]["y_numpy"].mean())) if name.endswith("_neutral") else None
                    model, weights, fit = fit_one(physical_name, batches["train"], batches["validation"],
                                                  seed, maxiter, plan["batch_size"], plan["validation_every"],
                                                  initialization_override=override)
                    scores = {}
                    for split, batch in batches.items():
                        raw = predict_raw(model, batch, plan["batch_size"])
                        scores[split] = metrics(batch["y_numpy"], raw["prediction"], batch["state_id"])
                        q = raw["correlation"]
                        p = np.stack([(1 + u * raw["heads"][:, 0] + v * raw["heads"][:, 1] + u * v * q) / 4
                                      for u, v in ((-1, -1), (-1, 1), (1, -1), (1, 1))], axis=1)
                        assert np.min(p) >= -2e-8 and np.max(p) <= 1 + 2e-8
                    # Reload into a fresh model and check the selected checkpoint before saving.
                    loaded = TorchQuantumModel(physical_name, initial_parameters=weights)
                    original = predict_raw(model, batches["validation"], plan["batch_size"])["prediction"]
                    restored = predict_raw(loaded, batches["validation"], plan["batch_size"])["prediction"]
                    reload_error = float(np.max(np.abs(original - restored)))
                    assert reload_error < 1e-12
                    output.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(output.with_suffix(".npz"), weights=weights)
                    row = {"dataset": dataset, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "budget": budget, "model": name, "physical_model": physical_name,
                           "seed": seed, "training_pairs": len(batches["train"]["y_numpy"]),
                           "validation_pairs": len(batches["validation"]["y_numpy"]),
                           "axis_only_seen_states": len(np.unique(batches["test_context"]["state_id"])),
                           "checkpoint_reload_max_abs": reload_error, "fit": fit, "metrics": scores}
                    output.write_text(json.dumps(row, indent=2) + "\n")
                    print(dataset, budget, name, seed, scores["test_joint"]["mse"], flush=True)


if __name__ == "__main__":
    main()
