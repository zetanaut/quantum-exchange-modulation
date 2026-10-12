#!/usr/bin/env python3
"""Fit frozen neural controls on independent datasets and save every run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.strong_classical import feature_family
from submission_classical_search import train


def transform(train_x, values):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1
    return [(value - mean) / scale for value in values], mean, scale


def score(y, prediction, ids):
    y = np.asarray(y).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    assert y.shape == prediction.shape
    residual = y - prediction
    state_errors = [float(np.mean(residual[ids == state] ** 2)) for state in np.unique(ids)]
    mse = float(np.mean(residual ** 2))
    variance = float(np.var(y))
    return {"mse": mse, "r2": 1 - mse / variance if variance else None,
            "mean_state_mse": float(np.mean(state_errors)), "state_mse": state_errors,
            "max_abs_error": float(np.max(np.abs(residual)))}


def run(path, manifest, model_name, config, budget, seed, updates, output, validation_every, patience):
    family = model_name.rsplit("__", 1)[0]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    subset = manifest["nested_subsets"][str(budget)]
    train_rows = np.asarray(subset["train_rows"], dtype=int)
    axis_rows = np.asarray(subset["axis_only_rows"], dtype=int)
    splits = {}
    for split in ("train", "validation", "test_state", "test_context", "test_joint"):
        x, context = feature_family(data[f"{split}_rho"], data[f"{split}_context"], family)
        rows = train_rows if split == "train" else axis_rows if split == "test_context" else slice(None)
        splits[split] = {"x": x[rows], "context": context[rows], "y": data[f"{split}_y"][rows],
                         "state_id": data[f"{split}_state_id"][rows]}
    xs, x_mean, x_scale = transform(splits["train"]["x"], [splits[s]["x"] for s in splits])
    cs, c_mean, c_scale = transform(splits["train"]["context"], [splits[s]["context"] for s in splits])
    for index, split in enumerate(splits):
        splits[split]["x_standardized"] = torch.as_tensor(xs[index], dtype=torch.float64)
        splits[split]["context_standardized"] = torch.as_tensor(cs[index], dtype=torch.float64)
    tr, va = splits["train"], splits["validation"]
    fit_data = (tr["x_standardized"], tr["context_standardized"], torch.as_tensor(tr["y"], dtype=torch.float64),
                va["x_standardized"], va["context_standardized"], torch.as_tensor(va["y"], dtype=torch.float64))
    fit, model = train(config, fit_data, updates, seed, validation_every, patience, return_model=True)
    metrics = {}
    with torch.no_grad():
        for split, values in splits.items():
            prediction = model(values["x_standardized"], values["context_standardized"]).numpy()
            metrics[split] = score(values["y"], prediction, values["state_id"])
    row = {"dataset": path.stem, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "model": model_name, "configuration": config, "budget": budget, "optimizer_seed": seed,
           "train_labels": len(tr["y"]), "validation_labels": len(va["y"]),
           "axis_only_test_states_seen_in_training": len(np.unique(splits["test_context"]["state_id"])),
           "fit": fit, "metrics": metrics,
           "test_access": "Predictions computed only after the validation-selected checkpoint was loaded"}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "configuration": config, "model": model_name,
                "x_mean": x_mean, "x_scale": x_scale, "c_mean": c_mean, "c_scale": c_scale,
                "dataset_sha256": row["dataset_sha256"], "budget": budget, "seed": seed},
               output.with_suffix(".pt"))
    output.write_text(json.dumps(row, indent=2) + "\n")
    return metrics["test_joint"]["mse"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/submission_revision/datasets"))
    parser.add_argument("--selection", type=Path, default=Path("results/submission_revision/classical_selection.json"))
    parser.add_argument("--plan", type=Path, default=Path("results/submission_revision/classical_search_plan.json"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/classical_confirmation"))
    parser.add_argument("--datasets", default="confirmation_1,confirmation_2,confirmation_3,confirmation_4,confirmation_5")
    parser.add_argument("--models", default="all")
    parser.add_argument("--budgets", default="8,16,32,64")
    parser.add_argument("--starts", default="17,31,73")
    parser.add_argument("--updates", type=int, default=5000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    selection = json.loads(args.selection.read_text())["selected"]
    plan = json.loads(args.plan.read_text())
    models = list(selection) if args.models == "all" else args.models.split(",")
    for name in args.datasets.split(","):
        dataset = args.data / f"{name}.npz"
        manifest = json.loads(dataset.with_suffix(".json").read_text())
        for budget in [int(value) for value in args.budgets.split(",")]:
            for model_name in models:
                for seed in [int(value) for value in args.starts.split(",")]:
                    output = args.output / name / str(budget) / f"{model_name}__seed{seed}.json"
                    if output.exists():
                        print("SKIP", output, flush=True)
                        continue
                    mse = run(dataset, manifest, model_name, selection[model_name], budget, seed,
                              args.updates, output, plan["validation_every"], plan["patience_checkpoints"])
                    print(name, budget, model_name, seed, mse, flush=True)


if __name__ == "__main__":
    main()
