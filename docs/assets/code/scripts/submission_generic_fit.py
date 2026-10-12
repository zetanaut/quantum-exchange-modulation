#!/usr/bin/env python3
"""Validation-selected generic quantum conditioner on the QIP axis split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.generic_conditioned import GenericConditionedExchange


def score(y, prediction, ids):
    error = (np.asarray(y).reshape(-1) - np.asarray(prediction).reshape(-1)) ** 2
    variance = float(np.var(y))
    mse = float(np.mean(error))
    return {"mse": mse, "r2": 1 - mse / variance if variance else None,
            "state_mse": [float(np.mean(error[ids == item])) for item in np.unique(ids)],
            "max_abs_error": float(np.sqrt(error.max()))}


def fit(path, seed, lr, plan, output, evaluate_tests, blind=False):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    model = GenericConditionedExchange()
    train = (torch.as_tensor(data["train_rho"], dtype=torch.complex128),
             torch.as_tensor(data["train_context"], dtype=torch.float64),
             torch.as_tensor(data["train_y"], dtype=torch.float64))
    validation = (torch.as_tensor(data["validation_rho"], dtype=torch.complex128),
                  torch.as_tensor(data["validation_context"], dtype=torch.float64),
                  torch.as_tensor(data["validation_y"], dtype=torch.float64))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    def predict(rho, context):
        control = torch.zeros_like(context) if blind else None
        return model(rho, context, control_context=control)["prediction"]
    history = []
    best = {"mse": float("inf"), "step": 0}
    best_state = None
    stale = 0
    start = time.perf_counter()
    for step in range(plan["max_updates"] + 1):
        if step % plan["validation_every"] == 0 or step == plan["max_updates"]:
            with torch.no_grad():
                tr_mse = float(torch.mean((predict(train[0], train[1]) - train[2]) ** 2))
                val_mse = float(torch.mean((predict(validation[0], validation[1]) - validation[2]) ** 2))
            history.append({"step": step, "train_mse": tr_mse, "validation_mse": val_mse})
            if val_mse < best["mse"]:
                best = {"mse": val_mse, "step": step}
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= plan["patience_checkpoints"]:
                break
        if step == plan["max_updates"]:
            break
        optimizer.zero_grad(set_to_none=True)
        prediction = predict(train[0], train[1])
        loss = torch.mean((prediction - train[2]) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite generic conditioner loss")
        loss.backward()
        optimizer.step()
    model.load_state_dict(best_state)
    metrics = {}
    if evaluate_tests:
        with torch.no_grad():
            for split in ("train", "validation", "test_state", "test_context", "test_joint"):
                rho = torch.as_tensor(data[f"{split}_rho"], dtype=torch.complex128)
                context = torch.as_tensor(data[f"{split}_context"], dtype=torch.float64)
                prediction = predict(rho, context).detach().numpy()
                metrics[split] = score(data[f"{split}_y"], prediction, data[f"{split}_state_id"])
    row = {"dataset": path.stem, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "model": "context_blind_source_fixed_P_readout" if blind else "generic_source_conditioner_fixed_P_readout", "seed": seed, "learning_rate": lr,
           "training_pairs": len(train[2]), "validation_pairs": len(validation[2]),
           "parameters": sum(value.numel() for value in model.parameters()),
           "fit": {"best_validation_mse": best["mse"], "best_validation_step": best["step"],
                   "updates_executed": step, "elapsed_seconds": time.perf_counter() - start,
                   "history": history}, "test_evaluated": evaluate_tests, "metrics": metrics}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "dataset_sha256": row["dataset_sha256"],
                "seed": seed, "learning_rate": lr}, output.with_suffix(".pt"))
    output.write_text(json.dumps(row, indent=2) + "\n")
    return best["mse"], metrics.get("test_joint", {}).get("mse")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("results/submission_revision/generic_plan.json"))
    parser.add_argument("--data", type=Path, default=Path("results/submission_revision/datasets"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/generic_fits"))
    parser.add_argument("--phase", choices=["development", "confirmation"], required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--blind", action="store_true", help="Pass a constant zero context only to the source controller")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.max_updates is not None:
        if "smoke" not in str(args.output):
            raise ValueError("Nonprotocol update count requires a separate smoke output")
        plan["max_updates"] = args.max_updates
    if args.blind and "context_blind" not in str(args.output):
        raise ValueError("Context-blind runs require a separate context_blind output")
    torch.set_num_threads(1)
    if args.phase == "development":
        datasets = ["development"]
        rates = plan["development_learning_rates"]
    else:
        datasets = plan["confirmation_datasets"] if args.datasets == "all" else args.datasets.split(",")
        selection = json.loads((args.output.parent / "generic_selection.json").read_text())
        rates = [selection["learning_rate"]]
    for dataset in datasets:
        path = args.data / f"{dataset}.npz"
        for lr in rates:
            for seed in plan["seeds"]:
                output = args.output / dataset / f"lr{lr:g}__seed{seed}.json"
                if output.exists():
                    print("SKIP", output, flush=True)
                    continue
                val, test = fit(path, seed, lr, plan, output, args.phase == "confirmation", blind=args.blind)
                print(dataset, lr, seed, val, test, flush=True)


if __name__ == "__main__":
    main()
