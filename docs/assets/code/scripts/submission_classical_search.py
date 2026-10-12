#!/usr/bin/env python3
"""Validation-only staged search for stronger MLP and deep-FiLM controls."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.strong_classical import feature_family, make_model


def standardize(train, validation):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1
    return (train - mean) / scale, (validation - mean) / scale


def load_features(path, family):
    with np.load(path, allow_pickle=False) as data:
        x_train, c_train = feature_family(data["train_rho"], data["train_context"], family)
        x_val, c_val = feature_family(data["validation_rho"], data["validation_context"], family)
        y_train = data["train_y"].copy()
        y_val = data["validation_y"].copy()
    x_train, x_val = standardize(x_train, x_val)
    c_train, c_val = standardize(c_train, c_val)
    return tuple(torch.as_tensor(item, dtype=torch.float64) for item in
                 (x_train, c_train, y_train, x_val, c_val, y_val))


def candidates(plan, family):
    grid = list(itertools.product(plan["architectures"], plan["widths"], plan["depths"],
                                  plan["activations"], plan["adam_learning_rates"], plan["weight_decay"]))
    rng = np.random.default_rng(plan["candidate_rng_seed"] + plan["feature_families"].index(family))
    output = []
    for kind in plan["architectures"]:
        subset = [values for values in grid if values[0] == kind]
        choices = rng.choice(len(subset), plan["candidate_count_per_family"] // 2, replace=False)
        for index in choices:
            k, width, depth, activation, lr, decay = subset[index]
            output.append({"kind": k, "width": width, "depth": depth, "activation": activation,
                           "learning_rate": lr, "weight_decay": decay})
    return output


def train(config, data, updates, seed, validation_every, patience, return_model=False):
    x, context, y, xv, cv, yv = data
    torch.manual_seed(seed)
    model = make_model(config["kind"], x.shape[1], context.shape[1], config["width"],
                       config["depth"], config["activation"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    best = {"validation_mse": float("inf"), "step": 0, "train_mse": None}
    history = []
    state = None
    stale = 0
    start = time.perf_counter()
    evaluations = 0
    for step in range(updates + 1):
        if step % validation_every == 0 or step == updates:
            with torch.no_grad():
                train_mse = float(torch.mean((model(x, context) - y) ** 2))
                val_mse = float(torch.mean((model(xv, cv) - yv) ** 2))
            history.append({"step": step, "train_mse": train_mse, "validation_mse": val_mse})
            if val_mse < best["validation_mse"]:
                best = {"validation_mse": val_mse, "step": step, "train_mse": train_mse}
                state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if step == updates:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x, context) - y) ** 2)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite classical training objective")
        loss.backward()
        optimizer.step()
        evaluations += 1
    assert state is not None
    model.load_state_dict(state)
    result = {"best": best, "history": history, "updates_executed": evaluations,
              "parameter_count": sum(value.numel() for value in model.parameters()),
              "elapsed_seconds": time.perf_counter() - start}
    return (result, model) if return_model else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("results/submission_revision/classical_search_plan.json"))
    parser.add_argument("--data", type=Path, default=Path("results/submission_revision/datasets/development.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/classical_search"))
    parser.add_argument("--smoke", action="store_true", help="One configuration per family, 20 updates; separate output only")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.smoke and "smoke" not in str(args.output):
        raise ValueError("Smoke output path must contain 'smoke'")
    for family in plan["feature_families"]:
        path = args.output / f"{family}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        data = load_features(args.data, family)
        configs = candidates(plan, family)
        if args.smoke:
            configs = configs[:1]
        rows = []
        for stage, updates in enumerate(plan["stage_updates"] if not args.smoke else [20]):
            for index, config in enumerate(configs):
                try:
                    result = train(config, data, updates, plan["candidate_start_seed"],
                                   plan["validation_every"] if not args.smoke else 5,
                                   plan["patience_checkpoints"])
                    row = {"stage": stage, "candidate": config, "result": result, "error": None}
                except Exception as exc:
                    row = {"stage": stage, "candidate": config, "result": None,
                           "error": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                path.write_text(json.dumps({"family": family, "smoke": args.smoke,
                                            "candidate_seed": plan["candidate_rng_seed"],
                                            "runs": rows}, indent=2) + "\n")
                print(family, stage, index + 1, len(configs),
                      row["result"]["best"]["validation_mse"] if row["result"] else row["error"],
                      flush=True)
            if args.smoke or stage == len(plan["stage_updates"]) - 1:
                break
            completed = [row for row in rows if row["stage"] == stage and row["result"]]
            next_count = plan["stage_survivors_per_family"][stage + 1] // 2
            configs = []
            for kind in plan["architectures"]:
                ordered = sorted((row for row in completed if row["candidate"]["kind"] == kind),
                                 key=lambda row: row["result"]["best"]["validation_mse"])
                configs.extend(row["candidate"] for row in ordered[:next_count])
            if not configs:
                raise RuntimeError(f"No valid candidates in {family}")


if __name__ == "__main__":
    main()
