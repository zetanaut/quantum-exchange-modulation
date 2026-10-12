#!/usr/bin/env python3
"""Independent checks for the QIP state and physical-axis split records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.data import fragility, gram_features, observable


def check(path, plan):
    with np.load(path, allow_pickle=False) as archive:
        d = {key: archive[key].copy() for key in archive.files}
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["dataset_sha256"]
    splits = ("train", "validation", "test_state", "test_context", "test_joint")
    label_error = gram_error = antipodal_error = 0.
    states = {}
    for split in splits:
        rho, n, y, ids = (d[f"{split}_{key}"] for key in ("rho", "context", "y", "state_id"))
        assert y.ndim == 1 and len(y) == len(rho) == len(n) == len(ids)
        assert np.max(np.abs(np.trace(rho, axis1=-2, axis2=-1) - 1)) < 1e-12
        assert np.min(np.linalg.eigvalsh(rho)) > -1e-12
        P = observable(n)
        label_error = max(label_error, float(np.max(np.abs(fragility(rho, P) - y))))
        gram_error = max(gram_error, float(np.max(np.abs(gram_features(rho, n).sum(axis=1) - y))))
        antipodal_error = max(antipodal_error, float(np.max(np.abs(fragility(rho, observable(-n)) - y))))
        states[split] = set(map(int, ids))
    assert states["train"].isdisjoint(states["validation"])
    assert states["train"].isdisjoint(states["test_joint"])
    assert states["validation"].isdisjoint(states["test_joint"])
    assert states["test_joint"] == states["test_state"]
    assert states["test_context"] == states["train"]
    unique_states = []
    for split in ("train", "validation", "test_joint"):
        _, first = np.unique(d[f"{split}_state_id"], return_index=True)
        unique_states.append(d[f"{split}_rho"][first])
    unique_states = np.concatenate(unique_states)
    distances = np.linalg.norm((unique_states[:, None] - unique_states[None, :]).reshape(len(unique_states), len(unique_states), -1), axis=-1)
    np.fill_diagonal(distances, np.inf)
    minimum_state_distance = float(distances.min())
    assert minimum_state_distance > 1e-10
    threshold = plan["axis_threshold"]
    train_axes = np.unique(d["train_context"], axis=0)
    val_axes = np.unique(d["validation_context"], axis=0)
    held_axes = np.unique(d["test_joint_context"], axis=0)
    assert np.all(np.abs(train_axes[:, 0]) <= threshold)
    assert np.all(np.abs(val_axes[:, 0]) <= threshold)
    assert np.all(np.abs(held_axes[:, 0]) > threshold)
    nearest = np.arccos(np.clip(np.max(np.abs(np.vstack((train_axes, val_axes)) @ held_axes.T), axis=0), -1, 1))
    assert nearest.min() > 1e-10
    for budget, subset in manifest["nested_subsets"].items():
        train_rows = np.asarray(subset["train_rows"], dtype=int)
        axis_rows = np.asarray(subset["axis_only_rows"], dtype=int)
        selected = set(subset["state_ids"])
        assert set(map(int, d["train_state_id"][train_rows])) == selected
        assert set(map(int, d["test_context_state_id"][axis_rows])) == selected
        assert len(train_rows) == int(budget) * plan["counts"]["train_contexts_per_state"]
        assert len(axis_rows) == int(budget) * plan["counts"]["test_contexts_per_state"]
    assert max(label_error, gram_error, antipodal_error) < 1e-12
    return {"dataset": path.stem, "status": "PASS", "label_max_abs_error": label_error,
            "gram_max_abs_error": gram_error, "antipodal_max_abs_error": antipodal_error,
            "nearest_heldout_axis_angle_radians_min": float(nearest.min()),
            "minimum_distinct_state_frobenius_distance": minimum_state_distance,
            "train_states": len(states["train"]), "validation_states": len(states["validation"]),
            "new_test_states": len(states["test_joint"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/submission_revision/datasets"))
    parser.add_argument("--plan", type=Path, default=Path("results/submission_revision/protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/dataset_checks.json"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows = [check(path, plan) for path in sorted(args.data.glob("*.npz"))]
    args.output.write_text(json.dumps({"status": "PASS", "datasets": rows}, indent=2) + "\n")
    for row in rows:
        print(row["dataset"], row["status"], row["nearest_heldout_axis_angle_radians_min"])


if __name__ == "__main__":
    main()
