#!/usr/bin/env python3
"""Generate independent, antipodally separated QIP confirmation datasets.

This leaves the published benchmark untouched. The locked plan is stored in
results/submission_revision/protocol.json and must exist before generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.data import fragility, observable, random_state


def directions(rng: np.random.Generator, count: int, heldout: bool, threshold: float) -> np.ndarray:
    rows = []
    while len(rows) < count:
        candidate = rng.normal(size=(max(32, count * 2), 3))
        candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
        accepted = candidate[np.abs(candidate[:, 0]) > threshold if heldout else
                             np.abs(candidate[:, 0]) <= threshold]
        rows.extend(accepted[:count - len(rows)])
    return np.asarray(rows)


def make_rows(states, families, ids, pool, per_state, rng, context_offset):
    state_ids, context_ids = [], []
    for state_id in ids:
        chosen = rng.choice(len(pool), per_state, replace=False)
        state_ids.extend([int(state_id)] * per_state)
        context_ids.extend(chosen.tolist())
    state_ids = np.asarray(state_ids, dtype=np.int64)
    context_ids = np.asarray(context_ids, dtype=np.int64)
    rho = states[state_ids]
    context = pool[context_ids]
    P = observable(context)
    return {
        "rho": rho, "context": context,
        "angles": np.column_stack((np.arccos(context[:, 2]), np.arctan2(context[:, 1], context[:, 0]))),
        "P": P, "y": fragility(rho, P), "state_id": state_ids,
        "context_id": context_ids + context_offset,
        "family": np.asarray([families[i] for i in state_ids]),
    }


def make_dataset(entry: dict, plan: dict, destination: Path) -> dict:
    root = np.random.SeedSequence(entry["seed"])
    state_seed, context_seed, row_seed, order_seed = root.spawn(4)
    state_rng = np.random.default_rng(state_seed)
    context_rng = np.random.default_rng(context_seed)
    row_rng = np.random.default_rng(row_seed)
    order_rng = np.random.default_rng(order_seed)
    counts = plan["counts"]
    n_train, n_val, n_test = (counts[k] for k in ("train_states", "validation_states", "test_states"))
    total = n_train + n_val + n_test
    states, families = zip(*(random_state(state_rng, i) for i in range(total)))
    states = np.stack(states)
    train_pool = directions(context_rng, counts["train_axes"], False, plan["axis_threshold"])
    val_pool = directions(context_rng, counts["validation_axes"], False, plan["axis_threshold"])
    held_pool = directions(context_rng, counts["heldout_axes"], True, plan["axis_threshold"])
    train_ids = np.arange(n_train)
    val_ids = np.arange(n_train, n_train + n_val)
    test_ids = np.arange(n_train + n_val, total)
    specs = {
        "train": (train_ids, train_pool, counts["train_contexts_per_state"], 0),
        "validation": (val_ids, val_pool, counts["validation_contexts_per_state"], len(train_pool)),
        "test_state": (test_ids, train_pool, counts["test_contexts_per_state"], 0),
        "test_context": (train_ids, held_pool, counts["test_contexts_per_state"], len(train_pool) + len(val_pool)),
        "test_joint": (test_ids, held_pool, counts["test_contexts_per_state"], len(train_pool) + len(val_pool)),
    }
    out = {}
    for split, (ids, pool, per_state, offset) in specs.items():
        for key, value in make_rows(states, families, ids, pool, per_state, row_rng, offset).items():
            out[f"{split}_{key}"] = value
    # One family-balanced nested state sequence, fixed before any fit.
    groups = {}
    for family in sorted(set(families[:n_train])):
        group = np.asarray([i for i in train_ids if families[i] == family])
        order_rng.shuffle(group)
        groups[family] = group.tolist()
    ordered = [int(group[pos]) for pos in range(max(map(len, groups.values())))
               for group in groups.values() if pos < len(group)]
    subsets = {}
    for budget in plan["state_budgets"]:
        chosen = np.asarray(ordered[:budget])
        train_rows = np.flatnonzero(np.isin(out["train_state_id"], chosen))
        axis_rows = np.flatnonzero(np.isin(out["test_context_state_id"], chosen))
        assert len(train_rows) == budget * counts["train_contexts_per_state"]
        assert len(axis_rows) == budget * counts["test_contexts_per_state"]
        subsets[str(budget)] = {"state_ids": chosen.tolist(), "train_rows": train_rows.tolist(),
                                "axis_only_rows": axis_rows.tolist()}
    assert not set(out["train_state_id"]) & set(out["test_joint_state_id"])
    assert not set(out["validation_state_id"]) & set(out["test_joint_state_id"])
    assert np.all(np.abs(out["train_context"][:, 0]) <= plan["axis_threshold"])
    assert np.all(np.abs(out["validation_context"][:, 0]) <= plan["axis_threshold"])
    assert np.all(np.abs(out["test_joint_context"][:, 0]) > plan["axis_threshold"])
    assert np.all(np.abs(out["test_context_context"][:, 0]) > plan["axis_threshold"])
    assert max(np.max(np.abs(np.linalg.norm(out[f"{s}_context"], axis=1) - 1)) for s in specs) < 1e-12
    eigenvalues = np.linalg.eigvalsh(states)
    assert eigenvalues.min() >= -1e-12
    assert np.max(np.abs(np.trace(states, axis1=-2, axis2=-1) - 1)) < 1e-12
    overlap = np.abs(train_pool @ held_pool.T)
    nearest_axis_angle = np.arccos(np.clip(overlap.max(axis=0), -1, 1))
    assert nearest_axis_angle.min() > 1e-9
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **out)
    manifest = {"name": entry["name"], "root_seed": entry["seed"],
                "rng_stream_spawn_keys": {"states": list(state_seed.spawn_key),
                    "context_pools": list(context_seed.spawn_key), "row_selection": list(row_seed.spawn_key),
                    "training_order": list(order_seed.spawn_key)},
                "source_families": sorted(set(families)), "nested_subsets": subsets,
                "split_pairs": {s: int(len(out[f"{s}_y"])) for s in specs},
                "heldout_axis_nearest_training_angle_radians_min": float(nearest_axis_angle.min()),
                "heldout_axis_nearest_training_angle_radians_mean": float(nearest_axis_angle.mean()),
                "min_state_eigenvalue": float(eigenvalues.min()),
                "dataset_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("results/submission_revision/protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/datasets"))
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_text())
    for entry in plan["datasets"]:
        destination = args.output / f"{entry['name']}.npz"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        print(json.dumps(make_dataset(entry, plan, destination), indent=2), flush=True)


if __name__ == "__main__":
    main()
