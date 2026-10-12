#!/usr/bin/env python3
"""Evaluate predeclared constant, product-feature, and exact Gram controls.

All fitted coefficients use training labels only. No model is selected by test error.
The 144- and 36-column maps encode the known same-spectator operator structure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.data import pauli_features

SPLITS = ("train", "validation", "test_state", "test_context", "test_joint")


def sym2(x, affine=False):
    pairs = [(i, j) for i in range(x.shape[1]) for j in range(i, x.shape[1])]
    quadratic = np.stack([x[:, i] * x[:, j] * (1 if i == j else np.sqrt(2))
                          for i, j in pairs], axis=1)
    return np.column_stack((np.ones(len(x)), np.sqrt(2) * x, quadratic)) if affine else quadratic


def products(a, b):
    return np.einsum("ni,nj->nij", a, b).reshape(len(a), -1)


def representations(rho, direction):
    pauli = pauli_features(rho)
    r = pauli[:, 1:]
    blocks = pauli.reshape(-1, 4, 4)[:, 1:, :]
    nq = sym2(direction)
    within = np.concatenate([sym2(blocks[:, :, b]) for b in range(4)], axis=1)
    gram = np.einsum("nib,njb->nij", blocks, blocks)
    gram_symmetric = np.stack([gram[:, i, j] * (1 if i == j else np.sqrt(2))
                               for i in range(3) for j in range(i, 3)], axis=1)
    oracle = .5 * (np.trace(gram, axis1=1, axis2=2) -
                   np.einsum("ni,nij,nj->n", direction, gram, direction))
    return {"full_product_816": products(sym2(r, True), nq),
            "same_spectator_144": products(within, nq),
            "gram_product_36": products(gram_symmetric, nq)}, oracle


def score(y, prediction, state_ids):
    y = np.asarray(y).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    assert y.shape == prediction.shape and np.all(np.isfinite(prediction))
    errors = (y - prediction) ** 2
    state_mse = [float(errors[state_ids == item].mean()) for item in np.unique(state_ids)]
    mse = float(errors.mean())
    variance = float(y.var())
    return {"mse": mse, "r2": float(1 - mse / variance) if variance else None,
            "max_abs_error": float(np.max(np.abs(y - prediction))),
            "state_count": len(state_mse), "mean_state_mse": float(np.mean(state_mse)),
            "state_mse": state_mse}


def run_dataset(path: Path, manifest: dict):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    features, oracles = {}, {}
    for split in SPLITS:
        features[split], oracles[split] = representations(data[f"{split}_rho"], data[f"{split}_context"])
        assert np.max(np.abs(oracles[split] - data[f"{split}_y"])) < 1e-12
    report = {"dataset": path.stem, "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "fitting": "NumPy least squares with rcond=1e-12, training labels only",
              "feature_scope": "144 and 36 encode known physics; their selection preceded these datasets",
              "budgets": {}}
    coefficients = {}
    for budget, subset in manifest["nested_subsets"].items():
        train_rows = np.asarray(subset["train_rows"], dtype=int)
        axis_rows = np.asarray(subset["axis_only_rows"], dtype=int)
        y_train = data["train_y"][train_rows]
        train_mean = float(y_train.mean())
        budget_report = {"training_states": int(budget), "training_pairs": len(train_rows),
                         "validation_pairs": len(data["validation_y"]), "models": {}}
        for model in ("constant", "gram_oracle", *features["train"]):
            fitted = None
            if model in features["train"]:
                matrix = features["train"][model][train_rows]
                weights, _, rank, singular = np.linalg.lstsq(matrix, y_train, rcond=1e-12)
                fitted = {"columns": matrix.shape[1], "rank": int(rank),
                          "smallest_returned_singular_value": float(singular[-1]),
                          "train_interpolation_mse": float(np.mean((matrix @ weights - y_train) ** 2))}
                coefficients[f"{budget}__{model}"] = weights
            model_row = {"fit": fitted, "splits": {}}
            for split in SPLITS:
                rows = train_rows if split == "train" else axis_rows if split == "test_context" else slice(None)
                y = data[f"{split}_y"][rows]
                ids = data[f"{split}_state_id"][rows]
                if model == "constant":
                    prediction = np.full_like(y, train_mean)
                elif model == "gram_oracle":
                    prediction = oracles[split][rows]
                else:
                    prediction = features[split][model][rows] @ weights
                model_row["splits"][split] = score(y, prediction, ids)
            budget_report["models"][model] = model_row
        report["budgets"][budget] = budget_report
    return report, coefficients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("results/submission_revision/datasets"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_revision/product_baselines"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.data.glob("*.npz")):
        manifest = json.loads(path.with_suffix(".json").read_text())
        report, coefficients = run_dataset(path, manifest)
        (args.output / f"{path.stem}.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(args.output / f"{path.stem}_coefficients.npz", **coefficients)
        print(path.stem, {name: report["budgets"]["64"]["models"][name]["splits"]["test_joint"]["mse"]
                          for name in report["budgets"]["64"]["models"]}, flush=True)


if __name__ == "__main__":
    main()
