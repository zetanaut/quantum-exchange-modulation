#!/usr/bin/env python3
"""Fit only the affine head on fixed analytic two-copy exchange features."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.data import fragility


def features(rho, P):
    m = np.einsum("nij,nji->n", rho, P).real
    d = fragility(rho, P)
    return np.column_stack((m - d / 2, m + d / 2, np.ones(len(m))))


def main():
    root = Path("results/submission_revision")
    output = root / "analytic_head"
    output.mkdir(exist_ok=True)
    for dataset in ["development", *(f"confirmation_{i}" for i in range(1, 6))]:
        path = root / "datasets" / f"{dataset}.npz"
        manifest = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        matrices = {split: features(data[f"{split}_rho"], data[f"{split}_P"])
                    for split in ("train", "validation", "test_state", "test_context", "test_joint")}
        report = {"dataset": dataset, "description": "Fixed analytic V=exp(-i pi P/4), eta=pi/4, chi=0; only 2 head weights plus bias fitted",
                  "resource_scope": "Exact simulator port expectations, not zero-copy measured features", "budgets": {}}
        for budget, subset in manifest["nested_subsets"].items():
            rows = np.asarray(subset["train_rows"], dtype=int)
            axis_rows = np.asarray(subset["axis_only_rows"], dtype=int)
            weights, _, rank, singular = np.linalg.lstsq(matrices["train"][rows], data["train_y"][rows], rcond=1e-12)
            entry = {"parameters": weights.tolist(), "train_rank": int(rank),
                     "smallest_singular_value": float(singular[-1]), "mse": {}}
            for split in matrices:
                selected = rows if split == "train" else axis_rows if split == "test_context" else slice(None)
                prediction = matrices[split][selected] @ weights
                y = data[f"{split}_y"][selected]
                assert prediction.shape == y.shape
                entry["mse"][split] = float(np.mean((prediction - y) ** 2))
            report["budgets"][budget] = entry
        target = output / f"{dataset}.json"
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(dataset, report["budgets"]["8"]["mse"]["test_joint"])


if __name__ == "__main__":
    main()
