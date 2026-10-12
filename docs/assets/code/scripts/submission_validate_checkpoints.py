#!/usr/bin/env python3
"""Reload every saved classical confirmation checkpoint and verify validation MSE."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.strong_classical import feature_family, make_model


def main():
    root = Path("results/submission_revision")
    rows = sorted((root / "classical_confirmation").glob("**/*.json"))
    torch.set_num_threads(1)
    cache = {}
    largest = 0.
    for path in rows:
        report = json.loads(path.read_text())
        dataset = report["dataset"]
        if dataset not in cache:
            with np.load(root / "datasets" / f"{dataset}.npz", allow_pickle=False) as archive:
                cache[dataset] = {key: archive[key].copy() for key in archive.files}
        data = cache[dataset]
        checkpoint = torch.load(path.with_suffix(".pt"), weights_only=False, map_location="cpu")
        assert checkpoint["dataset_sha256"] == report["dataset_sha256"]
        config = checkpoint["configuration"]
        family = report["model"].rsplit("__", 1)[0]
        x, c = feature_family(data["validation_rho"], data["validation_context"], family)
        x = (x - checkpoint["x_mean"]) / checkpoint["x_scale"]
        c = (c - checkpoint["c_mean"]) / checkpoint["c_scale"]
        model = make_model(config["kind"], x.shape[1], c.shape[1], config["width"],
                           config["depth"], config["activation"])
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            prediction = model(torch.as_tensor(x, dtype=torch.float64),
                               torch.as_tensor(c, dtype=torch.float64)).numpy()
        mse = float(np.mean((prediction - data["validation_y"]) ** 2))
        difference = abs(mse - report["metrics"]["validation"]["mse"])
        largest = max(largest, difference)
        assert difference < 1e-12
    result = {"status": "PASS", "checkpoints": len(rows),
              "maximum_validation_mse_reload_difference": largest}
    (root / "classical_checkpoint_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
