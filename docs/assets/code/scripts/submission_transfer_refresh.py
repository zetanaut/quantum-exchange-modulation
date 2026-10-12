#!/usr/bin/env python3
"""Refresh the finite-time and direction-map diagnostics from a fresh-split fit.

The checkpoint is selected solely by validation MSE among the three fixed
64-state confirmation_1 starts. Test states and labels play no selection role.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qem.data import fragility, observable
from submission_amortization import selected_model
from train_torch import predict_raw


def main() -> None:
    torch.set_num_threads(1)
    name = "confirmation_1"
    source = ROOT / "results/submission_revision/datasets" / f"{name}.npz"
    fit_root = ROOT / "results/submission_revision/quantum_confirmation"
    output = ROOT / "results/submission_revision/transfer_checks"
    output.mkdir(parents=True, exist_ok=True)
    fit, model = selected_model(fit_root, name, "joint_exchange")
    with np.load(source, allow_pickle=False) as archive:
        ids, indices = np.unique(archive["test_joint_state_id"], return_index=True)
        states = archive["test_joint_rho"][indices].copy()
        contexts = archive["test_joint_context"][indices].copy()
        angles = archive["test_joint_angles"][indices].copy()
        labels = archive["test_joint_y"][indices].copy()
        family = str(archive["test_joint_family"][indices[0]])
    assert len(ids) == 128 and np.all(np.abs(contexts[:, 0]) > .55)
    batch = {"rho": torch.as_tensor(states, dtype=torch.complex128),
             "angles": torch.as_tensor(angles, dtype=torch.float64),
             "y": torch.as_tensor(labels, dtype=torch.float64)}
    prediction = predict_raw(model, batch, 128)["prediction"]
    rates = np.array([.2, .7, 1.3])
    times = np.linspace(0, 3, 21)
    factor = .5 * (1 - np.exp(-4 * rates[:, None] * times[None, :]))
    error = (prediction - labels)[:, None, None] * factor[None, :, :]
    finite_time = {"dataset": name, "checkpoint_seed": fit["seed"],
                   "checkpoint_selection": "minimum validation MSE among seeds 17, 31, 73",
                   "checkpoint_validation_mse": fit["fit"]["best_validation_mse"],
                   "state_count": len(ids), "pairs": len(ids), "rates": rates.tolist(),
                   "times": times.tolist(), "queries": error.size,
                   "heldout_axis_min_abs_nx": float(np.min(np.abs(contexts[:, 0]))),
                   "response_mse": float(np.mean((prediction - labels) ** 2)),
                   "purity_loss_mse": float(np.mean(error ** 2)),
                   "purity_loss_max_abs_error": float(np.max(np.abs(error))),
                   "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (output / "finite_time.json").write_text(json.dumps(finite_time, indent=2) + "\n")

    # The first new test state is chosen by state ID, before inspecting errors.
    theta = np.linspace(0, np.pi, 11)
    phi = np.linspace(-np.pi, np.pi, 21)
    angle_rows = np.array([(t, p) for t in theta for p in phi])
    directions = np.column_stack((np.sin(angle_rows[:, 0]) * np.cos(angle_rows[:, 1]),
                                  np.sin(angle_rows[:, 0]) * np.sin(angle_rows[:, 1]),
                                  np.cos(angle_rows[:, 0])))
    rho = np.repeat(states[:1], len(angle_rows), axis=0)
    exact = fragility(rho, observable(directions))
    grid_batch = {"rho": torch.as_tensor(rho, dtype=torch.complex128),
                  "angles": torch.as_tensor(angle_rows, dtype=torch.float64),
                  "y": torch.as_tensor(exact, dtype=torch.float64)}
    estimated = predict_raw(model, grid_batch, 231)["prediction"]
    with (output / "directional_response_map.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state_id", "family", "theta_rad", "phi_rad", "nx", "ny", "nz",
                         "D_exact", "D_frozen_prediction", "prediction_minus_exact"])
        for a, n, y, yhat in zip(angle_rows, directions, exact, estimated):
            writer.writerow([int(ids[0]), family, *a, *n, y, yhat, yhat - y])
    grid = {"dataset": name, "checkpoint_seed": fit["seed"],
            "state_id": int(ids[0]), "family": family, "grid_rows": len(angle_rows),
            "grid_mse": float(np.mean((estimated - exact) ** 2)),
            "grid_max_abs_error": float(np.max(np.abs(estimated - exact))),
            "heldout_axis_grid_rows": int(np.count_nonzero(np.abs(directions[:, 0]) > .55)),
            "selection": "smallest new test state ID; full-sphere exploratory grid; no model selection"}
    (output / "directional_response_map.json").write_text(json.dumps(grid, indent=2) + "\n")
    print(json.dumps({"finite_time": finite_time, "grid": grid}, indent=2))


if __name__ == "__main__":
    main()
