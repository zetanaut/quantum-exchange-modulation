#!/usr/bin/env python3
"""Small, deterministic directional and spectator-purity diagnostics.

Run from the repository root with Python 3.12::

    python scripts/physics_application_diagnostics.py --output results/physics_applications

The direction map evaluates an already frozen checkpoint on a new grid; it is
not a training run or a new held-out population study. Matrix references and
selected actual PennyLane exchange circuits are checked independently.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.physics_examples import (ExactExchange, I2, X, Z, direction_observable,
                                       response, spectral_qfi, save_figure as _save_figure)
from qem.pennylane_models import TorchQuantumModel

TOL = 2e-10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(name, values):
    maximum = float(np.max(np.abs(np.asarray(values))))
    if not np.isfinite(maximum) or maximum > TOL:
        raise AssertionError(f"{name}: max error {maximum} exceeds {TOL}")
    return {"name": name, "max_abs_error": maximum, "tolerance": TOL, "passed": True}


def save_figure(figure, output, stem):
    _save_figure(figure, output, stem)
    svg = output / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def directional_map(output):
    data_path = ROOT / "data/benchmark.npz"
    weights_path = ROOT / "results/torch_quantum/weights.npz"
    data = np.load(data_path)
    rho = data["test_joint_rho"][0]
    state_id = int(data["test_joint_state_id"][0])
    family = str(data["test_joint_family"][0])
    theta_grid = np.linspace(0, np.pi, 11)
    phi_grid = np.linspace(-np.pi, np.pi, 21)
    rows, angles, directions, exact, qfi = [], [], [], [], []
    for theta in theta_grid:
        for phi in phi_grid:
            direction = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            p = direction_observable(direction, 2)
            directions.append(direction)
            angles.append((theta, phi))
            exact.append(response(rho, p))
            qfi.append(spectral_qfi(rho, p/2))
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    weights = np.load(weights_path)["joint_exchange__selected"]
    model = TorchQuantumModel("joint_exchange", initial_parameters=weights)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(angles), 16):
            stop = min(start + 16, len(angles))
            batch_rho = torch.tensor(np.repeat(rho[None], stop-start, axis=0), dtype=torch.complex128)
            batch_angles = torch.tensor(np.asarray(angles[start:stop]), dtype=torch.float64)
            predictions.extend(model(batch_rho, batch_angles)["prediction"].detach().numpy().tolist())
    predictions = np.asarray(predictions, dtype=float)
    exact, qfi = np.asarray(exact), np.asarray(qfi)
    for index, ((theta, phi), direction) in enumerate(zip(angles, directions)):
        rows.append({"state_id": state_id, "family": family, "theta_rad": float(theta),
                     "phi_rad": float(phi), "nx": float(direction[0]), "ny": float(direction[1]),
                     "nz": float(direction[2]), "D_exact": float(exact[index]),
                     "D_frozen_prediction": float(predictions[index]),
                     "prediction_minus_exact": float(predictions[index]-exact[index]),
                     "F_Q_P_over_2": float(qfi[index])})
    write_csv(output / "directional_response_map.csv", rows)
    check = [checked("directional_sub_qfi_bound", np.maximum(exact-qfi, 0))]
    circuit = ExactExchange(2)
    circuit_errors = []
    for index in [0, 37, 115, 184, 230]:
        circuit_errors.append(circuit(rho, directions[index])["D"] - exact[index])
    check.append(checked("directional_actual_exchange_qnodes", circuit_errors))
    zz = np.kron(Z, Z); zi = np.kron(Z, I2); iz = np.kron(I2, Z)
    connected_zz = float(np.trace(rho@zz).real - np.trace(rho@zi).real*np.trace(rho@iz).real)
    assert abs(connected_zz) > 0.1, "The selected state must have a visible connected correlation"
    shape = (len(theta_grid), len(phi_grid))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    extent = [-180, 180, 0, 180]
    plots = [(exact.reshape(shape), r"Exact $D_P$", "viridis"),
             (predictions.reshape(shape), "Frozen predictor", "viridis"),
             ((predictions-exact).reshape(shape), "Prediction − exact", "coolwarm")]
    for ax, (values, title, cmap) in zip(axes, plots):
        image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap=cmap)
        ax.set(xlabel=r"Azimuth $\varphi$ (degrees)", title=title)
        fig.colorbar(image, ax=ax, shrink=.82)
    axes[0].set_ylabel(r"Polar angle $\theta$ (degrees)")
    fig.suptitle(f"One fixed correlated two-spin test state (ID {state_id}); directional query")
    save_figure(fig, output, "directional_response_map")
    return {"status": "frozen trained predictor on an exploratory direction grid, plus exact matrix and selected circuit checks",
            "state_split": "test_joint", "state_index": 0, "state_id": state_id, "family": family,
            "connected_Z1Z2": connected_zz, "state_purity": float(np.trace(rho@rho).real),
            "theta_points": len(theta_grid), "phi_points": len(phi_grid),
            "grid_cases": len(rows), "predictor_checkpoint_key": "joint_exchange__selected",
            "checkpoint_sha256": sha256(weights_path), "dataset_sha256": sha256(data_path),
            "mean_squared_prediction_error_on_this_grid": float(np.mean((predictions-exact)**2)),
            "max_abs_prediction_error_on_this_grid": float(np.max(np.abs(predictions-exact))),
            "D_range": [float(exact.min()), float(exact.max())],
            "F_Q_range": [float(qfi.min()), float(qfi.max())], "checks": check}


def spectator_purity(output):
    # Fixed rho_A=(I+0.8 X)/2, P_A=Z; independent spectator sigma_s=(I+s Z)/2.
    # Both states are numerical density matrices, not trained or sampled data.
    r = 0.8
    rho_a = (I2 + r*X)/2
    s_grid = np.linspace(0, 1, 21)
    d_a = response(rho_a, Z)
    qfi_a = spectral_qfi(rho_a, Z/2)
    rows, factor_errors, qfi_errors, analytic_errors = [], [], [], []
    circuit_errors = []
    circuit = ExactExchange(2)
    for index, s in enumerate(s_grid):
        sigma = (I2 + s*Z)/2
        rho = np.kron(rho_a, sigma)
        p = np.kron(Z, I2)
        purity = float(np.trace(sigma@sigma).real)
        d = response(rho, p)
        fq = spectral_qfi(rho, p/2)
        factor_errors.append(d - d_a*purity)
        qfi_errors.append(fq - qfi_a)
        analytic_errors.extend([d_a-r*r, purity-(1+s*s)/2])
        if index in (0, 10, 20):
            circuit_errors.append(circuit(rho, [0, 0, 1])["D"]-d)
        rows.append({"spectator_bloch_z": float(s), "spectator_purity": purity,
                     "D_A": d_a, "D_A_tensor_spectator": d,
                     "D_A_times_spectator_purity": d_a*purity,
                     "F_Q_A_P_over_2": qfi_a, "F_Q_A_tensor_spectator_P_over_2": fq})
    write_csv(output / "spectator_purity.csv", rows)
    checks = [checked("spectator_response_factorization", factor_errors),
              checked("qfi_invariant_under_uncorrelated_spectator", qfi_errors),
              checked("spectator_closed_forms", analytic_errors),
              checked("spectator_actual_exchange_qnodes", circuit_errors)]
    fig, ax = plt.subplots(figsize=(7.2, 4), layout="constrained")
    purity_values = [row["spectator_purity"] for row in rows]
    ax.plot(purity_values, [row["D_A_tensor_spectator"] for row in rows], color="#1768AC",
            label=r"$D_{Z_A\otimes I}(\rho_A\otimes\sigma)$")
    ax.plot(purity_values, [row["F_Q_A_tensor_spectator_P_over_2"] for row in rows],
            color="#29866C", label=r"$F_Q(\rho_A\otimes\sigma,Z_A\otimes I/2)$")
    ax.scatter([purity_values[i] for i in (0, 10, 20)],
               [rows[i]["D_A_tensor_spectator"] for i in (0, 10, 20)],
               facecolors="none", edgecolors="#C96F16", s=70, label="Actual exchange QNode checks")
    ax.set(xlabel=r"Spectator purity $\mathrm{Tr}(\sigma^2)$", ylabel="Dimensionless sensitivity",
           title="An uncorrelated spectator suppresses sub-QFI but leaves QFI unchanged",
           xlim=(.48, 1.02), ylim=(0, .7))
    ax.legend(fontsize=8)
    save_figure(fig, output, "spectator_purity")
    return {"status": "exact matrix and selected prescribed PennyLane circuit diagnostic; no training",
            "rho_A": "(I+0.8 X)/2", "P_A": "Z", "spectator": "(I+s Z)/2",
            "spectator_bloch_z_grid": s_grid.tolist(), "D_A": d_a, "F_Q_A": qfi_a,
            "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/physics_applications")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    direction = directional_map(output)
    spectator = spectator_purity(output)
    manifest = {"command": "python scripts/physics_application_diagnostics.py --output results/physics_applications",
                "purpose": "Explanatory diagnostics; frozen prediction and prescribed exact circuits, not new training",
                "seed": None, "randomness": "none; fixed state index and deterministic grids",
                "shots": None, "units": "angles in radians in CSV; response and QFI dimensionless; spectator Bloch coordinate dimensionless",
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                             "torch": torch.__version__, "pennylane": qml.__version__},
                "directional_map": direction, "spectator_purity": spectator,
                "runtime_seconds": round(time.perf_counter()-start, 3)}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"output": str(output), "directional_grid_mse": direction["mean_squared_prediction_error_on_this_grid"],
                      "all_checks_passed": True, "runtime_seconds": manifest["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
