#!/usr/bin/env python3
"""Execute four physical examples using real PennyLane circuits.

Run from the repository root::

    python examples/physics_examples.py

The frozen two-qubit model is used only for analytic transfer to finite-time
dephasing.  The other examples use the prescribed exact exchange geometry;
none is a new optimization run, hardware experiment, or advantage claim.
All output is confined to results/physics_examples/ (or --output).
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, solve_sylvester
import scipy
import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qem.pennylane_models import TorchQuantumModel
from qem.quantum_models import QuantumBatch, make_model as make_numpy_model

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
PAULIS = (X, Y, Z)
BLUE, ORANGE, GREEN, GRAY = "#1768AC", "#C96F16", "#29866C", "#67727E"
EXACT_TOL = 2e-10


def tensor(*operators):
    result = np.ones((1, 1), dtype=np.complex128)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def spin(operator, site, nqubits):
    return tensor(*(operator if k == site else I2 for k in range(nqubits)))


def projector(state):
    return np.outer(state, state.conj())


def purity(rho):
    return float(np.trace(rho @ rho).real)


def response(rho, observable):
    return float((np.trace(rho @ rho) - np.trace(rho @ observable @ rho @ observable)).real)


def direction_observable(direction, nqubits):
    local = sum(float(c) * p for c, p in zip(direction, PAULIS))
    return tensor(local, np.eye(2 ** (nqubits - 1)))


def local_marginal(rho, nqubits, site):
    state = rho.reshape((2,) * (2 * nqubits))
    remaining = nqubits
    for other in reversed(range(nqubits)):
        if other != site:
            state = np.trace(state, axis1=other, axis2=other + remaining)
            remaining -= 1
    return state.reshape(2, 2)


@lru_cache(maxsize=8)
def swap_matrix(dimension):
    result = np.zeros((dimension**2, dimension**2), dtype=np.complex128)
    for first in range(dimension):
        for second in range(dimension):
            result[second * dimension + first, first * dimension + second] = 1
    return result


@lru_cache(maxsize=8)
def direct_exchange_unitary(dimension):
    """Cache a context-independent matrix exponential for the reference path."""
    return expm(-1j * np.pi * swap_matrix(dimension) / 4)


def angles_from_direction(direction):
    direction = np.asarray(direction, dtype=float)
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-12):
        raise ValueError("Context direction must have unit norm")
    return np.arccos(np.clip(direction[2], -1, 1)), np.arctan2(direction[1], direction[0])


def queue_source_rotation(theta, phi, first_wire):
    # A=RZ(phi) RY(theta); V=A RZ(pi/2) A^dagger=exp(-i*pi*P/4).
    qml.RZ(-phi, wires=first_wire)
    qml.RY(-theta, wires=first_wire)
    qml.RZ(np.pi / 2, wires=first_wire)
    qml.RY(theta, wires=first_wire)
    qml.RZ(phi, wires=first_wire)


class ExactExchange:
    """Actual mixed-state QNode with two physical input registers.

    The default queues one full-register unitary. The controlled alternative
    queues two controlled *global* swaps (2m Fredkin gates) and phase kickback.
    The local-pairs alternative is intentionally a different architecture.
    """

    def __init__(self, nqubits, implementation="global_matrix"):
        self.nqubits = nqubits
        self.implementation = implementation
        extra = int(implementation == "controlled_global")
        device = qml.device("default.mixed", wires=2 * nqubits + extra, shots=None)
        d = 2**nqubits
        # Explicit exponential, independently checked against direct matrices.
        unitary = expm(-1j * np.pi / 4 * swap_matrix(d))
        local_unitary = expm(-1j * np.pi / 4 * swap_matrix(2))

        @qml.qnode(device, interface=None, diff_method=None)
        def circuit(rho, theta, phi):
            qml.QubitDensityMatrix(np.kron(rho, rho), wires=range(2 * nqubits))
            queue_source_rotation(theta, phi, nqubits)
            if implementation == "global_matrix":
                qml.QubitUnitary(unitary, wires=range(2 * nqubits))
            elif implementation == "controlled_global":
                ancilla = 2 * nqubits
                qml.Hadamard(wires=ancilla)
                for j in range(nqubits):
                    qml.CSWAP(wires=[ancilla, j, nqubits + j])
                qml.Hadamard(wires=ancilla)
                qml.RZ(np.pi / 2, wires=ancilla)
                qml.Hadamard(wires=ancilla)
                for j in range(nqubits):
                    qml.CSWAP(wires=[ancilla, j, nqubits + j])
                qml.Hadamard(wires=ancilla)
            elif implementation == "local_pairs":
                for j in range(nqubits):
                    qml.QubitUnitary(local_unitary, wires=[j, nqubits + j])
            else:
                raise ValueError(implementation)
            for wire in (0, nqubits):
                qml.RZ(-phi, wires=wire)
                qml.RY(-theta, wires=wire)
            return qml.probs(wires=[0, nqubits])

        self.circuit = circuit

    def __call__(self, rho, direction):
        theta, phi = angles_from_direction(direction)
        probabilities = np.asarray(self.circuit(rho, theta, phi), dtype=float)
        first_sign = np.array([1, 1, -1, -1])
        second_sign = np.array([1, -1, 1, -1])
        return {
            "probabilities": probabilities,
            "first": float(probabilities @ first_sign),
            "second": float(probabilities @ second_sign),
            "correlation": float(probabilities @ (first_sign * second_sign)),
            "D": float(probabilities @ (second_sign - first_sign)),
        }


def direct_exchange(rho, observable):
    """Independent full-matrix propagation, without the reduced-state formula."""
    d = len(rho)
    rotation = expm(-1j * np.pi * observable / 4)
    exchange = direct_exchange_unitary(d)
    preparation = np.kron(np.eye(d), rotation)
    joint = preparation @ np.kron(rho, rho) @ preparation.conj().T
    joint = exchange @ joint @ exchange.conj().T
    first = float(np.trace(joint @ np.kron(observable, np.eye(d))).real)
    second = float(np.trace(joint @ np.kron(np.eye(d), observable)).real)
    correlation = float(np.trace(joint @ np.kron(observable, observable)).real)
    signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    probabilities = np.array([(1 + s * first + t * second + s * t * correlation) / 4
                              for s, t in signs])
    return second - first, probabilities


class Checks:
    def __init__(self):
        self.rows = []

    def error(self, name, values, tolerance=EXACT_TOL, **details):
        values = np.asarray(values, dtype=float)
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        self.rows.append({"name": name, "passed": bool(np.isfinite(maximum) and maximum <= tolerance),
                          "max_error": maximum, "tolerance": tolerance,
                          "comparisons": int(values.size), **details})

    @property
    def passed(self):
        return all(row["passed"] for row in self.rows)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, output, stem):
    figure.savefig(output / f"{stem}.png", dpi=190, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def configure_plotting():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.2, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})


def correlation_example(output, checks, rng):
    tgrid = np.linspace(0, 1, 41)
    rows, arrays, errors, marginal_errors, pauli_errors = [], {}, [], [], []
    for nqubits in (2, 3):
        d = 2**nqubits
        positive = np.zeros(d, dtype=complex)
        positive[[0, -1]] = 1 / np.sqrt(2)
        negative = positive.copy()
        negative[-1] *= -1
        plus, minus = projector(positive), projector(negative)
        observable = spin(Z, 0, nqubits)
        circuit = ExactExchange(nqubits)
        device = qml.device("default.mixed", wires=nqubits, shots=None)

        @qml.qnode(device, interface=None, diff_method=None)
        def global_pauli(rho):
            qml.QubitDensityMatrix(rho, wires=range(nqubits))
            return qml.expval(qml.prod(*(qml.X(j) for j in range(nqubits))))

        states, estimates, pauli_means, probabilities = [], [], [], []
        for t in tgrid:
            rho = t * plus + (1 - t) * minus
            theoretical = (2 * t - 1)**2
            exact = response(rho, observable)
            readout = circuit(rho, [0, 0, 1])
            direct, direct_prob = direct_exchange(rho, observable)
            pauli_mean = float(global_pauli(rho))
            marginal_error = max(np.max(np.abs(local_marginal(rho, nqubits, j) - I2 / 2))
                                 for j in range(nqubits))
            errors.extend([exact - theoretical, readout["D"] - exact, direct - exact,
                           *list(readout["probabilities"] - direct_prob)])
            marginal_errors.append(marginal_error)
            pauli_errors.extend([pauli_mean - (2 * t - 1), pauli_mean**2 - theoretical])
            rows.append({"nqubits": nqubits, "mixture_t": float(t), "D_exact": exact,
                         "D_exchange_qnode": readout["D"], "global_X_expectation": pauli_mean,
                         "D_global_X_squared": pauli_mean**2,
                         "max_local_marginal_error_from_I_over_2": float(marginal_error)})
            states.append(rho)
            estimates.append(readout["D"])
            pauli_means.append(pauli_mean)
            probabilities.append(readout["probabilities"])
        arrays[f"rho_{nqubits}q"] = np.asarray(states)
        arrays[f"exchange_{nqubits}q"] = np.asarray(estimates)
        arrays[f"global_X_{nqubits}q"] = np.asarray(pauli_means)
        arrays[f"joint_probabilities_{nqubits}q"] = np.asarray(probabilities)
    checks.error("bell_ghz_exchange_vs_matrices_and_exact_polynomial", errors)
    checks.error("bell_ghz_constant_local_marginals", marginal_errors)
    checks.error("bell_ghz_known_global_pauli_comparator", pauli_errors)
    arrays["mixture_t"] = tgrid
    arrays["D_exact"] = (2 * tgrid - 1)**2
    np.savez_compressed(output / "correlation.npz", **arrays)
    write_csv(output / "correlation.csv", rows)

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained")
    axes[0].plot(tgrid, arrays["D_exact"], color=GRAY, linewidth=2.5, label=r"Exact $(2t-1)^2$")
    axes[0].plot(tgrid, arrays["global_X_3q"]**2, color=ORANGE, linestyle="--", label=r"Known $\langle X^{\otimes m}\rangle^2$")
    axes[0].plot(tgrid[::4], arrays["exchange_2q"][::4], "o", fillstyle="none", color=BLUE, label="Bell: 2-qubit exchange")
    axes[0].plot(tgrid[2::4], arrays["exchange_3q"][2::4], "^", fillstyle="none", color=GREEN, label="GHZ: 3-qubit exchange")
    axes[0].set(xlabel=r"Mixture weight $t$", ylabel=r"Dephasing response $D_{Z_1}$", title="Correlations determine the response", ylim=(-0.04, 1.08))
    axes[0].legend(fontsize=8, loc="upper center")
    axes[1].plot(tgrid, arrays["global_X_3q"], color=ORANGE, linewidth=2, label=r"Global parity $\langle X^{\otimes m}\rangle$")
    axes[1].plot(tgrid, np.zeros_like(tgrid), color=BLUE, linewidth=2, label=r"All local $\langle X_j\rangle,\langle Y_j\rangle,\langle Z_j\rangle$")
    axes[1].set(xlabel=r"Mixture weight $t$", ylabel="Expectation", title="Every one-qubit marginal stays maximally mixed", ylim=(-1.12, 1.12))
    axes[1].legend(fontsize=8, loc="upper left")
    save_figure(figure, output, "correlation_sensitivity")

    # Enumerate both estimators before any Monte Carlo: a two-copy exchange
    # pair yields t-s; two independent global-X parities yield their product.
    values = np.array([0, -2, 2, 0])
    distribution_errors = []
    for k, dvalue in enumerate(arrays["D_exact"]):
        prob = arrays["joint_probabilities_2q"][k]
        distribution_errors.append(prob @ values**2 - (prob @ values)**2 - (2 - dvalue**2))
        mean = arrays["global_X_2q"][k]
        single = np.array([(1 + mean) / 2, (1 - mean) / 2])
        pair_prob = np.kron(single, single)
        products = np.array([1, -1, -1, 1])
        distribution_errors.append(pair_prob @ products**2 - (pair_prob @ products)**2 - (1 - dvalue**2))
    checks.error("correlation_exact_estimator_variances", distribution_errors)
    selected_t, repetitions = 0.85, 4096
    dvalue = (2 * selected_t - 1)**2
    copies = np.array([128, 512, 2048, 8192])
    joint_prob = np.array([0.25, (1 - dvalue) / 4, (1 + dvalue) / 4, 0.25])
    mc_rows, mc_mean_z, mc_variance_z = [], [], []
    mc_arrays = {"raw_copies": copies, "mixture_t": np.asarray(selected_t), "D_exact": np.asarray(dvalue)}
    for estimator in ("exchange", "global_pauli_product"):
        for cost in copies:
            pairs = int(cost // 2)
            expected_variance = (2 - dvalue**2 if estimator == "exchange" else 1 - dvalue**2) / pairs
            if estimator == "exchange":
                draws = rng.multinomial(pairs, joint_prob, size=repetitions) @ values / pairs
            else:
                draws = 2 * rng.binomial(pairs, (1 + dvalue) / 2, size=repetitions) / pairs - 1
            observed_mean, observed_variance = float(draws.mean()), float(draws.var(ddof=1))
            mean_z = abs(observed_mean - dvalue) / np.sqrt(expected_variance / repetitions)
            variance_z = abs(observed_variance - expected_variance) / (expected_variance * np.sqrt(2 / (repetitions - 1)))
            mc_mean_z.append(mean_z)
            mc_variance_z.append(variance_z)
            mc_arrays[f"{estimator}_copies_{cost}"] = draws
            mc_rows.append({"estimator": estimator, "mixture_t": selected_t, "D_exact": dvalue,
                            "raw_copies": int(cost), "independent_pairs": pairs,
                            "repetitions": repetitions, "expected_variance": float(expected_variance),
                            "monte_carlo_mean": observed_mean, "monte_carlo_variance": observed_variance,
                            "mean_standard_errors": float(mean_z),
                            "variance_normal_approx_standard_errors": float(variance_z)})
    checks.error("correlation_monte_carlo_mean_standard_errors", mc_mean_z, tolerance=6.0)
    checks.error("correlation_monte_carlo_variance_standard_errors", mc_variance_z, tolerance=6.0,
                 note="Normal approximation for sample-variance standard errors; deterministic diagnostic, not a hardware experiment")
    write_csv(output / "correlation_copy_cost.csv", mc_rows)
    np.savez_compressed(output / "correlation_copy_cost.npz", **mc_arrays)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    axes[0].plot(tgrid, 4 - 2 * arrays["D_exact"]**2, color=BLUE, label="Terminal exchange")
    axes[0].plot(tgrid, 2 - 2 * arrays["D_exact"]**2, color=ORANGE, label="Known global-Pauli product")
    axes[0].set(xlabel=r"Mixture weight $t$", ylabel=r"$C\,\mathrm{Var}(\widehat D)$", title="Known Pauli estimator uses fewer copies", ylim=(-0.1, 4.3))
    axes[0].legend(fontsize=8)
    for name, color, label in [("exchange", BLUE, "Exchange"), ("global_pauli_product", ORANGE, "Global-Pauli product")]:
        selected = [r for r in mc_rows if r["estimator"] == name]
        axes[1].loglog(copies, np.sqrt([r["expected_variance"] for r in selected]), color=color, label=label + " (exact)")
        axes[1].loglog(copies, np.sqrt([r["monte_carlo_variance"] for r in selected]), "o", color=color, fillstyle="none")
    axes[1].set(xlabel=r"Total raw-state preparations $C$", ylabel="Standard deviation", title=r"Monte Carlo check at $t=0.85$")
    axes[1].text(0.02, 0.03, "Markers: 4,096 simulated repetitions", transform=axes[1].transAxes, fontsize=8)
    axes[1].legend(fontsize=8)
    save_figure(figure, output, "correlation_copy_cost")
    return {"status": "exact prescribed circuit; no training", "nqubits": [2, 3],
            "mixture_grid": tgrid.tolist(), "state_context_cases": 82,
            "copies_per_joint_shot": 2, "global_pauli_readout": "Measure every qubit in X and multiply signs; no entangling readout required",
            "interpretation": "Constant local marginals hide the response. A known global-Pauli estimator already solves this family and has strictly smaller variance than exchange.",
            "monte_carlo_repetitions": repetitions}


def dephasing_example(output, checks):
    source = np.load(ROOT / "data/benchmark.npz")
    rho, directions = source["test_joint_rho"], source["test_joint_context"]
    angles, observables = source["test_joint_angles"], source["test_joint_P"]
    exact_d = source["test_joint_y"]
    checkpoint = ROOT / "results/torch_quantum/weights.npz"
    weights = np.load(checkpoint)["joint_exchange__selected"]
    model = TorchQuantumModel("joint_exchange", initial_parameters=weights)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(rho), torch.as_tensor(angles))["prediction"].numpy()
    batch = QuantumBatch.from_npz(source, "test_joint")
    reference_prediction = make_numpy_model("joint_exchange").predict(weights, batch)
    checks.error("frozen_predictor_pennylane_vs_independent_numpy", prediction - reference_prediction)
    rates, time_grid = np.array([0.15, 0.5, 1.25]), np.linspace(0, 3, 21)
    factor = 0.5 * (1 - np.exp(-4 * rates[:, None] * time_grid[None, :]))
    exact_loss = np.zeros((len(rho), len(rates), len(time_grid)))
    transferred = prediction[:, None, None] * factor[None, :, :]
    direct_errors, qnode_errors, physical_errors, identity_errors = [], [], [], []
    # One representative query for every second held-out state: fixed indices,
    # chosen before examining labels or predictions.
    explicit_indices = np.arange(0, len(rho), 8)
    device = qml.device("default.mixed", wires=2, shots=None)

    @qml.qnode(device, interface=None, diff_method=None)
    def dephasing_channel(state, observable, damping):
        qml.QubitDensityMatrix(state, wires=[0, 1])
        alpha = (1 + np.exp(-2 * damping)) / 2
        qml.QubitChannel([np.sqrt(alpha) * np.eye(4), np.sqrt(1 - alpha) * observable], wires=[0, 1])
        return qml.density_matrix(wires=[0, 1])

    rows = []
    for case, (state, p) in enumerate(zip(rho, observables)):
        for irate, rate in enumerate(rates):
            generator = rate * (np.kron(p.T, p) - np.eye(16))
            for itime, duration in enumerate(time_grid):
                alpha = (1 + np.exp(-2 * rate * duration)) / 2
                evolved = alpha * state + (1 - alpha) * (p @ state @ p)
                lost = purity(state) - purity(evolved)
                exact_loss[case, irate, itime] = lost
                identity_errors.append(lost - factor[irate, itime] * exact_d[case])
                if case in explicit_indices:
                    vector = expm(duration * generator) @ state.reshape(-1, order="F")
                    direct = vector.reshape(4, 4, order="F")
                    direct_errors.append(np.max(np.abs(direct - evolved)))
                    if itime in (0, 5, 10, 20):
                        physical = np.asarray(dephasing_channel(state, p, rate * duration))
                        qnode_errors.append(np.max(np.abs(physical - direct)))
                        physical_errors.extend([abs(np.trace(physical) - 1),
                                                max(0, -np.linalg.eigvalsh(physical).min())])
                rows.append({"case": case, "state_id": int(source["test_joint_state_id"][case]),
                             "context_id": int(source["test_joint_context_id"][case]),
                             "gamma": float(rate), "time": float(duration),
                             "D_exact": float(exact_d[case]), "D_frozen_prediction": float(prediction[case]),
                             "purity_loss_exact_channel": lost,
                             "purity_loss_transferred_prediction": float(transferred[case, irate, itime])})
    checks.error("finite_dephasing_purity_loss_identity", identity_errors)
    checks.error("finite_dephasing_kraus_vs_liouvillian_expm", direct_errors)
    checks.error("finite_dephasing_pennylane_vs_liouvillian_expm", qnode_errors)
    checks.error("finite_dephasing_output_physicality", physical_errors)
    write_csv(output / "dephasing_transfer.csv", rows)
    np.savez_compressed(output / "dephasing_transfer.npz", rho=rho, directions=directions,
                        angles=angles, observables=observables, gamma=rates, time=time_grid,
                        D_exact=exact_d, D_frozen_prediction=prediction,
                        purity_loss_exact=exact_loss, purity_loss_prediction=transferred,
                        explicit_expm_case_indices=explicit_indices,
                        state_id=source["test_joint_state_id"], context_id=source["test_joint_context_id"])
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    colors = [BLUE, ORANGE, GREEN]
    for i, (rate, color) in enumerate(zip(rates, colors)):
        axes[0].plot(time_grid, exact_loss[:, i].mean(axis=0), color=color, label=fr"$\gamma={rate}$: exact channel")
        axes[0].plot(time_grid[::3], transferred[:, i, ::3].mean(axis=0), "o", fillstyle="none", color=color)
    axes[0].plot([], [], "o", fillstyle="none", color=GRAY, label="Frozen D predictor + known decay law")
    axes[0].set(xlabel=r"Evolution time $T$", ylabel="Mean purity loss", title="Frozen D predicts finite-time purity loss")
    axes[0].legend(fontsize=8, loc="lower right")
    for i, (rate, color) in enumerate(zip(rates, colors)):
        rmse = np.sqrt(np.mean((transferred[:, i] - exact_loss[:, i])**2, axis=0))
        axes[1].plot(time_grid, rmse, color=color, label=fr"$\gamma={rate}$")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set(xlabel=r"Evolution time $T$", ylabel="RMSE of transferred purity loss", title="Prediction error after analytic transfer")
    axes[1].legend(fontsize=8)
    save_figure(figure, output, "dephasing_transfer")
    return {"status": "analytic transfer of a frozen trained D predictor; no retraining",
            "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_key": "joint_exchange__selected",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "selected_seed": 31, "original_split": "test_joint", "state_context_cases": len(rho),
            "unique_states": int(len(np.unique(source["test_joint_state_id"]))),
            "rates": rates.tolist(), "time_grid": time_grid.tolist(),
            "finite_time_cases": len(rows), "explicit_expm_cases": len(direct_errors),
            "pennylane_channel_cases": len(qnode_errors),
            "D_prediction_mse": float(np.mean((prediction - exact_d)**2)),
            "transferred_purity_loss_mse": float(np.mean((transferred - exact_loss)**2)),
            "max_transferred_absolute_error": float(np.max(np.abs(transferred - exact_loss))),
            "interpretation": "The known closed-form decay law multiplies the previously learned D. This checks a useful application, not learning an unseen dynamical law."}


def spectral_qfi(rho, generator):
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)
    elements = eigenvectors.conj().T @ generator @ eigenvectors
    difference = eigenvalues[:, None] - eigenvalues[None, :]
    total = eigenvalues[:, None] + eigenvalues[None, :]
    weight = np.divide(difference**2, total, out=np.zeros_like(total), where=total > 1e-13)
    return float(2 * np.sum(weight * np.abs(elements)**2))


def random_density(rng, d, pure=False):
    amplitude = rng.normal(size=(d, 1 if pure else d)) + 1j * rng.normal(size=(d, 1 if pure else d))
    rho = amplitude @ amplitude.conj().T
    return rho / np.trace(rho).real


def qfi_example(output, checks, rng):
    rows, arrays, circuit_errors, pure_errors, qubit_errors, violations, sld_errors = [], {}, [], [], [], [], []
    for category, nqubits, count, pure in [("pure_two_qubit", 2, 60, True),
                                          ("mixed_qubit", 1, 60, False),
                                          ("mixed_two_qubit", 2, 100, False)]:
        states, contexts, dvalues, fvalues = [], [], [], []
        circuit = ExactExchange(nqubits)
        for case in range(count):
            rho = random_density(rng, 2**nqubits, pure)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            p = direction_observable(direction, nqubits)
            dvalue, fvalue = response(rho, p), spectral_qfi(rho, p / 2)
            readout = circuit(rho, direction)
            circuit_errors.append(readout["D"] - dvalue)
            violations.append(max(0, dvalue - fvalue))
            if pure:
                pure_errors.extend([dvalue - fvalue, fvalue - (1 - np.trace(rho @ p).real**2)])
            elif nqubits == 1:
                qubit_errors.append(dvalue - fvalue)
            if not pure:
                derivative = -1j * ((p / 2) @ rho - rho @ (p / 2))
                sld = solve_sylvester(rho, rho, 2 * derivative)
                sld_errors.extend([np.trace(rho @ sld @ sld).real - fvalue,
                                   np.max(np.abs((rho @ sld + sld @ rho) / 2 - derivative))])
            rows.append({"family": category, "case": case, "nqubits": nqubits,
                         "D_exact": dvalue, "D_exchange_qnode": readout["D"],
                         "F_Q_generator_P_over_2": fvalue, "F_Q_minus_D": fvalue - dvalue})
            states.append(rho)
            contexts.append(direction)
            dvalues.append(dvalue)
            fvalues.append(fvalue)
        arrays[f"{category}_rho"] = np.asarray(states)
        arrays[f"{category}_directions"] = np.asarray(contexts)
        arrays[f"{category}_D"] = np.asarray(dvalues)
        arrays[f"{category}_FQ"] = np.asarray(fvalues)
    checks.error("qfi_exchange_qnodes_vs_D", circuit_errors)
    checks.error("qfi_lower_bound_violations", violations)
    checks.error("qfi_pure_state_equality_and_generator_normalization", pure_errors)
    checks.error("qfi_arbitrary_single_qubit_equality", qubit_errors)
    checks.error("qfi_spectral_vs_independent_sld_sylvester", sld_errors)
    pgrid = np.linspace(0, 1, 41)
    phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    p = np.kron(Z, I2)
    circuit = ExactExchange(2)
    depolarized_d, depolarized_f, depolarized_circuit = [], [], []
    family_errors = []
    for fraction in pgrid:
        rho = fraction * projector(phi) + (1 - fraction) * np.eye(4) / 4
        dvalue, fvalue = response(rho, p), spectral_qfi(rho, p / 2)
        readout = circuit(rho, [0, 0, 1])
        family_errors.extend([dvalue - fraction**2, fvalue - 2 * fraction**2 / (1 + fraction), readout["D"] - dvalue])
        depolarized_d.append(dvalue)
        depolarized_f.append(fvalue)
        depolarized_circuit.append(readout["D"])
        rows.append({"family": "depolarized_bell", "case": float(fraction), "nqubits": 2,
                     "D_exact": dvalue, "D_exchange_qnode": readout["D"],
                     "F_Q_generator_P_over_2": fvalue, "F_Q_minus_D": fvalue - dvalue})
    checks.error("qfi_depolarized_bell_closed_forms", family_errors)
    arrays.update(depolarized_bell_fraction=pgrid, depolarized_bell_D=np.asarray(depolarized_d),
                  depolarized_bell_FQ=np.asarray(depolarized_f), depolarized_bell_exchange=np.asarray(depolarized_circuit))
    np.savez_compressed(output / "qfi.npz", **arrays)
    write_csv(output / "qfi.csv", rows)
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    axes[0].plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1)
    for category, color, marker, label in [("mixed_two_qubit", BLUE, "o", "Mixed 2-qubit states"),
                                          ("pure_two_qubit", GREEN, "+", "Pure 2-qubit states"),
                                          ("mixed_qubit", ORANGE, "x", "Arbitrary mixed qubits")]:
        axes[0].scatter(arrays[f"{category}_D"], arrays[f"{category}_FQ"], s=20, alpha=0.72, color=color, marker=marker, label=label)
    axes[0].set(xlabel=r"Exchange response $D_P$", ylabel=r"Quantum Fisher information $F_Q(\rho,P/2)$", title=r"A lower bound, with exact equality cases", xlim=(-0.025, 1.03), ylim=(-0.025, 1.03))
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].plot(pgrid, depolarized_f, color=GREEN, label=r"$F_Q=2p^2/(1+p)$")
    axes[1].plot(pgrid, depolarized_d, color=BLUE, label=r"$D_P=p^2$")
    axes[1].plot(pgrid[::4], np.asarray(depolarized_circuit)[::4], "o", color=BLUE, fillstyle="none", label="Actual exchange QNode")
    axes[1].set(xlabel=r"Bell fraction $p$ in $p|\Phi^+\rangle\langle\Phi^+|+(1-p)I/4$", ylabel="Sensitivity", title="Mixed-state sensitivity generally exceeds D")
    axes[1].legend(fontsize=8)
    save_figure(figure, output, "qfi_sensitivity")
    return {"status": "exact prescribed exchange circuit and independent spectral/SLD analysis; no training",
            "state_context_cases": len(rows), "pure_two_qubit_cases": 60,
            "mixed_qubit_cases": 60, "mixed_two_qubit_cases": 100, "depolarized_bell_cases": 41,
            "generator": "G=P/2 in rho_theta=exp(-i theta G) rho exp(+i theta G)",
            "normalization_source": "https://arxiv.org/abs/0804.2981",
            "journal_doi": "10.1142/S0219749909004839",
            "sub_qfi_source": "https://arxiv.org/abs/2101.10144",
            "sub_qfi_identity": "Cerezo et al. Theorem 1: sub-QFI=4[Tr(rho^2 G^2)-Tr(rho G rho G)]=D_P for G=P/2 and P^2=I",
            "proof": "D=(1/2)sum_ij (lambda_i-lambda_j)^2 |P_ij|^2; FQ inserts 1/(lambda_i+lambda_j). Every off-diagonal lambda_i+lambda_j<=1. Pure-state active pairs, and every qubit off-diagonal pair, have sum one.",
            "interpretation": "D is the established sub-QFI in this normalization, not a new sensitivity bound. It is not the QFI for general mixed multiqubit states. Measuring D does not assert an optimal phase-estimation protocol."}


def gibbs_state(hamiltonian, beta):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    weights = np.exp(-beta * (eigenvalues - eigenvalues.min()))
    return (eigenvectors * (weights / weights.sum())) @ eigenvectors.conj().T


def spin_example(output, checks):
    nqubits, interaction, staggered = 3, 1.0, 0.13
    betas, fields = np.array([0, 0.25, 0.5, 1, 2, 4.0]), np.linspace(0, 1.8, 19)
    sz, sx = [spin(Z, j, 3) for j in range(3)], [spin(X, j, 3) for j in range(3)]
    circuit, controlled, pairwise = ExactExchange(3), ExactExchange(3, "controlled_global"), ExactExchange(3, "local_pairs")
    responses = np.zeros((len(betas), len(fields), 2))
    correlations = np.zeros((len(betas), len(fields)))
    states = np.zeros((len(betas), len(fields), 8, 8), dtype=complex)
    local_values = np.zeros(len(fields))
    rows, exact_errors, fredkin_errors, local_errors, physical_errors = [], [], [], [], []
    for ibeta, beta in enumerate(betas):
        for ifield, field in enumerate(fields):
            hamiltonian = -interaction * (sz[0] @ sz[1] + sz[1] @ sz[2]) - field * sum(sx) - staggered * (sz[0] - sz[2])
            rho = gibbs_state(hamiltonian, beta)
            states[ibeta, ifield] = rho
            czz = float(np.trace(rho @ sz[0] @ sz[1]).real - np.trace(rho @ sz[0]).real * np.trace(rho @ sz[1]).real)
            correlations[ibeta, ifield] = czz
            physical_errors.extend([abs(np.trace(rho) - 1), np.max(np.abs(rho-rho.conj().T)), max(0, -np.linalg.eigvalsh(rho).min())])
            for axis, direction, p in [(0, [0, 0, 1], sz[0]), (1, [1, 0, 0], sx[0])]:
                exact = response(rho, p)
                readout = circuit(rho, direction)
                direct, probability = direct_exchange(rho, p)
                exact_errors.extend([readout["D"] - exact, direct - exact,
                                     np.max(np.abs(readout["probabilities"] - probability))])
                responses[ibeta, ifield, axis] = readout["D"]
                rows.append({"nqubits": 3, "J": interaction, "staggered_z_field": staggered,
                             "inverse_temperature_beta": float(beta), "transverse_field_h": float(field),
                             "query_axis": "Z_1" if axis == 0 else "X_1", "D_exact": exact,
                             "D_global_exchange_qnode": readout["D"], "connected_Z1Z2": czz})
                if axis == 0 and ibeta == 4 and ifield in (0, 8, 18):
                    fredkin = controlled(rho, direction)
                    fredkin_errors.extend(fredkin["probabilities"] - readout["probabilities"])
            if ibeta == 4:
                local_readout = pairwise(rho, [0, 0, 1])["D"]
                marginal = local_marginal(rho, 3, 0)
                local_values[ifield] = local_readout
                local_errors.append(local_readout - response(marginal, Z))
    checks.error("three_spin_global_exchange_vs_independent_matrices", exact_errors)
    checks.error("three_spin_six_fredkin_phase_kickback_vs_full_global_gate", fredkin_errors)
    checks.error("three_spin_pairwise_exchange_matches_local_marginal_only", local_errors)
    checks.error("three_spin_gibbs_states_physicality", physical_errors)
    np.savez_compressed(output / "spin_gibbs.npz", inverse_temperature=betas, transverse_field=fields,
                        rho=states, D_global_exchange=responses, connected_Z1Z2=correlations,
                        pairwise_exchange_beta2=local_values)
    write_csv(output / "spin_gibbs.csv", rows)
    write_csv(output / "spin_gibbs_local_comparison.csv", [
        {"inverse_temperature_beta": 2.0, "transverse_field_h": float(h),
         "D_global_Z1": float(responses[4, k, 0]), "D_global_X1": float(responses[4, k, 1]),
         "D_local_pair_exchange_Z1": float(local_values[k]),
         "connected_Z1Z2": float(correlations[4, k])}
        for k, h in enumerate(fields)])
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    mesh = axes[0].pcolormesh(fields, betas, responses[:, :, 0], shading="nearest", cmap="viridis", vmin=0)
    axes[0].grid(False)
    axes[0].set(xlabel=r"Transverse field $h/J$", ylabel=r"Inverse temperature $\beta J$", title=r"Three-spin Gibbs response $D_{Z_1}$", xlim=(0, 1.8), ylim=(0, 4))
    figure.colorbar(mesh, ax=axes[0], label=r"$D_{Z_1}$")
    axes[1].plot(fields, responses[4, :, 0], color=BLUE, label=r"Global exchange: $D_{Z_1}$")
    axes[1].plot(fields[::3], responses[4, ::3, 0], "o", color=BLUE, fillstyle="none")
    axes[1].plot(fields, responses[4, :, 1], color=GREEN, label=r"Global exchange: $D_{X_1}$")
    axes[1].plot(fields, local_values, color=ORANGE, linestyle="--", label=r"Local pair swaps: marginal $D_{Z_1}$")
    axes[1].set(xlabel=r"Transverse field $h/J$", ylabel="Measured response", title=r"Full-register and local exchange differ ($\beta J=2$)")
    axes[1].legend(fontsize=8)
    save_figure(figure, output, "spin_gibbs")
    return {"status": "exact three-qubit circuit demonstration; no three-qubit model trained",
            "nqubits_per_input": 3, "physical_data_qubits": 6, "copies_per_joint_shot": 2,
            "hamiltonian": "H=-J(Z1 Z2+Z2 Z3)-h(X1+X2+X3)-epsilon(Z1-Z3)",
            "J": interaction, "epsilon": staggered, "inverse_temperatures": betas.tolist(),
            "transverse_fields": fields.tolist(), "states": int(len(betas)*len(fields)),
            "state_context_cases": len(rows), "query_axes": ["Z_1", "X_1"],
            "controlled_global_verification_cases": 3, "controlled_global_fredkin_count": 6,
            "clean_control_ancillas_in_verification": 1,
            "max_abs_connected_Z1Z2": float(np.max(np.abs(correlations))),
            "max_global_vs_pairwise_gap_beta2": float(np.max(np.abs(responses[4, :, 0]-local_values))),
            "interpretation": "The exact identity extends to a larger correlated register. Dense simulator gates are not constant-cost hardware gates, and local partial swaps are a different circuit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/physics_examples")
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    configure_plotting()
    random_streams = np.random.SeedSequence(args.seed).spawn(2)
    checks = Checks()
    started = time.perf_counter()
    manifest = {"seed": args.seed, "numpy_seed_sequence_children": [0, 1],
                "command": "python examples/physics_examples.py", "shots": None,
                "execution": "Actual PennyLane default.mixed QNodes for exact examples; actual Torch default.qubit QNodes for the frozen saved predictor",
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                             "torch": torch.__version__, "pennylane": qml.__version__},
                "physical_protocol": "Independent rho tensor rho; V=exp(-i*pi*P/4) on source; one U=exp(-i*pi*S/4) global exchange; jointly measure P on both output ports; output second-first",
                "resource_note": "Two independently prepared input registers per joint exchange shot. Circuit preparation uses exact simulator density matrices. No hardware preparation procedure, finite-shot training, or quantum advantage is claimed.",
                "examples": {}}
    stages = [("correlation", lambda: correlation_example(output, checks, np.random.default_rng(random_streams[0]))),
              ("dephasing_transfer", lambda: dephasing_example(output, checks)),
              ("qfi", lambda: qfi_example(output, checks, np.random.default_rng(random_streams[1]))),
              ("spin_gibbs", lambda: spin_example(output, checks))]
    for name, run in stages:
        print(f"Running {name}...", flush=True)
        stage_started = time.perf_counter()
        manifest["examples"][name] = run()
        manifest["examples"][name]["runtime_seconds"] = round(time.perf_counter()-stage_started, 3)
        print(f"Completed {name} in {time.perf_counter()-stage_started:.2f}s", flush=True)
    manifest["runtime_seconds"] = round(time.perf_counter()-started, 3)
    manifest["all_checks_passed"] = checks.passed
    manifest["check_count"] = len(checks.rows)
    manifest["files"] = sorted({p.name for p in output.iterdir() if p.is_file()} | {"manifest.json", "checks.json"})
    (output / "checks.json").write_text(json.dumps({"all_passed": checks.passed, "checks": checks.rows}, indent=2)+"\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"all_checks_passed": checks.passed, "check_count": len(checks.rows),
                      "runtime_seconds": manifest["runtime_seconds"], "output": str(output)}, indent=2))
    if not checks.passed:
        for row in checks.rows:
            if not row["passed"]:
                print("FAILED", json.dumps(row), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
