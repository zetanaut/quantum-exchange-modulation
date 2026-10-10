#!/usr/bin/env python3
"""Verify actual Torch/PennyLane models against independent NumPy circuits.

The test exits nonzero on a failed metric and writes a machine-readable report.
It checks full values and all parameter gradients, not just a successful import.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pennylane as qml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qem.pennylane_models import (  # noqa: E402
    TorchQuantumModel,
    global_exchange,
    prepare_density_batch,
    two_copy_purification,
)
from qem.quantum_models import QuantumBatch, batched_kron, make_model  # noqa: E402


MODELS = ("joint_exchange", "no_commutator", "exchange_matched_qnn")
VALUE_TOLERANCE = 2e-11
GRADIENT_TOLERANCE = 2e-9


def numpy_value(tensor):
    return tensor.detach().cpu().numpy()


def error(first, second):
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def metric(value, tolerance):
    return {"value": float(value), "tolerance": float(tolerance), "passed": bool(value <= tolerance)}


def result(name, cases, metrics, details=None):
    return {
        "name": name,
        "status": "PASS" if all(item["passed"] for item in metrics.values()) else "FAIL",
        "cases": cases,
        "metrics": metrics,
        "details": details or {},
    }


def make_fixture(source):
    selections = (("train", [0, 9, 20, 45]), ("validation", [0, 7, 18]),
                  ("test_joint", [0, 5, 17]), ("bell", [0, 13, 25, 38, 50]))
    arrays = []
    for name in ("rho", "context", "angles", "P", "y"):
        arrays.append(np.concatenate([source[f"{split}_{name}"][indices] for split, indices in selections]))
    return QuantumBatch.from_arrays(*arrays)


def subset(batch, indices):
    return QuantumBatch.from_arrays(*(getattr(batch, name)[indices] for name in ("rho", "context", "angles", "P", "y")))


def loss_and_gradient(model, batch):
    model.zero_grad(set_to_none=True)
    raw = model(batch.rho, batch.angles)
    labels = torch.as_tensor(batch.y, dtype=torch.float64)
    loss = ((raw["prediction"] - labels) ** 2).mean()
    loss.backward()
    return raw, float(loss.detach()), numpy_value(model.parameters_vector.grad).copy()


def verify_purification(batch):
    rho = torch.as_tensor(batch.rho, dtype=torch.complex128)
    amplitudes = two_copy_purification(rho)
    amplitude_matrix = amplitudes.reshape(-1, 16, 16)
    reduced = amplitude_matrix @ amplitude_matrix.mH
    expected = batched_kron(batch.rho, batch.rho)
    norms = torch.sum(torch.abs(amplitudes) ** 2, dim=-1)
    return result("exact_input_purification", len(batch.rho), {
        "maximum_reduced_density_matrix_error": metric(error(numpy_value(reduced), expected), VALUE_TOLERANCE),
        "maximum_state_norm_error": metric(error(numpy_value(norms), np.ones(len(rho))), VALUE_TOLERANCE),
    }, {"data_wires": 4, "simulator_environment_wires": 4,
        "environment_touched_by_learned_gates": False, "additional_physical_input_copies": 0})


def verify_numpy_parity(batch, weight_files):
    maximum_values = {"heads": 0.0, "correlation": 0.0, "prediction": 0.0}
    loss_error = gradient_error = relative_gradient_error = 0.0
    fixtures = []
    for name in MODELS:
        reference = make_model(name)
        fixtures.append((name, "initial_seed19", reference.initial_parameters(19, float(batch.y.mean()))))
    loaded_files = []
    for path in weight_files:
        if path is None or not Path(path).exists():
            continue
        path = Path(path)
        archive = np.load(path, allow_pickle=False)
        loaded_files.append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
        for key in archive.files:
            name = key.split("__", 1)[0]
            if name in MODELS:
                fixtures.append((name, f"{path.parent.name}/{key}", archive[key]))
    per_fixture = []
    for name, label, parameters in fixtures:
        reference = make_model(name)
        model = TorchQuantumModel(name, initial_parameters=parameters)
        raw, actual_loss, actual_gradient = loss_and_gradient(model, batch)
        expected = reference.raw(parameters, batch)
        expected_loss, expected_gradient = reference.loss_and_gradient(parameters, batch)
        per_errors = {}
        for field in maximum_values:
            discrepancy = error(numpy_value(raw[field]), expected[field])
            maximum_values[field] = max(maximum_values[field], discrepancy)
            per_errors[field] = discrepancy
        loss_error = max(loss_error, abs(actual_loss - expected_loss))
        gradient_error = max(gradient_error, error(actual_gradient, expected_gradient))
        relative = np.linalg.norm(actual_gradient - expected_gradient) / max(np.linalg.norm(expected_gradient), 1e-8)
        relative_gradient_error = max(relative_gradient_error, float(relative))
        per_fixture.append({"model": name, "fixture": label, "maximum_value_error": max(per_errors.values()),
                            "maximum_gradient_error": error(actual_gradient, expected_gradient)})
    metrics = {f"maximum_{field}_error": metric(value, VALUE_TOLERANCE) for field, value in maximum_values.items()}
    metrics.update({
        "maximum_loss_error": metric(loss_error, VALUE_TOLERANCE),
        "maximum_all_parameter_gradient_error": metric(gradient_error, GRADIENT_TOLERANCE),
        "maximum_gradient_relative_L2_error": metric(relative_gradient_error, 2e-7),
    })
    return result("independent_NumPy_value_and_gradient_parity", len(fixtures), metrics,
                  {"loaded_weight_files": loaded_files, "fixtures": per_fixture,
                   "examples_per_fixture": len(batch.rho), "gradient_reference": "Independent hand-derived NumPy circuit adjoint"})


def verify_finite_differences(batch):
    finite_batch = subset(batch, [0, 4, 8])
    maximum_error = maximum_relative = 0.0
    checks = []
    step = 1e-6
    for name in MODELS:
        reference = make_model(name)
        parameters = reference.initial_parameters(43, float(finite_batch.y.mean()))
        model = TorchQuantumModel(name, parameters)
        _, _, automatic = loss_and_gradient(model, finite_batch)
        numerical = np.empty_like(parameters)
        labels = torch.as_tensor(finite_batch.y, dtype=torch.float64)
        for index in range(len(parameters)):
            values = []
            for sign in (1, -1):
                shifted = parameters.copy()
                shifted[index] += sign * step
                with torch.no_grad():
                    model.parameters_vector.copy_(torch.as_tensor(shifted, dtype=torch.float64))
                    prediction = model(finite_batch.rho, finite_batch.angles)["prediction"]
                    values.append(float(((prediction - labels) ** 2).mean()))
            numerical[index] = (values[0] - values[1]) / (2 * step)
        absolute = error(automatic, numerical)
        relative = np.linalg.norm(automatic - numerical) / max(np.linalg.norm(numerical), 1e-8)
        maximum_error = max(maximum_error, absolute)
        maximum_relative = max(maximum_relative, float(relative))
        checks.append({"model": name, "parameters_checked": len(parameters), "maximum_error": absolute})
    return result("PennyLane_all_parameter_central_finite_differences", sum(item["parameters_checked"] for item in checks), {
        "maximum_gradient_absolute_error": metric(maximum_error, 2e-6),
        "maximum_gradient_relative_L2_error": metric(maximum_relative, 2e-5),
    }, {"step": step, "models": checks})


def verify_batching_and_gate_tapes(batch):
    batch = subset(batch, [0, 2, 4, 7, 10])
    value_error = gradient_error = missing_gate_violation = 0.0
    tape_details = []
    for name in MODELS:
        reference = make_model(name)
        parameters = reference.initial_parameters(29, float(batch.y.mean()))
        model = TorchQuantumModel(name, parameters)
        batched, _, batched_gradient = loss_and_gradient(model, batch)
        model.zero_grad(set_to_none=True)
        rows = [model(batch.rho[i], batch.angles[i]) for i in range(len(batch.rho))]
        for field in ("heads", "correlation", "prediction"):
            individual = torch.cat([row[field] for row in rows], dim=0)
            value_error = max(value_error, error(numpy_value(individual), numpy_value(batched[field])))
        predictions = torch.cat([row["prediction"] for row in rows])
        loss = ((predictions - torch.as_tensor(batch.y, dtype=torch.float64)) ** 2).mean()
        loss.backward()
        gradient_error = max(gradient_error, error(numpy_value(model.parameters_vector.grad), batched_gradient))
        purification = two_copy_purification(torch.as_tensor(batch.rho, dtype=torch.complex128))
        tape = model.qnode.construct((purification, torch.as_tensor(batch.angles, dtype=torch.float64), model.parameters_vector, 1.0), {})
        names = [operation.name for operation in tape.operations]
        required = {"StatePrep", "RY", "RZ", "QubitUnitary", "IsingXX", "IsingZZ"}
        missing_gate_violation = max(missing_gate_violation, float(bool(required - set(names))))
        tape_details.append({"model": name, "operation_count_per_branch": len(names),
                             "operation_names": names, "interface": str(model.qnode.interface),
                             "differentiation": str(model.qnode.diff_method),
                             "environment_wires": [4, 5, 6, 7],
                             "environment_gates_after_StatePrep": [operation.name for operation in tape.operations[1:] if set(operation.wires) & {4, 5, 6, 7}]})
        if tape_details[-1]["environment_gates_after_StatePrep"]:
            missing_gate_violation = 1.0
    return result("native_batching_and_actual_gate_tapes", 3 * len(batch.rho), {
        "maximum_batched_vs_individual_value_error": metric(value_error, VALUE_TOLERANCE),
        "maximum_batched_vs_individual_gradient_error": metric(gradient_error, GRADIENT_TOLERANCE),
        "missing_gate_or_environment_access_violation": metric(missing_gate_violation, 0),
    }, {"tapes": tape_details})


def verify_gain_endpoints(batch):
    batch = subset(batch, [1, 5, 11])
    maximum_value = maximum_gradient = inactive_gradient = 0.0
    cases = 0
    for name in ("joint_exchange", "no_commutator"):
        reference = make_model(name)
        for eta, chi in ((0, 0), (np.pi / 4, 0), (np.pi / 4, np.pi / 2),
                         (np.pi / 4, np.pi), (np.pi / 2, -np.pi / 2)):
            parameters = reference.initial_parameters(59, float(batch.y.mean()))
            parameters[20:22] = [eta, chi]
            model = TorchQuantumModel(name, parameters)
            raw, _, gradient = loss_and_gradient(model, batch)
            expected = reference.raw(parameters, batch)
            _, expected_gradient = reference.loss_and_gradient(parameters, batch)
            maximum_value = max(maximum_value, error(numpy_value(raw["prediction"]), expected["prediction"]))
            maximum_gradient = max(maximum_gradient, error(gradient, expected_gradient))
            if name == "no_commutator":
                inactive_gradient = max(inactive_gradient, abs(float(gradient[21])))
            cases += 1
    return result("signed_exchange_boundary_gradients", cases, {
        "maximum_prediction_error": metric(maximum_value, VALUE_TOLERANCE),
        "maximum_gradient_error": metric(maximum_gradient, GRADIENT_TOLERANCE),
        "maximum_no_commutator_chi_gradient": metric(inactive_gradient, 1e-14),
    }, {"reason": "Cosine/sine mixture probabilities avoid square-root derivative singularities at zero branch probability"})


def verify_controlled_swap_implementation(batch):
    batch = subset(batch, [0, 5, 12])
    value_error = gradient_error = count_error = 0.0
    for name in MODELS:
        reference = make_model(name)
        parameters = reference.initial_parameters(61, float(batch.y.mean()))
        dense = TorchQuantumModel(name, parameters)
        native = TorchQuantumModel(name, parameters, exchange_implementation="controlled_swap")
        direct, _, direct_gradient = loss_and_gradient(dense, batch)
        compiled, _, compiled_gradient = loss_and_gradient(native, batch)
        for field in ("heads", "correlation", "prediction"):
            value_error = max(value_error, error(numpy_value(direct[field]), numpy_value(compiled[field])))
        gradient_error = max(gradient_error, error(direct_gradient, compiled_gradient))
        purification = two_copy_purification(torch.as_tensor(batch.rho, dtype=torch.complex128))
        tape = native.qnode.construct((purification, torch.as_tensor(batch.angles, dtype=torch.float64), native.parameters_vector, 1.0), {})
        count_error = max(count_error, abs(sum(operation.name == "CSWAP" for operation in tape.operations) - 4))

    device = qml.device("default.qubit", wires=9)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def ancilla_check(joint_density, eta):
        prepare_density_batch(joint_density, wires=range(4), environment_wires=range(4, 8))
        global_exchange(eta, range(4), implementation="controlled_swap", control_wire=8)
        return qml.density_matrix(wires=[8])

    joint = torch.as_tensor(batched_kron(batch.rho, np.roll(batch.rho, 1, axis=0)), dtype=torch.complex128)
    observed_ancilla = ancilla_check(joint, torch.tensor(0.73, dtype=torch.float64))
    target_ancilla = np.broadcast_to(np.diag([1.0, 0.0]), (len(batch.rho), 2, 2))
    return result("actual_PennyLane_Fredkin_exchange_implementation", len(MODELS), {
        "maximum_matrix_vs_Fredkin_value_error": metric(value_error, VALUE_TOLERANCE),
        "maximum_matrix_vs_Fredkin_gradient_error": metric(gradient_error, GRADIENT_TOLERANCE),
        "maximum_control_ancilla_reset_error": metric(error(numpy_value(observed_ancilla), target_ancilla), VALUE_TOLERANCE),
        "CSWAP_count_error": metric(count_error, 0),
    }, {"physical_phase_control_ancillas": 1, "Fredkins_per_branch": 4,
        "simulator_environment_wires": 4, "hardware_optimality_claim": False})


def verify_analytic_containment(batch):
    maximum_error = 0.0
    for name in ("joint_exchange", "exchange_matched_qnn"):
        reference = make_model(name)
        if name == "exchange_matched_qnn":
            parameters = reference.analytic_parameters()
        else:
            parameters = np.zeros(reference.nparameters)
            parameters[[0, 2]] = 1
            parameters[5] = np.pi / 2
            parameters[20] = np.pi / 4
            parameters[22:24] = [-1, 1]
        model = TorchQuantumModel(name, parameters)
        with torch.no_grad():
            prediction = numpy_value(model(batch.rho, batch.angles)["prediction"])
        maximum_error = max(maximum_error, error(prediction, batch.y))
    return result("analytic_solution_containment_without_training", 2, {
        "maximum_exact_dephasing_target_error": metric(maximum_error, VALUE_TOLERANCE),
    }, {"analytic_parameters_used_for_random_training_initialization": False,
        "purpose": "Expressivity and convention check only"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/benchmark.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "results/quantum/weights.npz")
    parser.add_argument("--torch-checkpoints", type=Path, default=ROOT / "results/torch_quantum/weights.npz")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pennylane_results.json"))
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    source = np.load(arguments.data, allow_pickle=False)
    batch = make_fixture(source)
    jobs = (
        lambda: verify_purification(batch),
        lambda: verify_numpy_parity(batch, [arguments.weights, arguments.torch_checkpoints]),
        lambda: verify_finite_differences(batch),
        lambda: verify_batching_and_gate_tapes(batch),
        lambda: verify_gain_endpoints(batch),
        lambda: verify_controlled_swap_implementation(batch),
        lambda: verify_analytic_containment(batch),
    )
    checks = []
    started = time.perf_counter()
    for job in jobs:
        try:
            check = job()
        except Exception as exc:
            check = {"name": "verification_exception", "status": "FAIL", "cases": 0,
                     "metrics": {}, "exception": f"{type(exc).__name__}: {exc}"}
        checks.append(check)
        print(f"{check['status']}: {check['name']}", flush=True)
    failures = [check for check in checks if check["status"] != "PASS"]
    report = {
        "schema_version": 1,
        "status": "FAIL" if failures else "PASS",
        "seed": 20261009,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "torch": torch.__version__, "pennylane": qml.__version__,
                    "torch_threads": torch.get_num_threads(), "device": "default.qubit"},
        "summary": {"checks": len(checks), "passed": len(checks) - len(failures),
                    "failed": len(failures), "cases": sum(check["cases"] for check in checks),
                    "elapsed_seconds": time.perf_counter() - started},
        "checks": checks,
        "scope": "Actual PennyLane circuit values, batching, Torch gradients, and gate implementation verified against independent NumPy and finite-difference references. No learning advantage is inferred from these checks.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {report['summary']['passed']}/{len(checks)} checks; {arguments.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
