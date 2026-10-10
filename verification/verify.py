#!/usr/bin/env python3
"""Independent matrix checks for quantum exchange feature modulation.

Run with Python, NumPy, and SciPy only.  By default this writes theory_results.json
beside the script and exits with a nonzero status if any declared check fails.
The checks compare literal joint-register circuits with independently stated
operator identities; no learning or performance claims are tested here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.linalg import expm


SEED = 20261009
TOL = 5.0e-12
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
PAULIS = (X, Y, Z)


def kron_all(*matrices):
    result = np.ones((1, 1), dtype=complex)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def partial_trace(matrix, dimensions, keep):
    """Trace a density matrix, keeping subsystems in their original order."""
    dimensions = list(dimensions)
    result = matrix.reshape(tuple(dimensions + dimensions))
    for axis in sorted(set(range(len(dimensions))) - set(keep), reverse=True):
        result = np.trace(result, axis1=axis, axis2=axis + len(dimensions))
        dimensions.pop(axis)
    retained = math.prod(dimensions)
    return result.reshape(retained, retained)


def random_density(rng, dimension, pure=False):
    columns = 1 if pure else dimension
    matrix = rng.normal(size=(dimension, columns)) + 1j * rng.normal(
        size=(dimension, columns)
    )
    state = matrix @ matrix.conj().T
    return state / np.trace(state)


def random_unitary(rng, dimension):
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    phases = diagonal / np.abs(diagonal)
    return q @ np.diag(phases)


def swap_registers(dimension):
    """Literal permutation |i>|j> -> |j>|i>, without a Pauli expansion."""
    result = np.zeros((dimension * dimension,) * 2, dtype=complex)
    for i in range(dimension):
        for j in range(dimension):
            result[j * dimension + i, i * dimension + j] = 1
    return result


def swap_qubits(number, first, second):
    dimension = 2**number
    result = np.zeros((dimension, dimension), dtype=complex)
    for source in range(dimension):
        bits = list(format(source, f"0{number}b"))
        bits[first], bits[second] = bits[second], bits[first]
        target = int("".join(bits), 2)
        result[target, source] = 1
    return result


def cnot(number, control, target):
    dimension = 2**number
    result = np.zeros((dimension, dimension), dtype=complex)
    for source in range(dimension):
        bits = list(format(source, f"0{number}b"))
        if bits[control] == "1":
            bits[target] = "0" if bits[target] == "1" else "1"
        result[int("".join(bits), 2), source] = 1
    return result


def rotation(generator, angle):
    return expm(-0.5j * angle * generator)


def exchange_circuit(rho, tau, eta):
    dimension = len(rho)
    unitary = expm(-1j * eta * swap_registers(dimension))
    joint = unitary @ np.kron(rho, tau) @ unitary.conj().T
    return partial_trace(joint, [dimension, dimension], [0])


def gain_circuit(rho, tau, eta, kappa):
    probability = (1 + kappa) / 2
    return probability * exchange_circuit(rho, tau, eta) + (
        1 - probability
    ) * exchange_circuit(rho, tau, -eta)


def exchange_formula(rho, tau, a, b):
    return a * rho + (1 - a) * tau + 1j * b * (rho @ tau - tau @ rho)


def expectation(observable, state):
    value = np.trace(observable @ state)
    if abs(value.imag) > TOL:
        raise ValueError(f"Unexpected imaginary expectation: {value.imag}")
    return float(value.real)


def maximum_entry(matrix):
    return float(np.max(np.abs(matrix)))


def metric(value, tolerance=TOL, comparison="<="):
    value = float(value)
    passed = value <= tolerance if comparison == "<=" else value >= tolerance
    return {
        "value": value,
        "comparison": comparison,
        "tolerance": float(tolerance),
        "passed": bool(passed),
    }


def check_result(name, cases, metrics, explanation):
    return {
        "name": name,
        "status": "PASS" if all(m["passed"] for m in metrics.values()) else "FAIL",
        "cases": cases,
        "metrics": metrics,
        "explanation": explanation,
    }


def check_global_identity(rng):
    maximum = 0.0
    cases = 96
    for case in range(cases):
        rho = random_density(rng, 4, pure=case % 4 == 0)
        tau = random_density(rng, 4, pure=case % 4 == 1)
        eta = rng.uniform(-math.pi, math.pi)
        a = math.cos(eta) ** 2
        b = math.cos(eta) * math.sin(eta)
        maximum = max(
            maximum,
            maximum_entry(exchange_circuit(rho, tau, eta) - exchange_formula(rho, tau, a, b)),
        )
    return check_result(
        "global_partial_swap_identity_16_by_16", cases,
        {"maximum_matrix_entry_error": metric(maximum)},
        "A literal 16-by-16 matrix exponential acts on two 2-qubit registers; "
        "the retained register is compared with a rho + (1-a) tau + i b [rho,tau].",
    )


def check_gain_and_control_ancilla(rng):
    mixture_error = ancilla_error = phase_circuit_error = ancilla_leakage = 0.0
    cases = 40
    h = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    p0, p1 = np.diag([1, 0]), np.diag([0, 1])
    for case in range(cases):
        dimension = 2 if case % 2 else 4
        rho = random_density(rng, dimension)
        tau = random_density(rng, dimension)
        eta = rng.uniform(0, math.pi / 2)
        kappa = [-1.0, 0.0, 1.0][case] if case < 3 else rng.uniform(-1, 1)
        probability = (1 + kappa) / 2
        a, b = math.cos(eta) ** 2, kappa * math.cos(eta) * math.sin(eta)
        target = exchange_formula(rho, tau, a, b)
        mixture_error = max(mixture_error, maximum_entry(gain_circuit(rho, tau, eta, kappa) - target))

        swap = swap_registers(dimension)
        plus = expm(-1j * eta * swap)
        minus = expm(1j * eta * swap)
        controlled_sign = np.kron(p0, plus) + np.kron(p1, minus)
        ancilla_vector = np.array([math.sqrt(probability), math.sqrt(1 - probability)])
        ancilla_state = np.outer(ancilla_vector, ancilla_vector)
        input_state = kron_all(ancilla_state, rho, tau)
        output = controlled_sign @ input_state @ controlled_sign.conj().T
        retained = partial_trace(output, [2, dimension, dimension], [1])
        ancilla_error = max(ancilla_error, maximum_entry(retained - target))

        # Independent phase-kickback construction of the full-register exchange.
        size = dimension**2
        controlled_swap = np.kron(p0, np.eye(size)) + np.kron(p1, swap)
        parity_compute = np.kron(h, np.eye(size)) @ controlled_swap @ np.kron(h, np.eye(size))
        phase = np.kron(rotation(Z, 2 * eta), np.eye(size))
        implemented = parity_compute @ phase @ parity_compute
        phase_circuit_error = max(phase_circuit_error, maximum_entry(implemented[:size, :size] - plus))
        ancilla_leakage = max(ancilla_leakage, maximum_entry(implemented[size:, :size]))
    return check_result(
        "bounded_gain_mixture_and_ancilla_implementations", cases,
        {
            "maximum_randomized_mixture_error": metric(mixture_error),
            "maximum_control_ancilla_error": metric(ancilla_error),
            "maximum_phase_kickback_unitary_error": metric(phase_circuit_error),
            "maximum_phase_kickback_ancilla_leakage": metric(ancilla_leakage),
        },
        "The signed-exchange mixture and a coherent control ancilla followed by "
        "tracing give the same bounded gain. Two controlled global SWAPs also "
        "implement exp(-i eta SWAP) with an ancilla returned to zero.",
    )


def check_physicality(rng):
    hermiticity = trace_error = positivity_violation = consistency = 0.0
    cases = 160
    for case in range(cases):
        dimension = 2 if case % 2 else 4
        rho = random_density(rng, dimension, pure=case % 3 == 0)
        tau = random_density(rng, dimension, pure=case % 3 == 1)
        if case % 11 == 0:
            rho = np.eye(dimension) / dimension
        a = [0.0, 1.0, 0.5][case] if case < 3 else rng.uniform(0, 1)
        kappa = [-1.0, 0.0, 1.0][case % 3]
        b = kappa * math.sqrt(a * (1 - a))
        output = exchange_formula(rho, tau, a, b)
        hermiticity = max(hermiticity, maximum_entry(output - output.conj().T))
        trace_error = max(trace_error, abs(np.trace(output) - 1))
        positivity_violation = max(positivity_violation, -float(np.linalg.eigvalsh(output)[0]))
        eta = math.acos(math.sqrt(a))
        consistency = max(consistency, maximum_entry(output - gain_circuit(rho, tau, eta, kappa)))
    return check_result(
        "hermiticity_trace_and_positivity", cases,
        {
            "maximum_hermiticity_error": metric(hermiticity),
            "maximum_trace_error": metric(trace_error),
            "maximum_negative_eigenvalue_magnitude": metric(positivity_violation),
            "maximum_independent_circuit_error": metric(consistency),
        },
        "Valid gains include endpoints, pure inputs, mixed inputs, and maximally "
        "mixed inputs. Positivity is checked by the smallest matrix eigenvalue.",
    )


def check_sharp_parameter_domain(rng):
    del rng
    minimum_negativity = math.inf
    eigenvalue_identity_error = 0.0
    rho = np.diag([1.0, 0.0]).astype(complex)
    orthogonal = np.diag([0.0, 1.0]).astype(complex)
    cases = 0
    for a in (-0.2, 1.2):
        output = exchange_formula(rho, orthogonal, a, 0)
        minimum_negativity = min(minimum_negativity, -np.linalg.eigvalsh(output)[0])
        cases += 1
    for a in (0.01, 0.1, 0.5, 0.9, 0.99):
        for sign in (-1, 1):
            b = sign * 1.15 * math.sqrt(a * (1 - a))
            threshold = 2 * a * (1 - a) / (b * b) - 1
            dot = (1 + threshold) / 2
            tau = (I2 + math.sqrt(1 - dot * dot) * X + dot * Z) / 2
            output = exchange_formula(rho, tau, a, b)
            eigenvalue = float(np.linalg.eigvalsh(output)[0])
            squared_norm = 1 + (1 - dot) * (-2 * a * (1 - a) + b * b * (1 + dot))
            predicted = (1 - math.sqrt(squared_norm)) / 2
            eigenvalue_identity_error = max(eigenvalue_identity_error, abs(eigenvalue - predicted))
            minimum_negativity = min(minimum_negativity, -eigenvalue)
            cases += 1
    return check_result(
        "sharp_gain_domain_counterexamples", cases,
        {
            "minimum_counterexample_negative_eigenvalue_magnitude": metric(minimum_negativity, 1.0e-6, ">="),
            "maximum_analytic_eigenvalue_error": metric(eigenvalue_identity_error),
        },
        "Orthogonal pure states exclude a outside [0,1]. Nearly aligned pure "
        "states explicitly exclude b^2 > a(1-a); the proof in theory.md treats "
        "arbitrarily small violations and establishes necessity and sufficiency.",
    )


def check_simultaneous_covariance(rng):
    covariance_error = swap_commutator_error = conditioned_error = 0.0
    cases = 64
    for case in range(cases):
        dimension = 2 if case % 2 else 4
        rho, tau = random_density(rng, dimension), random_density(rng, dimension)
        w, v = random_unitary(rng, dimension), random_unitary(rng, dimension)
        eta, kappa = rng.uniform(0, math.pi / 2), rng.uniform(-1, 1)
        wdag = w.conj().T
        lhs = gain_circuit(w @ rho @ wdag, w @ tau @ wdag, eta, kappa)
        rhs = w @ gain_circuit(rho, tau, eta, kappa) @ wdag
        covariance_error = max(covariance_error, maximum_entry(lhs - rhs))
        tensor_w, swap = np.kron(w, w), swap_registers(dimension)
        swap_commutator_error = max(swap_commutator_error, maximum_entry(swap @ tensor_w - tensor_w @ swap))
        transformed_v = w @ v @ wdag
        rotated_rho = w @ rho @ wdag
        lhs = gain_circuit(rotated_rho, transformed_v @ rotated_rho @ transformed_v.conj().T, eta, kappa)
        rhs = w @ gain_circuit(rho, v @ rho @ v.conj().T, eta, kappa) @ wdag
        conditioned_error = max(conditioned_error, maximum_entry(lhs - rhs))
    return check_result(
        "simultaneous_unitary_and_conditioned_covariance", cases,
        {
            "maximum_channel_covariance_error": metric(covariance_error),
            "maximum_swap_commutation_error": metric(swap_commutator_error),
            "maximum_conditioned_rotation_covariance_error": metric(conditioned_error),
        },
        "Both copies and the context rotation transform together. This is "
        "covariance, not a claim that every individual register charge is conserved.",
    )


def dephasing_readouts(rho, p):
    plus_tau = rotation(p, math.pi / 2) @ rho @ rotation(p, math.pi / 2).conj().T
    minus_tau = rotation(p, -math.pi / 2) @ rho @ rotation(p, -math.pi / 2).conj().T
    plus = expectation(p, exchange_circuit(rho, plus_tau, math.pi / 4))
    minus = expectation(p, exchange_circuit(rho, minus_tau, math.pi / 4))
    return plus, minus


def check_dephasing_response(rng):
    response_error = branch_error = commutator_error = derivative_error = bell_error = 0.0
    minimum_d = math.inf
    cases = 72
    for case in range(cases):
        rho = random_density(rng, 4, pure=case % 3 == 0)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        p = np.kron(sum(axis[j] * PAULIS[j] for j in range(3)), I2)
        d_value = float(np.trace(rho @ rho - rho @ p @ rho @ p).real)
        commutator = p @ rho - rho @ p
        commutator_error = max(commutator_error, abs(d_value - 0.5 * np.linalg.norm(commutator, "fro") ** 2))
        plus, minus = dephasing_readouts(rho, p)
        mean = expectation(p, rho)
        branch_error = max(branch_error, abs(plus - (mean - d_value / 2)), abs(minus - (mean + d_value / 2)))
        response_error = max(response_error, abs((minus - plus) - d_value))
        gamma = rng.uniform(0.1, 2)
        purity_derivative = 2 * np.trace(rho @ (gamma * (p @ rho @ p - rho))).real
        derivative_error = max(derivative_error, abs(purity_derivative + 2 * gamma * d_value))
        minimum_d = min(minimum_d, d_value)
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    phi_minus = np.array([1, 0, 0, -1], dtype=complex) / math.sqrt(2)
    bell_plus = np.outer(phi_plus, phi_plus.conj())
    bell_minus = np.outer(phi_minus, phi_minus.conj())
    for t in np.linspace(-1, 1, 41):
        rho = (1 + t) * bell_plus / 2 + (1 - t) * bell_minus / 2
        plus, minus = dephasing_readouts(rho, np.kron(Z, I2))
        bell_error = max(bell_error, abs(minus - plus - t * t), abs(plus + t * t / 2), abs(minus - t * t / 2))
        cases += 1
    return check_result(
        "nonlinear_dephasing_D_and_Bell_mixtures", cases,
        {
            "maximum_branch_difference_D_error": metric(response_error),
            "maximum_individual_branch_error": metric(branch_error),
            "maximum_commutator_norm_identity_error": metric(commutator_error),
            "maximum_purity_derivative_identity_error": metric(derivative_error),
            "maximum_Bell_mixture_polynomial_error": metric(bell_error),
            "maximum_D_nonnegativity_violation": metric(max(0.0, -minimum_d)),
        },
        "Official normalization: D=Tr(rho^2)-Tr(rho P rho P)=||[P,rho]||_F^2/2. "
        "Quarter rotations V_+/- and a half exchange give z_+=m-D/2, z_-=m+D/2. "
        "For rho_t=(1+t)Phi+/2+(1-t)Phi-/2 and P=Z1, D=t^2.",
    )


def check_joint_port_dephasing_readout(rng):
    operator_error = port_error = correlation_error = variance_error = probability_error = 0.0
    maximum_mean_zscore = maximum_variance_zscore = 0.0
    matrix_cases, monte_carlo_cases = 72, 6
    shots, batches = 512, 8000
    for case in range(matrix_cases):
        dimension = 2 if case % 2 else 4
        rho = random_density(rng, dimension, pure=case % 3 == 0)
        basis = random_unitary(rng, dimension)
        diagonal_p = np.diag([1.0] * (dimension // 2) + [-1.0] * (dimension // 2))
        p = basis @ diagonal_p @ basis.conj().T
        identity = np.eye(dimension)
        pa, pb = np.kron(p, identity), np.kron(identity, p)
        swap = swap_registers(dimension)
        v = rotation(p, math.pi / 2)
        circuit = expm(-1j * math.pi * swap / 4) @ np.kron(identity, v)
        input_state = np.kron(rho, rho)
        output = circuit @ input_state @ circuit.conj().T
        delta = pb - pa
        direct_k = (np.eye(dimension**2) - np.kron(p, p)) @ swap
        operator_error = max(operator_error, maximum_entry(circuit.conj().T @ delta @ circuit - direct_k))
        m = expectation(p, rho)
        d_value = float(np.trace(rho @ rho - rho @ p @ rho @ p).real)
        za, zb = expectation(pa, output), expectation(pb, output)
        correlation = expectation(pa @ pb, output)
        port_error = max(port_error, abs(za - m + d_value / 2), abs(zb - m - d_value / 2))
        correlation_error = max(correlation_error, abs(correlation - m * m))
        predicted_variance = 2 * (1 - m * m) - d_value * d_value
        measured_variance = expectation(delta @ delta, output) - expectation(delta, output)**2
        direct_variance = expectation(direct_k @ direct_k, input_state) - expectation(direct_k, input_state)**2
        variance_error = max(variance_error, abs(measured_variance - predicted_variance), abs(direct_variance - predicted_variance))
        probabilities, outcomes = [], []
        for first_sign, second_sign in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            projector = np.kron((identity + first_sign * p) / 2, (identity + second_sign * p) / 2)
            probability = expectation(projector, output)
            predicted = (1 + first_sign * za + second_sign * zb + first_sign * second_sign * m * m) / 4
            probability_error = max(probability_error, abs(probability - predicted))
            probabilities.append(max(0.0, probability))
            outcomes.append(second_sign - first_sign)
        if case < monte_carlo_cases:
            probabilities = np.asarray(probabilities)
            probabilities /= probabilities.sum()
            outcomes = np.asarray(outcomes)
            counts = rng.multinomial(shots, probabilities, size=batches)
            estimates = counts @ outcomes / shots
            theoretical_batch_variance = predicted_variance / shots
            empirical_mean = float(estimates.mean())
            empirical_variance = float(estimates.var(ddof=1))
            mean_standard_error = math.sqrt(theoretical_batch_variance / batches)
            centered_fourth = float(np.sum(probabilities * (outcomes - d_value)**4))
            batch_fourth = (centered_fourth + 3 * (shots - 1) * predicted_variance**2) / shots**3
            variance_standard_error = math.sqrt(
                (batch_fourth - (batches - 3) / (batches - 1) * theoretical_batch_variance**2) / batches
            )
            maximum_mean_zscore = max(maximum_mean_zscore, abs(empirical_mean - d_value) / mean_standard_error)
            maximum_variance_zscore = max(maximum_variance_zscore, abs(empirical_variance - theoretical_batch_variance) / variance_standard_error)
    return check_result(
        "joint_port_D_readout_and_exact_shot_variance", matrix_cases + monte_carlo_cases,
        {
            "maximum_direct_K_operator_identity_error": metric(operator_error),
            "maximum_port_mean_error": metric(port_error),
            "maximum_port_correlation_error": metric(correlation_error),
            "maximum_shot_variance_identity_error": metric(variance_error),
            "maximum_joint_outcome_probability_error": metric(probability_error),
            "maximum_Monte_Carlo_mean_standard_errors": metric(maximum_mean_zscore, 6.0),
            "maximum_Monte_Carlo_variance_standard_errors": metric(maximum_variance_zscore, 6.0),
        },
        "Keep both output ports of one quarter-rotation, half-exchange circuit. "
        "The commuting P_A,P_B measurements give D from B-A using two copies "
        "per joint shot. Exactly Wdagger(P_B-P_A)W=K=(I-P tensor P)SWAP, and "
        "Var(Dhat)=[2(1-m^2)-D^2]/N. Thus the direct K measurement has identical "
        f"variance. Monte Carlo uses {monte_carlo_cases} cases, {batches} batches "
        f"per case, and {shots} joint shots per batch; six-standard-error checks "
        "validate empirical means and variances without asserting a learning advantage.",
    )


def check_quadratic_qubit_span(rng):
    bloch_error = span_error = polynomial_error = 0.0
    cases = 160
    for _ in range(cases):
        direction = rng.normal(size=3)
        r = direction / np.linalg.norm(direction) * rng.uniform(0, 1) ** (1 / 3)
        rho = (I2 + sum(r[j] * PAULIS[j] for j in range(3))) / 2
        outputs = {}
        settings = [("z_pi", 2, math.pi), ("y_pi", 1, math.pi)]
        settings += [(f"{j}_quarter", j, math.pi / 2) for j in range(3)]
        for name, j, angle in settings:
            v = rotation(PAULIS[j], angle)
            tau = v @ rho @ v.conj().T
            t = np.array([expectation(p, tau) for p in PAULIS])
            output = exchange_circuit(rho, tau, math.pi / 4)
            observed = np.array([expectation(p, output) for p in PAULIS])
            predicted = (r + t - np.cross(r, t)) / 2
            bloch_error = max(bloch_error, maximum_entry(observed - predicted))
            outputs[name] = observed
        sums = np.array([2 * (r[j] - outputs[f"{j}_quarter"][j]) for j in range(3)])
        squares = np.array([(sums[1] + sums[2] - sums[0]) / 2,
                            (sums[2] + sums[0] - sums[1]) / 2,
                            (sums[0] + sums[1] - sums[2]) / 2])
        xy, xz, yz = -outputs["y_pi"][2], outputs["z_pi"][1], -outputs["z_pi"][0]
        recovered = np.r_[1.0, r, squares, xy, xz, yz]
        x, y, z = r
        expected = np.array([1, x, y, z, x*x, y*y, z*z, x*y, x*z, y*z])
        span_error = max(span_error, maximum_entry(recovered - expected))
        coefficients = rng.normal(size=10)
        polynomial_error = max(polynomial_error, abs(coefficients @ recovered - coefficients @ expected))
    return check_result(
        "qubit_quadratic_feature_span", cases,
        {
            "maximum_cross_product_Bloch_identity_error": metric(bloch_error),
            "maximum_reconstructed_monomial_error": metric(span_error),
            "maximum_arbitrary_quadratic_readout_error": metric(polynomial_error),
        },
        "Five specified rotated-copy exchange settings, Pauli measurements, "
        "linear input features, and a constant span the ten scalar monomials "
        "through degree two. Each block is evaluated as an actual 4-by-4 circuit.",
    )


def check_purity_contraction(rng):
    contraction_violation = identity_error = commutator_bound_violation = fixedpoint_error = 0.0
    cases = 100
    for case in range(cases):
        dimension = 2 if case % 2 else 4
        rho = random_density(rng, dimension, pure=case % 3 == 0)
        v = random_unitary(rng, dimension)
        tau = v @ rho @ v.conj().T
        eta, kappa = rng.uniform(0, math.pi / 2), rng.uniform(-1, 1)
        a, b = math.cos(eta) ** 2, kappa * math.cos(eta) * math.sin(eta)
        output = gain_circuit(rho, tau, eta, kappa)
        input_purity = float(np.trace(rho @ rho).real)
        output_purity = float(np.trace(output @ output).real)
        difference_norm = np.linalg.norm(rho - tau, "fro") ** 2
        commutator_norm = np.linalg.norm(rho @ tau - tau @ rho, "fro") ** 2
        predicted_delta = -a * (1 - a) * difference_norm + b * b * commutator_norm
        identity_error = max(identity_error, abs(output_purity - input_purity - predicted_delta))
        contraction_violation = max(contraction_violation, output_purity - input_purity)
        eigenvalues = np.linalg.eigvalsh(rho)
        spectral_width = eigenvalues[-1] - eigenvalues[0]
        commutator_bound_violation = max(commutator_bound_violation, commutator_norm - spectral_width**2 * difference_norm)
        mixed = np.eye(dimension) / dimension
        fixedpoint_error = max(fixedpoint_error, maximum_entry(gain_circuit(mixed, v @ mixed @ v.conj().T, eta, kappa) - mixed))
    return check_result(
        "self_rotated_copy_purity_contraction", cases,
        {
            "maximum_purity_increase": metric(contraction_violation),
            "maximum_exact_purity_delta_error": metric(identity_error),
            "maximum_spectral_commutator_bound_violation": metric(commutator_bound_violation),
            "maximum_maximally_mixed_fixed_point_error": metric(fixedpoint_error),
        },
        "Only isospectral inputs tau=V rho Vdagger are used. The bound proves "
        "that this self-rotated module cannot purify its input.",
    )


def check_exchange_parameter_shift(rng):
    eta_error = theta_error = formula_derivative_error = 0.0
    cases = 72
    for case in range(cases):
        dimension = 2 if case % 2 else 4
        rho, tau = random_density(rng, dimension), random_density(rng, dimension)
        observable = random_density(rng, dimension) - np.eye(dimension) / dimension
        eta = rng.uniform(-math.pi, math.pi)
        swap = swap_registers(dimension)
        u = expm(-1j * eta * swap)
        evolved = u @ np.kron(rho, tau) @ u.conj().T
        derivative_joint = -1j * (swap @ evolved - evolved @ swap)
        derivative = expectation(np.kron(observable, np.eye(dimension)), derivative_joint)
        shifted = expectation(observable, exchange_circuit(rho, tau, eta + math.pi / 4)) - expectation(observable, exchange_circuit(rho, tau, eta - math.pi / 4))
        eta_error = max(eta_error, abs(derivative - shifted))
        # theta=2 eta: the usual half-generator convention has prefactor 1/2.
        theta = 2 * eta
        shifted_theta = 0.5 * (expectation(observable, exchange_circuit(rho, tau, (theta + math.pi / 2) / 2)) - expectation(observable, exchange_circuit(rho, tau, (theta - math.pi / 2) / 2)))
        theta_error = max(theta_error, abs(0.5 * derivative - shifted_theta))
        derivative_formula = math.sin(2 * eta) * (tau - rho) + 1j * math.cos(2 * eta) * (rho @ tau - tau @ rho)
        formula_derivative_error = max(formula_derivative_error, abs(derivative - expectation(observable, derivative_formula)))
    return check_result(
        "exchange_parameter_shift_conventions", cases,
        {
            "maximum_eta_shift_error": metric(eta_error),
            "maximum_theta_half_generator_shift_error": metric(theta_error),
            "maximum_operator_derivative_error": metric(formula_derivative_error),
        },
        "For U_eta=exp(-i eta S), df/deta=f(eta+pi/4)-f(eta-pi/4), with no "
        "factor 1/2. For exp(-i theta S/2), shifts are +/-pi/2 and prefactor 1/2. "
        "Other gates and inputs are held fixed during each gate derivative.",
    )


def check_global_versus_pairwise(rng):
    del rng
    rho = np.diag([1, 0, 0, 0]).astype(complex)
    tau = np.diag([0, 0, 0, 1]).astype(complex)
    eta = math.pi / 4
    global_output = exchange_circuit(rho, tau, eta)
    s02, s13 = swap_qubits(4, 0, 2), swap_qubits(4, 1, 3)
    pairwise = expm(-1j * eta * s02) @ expm(-1j * eta * s13)
    output = pairwise @ np.kron(rho, tau) @ pairwise.conj().T
    pairwise_output = partial_trace(output, [4, 4], [0])
    distance = float(np.sum(np.abs(np.linalg.eigvalsh(global_output - pairwise_output))))
    return check_result(
        "global_partial_swap_is_not_pairwise_partial_swaps", 1,
        {
            "maximum_global_swap_product_identity_error": metric(maximum_entry(s02 @ s13 - swap_registers(4))),
            "maximum_global_expected_state_error": metric(maximum_entry(global_output - np.diag([0.5, 0, 0, 0.5]))),
            "maximum_pairwise_expected_state_error": metric(maximum_entry(pairwise_output - np.eye(4) / 4)),
            "trace_norm_separation_error_from_one": metric(abs(distance - 1)),
            "observed_trace_norm_separation": metric(distance, 0.9, ">="),
        },
        "At eta=pi/4, |00> and |11> yield diag(1/2,0,0,1/2) under a global "
        "partial SWAP, but I/4 under independent pairwise partial SWAPs. "
        "Their trace norm distance is 1 (trace distance is 1/2).",
    )


def check_fixed_copy_polynomial_degree(rng):
    upper_degree_error = exact_degree_error = exact_circuit_error = 0.0
    highest_difference_minimum = math.inf
    cases = 0
    zero_ancilla = np.diag([1.0, 0.0]).astype(complex)
    for copies in (1, 2, 3):
        dimension = 2**copies
        first, second = random_density(rng, 2), random_density(rng, 2)
        for depth in (1, 4, 9):
            total = np.eye(2 * dimension, dtype=complex)
            for _ in range(depth):
                total = random_unitary(rng, 2 * dimension) @ total
            # A fixed ancilla and arbitrary circuit depth cannot add input copies.
            output_observable = kron_all(Z, *([I2] * (copies - 1)))
            values = []
            for t in np.linspace(0.05, 0.95, copies + 2):
                rho = (1 - t) * first + t * second
                joint = kron_all(*([rho] * copies), zero_ancilla)
                output = total @ joint @ total.conj().T
                retained = partial_trace(output, [dimension, 2], [0])
                values.append(expectation(output_observable, retained))
            upper_degree_error = max(upper_degree_error, abs(np.diff(values, n=copies + 1)[0]))
            cases += 1

        # A parity-to-local-readout circuit reaches the upper degree exactly.
        parity = np.eye(dimension, dtype=complex)
        for control in range(copies - 1):
            parity = cnot(copies, control, copies - 1) @ parity
        local_readout = kron_all(*([I2] * (copies - 1)), Z)
        z_grid = np.linspace(-0.8, 0.8, copies + 2)
        values = []
        for z in z_grid:
            rho = (I2 + z * Z) / 2
            input_state = kron_all(*([rho] * copies))
            output = parity @ input_state @ parity.conj().T
            observed = expectation(local_readout, output)
            exact_circuit_error = max(exact_circuit_error, abs(observed - z**copies))
            values.append(observed)
        exact_degree_error = max(exact_degree_error, abs(np.diff(values, n=copies + 1)[0]))
        step = z_grid[1] - z_grid[0]
        highest = np.diff(values, n=copies)[0]
        highest_difference_minimum = min(highest_difference_minimum, abs(highest))
        exact_degree_error = max(exact_degree_error, abs(highest - math.factorial(copies) * step**copies))
        cases += 1
    return check_result(
        "fixed_copy_polynomial_degree", cases,
        {
            "maximum_order_k_plus_one_finite_difference": metric(upper_degree_error, 5.0e-11),
            "maximum_exact_degree_finite_difference_error": metric(exact_degree_error, 5.0e-11),
            "maximum_parity_circuit_monomial_error": metric(exact_circuit_error),
            "minimum_nonzero_order_k_difference": metric(highest_difference_minimum, 0.01, ">="),
        },
        "One, two, and three fixed input copies are tested at circuit depths "
        "1,4,9 with an independent ancilla. Finite differences above degree k "
        "vanish; CNOT parity circuits independently attain exactly z^k. "
        "No postselection normalization or nonlinear classical output head is used.",
    )


def check_common_gate_resource_certificate(rng):
    h = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    t_gate = np.diag([1, np.exp(1j * math.pi / 4)])
    s_gate = np.diag([1, 1j])
    gates = {"H": h, "T": t_gate, "TDG": t_gate.conj().T}

    def single(matrix, number, wire):
        return kron_all(*(matrix if j == wire else I2 for j in range(number)))

    def literal_toffoli(number, first, second, target):
        result = np.zeros((2**number,) * 2, dtype=complex)
        for source in range(2**number):
            bits = list(format(source, f"0{number}b"))
            if bits[first] == bits[second] == "1":
                bits[target] = "0" if bits[target] == "1" else "1"
            result[int("".join(bits), 2), source] = 1
        return result

    def toffoli_sequence(first, second, target):
        return [("H", target), ("CX", second, target), ("TDG", target),
                ("CX", first, target), ("T", target), ("CX", second, target),
                ("TDG", target), ("CX", first, target), ("T", second),
                ("T", target), ("H", target), ("CX", first, second),
                ("T", first), ("TDG", second), ("CX", first, second)]

    def compile_sequence(number, sequence):
        result = np.eye(2**number, dtype=complex)
        for instruction in sequence:
            gate = cnot(number, instruction[1], instruction[2]) if instruction[0] == "CX" else single(gates[instruction[0]], number, instruction[1])
            result = gate @ result
        return result

    def fredkin_sequence(control, first, second):
        return [("CX", second, first)] + toffoli_sequence(control, first, second) + [("CX", second, first)]

    def literal_fredkin(number, control, first, second):
        result = np.zeros((2**number,) * 2, dtype=complex)
        for source in range(2**number):
            bits = list(format(source, f"0{number}b"))
            if bits[control] == "1":
                bits[first], bits[second] = bits[second], bits[first]
            result[int("".join(bits), 2), source] = 1
        return result

    toffoli_error = maximum_entry(compile_sequence(3, toffoli_sequence(0, 1, 2)) - literal_toffoli(3, 0, 1, 2))
    fredkin_error = maximum_entry(compile_sequence(3, fredkin_sequence(0, 1, 2)) - literal_fredkin(3, 0, 1, 2))
    # Wire order: control, A1, A2, B1, B2.
    controlled_global_sequence = fredkin_sequence(0, 1, 3) + fredkin_sequence(0, 2, 4)
    controlled_global = compile_sequence(5, controlled_global_sequence)
    p0, p1 = np.diag([1, 0]), np.diag([0, 1])
    ideal_controlled_global = np.kron(p0, np.eye(16)) + np.kron(p1, swap_registers(4))
    controlled_error = maximum_entry(controlled_global - ideal_controlled_global)
    compute = single(h, 5, 0) @ controlled_global @ single(h, 5, 0)
    exchange_error = leakage_error = 0.0
    angles = [0.0, math.pi / 4, math.pi / 2] + list(rng.uniform(-math.pi, math.pi, size=9))
    for eta in angles:
        circuit = compute @ single(rotation(Z, 2 * eta), 5, 0) @ compute
        exchange_error = max(exchange_error, maximum_entry(circuit[:16, :16] - expm(-1j * eta * swap_registers(4))))
        leakage_error = max(leakage_error, maximum_entry(circuit[16:, :16]))

    pauli_error = 0.0
    basis_changes = (h, h @ s_gate.conj().T, I2)
    cx = cnot(2, 0, 1)
    for first in range(3):
        for second in range(3):
            angle = rng.uniform(-math.pi, math.pi)
            basis = np.kron(basis_changes[first], basis_changes[second])
            circuit = basis.conj().T @ cx @ np.kron(I2, rotation(Z, angle)) @ cx @ basis
            ideal = rotation(np.kron(PAULIS[first], PAULIS[second]), angle)
            pauli_error = max(pauli_error, maximum_entry(circuit - ideal))

    toffoli_count = sum(item[0] == "CX" for item in toffoli_sequence(0, 1, 2))
    fredkin_count = sum(item[0] == "CX" for item in fredkin_sequence(0, 1, 2))
    global_count = 2 * sum(item[0] == "CX" for item in controlled_global_sequence)
    result = check_result(
        "common_gate_resource_upper_construction", 2 + len(angles) + 9 + 1,
        {
            "maximum_literal_Toffoli_matrix_error": metric(toffoli_error),
            "maximum_literal_Fredkin_matrix_error": metric(fredkin_error),
            "maximum_controlled_global_SWAP_matrix_error": metric(controlled_error),
            "maximum_full_exchange_matrix_error": metric(exchange_error),
            "maximum_control_ancilla_leakage": metric(leakage_error),
            "maximum_two_CNOT_Pauli_rotation_error": metric(pauli_error),
            "Toffoli_CNOT_count_error": metric(abs(toffoli_count - 6), 0),
            "Fredkin_CNOT_count_error": metric(abs(fredkin_count - 8), 0),
            "global_exchange_CNOT_count_error": metric(abs(global_count - 32), 0),
        },
        "Exact matrix verification includes phases, not only classical truth "
        "tables. Fredkin(c,a,b) is chronological CX(b,a), CCX(c,a,b), CX(b,a). "
        "A six-CNOT ancilla-free Toffoli gives eight CNOTs per Fredkin. Two "
        "controlled global SWAPs on two-qubit registers use four Fredkins, "
        "hence 32 CNOTs; four extra weight-two Pauli rotations add eight. "
        "This is a constructive upper bound, not an optimal compiled count.",
    )
    result["resource_counts"] = {
        "logical_data_qubits_per_copy": 2,
        "copies": 2,
        "exchange_control_ancillas": 1,
        "Toffoli_CNOTs": toffoli_count,
        "Fredkin_CNOTs": fredkin_count,
        "Fredkins_per_global_exchange": 4,
        "global_exchange_CNOTs": global_count,
        "additional_weight_two_Pauli_rotations": 4,
        "CNOTs_per_weight_two_Pauli_rotation": 2,
        "module_CNOT_upper_construction": global_count + 4 * 2,
        "assumptions": "All-to-all logical CNOT availability; arbitrary required "
                       "single-qubit rotations; no routing, error correction, "
                       "state preparation, readout, cancellation optimization, "
                       "or additional unlisted layers are counted. Four Pauli "
                       "rotations are an explicit module assumption.",
        "not_claimed": "An optimal count, a lower bound, equality with a generic "
                       "QNN gate budget, or a hardware-level runtime comparison.",
    }
    return result


CHECKS = (
    check_global_identity,
    check_gain_and_control_ancilla,
    check_physicality,
    check_sharp_parameter_domain,
    check_simultaneous_covariance,
    check_dephasing_response,
    check_joint_port_dephasing_readout,
    check_quadratic_qubit_span,
    check_purity_contraction,
    check_exchange_parameter_shift,
    check_global_versus_pairwise,
    check_fixed_copy_polynomial_degree,
    check_common_gate_resource_certificate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("theory_results.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    arguments = parser.parse_args()
    seeds = np.random.SeedSequence(arguments.seed).spawn(len(CHECKS))
    results = []
    for function, seed in zip(CHECKS, seeds):
        try:
            result = function(np.random.default_rng(seed))
        except Exception as exc:
            result = {"name": function.__name__, "status": "FAIL", "cases": 0,
                      "metrics": {}, "exception": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(f"{result['status']}: {result['name']}")
    failures = [item["name"] for item in results if item["status"] != "PASS"]
    report = {
        "package": "quantum_exchange_modulation independent mathematical verification",
        "schema_version": 1,
        "status": "FAIL" if failures else "PASS",
        "seed": arguments.seed,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "summary": {"checks": len(results), "passed": len(results) - len(failures),
                    "failed": len(failures), "matrix_cases": sum(item["cases"] for item in results)},
        "checks": results,
        "scope": "Numerical validation of exact mathematical identities and counterexamples; "
                 "not a statistical learning benchmark or a proof of quantum advantage.",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    resource_result = next(item for item in results if item["name"] == "common_gate_resource_upper_construction")
    resource_path = arguments.output.with_name("resource_certificate.json")
    resource_path.write_text(json.dumps({"schema_version": 1, "seed": arguments.seed, **resource_result}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report['summary']['passed']}/{len(results)} checks; report: {arguments.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
