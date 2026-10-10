"""Trainable two-copy models for the conditional purity-loss experiment.

The simulation is exact-density-matrix training.  No finite-shot gradient or
training advantage is claimed.  All models are given the same query P(c), and
their Euler frames have learned scales and offsets rather than exact alignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1., -1.]).astype(complex)
PAULIS = {"I": I2, "X": X, "Y": Y, "Z": Z}


def batched_kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("nab,ncd->nacbd", a, b).reshape(
        len(a), a.shape[1] * b.shape[1], a.shape[2] * b.shape[2])


def expectation(observable: np.ndarray, state: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nji->n", observable, state).real


@dataclass
class QuantumBatch:
    rho: np.ndarray
    context: np.ndarray
    angles: np.ndarray
    P: np.ndarray
    y: np.ndarray
    features: np.ndarray
    rho2: np.ndarray
    readout2: tuple[np.ndarray, np.ndarray]
    correlation2: np.ndarray

    @classmethod
    def from_arrays(cls, rho, context, angles, P, y):
        rho = np.asarray(rho, complex)
        P = np.asarray(P, complex)
        n = len(rho)
        identity = np.broadcast_to(np.eye(4, dtype=complex), (n, 4, 4))
        return cls(rho, np.asarray(context), np.asarray(angles), P,
                   np.asarray(y, float),
                   np.column_stack((np.ones(n), angles[:, 0], angles[:, 1])),
                   batched_kron(rho, rho),
                   (batched_kron(P, identity), batched_kron(identity, P)),
                   batched_kron(P, P))

    @classmethod
    def from_npz(cls, source, split):
        return cls.from_arrays(*(source[f"{split}_{name}"] for name in
                                 ("rho", "context", "angles", "P", "y")))


@dataclass(frozen=True)
class Gate:
    name: str
    permutation: np.ndarray
    phase: np.ndarray
    # Each triple contributes coefficient * parameter[index] * feature[column].
    links: tuple[tuple[int, int, float], ...] = ()
    fixed: bool = False


def pauli_gate(word, links, name=None):
    matrix = np.array([[1]], complex)
    for letter in word:
        matrix = np.kron(matrix, PAULIS[letter])
    permutation = np.argmax(np.abs(matrix), axis=1)
    phase = matrix[np.arange(len(matrix)), permutation]
    return Gate(name or "R" + word, permutation, phase, tuple(links))


def cnot_gate(nqubits, control, target):
    values = np.arange(2 ** nqubits)
    permutation = values ^ (((values >> (nqubits - 1 - control)) & 1)
                            << (nqubits - 1 - target))
    return Gate(f"CNOT({control},{target})", permutation,
                np.ones(len(values), complex), (), True)


def local_word(nqubits, qubit, axis):
    return "I" * qubit + axis + "I" * (nqubits - qubit - 1)


def _left(state, gate):
    return gate.phase[None, :, None] * state[:, gate.permutation, :]


def _right(state, gate):
    return state[:, :, gate.permutation] * gate.phase[gate.permutation][None, None, :]


def _conjugate_fixed(state, gate):
    return _right(_left(state, gate), gate)


def _rotate(state, gate, angle):
    left = _left(state, gate)
    right = _right(state, gate)
    both = _right(left, gate)
    c = np.cos(angle / 2)[:, None, None]
    s = np.sin(angle / 2)[:, None, None]
    return c * c * state + s * s * both + 1j * c * s * (right - left)


def _angle(gate, parameters, features):
    result = np.zeros(len(features))
    for index, column, coefficient in gate.links:
        result += coefficient * parameters[index] * features[:, column]
    return result


def circuit_forward(gates, parameters, features, state, keep_states=False):
    history = [state] if keep_states else None
    angles = []
    for gate in gates:
        if gate.fixed:
            state = _conjugate_fixed(state, gate)
            angles.append(None)
        else:
            angle = _angle(gate, parameters, features)
            state = _rotate(state, gate, angle)
            angles.append(angle)
        if history is not None:
            history.append(state)
    return state, history, angles


def circuit_backward(gates, features, history, angles, adjoint, nparameters):
    gradient = np.zeros(nparameters)
    for index in range(len(gates) - 1, -1, -1):
        gate = gates[index]
        if gate.fixed:
            adjoint = _conjugate_fixed(adjoint, gate)
            continue
        state_after = history[index + 1]
        derivative = -0.5j * (_left(state_after, gate) - _right(state_after, gate))
        gate_derivative = expectation(adjoint, derivative)
        for parameter, column, coefficient in gate.links:
            gradient[parameter] += coefficient * np.dot(gate_derivative, features[:, column])
        adjoint = _rotate(adjoint, gate, -angles[index])
    return gradient


def frame_gates(nqubits, qubits, inverse=False):
    # A(c)=Rz(s_phi phi+b_phi) Ry(s_theta theta+b_theta).
    theta = ((0, 1, 1.), (1, 0, 1.))
    phi = ((2, 2, 1.), (3, 0, 1.))
    if inverse:
        order = (("Z", tuple((i, f, -v) for i, f, v in phi)),
                 ("Y", tuple((i, f, -v) for i, f, v in theta)))
    else:
        order = (("Y", theta), ("Z", phi))
    return [pauli_gate(local_word(nqubits, q, axis), links,
                       f"learned_frame_{axis}_{q}")
            for q in qubits for axis, links in order]


def metrics(y, prediction):
    error = prediction - y
    variance = np.mean((y - np.mean(y)) ** 2)
    return {"mse": float(np.mean(error ** 2)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "max_abs_error": float(np.max(np.abs(error))),
            "r2": float(1 - np.mean(error ** 2) / variance) if variance > 0 else None,
            "prediction_min": float(np.min(prediction)),
            "prediction_max": float(np.max(prediction)),
            "n": int(len(y))}


class JointExchangeModel:
    """One transformed fresh copy, one exchange, and both output-port readouts."""
    nparameters = 25

    def __init__(self, commutator=True):
        self.commutator = commutator
        self.name = "joint_exchange" if commutator else "no_commutator"
        words = ("YI", "ZI", "IY", "IZ", "XX", "ZZ", "XI", "IX") * 2
        self.gates = (frame_gates(2, [0], inverse=True) +
                      [pauli_gate(word, ((4 + i, 0, 1.),)) for i, word in enumerate(words)] +
                      frame_gates(2, [0], inverse=False))

    def initial_parameters(self, seed, label_mean=0.):
        rng = np.random.default_rng(seed)
        p = np.zeros(self.nparameters)
        p[:4] = [1 + rng.normal(0, .12), rng.normal(0, .18),
                 1 + rng.normal(0, .12), rng.normal(0, .18)]
        p[4:20] = rng.normal(0, .45, 16)
        p[20] = rng.uniform(.35, 1.15)
        p[21] = rng.uniform(.15, .65)
        p[22:24] = [-.5, .5] + rng.normal(0, .1, 2)
        p[24] = label_mean
        return p

    def bounds(self):
        return ([(.4, 1.6), (-np.pi, np.pi), (.4, 1.6), (-np.pi, np.pi)] +
                [(-np.pi, np.pi)] * 16 +
                [(.08, np.pi / 2 - .08), (-np.pi / 2, np.pi / 2)] +
                [(-8., 8.), (-8., 8.), (-2., 2.)])

    def _forward(self, parameters, batch, keep_states=False):
        tau, history, angles = circuit_forward(self.gates, parameters,
                                               batch.features, batch.rho, keep_states)
        eta, chi = parameters[20:22]
        a = np.cos(eta) ** 2
        b = .5 * np.sin(2 * eta) * np.cos(chi) if self.commutator else 0.
        m = expectation(batch.P, batch.rho)
        t = expectation(batch.P, tau)
        commutator = batch.rho @ tau - tau @ batch.rho
        kappa = expectation(batch.P, 1j * commutator)
        z_a = a * m + (1 - a) * t + b * kappa
        z_b = (1 - a) * m + a * t - b * kappa
        heads = np.column_stack((z_a, z_b))
        prediction = heads @ parameters[22:24] + parameters[24]
        raw = {"prediction": prediction, "heads": heads, "correlation": m * t,
               "source_mean": m, "transformed_mean": t,
               "commutator_readout": kappa}
        return raw, (tau, history, angles, m, t, kappa, a, b)

    def raw(self, parameters, batch):
        return self._forward(parameters, batch)[0]

    def predict(self, parameters, batch):
        return self.raw(parameters, batch)["prediction"]

    def loss_and_gradient(self, parameters, batch):
        raw, cache = self._forward(parameters, batch, keep_states=True)
        tau, history, angles, m, t, kappa, a, b = cache
        error = raw["prediction"] - batch.y
        delta = 2 * error / len(error)
        u, v = parameters[22:24]
        commutator_observable = 1j * (batch.P @ batch.rho - batch.rho @ batch.P)
        adjoint = delta[:, None, None] * (
            (u * (1 - a) + v * a) * batch.P + b * (u - v) * commutator_observable)
        gradient = circuit_backward(self.gates, batch.features, history, angles,
                                    adjoint, self.nparameters)
        eta, chi = parameters[20:22]
        derivative_eta = np.sin(2 * eta) * (t - m)
        if self.commutator:
            derivative_eta += np.cos(2 * eta) * np.cos(chi) * kappa
            gradient[21] = np.dot(delta * (u - v),
                                   -.5 * np.sin(2 * eta) * np.sin(chi) * kappa)
        gradient[20] = np.dot(delta * (u - v), derivative_eta)
        gradient[22:24] = raw["heads"].T @ delta
        gradient[24] = np.sum(delta)
        return float(np.mean(error ** 2)), gradient

    def describe(self, parameters=None):
        description = {
            "name": self.name, "trainable_parameters": self.nparameters,
            "inactive_parameters": [] if self.commutator else [21],
            "active_parameters": self.nparameters if self.commutator else self.nparameters - 1,
            "input_qubits": 2, "simultaneous_input_copies": 2,
            "output_heads": "P(c) on both output registers; same execution",
            "conditioner": "shared learned Euler frame, 4 scale/offset parameters",
            "input_preprocessing": "no exact query alignment; learned frame is part of V(c)",
            "inner_circuit": [g.name for g in self.gates],
            "readout": "learned affine combination of both port means",
            "physical_coefficients": "a=cos(eta)^2; b=sin(2eta)cos(chi)/2" if self.commutator else
                                     "a=cos(eta)^2; b=0 (equal mixture of opposite exchange signs)",
            "chi_realization": "classical random choice of exchange sign, no postselection",
            "primitive_gates_per_pair": {
                "one_qubit_parameterized_rotations_before_readout": 16,
                "two_qubit_Pauli_rotations_before_readout": 4,
                "global_two_register_partial_swap": 1,
                "one_qubit_rotations_for_both_P_readouts": 4},
            "global_partial_swap_exact_ancilla_construction": {
                "extra_clean_ancillas": 1, "controlled_full_register_swaps": 2,
                "controlled_pair_swaps_Fredkin": 4, "ancilla_H": 4, "ancilla_Rz": 1},
            "compiled_two_qubit_gate_budget_matched": False,
            "resource_note": "Global register partial-SWAP is not a product of pair partial-SWAPs. "
                             "Fredkin decomposition is not compiled or costed as a CNOT here.",
            "training": "exact density matrix, analytic gradients, full-batch L-BFGS-B",
            "identifiability": "a,b and affine-head scale are not separately physical estimates"}
        if parameters is not None:
            eta, chi = parameters[20:22]
            description["fitted_summary"] = {
                "eta": float(eta), "chi": float(chi),
                "a": float(np.cos(eta) ** 2),
                "b": float(.5 * np.sin(2 * eta) * np.cos(chi)) if self.commutator else 0.,
                "head_weights": parameters[22:24].tolist(), "head_bias": float(parameters[24]),
                "frame_parameters": parameters[:4].tolist()}
        return description


class TwoCopyQNN:
    """General conditional four-qubit circuit with jointly measurable P heads."""
    name = "two_copy_qnn"
    nparameters = 31
    head_start = 28

    def __init__(self):
        self.gates = frame_gates(4, [0, 2], inverse=True)
        parameter = 4
        entanglers = (((0, 2), (1, 3), (0, 1), (2, 3)),
                      ((2, 0), (3, 1), (1, 2), (3, 0)))
        for layer in range(3):
            for qubit in range(4):
                for axis in ("Y", "Z"):
                    self.gates.append(pauli_gate(local_word(4, qubit, axis),
                                                  ((parameter, 0, 1.),)))
                    parameter += 1
            if layer < 2:
                self.gates.extend(cnot_gate(4, i, j) for i, j in entanglers[layer])
        self.gates += frame_gates(4, [0, 2], inverse=False)

    def initial_parameters(self, seed, label_mean=0.):
        rng = np.random.default_rng(seed)
        p = np.zeros(self.nparameters)
        p[:4] = [1 + rng.normal(0, .12), rng.normal(0, .18),
                 1 + rng.normal(0, .12), rng.normal(0, .18)]
        p[4:28] = rng.normal(0, .45, 24)
        p[28:30] = [-.5, .5] + rng.normal(0, .1, 2)
        p[30] = label_mean
        return p

    def bounds(self):
        return ([(.4, 1.6), (-np.pi, np.pi), (.4, 1.6), (-np.pi, np.pi)] +
                [(-np.pi, np.pi)] * 24 + [(-8., 8.), (-8., 8.), (-2., 2.)])

    def _forward(self, parameters, batch, keep_states=False):
        final, history, angles = circuit_forward(self.gates, parameters, batch.features,
                                                 batch.rho2, keep_states)
        heads = np.column_stack([expectation(observable, final)
                                 for observable in batch.readout2])
        h = self.head_start
        raw = {"prediction": heads @ parameters[h:h + 2] + parameters[h + 2],
               "heads": heads, "correlation": expectation(batch.correlation2, final)}
        return raw, (history, angles)

    def raw(self, parameters, batch):
        return self._forward(parameters, batch)[0]

    def predict(self, parameters, batch):
        return self.raw(parameters, batch)["prediction"]

    def loss_and_gradient(self, parameters, batch):
        raw, (history, angles) = self._forward(parameters, batch, keep_states=True)
        error = raw["prediction"] - batch.y
        delta = 2 * error / len(error)
        h = self.head_start
        adjoint = delta[:, None, None] * (parameters[h] * batch.readout2[0] +
                                         parameters[h + 1] * batch.readout2[1])
        gradient = circuit_backward(self.gates, batch.features, history, angles,
                                    adjoint, self.nparameters)
        gradient[h:h + 2] = raw["heads"].T @ delta
        gradient[h + 2] = np.sum(delta)
        return float(np.mean(error ** 2)), gradient

    def describe(self, parameters=None):
        description = {
            "name": self.name, "trainable_parameters": self.nparameters,
            "active_parameters": self.nparameters, "input_qubits": 2,
            "simultaneous_input_copies": 2,
            "output_heads": "P(c) on first spin of each output register; same execution",
            "conditioner": "same learned Euler-frame family as exchange model",
            "input_preprocessing": "learned, not exact, query frame on each copy",
            "inner_circuit": [g.name for g in self.gates],
            "readout": "learned affine combination of two jointly measured heads",
            "primitive_gates_per_pair": {
                "one_qubit_parameterized_rotations_before_readout": 32,
                "CNOT_before_readout": 8,
                "one_qubit_rotations_for_both_P_readouts": 4},
            "compiled_two_qubit_gate_budget_matched": False,
            "resource_note": "All-to-all CNOT edges; no routing overhead or hardware compilation.",
            "training": "exact density matrix, analytic gradients, full-batch L-BFGS-B"}
        if parameters is not None:
            description["fitted_summary"] = {"head_weights": parameters[28:30].tolist(),
                                               "head_bias": float(parameters[30]),
                                               "frame_parameters": parameters[:4].tolist()}
        return description


class ExchangeMatchedQNN(TwoCopyQNN):
    """General two-copy pre/exchange/post circuit sharing the exchange primitive.

    This stronger control contains the analytic solution and uses the same
    number of global exchanges and two-qubit Pauli rotations as JointExchange.
    It is not a hardware-compiled equality of all gate counts.
    """
    name = "exchange_matched_qnn"
    nparameters = 28
    head_start = 25

    def __init__(self):
        self.gates = frame_gates(4, [0, 2], inverse=True)
        parameter = 4
        for qubit in range(4):
            for axis in ("Y", "Z"):
                self.gates.append(pauli_gate(local_word(4, qubit, axis), ((parameter, 0, 1.),)))
                parameter += 1
        for word in ("XIXI", "IXIX"):
            self.gates.append(pauli_gate(word, ((parameter, 0, 1.),)))
            parameter += 1
        values = np.arange(16)
        swap_permutation = ((values & 3) << 2) | (values >> 2)
        self.gates.append(Gate("global_register_partial_swap", swap_permutation,
                               np.ones(16, complex), ((14, 0, 2.),)))
        parameter = 15
        for qubit in range(4):
            for axis in ("Y", "Z"):
                self.gates.append(pauli_gate(local_word(4, qubit, axis), ((parameter, 0, 1.),)))
                parameter += 1
        for word in ("ZZII", "IIZZ"):
            self.gates.append(pauli_gate(word, ((parameter, 0, 1.),)))
            parameter += 1
        self.gates += frame_gates(4, [0, 2], inverse=False)

    def initial_parameters(self, seed, label_mean=0.):
        rng = np.random.default_rng(seed)
        p = np.zeros(self.nparameters)
        p[:4] = [1 + rng.normal(0, .12), rng.normal(0, .18),
                 1 + rng.normal(0, .12), rng.normal(0, .18)]
        p[4:25] = rng.normal(0, .45, 21)
        p[14] = rng.uniform(.35, 1.15)
        p[25:27] = [-.5, .5] + rng.normal(0, .1, 2)
        p[27] = label_mean
        return p

    def bounds(self):
        bounds = ([(.4, 1.6), (-np.pi, np.pi), (.4, 1.6), (-np.pi, np.pi)] +
                  [(-np.pi, np.pi)] * 21 + [(-8., 8.), (-8., 8.), (-2., 2.)])
        bounds[14] = (.08, np.pi / 2 - .08)
        return bounds

    def analytic_parameters(self):
        p = np.zeros(self.nparameters)
        p[[0, 2]] = 1.
        p[9] = np.pi / 2
        p[14] = np.pi / 4
        p[25:27] = [-1., 1.]
        return p

    def describe(self, parameters=None):
        description = {
            "name": self.name, "trainable_parameters": self.nparameters,
            "active_parameters": self.nparameters, "input_qubits": 2,
            "simultaneous_input_copies": 2,
            "output_heads": "P(c) on first spin of each output register; same execution",
            "conditioner": "same learned Euler-frame family as exchange model",
            "inner_circuit": [g.name for g in self.gates],
            "readout": "learned affine combination of two jointly measured heads",
            "analytic_solution_contained": True,
            "random_initialization": "analytic solution is verified but not used for training initialization",
            "primitive_gates_per_pair": {
                "one_qubit_parameterized_rotations_before_readout": 24,
                "two_qubit_Pauli_rotations_before_readout": 4,
                "global_two_register_partial_swap": 1,
                "one_qubit_rotations_for_both_P_readouts": 4},
            "global_partial_swap_exact_ancilla_construction": {
                "extra_clean_ancillas": 1, "controlled_full_register_swaps": 2,
                "controlled_pair_swaps_Fredkin": 4, "ancilla_H": 4, "ancilla_Rz": 1},
            "matched_to_exchange": "same 1 global exchange and 4 two-qubit Pauli rotations; 8 more single rotations",
            "compiled_two_qubit_gate_budget_matched": False,
            "resource_note": "Nonlocal primitive counts match; no hardware compilation or routing comparison.",
            "training": "exact density matrix, analytic gradients, full-batch L-BFGS-B"}
        if parameters is not None:
            description["fitted_summary"] = {"head_weights": parameters[25:27].tolist(),
                                               "head_bias": float(parameters[27]),
                                               "frame_parameters": parameters[:4].tolist(),
                                               "eta": float(parameters[14])}
        return description


def make_model(name):
    if name == "joint_exchange":
        return JointExchangeModel(True)
    if name == "no_commutator":
        return JointExchangeModel(False)
    if name == "two_copy_qnn":
        return TwoCopyQNN()
    if name == "exchange_matched_qnn":
        return ExchangeMatchedQNN()
    raise ValueError(f"Unknown quantum model: {name}")


def analytic_exchange_reference(batch):
    """Untrained exact reference, returning the same two-port raw interface."""
    identity = np.broadcast_to(np.eye(4), batch.rho.shape)
    V = (identity - 1j * batch.P) / np.sqrt(2)
    tau = V @ batch.rho @ np.swapaxes(V.conj(), 1, 2)
    output_a = .5 * (batch.rho + tau) + .5j * (batch.rho @ tau - tau @ batch.rho)
    output_b = .5 * (batch.rho + tau) - .5j * (batch.rho @ tau - tau @ batch.rho)
    heads = np.column_stack((expectation(batch.P, output_a), expectation(batch.P, output_b)))
    m = expectation(batch.P, batch.rho)
    return {"prediction": heads[:, 1] - heads[:, 0], "heads": heads,
            "correlation": m * m, "head_weights": np.array([-1., 1.]), "head_bias": 0.}


def gradient_check(model, batch, seed=123, step=1e-6):
    parameters = model.initial_parameters(seed, float(np.mean(batch.y)))
    value, analytic = model.loss_and_gradient(parameters, batch)
    numerical = np.empty_like(analytic)
    for i in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[i] += step
        minus[i] -= step
        numerical[i] = (np.mean((model.predict(plus, batch) - batch.y) ** 2) -
                        np.mean((model.predict(minus, batch) - batch.y) ** 2)) / (2 * step)
    absolute = np.abs(analytic - numerical)
    report = {"loss": value, "max_absolute_error": float(np.max(absolute)),
              "relative_l2_error": float(np.linalg.norm(analytic - numerical) /
                                         max(np.linalg.norm(numerical), 1e-12)),
              "parameters_checked": len(parameters), "passed": bool(np.max(absolute) < 2e-6)}
    if not report["passed"]:
        raise AssertionError(f"Gradient check failed for {model.name}: {report}")
    return report
