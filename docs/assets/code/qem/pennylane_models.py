"""Actual PennyLane circuits with Torch autodifferentiation.

The primary backend is ``default.qubit`` with an exact purification of each
supplied mixed state.  Four data wires carry rho tensor rho; four untouched
environment wires represent the mixed input in the simulator.  They are not
additional input copies, are never read by the learner, and are not included in
the physical circuit-resource count.

This public-API backend avoids a batched-QubitDensityMatrix initialization bug
observed in PennyLane 0.45.1's default.mixed.  All learned operations are actual
PennyLane gates inside QNodes; no reduced density-matrix prediction formula is
used for training.  Signed exchange is an exact weighted average of two QNode
executions.  On hardware it corresponds to random selection of the exchange
sign, with one sign used per prepared pair.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Iterable

import numpy as np
import pennylane as qml
import torch
from torch import nn

from .quantum_models import make_model as make_numpy_model


@lru_cache(maxsize=8)
def _numpy_swap(register_dimension: int) -> np.ndarray:
    d = register_dimension
    matrix = np.zeros((d * d, d * d), dtype=np.complex128)
    for first in range(d):
        for second in range(d):
            matrix[second * d + first, first * d + second] = 1
    return matrix


def exchange_unitary(eta: torch.Tensor, register_dimension: int) -> torch.Tensor:
    """Return exp(-i eta SWAP), including a possible leading angle batch."""
    eta = torch.as_tensor(eta)
    dtype = torch.complex128 if eta.dtype == torch.float64 else torch.complex64
    swap = torch.as_tensor(_numpy_swap(register_dimension), dtype=dtype, device=eta.device)
    identity = torch.eye(register_dimension**2, dtype=dtype, device=eta.device)
    return torch.cos(eta)[..., None, None] * identity - 1j * torch.sin(eta)[..., None, None] * swap


def global_exchange(
    eta: torch.Tensor,
    wires: Iterable[int],
    implementation: str = "matrix",
    control_wire: int | None = None,
) -> None:
    """Queue a full-register exchange, never a product of local exchanges.

    ``matrix`` queues the exact differentiable QubitUnitary. ``controlled_swap``
    queues the phase-kickback circuit using two controlled global SWAPs and a
    clean control ancilla, which returns to zero.
    """
    wires = tuple(wires)
    if len(wires) == 0 or len(wires) % 2:
        raise ValueError("Exchange requires two equal, nonempty qubit registers")
    half = len(wires) // 2
    if implementation == "matrix":
        qml.QubitUnitary(exchange_unitary(eta, 2**half), wires=wires)
        return
    if implementation != "controlled_swap":
        raise ValueError(f"Unknown exchange implementation: {implementation}")
    if control_wire is None or control_wire in wires:
        raise ValueError("controlled_swap requires a distinct clean control_wire")
    qml.Hadamard(wires=control_wire)
    for first, second in zip(wires[:half], wires[half:]):
        qml.CSWAP(wires=[control_wire, first, second])
    qml.Hadamard(wires=control_wire)
    qml.RZ(2 * eta, wires=control_wire)
    qml.Hadamard(wires=control_wire)
    for first, second in zip(wires[:half], wires[half:]):
        qml.CSWAP(wires=[control_wire, first, second])
    qml.Hadamard(wires=control_wire)


def purify_density_matrix(density_matrix: torch.Tensor) -> torch.Tensor:
    """Vectorize the positive square root, with system indices before environment.

    The returned amplitudes have shape ``(..., d*d)`` and reduce to the input
    density matrix after tracing the environment.  Inputs are simulator state
    objects. Spectral preparation is not a learned classical feature extractor.
    Benchmark inputs do not require gradients. For trainable density inputs,
    users should be aware of eigenvector derivative degeneracies and prefer a
    direct mixed-state backend, as in the separate reusable layer module.
    """
    matrix = (density_matrix + density_matrix.mH) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    if bool(torch.any(eigenvalues.detach() < -1e-9)):
        raise ValueError("Input density matrix is not positive semidefinite")
    root = (eigenvectors * torch.sqrt(torch.clamp(eigenvalues, min=0))[..., None, :]) @ eigenvectors.mH
    return root.reshape(root.shape[:-2] + (root.shape[-1] ** 2,))


def two_copy_purification(rho: torch.Tensor) -> torch.Tensor:
    """Efficiently purify rho tensor rho using one 4-by-4 eigendecomposition."""
    one_root = purify_density_matrix(rho).reshape(rho.shape)
    joint_root = torch.einsum("...ab,...cd->...acbd", one_root, one_root)
    return joint_root.reshape(rho.shape[:-2] + (256,))


def prepare_density_batch(
    density_matrix: torch.Tensor,
    wires: Iterable[int],
    environment_wires: Iterable[int] | None = None,
) -> None:
    """Queue a public-API purified StatePrep for a supplied mixed-state batch."""
    wires = tuple(wires)
    if environment_wires is None:
        if wires != tuple(range(len(wires))):
            raise ValueError("Specify environment_wires for nonconsecutive data-wire labels")
        environment_wires = tuple(range(len(wires), 2 * len(wires)))
    environment_wires = tuple(environment_wires)
    if len(environment_wires) != len(wires) or set(wires) & set(environment_wires):
        raise ValueError("Purification requires the same number of distinct environment wires")
    amplitudes = purify_density_matrix(density_matrix)
    qml.StatePrep(amplitudes, wires=wires + environment_wires, normalize=False, validate_norm=True)


def _pauli_rotation(angle, word, wires):
    active = [(letter, wire) for letter, wire in zip(word, wires) if letter != "I"]
    letters = "".join(letter for letter, _ in active)
    active_wires = [wire for _, wire in active]
    if len(active) == 1:
        {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[letters](angle, wires=active_wires[0])
    elif letters == "XX":
        qml.IsingXX(angle, wires=active_wires)
    elif letters == "YY":
        qml.IsingYY(angle, wires=active_wires)
    elif letters == "ZZ":
        qml.IsingZZ(angle, wires=active_wires)
    else:
        qml.PauliRot(angle, pauli_word=letters, wires=active_wires)


def _decode_gate(gate, number_of_qubits):
    if gate.name == "global_register_partial_swap":
        return ("exchange", None, gate.links)
    if gate.fixed:
        inside = gate.name.partition("(")[2].rstrip(")")
        control, target = (int(item) for item in inside.split(","))
        return ("cnot", (control, target), gate.links)
    if gate.name.startswith("learned_frame_"):
        axis, wire = gate.name.split("_")[-2:]
        word = "I" * int(wire) + axis + "I" * (number_of_qubits - int(wire) - 1)
    else:
        word = gate.name[1:]
    if len(word) != number_of_qubits or set(word) - set("IXYZ"):
        raise ValueError(f"Unrecognized reference gate: {gate.name}")
    return ("pauli", word, gate.links)


def _linked_angle(parameters, angles, links):
    value = parameters.new_zeros(())
    for index, column, coefficient in links:
        feature = 1.0 if column == 0 else angles[..., column - 1]
        value = value + coefficient * parameters[index] * feature
    return value


class TorchQuantumModel(nn.Module):
    """Torch trainable wrapper around a real batched PennyLane QNode.

    API::

        model = TorchQuantumModel(name, initial_parameters=numpy_vector)
        raw = model(rho_complex128[N,4,4], angles_float64[N,2])
        loss = ((raw['prediction'] - labels) ** 2).mean()
        loss.backward()

    ``parameters_vector`` is the only nn.Parameter. Its ordering exactly matches
    the NumPy reference implementation. No gates inspect preparation labels or
    density-matrix entries as classical features. The fixed query readout is
    realized by actual RZ(-phi), RY(-theta) rotations on the first spin of each
    data register, followed by Pauli-Z measurements.
    """

    def __init__(
        self,
        name: str,
        initial_parameters=None,
        *,
        seed: int = 17,
        label_mean: float = 0.0,
        exchange_implementation: str = "matrix",
    ):
        super().__init__()
        self.name = name
        self._reference = make_numpy_model(name)
        self.nparameters = self._reference.nparameters
        self.head_start = 22 if name in ("joint_exchange", "no_commutator") else self._reference.head_start
        self.exchange_implementation = exchange_implementation
        if initial_parameters is None:
            initial_parameters = self._reference.initial_parameters(seed, label_mean)
        initial_parameters = torch.as_tensor(initial_parameters, dtype=torch.float64).detach().clone()
        if initial_parameters.shape != (self.nparameters,):
            raise ValueError(f"Expected {self.nparameters} parameters, got {tuple(initial_parameters.shape)}")
        self.parameters_vector = nn.Parameter(initial_parameters)
        is_exchange_family = name in ("joint_exchange", "no_commutator")
        gate_qubits = 2 if is_exchange_family else 4
        self._specifications = tuple(_decode_gate(gate, gate_qubits) for gate in self._reference.gates)
        self._is_exchange_family = is_exchange_family
        extra_control = int(exchange_implementation == "controlled_swap")
        self.simulator_wires = 8 + extra_control
        self._device = qml.device("default.qubit", wires=self.simulator_wires, shots=None)

        @qml.qnode(self._device, interface="torch", diff_method="backprop")
        def circuit(purification, angles, parameters, exchange_sign=1.0):
            qml.StatePrep(purification, wires=range(8), normalize=False, validate_norm=True)
            gate_wires = (2, 3) if is_exchange_family else (0, 1, 2, 3)
            for kind, payload, links in self._specifications:
                if kind == "cnot":
                    qml.CNOT(wires=payload)
                elif kind == "exchange":
                    # Reference Gate angle is 2*eta in the exp(-i angle*S/2) convention.
                    angle = _linked_angle(parameters, angles, links) / 2
                    global_exchange(angle, range(4), self.exchange_implementation, 8 if extra_control else None)
                else:
                    angle = _linked_angle(parameters, angles, links)
                    _pauli_rotation(angle, payload, gate_wires)
            if is_exchange_family:
                global_exchange(exchange_sign * parameters[20], range(4), self.exchange_implementation, 8 if extra_control else None)
            for wire in (0, 2):
                qml.RZ(-angles[..., 1], wires=wire)
                qml.RY(-angles[..., 0], wires=wire)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(2)), qml.expval(qml.Z(0) @ qml.Z(2))

        self.qnode = circuit

    def _coerce_inputs(self, rho, angles):
        device = self.parameters_vector.device
        rho = torch.as_tensor(rho, dtype=torch.complex128, device=device)
        angles = torch.as_tensor(angles, dtype=torch.float64, device=device)
        if rho.ndim == 2:
            rho = rho.unsqueeze(0)
        if angles.ndim == 1:
            angles = angles.unsqueeze(0)
        if rho.ndim != 3 or tuple(rho.shape[-2:]) != (4, 4):
            raise ValueError("rho must have shape (N,4,4), or (4,4) for one example")
        if angles.ndim != 2 or angles.shape[-1] != 2:
            raise ValueError("angles must have shape (N,2), ordered theta,phi")
        if len(angles) == 1 and len(rho) != 1:
            angles = angles.expand(len(rho), -1)
        if len(rho) != len(angles):
            raise ValueError("State and context batch sizes differ")
        return rho, angles

    def forward(self, rho_batch, angles_batch):
        rho, angles = self._coerce_inputs(rho_batch, angles_batch)
        purification = two_copy_purification(rho)
        parameters = self.parameters_vector
        positive = torch.stack(self.qnode(purification, angles, parameters, 1.0), dim=-1)
        if self._is_exchange_family:
            negative = torch.stack(self.qnode(purification, angles, parameters, -1.0), dim=-1)
            probability = 0.5 if self.name == "no_commutator" else torch.cos(parameters[21] / 2) ** 2
            values = probability * positive + (1 - probability) * negative
        else:
            values = positive
        heads = values[..., :2]
        h = self.head_start
        prediction = heads @ parameters[h:h + 2] + parameters[h + 2]
        return {"heads": heads, "correlation": values[..., 2], "prediction": prediction}

    def bounds(self):
        """Reference optimization bounds; all real gate angles remain physical."""
        return self._reference.bounds()

    def describe(self, parameters=None):
        if parameters is None:
            parameters = self.parameters_vector.detach().cpu().numpy()
        elif torch.is_tensor(parameters):
            parameters = parameters.detach().cpu().numpy()
        result = self._reference.describe(np.asarray(parameters))
        result["training"] = "Actual PennyLane QNodes with interface='torch', diff_method='backprop'; optimizer selected by training script"
        result["backend"] = {
            "device": "default.qubit",
            "interface": "torch",
            "differentiation": "backpropagation through actual parameterized PennyLane gates",
            "physical_data_wires": 4,
            "raw_state_copies_per_quantum_execution": 2,
            "simulator_environment_wires": 4,
            "environment_accessible_to_learner": False,
            "simulator_exchange_control_wires": int(self.exchange_implementation == "controlled_swap"),
            "simulator_total_wires": self.simulator_wires,
            "mixed_state_representation": "Exact positive-square-root purification; eigenvectors are input preparation only and never used as learned features",
            "batching": "Native PennyLane StatePrep and gate-parameter broadcasting",
            "signed_mixture": "Two actual QNode branches, averaged with cos(chi/2)^2 and sin(chi/2)^2" if self._is_exchange_family else "Not used",
            "exchange_implementation": self.exchange_implementation,
            "physical_resource_note": "Purification environments are simulator bookkeeping and do not change raw-copy or physical data-gate counts. On hardware the sign mixture randomly selects one exchange sign for each prepared pair.",
            "compatibility_note": "Public-API purification avoids the batched QubitDensityMatrix initialization bug observed in default.mixed in PennyLane 0.45.1",
        }
        return result


def make_model(name, initial_parameters=None, **kwargs):
    return TorchQuantumModel(name, initial_parameters=initial_parameters, **kwargs)
