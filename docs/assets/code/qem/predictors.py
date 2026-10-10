"""Attach a measured Pauli readout and a Torch head to quantum memory layers.

Density matrices are a simulator input format. A device implementation must
retain the memory and estimate these observables using repeated executions.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from .layers import StreamingExchangeNetwork


def pauli_matrix(word: str) -> torch.Tensor:
    """Matrix of a Pauli word, with its first letter acting on wire zero."""
    single = {
        "I": [[1, 0], [0, 1]], "X": [[0, 1], [1, 0]],
        "Y": [[0, -1j], [1j, 0]], "Z": [[1, 0], [0, -1]],
    }
    if not word or any(c not in single for c in word):
        raise ValueError("A Pauli word must contain one or more I, X, Y, Z letters.")
    result = torch.ones((1, 1), dtype=torch.complex128)
    for letter in word:
        result = torch.kron(result, torch.tensor(single[letter], dtype=torch.complex128))
    return result


class StreamingObservableRegressor(nn.Module):
    """Quantum streaming backbone, selected expectations, affine classical head.

    ``rho`` has shape [batch, 2**n, 2**n]; ``context`` is [batch, context_dim].
    The returned prediction has shape [batch, output_dim]. An affine head
    preserves the copy-order bound on quantum-input polynomial degree.
    """

    def __init__(self, n_qubits=1, context_dim=3, hidden=8, depth=2,
                 observables=None, output_dim=1):
        super().__init__()
        if observables is None:
            observables = ["Z" + "I" * (n_qubits - 1)]
        observables = list(observables)
        if not observables or any(len(p) != n_qubits for p in observables):
            raise ValueError("Every observable must have exactly n_qubits letters.")
        if output_dim < 1:
            raise ValueError("output_dim must be positive.")
        self.config = dict(n_qubits=int(n_qubits), context_dim=int(context_dim),
                           hidden=int(hidden), depth=int(depth),
                           observables=observables, output_dim=int(output_dim))
        self.backbone = StreamingExchangeNetwork(n_qubits, context_dim, hidden, depth)
        self.register_buffer("observables", torch.stack([pauli_matrix(w) for w in observables]))
        self.head = nn.Linear(len(observables), output_dim).double()

    @property
    def copies_per_execution(self):
        return self.backbone.copies_per_execution

    def features(self, rho, context):
        state = self.backbone(rho, context)
        return torch.einsum("...ij,kji->...k", state, self.observables).real

    def forward(self, rho, context):
        return self.head(self.features(rho, context))


def save_predictor(path, model):
    """Save configuration and tensors, not a pickled QNode or device."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "qem-streaming-regressor-v1", "config": model.config,
                "state_dict": model.state_dict()}, path)


def load_predictor(path):
    """Load this package's tensor checkpoint for CPU simulation."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "qem-streaming-regressor-v1":
        raise ValueError("Unsupported predictor checkpoint format.")
    model = StreamingObservableRegressor(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model
