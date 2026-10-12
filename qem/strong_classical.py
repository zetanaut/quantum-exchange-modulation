"""Broader Torch classical controls for the QIP submission study."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import pauli_features


def even_context(n):
    return np.stack([n[:, i] * n[:, j] * (np.sqrt(2) if i != j else 1)
                     for i in range(3) for j in range(i, 3)], axis=1)


def feature_family(rho, n, family):
    pauli = pauli_features(rho)
    r = pauli[:, 1:]
    if family == "raw_signed":
        return r, n
    if family == "raw_even":
        return r, even_context(n)
    if family == "within_even":
        blocks = pauli.reshape(-1, 4, 4)[:, 1:, :]
        x = np.stack([blocks[:, i, b] * blocks[:, j, b]
                      for b in range(4) for i in range(3) for j in range(i, 3)], axis=1)
        return x, even_context(n)
    if family == "joint_signed":
        return np.einsum("ni,nj->nij", r, n).reshape(len(r), -1), n
    raise ValueError(family)


class ConcatMLP(nn.Module):
    def __init__(self, x_dim, c_dim, width, depth, activation):
        super().__init__()
        act = nn.Tanh if activation == "tanh" else nn.SiLU
        layers = []
        input_dim = x_dim + c_dim
        for _ in range(depth):
            layers.extend((nn.Linear(input_dim, width), act()))
            input_dim = width
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x, context):
        return self.network(torch.cat((x, context), dim=1)).squeeze(-1)


class DeepFiLM(nn.Module):
    """Each depth step has its own hidden transform and context modulation."""
    def __init__(self, x_dim, c_dim, width, depth, activation):
        super().__init__()
        self.hidden = nn.ModuleList()
        self.modulators = nn.ModuleList()
        input_dim = x_dim
        for _ in range(depth):
            self.hidden.append(nn.Linear(input_dim, width))
            self.modulators.append(nn.Linear(c_dim, 2 * width))
            input_dim = width
        self.readout = nn.Linear(width, 1)
        self.activation = torch.tanh if activation == "tanh" else torch.nn.functional.silu

    def forward(self, x, context):
        h = x
        for layer, modulator in zip(self.hidden, self.modulators):
            gamma, beta = modulator(context).chunk(2, dim=1)
            h = self.activation((1 + gamma) * layer(h) + beta)
        return self.readout(h).squeeze(-1)


def make_model(kind, x_dim, c_dim, width, depth, activation):
    constructor = ConcatMLP if kind == "mlp" else DeepFiLM if kind == "film" else None
    if constructor is None:
        raise ValueError(kind)
    return constructor(x_dim, c_dim, width, depth, activation).double()
