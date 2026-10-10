"""PyTorch measured-feature FiLM comparator, with a portable flat parameter vector."""
from __future__ import annotations
import torch
from torch import nn
from .classical_models import FiLMRegressor


class TorchFiLM(nn.Module):
    """Classical FiLM with nonlinear preprocessing and post-modulation activation.

    Inputs are 15 measured nonidentity two-qubit Pauli expectations and a known
    three-component context direction. Exact-expectation training does not pay
    the hardware measurement cost; that cost is assessed separately.
    """
    def __init__(self, width=12, seed=17):
        super().__init__()
        self.width = int(width)
        self.reference = FiLMRegressor(width=self.width)
        self.weights = nn.Parameter(torch.tensor(self.reference.initial(seed), dtype=torch.float64))

    def unpack(self):
        values, offset = [], 0
        for shape, size in zip(self.reference.shapes, self.reference.sizes):
            values.append(self.weights[offset:offset+size].reshape(shape))
            offset += size
        return values

    def forward(self, features, context):
        W, bh, Wg, Wb, wo, bo = self.unpack()
        h = torch.tanh(features @ W + bh)
        z = torch.tanh(h * (1 + context @ Wg) + context @ Wb)
        return z @ wo + bo[0]

    @property
    def n_params(self):
        return self.weights.numel()
