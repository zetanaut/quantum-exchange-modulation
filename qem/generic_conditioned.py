"""Generic nonlinear source conditioner with the benchmark's fixed P(c) readout.

The two-copy exchange map is evaluated by its exact differentiable matrix formula.
The corresponding PennyLane branch is checked independently before learning.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def local_rot(angles):
    alpha, beta, gamma = angles.unbind(dim=-1)
    zero = torch.zeros_like(alpha)
    rz_a = torch.stack((torch.exp(-.5j * alpha), zero, zero, torch.exp(.5j * alpha)), dim=-1).reshape(-1, 2, 2)
    rz_g = torch.stack((torch.exp(-.5j * gamma), zero, zero, torch.exp(.5j * gamma)), dim=-1).reshape(-1, 2, 2)
    c, s = torch.cos(beta / 2), torch.sin(beta / 2)
    ry = torch.stack((c, -s, s, c), dim=-1).reshape(-1, 2, 2).to(torch.complex128)
    return rz_g @ ry @ rz_a


def batch_kron(a, b):
    return torch.einsum("nab,ncd->nacbd", a, b).reshape(len(a), a.shape[1] * b.shape[1], a.shape[2] * b.shape[2])


class GenericConditionedExchange(nn.Module):
    """Unconstrained context network for source gates and exchange, with fixed P(c)."""
    def __init__(self, context_dim=3, hidden=16):
        super().__init__()
        self.conditioner = nn.Sequential(nn.Linear(context_dim, hidden), nn.Tanh(), nn.Linear(hidden, 9)).double()
        self.head = nn.Linear(2, 1).double()
        with torch.no_grad():
            self.conditioner[-1].weight.mul_(.15)
            self.conditioner[-1].bias.zero_()
            self.conditioner[-1].bias[-1] = -math.log(3.)
        self.register_buffer("eye4", torch.eye(4, dtype=torch.complex128))
        self.register_buffer("zz", torch.diag(torch.tensor([1., -1., -1., 1.])).to(torch.complex128))
        self.register_buffer("paulis", torch.stack((torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
                                                    torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
                                                    torch.diag(torch.tensor([1., -1.])).to(torch.complex128))))
        self.register_buffer("eye2", torch.eye(2, dtype=torch.complex128))

    def gate_parameters(self, context):
        raw = self.conditioner(context)
        angles = math.pi * torch.tanh(raw[:, :7])
        eta = (math.pi / 2) * torch.sigmoid(raw[:, 7])
        chi = math.pi * torch.sigmoid(raw[:, 8])
        return angles, eta, chi

    def source(self, rho, angles):
        rotations = batch_kron(local_rot(angles[:, :3]), local_rot(angles[:, 3:6]))
        theta = angles[:, 6]
        zz_gate = torch.cos(theta / 2)[:, None, None] * self.eye4 - 1j * torch.sin(theta / 2)[:, None, None] * self.zz
        V = zz_gate @ rotations
        return V @ rho @ V.conj().transpose(-1, -2)

    def forward(self, rho, context, control_context=None):
        angles, eta, chi = self.gate_parameters(context if control_context is None else control_context)
        tau = self.source(rho, angles)
        a = torch.cos(eta) ** 2
        b = .5 * torch.sin(2 * eta) * torch.cos(chi)
        commutator = rho @ tau - tau @ rho
        first = a[:, None, None] * rho + (1 - a)[:, None, None] * tau + 1j * b[:, None, None] * commutator
        second = (1 - a)[:, None, None] * rho + a[:, None, None] * tau - 1j * b[:, None, None] * commutator
        P1 = torch.einsum("na,aij->nij", context.to(torch.complex128), self.paulis)
        P = batch_kron(P1, self.eye2.expand(len(context), -1, -1))
        heads = torch.stack((torch.einsum("nij,nji->n", first, P).real,
                             torch.einsum("nij,nji->n", second, P).real), dim=1)
        source_mean = torch.einsum("nij,nji->n", rho, P).real
        transformed_mean = torch.einsum("nij,nji->n", tau, P).real
        return {"prediction": self.head(heads).squeeze(-1), "heads": heads,
                "correlation": source_mean * transformed_mean, "first_state": first,
                "second_state": second, "source_state": tau}
