#!/usr/bin/env python3
"""Verify the generic control against an independent PennyLane branch."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qem.generic_conditioned import GenericConditionedExchange
from qem.layers import ContextExchangeLayer


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    with np.load("results/submission_revision/datasets/development.npz", allow_pickle=False) as data:
        rho = torch.as_tensor(data["train_rho"][:3], dtype=torch.complex128)
        context = torch.as_tensor(data["train_context"][:3], dtype=torch.float64)
    torch.manual_seed(139)
    generic = GenericConditionedExchange()
    independent = ContextExchangeLayer()
    independent.conditioner.load_state_dict(generic.conditioner.state_dict())
    raw = generic(rho, context)
    qnode = independent.self_modulate(rho, context)
    state_error = float((raw["first_state"] - qnode).abs().max().detach())
    assert state_error < 1e-12
    eig_min = min(float(torch.linalg.eigvalsh(raw[key]).real.min().detach()) for key in ("first_state", "second_state"))
    assert eig_min > -1e-12
    trace_error = max(float((torch.diagonal(raw[key], dim1=-2, dim2=-1).sum(-1) - 1).abs().max().detach())
                      for key in ("first_state", "second_state"))
    assert trace_error < 1e-12
    # Compare the same physical observable derivative through independent implementations.
    observable = torch.diag(torch.tensor([1., 1., -1., -1.])).to(torch.complex128)
    target_generic = torch.einsum("nij,ji->n", raw["first_state"], observable).real.mean()
    target_qnode = torch.einsum("nij,ji->n", qnode, observable).real.mean()
    weight = generic.conditioner[0].weight
    qweight = independent.conditioner[0].weight
    grad_generic = torch.autograd.grad(target_generic, weight)[0]
    grad_qnode = torch.autograd.grad(target_qnode, qweight)[0]
    grad_error = float((grad_generic - grad_qnode).abs().max().detach())
    assert grad_error < 1e-9
    result = {"status": "PASS", "state_max_abs_error": state_error,
              "gradient_max_abs_error": grad_error, "trace_max_abs_error": trace_error,
              "minimum_eigenvalue": eig_min,
              "trainable_parameters": sum(value.numel() for value in generic.parameters()),
              "fixed_readout": "P(c) on both exchange output ports",
              "copies_per_execution": 2,
              "conditioner": "Linear(3,16) -> tanh -> Linear(16,9); 7 gate angles plus eta/chi"}
    Path("results/submission_revision/generic_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
