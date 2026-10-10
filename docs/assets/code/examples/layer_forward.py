#!/usr/bin/env python3
"""Create physical mixed inputs, execute one conditioned layer, and inspect it."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from qem.layers import ContextExchangeLayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-qubits", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output", type=Path, default=Path("results/layer_forward"))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = 2 ** args.n_qubits
    a = rng.normal(size=(args.batch, d, d)) + 1j * rng.normal(size=(args.batch, d, d))
    rho = a @ a.conj().transpose(0, 2, 1)
    rho /= np.trace(rho, axis1=1, axis2=2)[:, None, None]
    context = rng.uniform(-1, 1, (args.batch, 3))
    layer = ContextExchangeLayer(args.n_qubits, context_dim=3, hidden=16)
    with torch.no_grad():
        state = layer.self_modulate(rho, context).numpy()
        controls = {k: v.numpy() for k, v in layer.gate_parameters(torch.tensor(context)).items()}
    report = dict(seed=args.seed, qubits_per_register=args.n_qubits,
                  batch=args.batch, input_shape=list(rho.shape), output_shape=list(state.shape),
                  source_copies_per_execution=2,
                  parameters=sum(p.numel() for p in layer.parameters()),
                  max_trace_error=float(np.max(np.abs(np.trace(state, axis1=1, axis2=2)-1))),
                  min_eigenvalue=float(np.linalg.eigvalsh(state).min()),
                  max_physical_region_violation=float(np.max(controls["b"]**2-controls["a"]*(1-controls["a"]))),
                  purpose="Forward execution and physical-state checks; no fitted advantage claim")
    assert report["max_trace_error"] < 1e-10 and report["min_eigenvalue"] > -1e-10
    assert report["max_physical_region_violation"] < 1e-12
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/"states_and_controls.npz", rho=rho, context=context, output=state, **controls)
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
