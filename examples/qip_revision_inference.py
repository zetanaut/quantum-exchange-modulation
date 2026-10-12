#!/usr/bin/env python3
"""Load a validation-selected QIP revision checkpoint and predict physical response.

By default this scores the full new-state/held-out-axis split of confirmation
dataset 1. An optional NPZ with ``rho`` and ``angles`` arrays can be used for
other two-qubit states; ``target`` is optional and used only for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qem.pennylane_models import TorchQuantumModel
from train_torch import predict_raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[f"confirmation_{i}" for i in range(1, 6)],
                        default="confirmation_1")
    parser.add_argument("--model", choices=("joint_exchange", "exchange_matched_qnn"),
                        default="joint_exchange")
    parser.add_argument("--split", choices=("test_joint", "test_state", "test_context"),
                        default="test_joint")
    parser.add_argument("--input", type=Path, help="Custom NPZ with rho, angles, optional target")
    parser.add_argument("--output", type=Path, default=Path("/tmp/qem-qip-revision-inference"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    fit_root = ROOT / "results/submission_revision/quantum_confirmation" / args.dataset / "64"
    records = [json.loads((fit_root / f"{args.model}__seed{seed}.json").read_text())
               for seed in (17, 31, 73)]
    selected = min(records, key=lambda item: item["fit"]["best_validation_mse"])
    checkpoint = fit_root / f"{args.model}__seed{selected['seed']}.npz"
    with np.load(checkpoint, allow_pickle=False) as saved:
        weights = saved["weights"].copy()
    if args.input:
        source = args.input
        with np.load(source, allow_pickle=False) as saved:
            rho = saved["rho"].copy()
            angles = saved["angles"].copy()
            target = saved["target"].copy() if "target" in saved.files else None
    else:
        source = ROOT / "results/submission_revision/datasets" / f"{args.dataset}.npz"
        with np.load(source, allow_pickle=False) as saved:
            rho = saved[f"{args.split}_rho"].copy()
            angles = saved[f"{args.split}_angles"].copy()
            target = saved[f"{args.split}_y"].copy()
    if rho.ndim != 3 or rho.shape[1:] != (4, 4) or angles.shape != (len(rho), 2):
        raise ValueError("Expected rho[B,4,4] and angles[B,2] for two-qubit inputs")
    if len(rho) == 0 or not np.all(np.isfinite(rho)) or not np.all(np.isfinite(angles)):
        raise ValueError("Inputs must be nonempty and finite")
    if np.max(np.abs(rho - rho.conj().transpose(0, 2, 1))) > 1e-9:
        raise ValueError("rho must be Hermitian")
    if np.max(np.abs(np.trace(rho, axis1=1, axis2=2) - 1)) > 1e-9:
        raise ValueError("rho must have unit trace")
    if np.min(np.linalg.eigvalsh(rho)) < -1e-9:
        raise ValueError("rho must be positive semidefinite")
    if target is not None and target.reshape(-1).shape != (len(rho),):
        raise ValueError("target must have one scalar per input pair")
    if target is not None and not np.all(np.isfinite(target)):
        raise ValueError("target must be finite")
    model = TorchQuantumModel(args.model, initial_parameters=weights)
    batch = {"rho": torch.as_tensor(rho, dtype=torch.complex128),
             "angles": torch.as_tensor(angles, dtype=torch.float64),
             "y": torch.zeros(len(rho), dtype=torch.float64)}
    raw = predict_raw(model, batch, batch_size=128)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "predictions.npz", prediction=raw["prediction"],
                        heads=raw["heads"], correlation=raw["correlation"],
                        **({"target": target.reshape(-1)} if target is not None else {}))
    report = {"dataset": args.dataset, "model": args.model,
              "checkpoint_seed": selected["seed"],
              "selection": "minimum validation MSE among seeds 17, 31, 73",
              "checkpoint_validation_mse": selected["fit"]["best_validation_mse"],
              "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "pairs": len(rho), "output": str(args.output)}
    if target is not None:
        difference = raw["prediction"] - target.reshape(-1)
        report.update(mse=float(np.mean(difference ** 2)),
                      max_abs_error=float(np.max(np.abs(difference))))
        if not args.input:
            expected = selected["metrics"][args.split]["mse"]
            if not np.isclose(report["mse"], expected, rtol=1e-8, atol=1e-12):
                raise AssertionError(f"Checkpoint score mismatch: {report['mse']} vs {expected}")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
