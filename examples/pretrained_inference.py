#!/usr/bin/env python3
"""Run the saved one-collision predictor on a benchmark split or new matrices."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from qem.pennylane_models import TorchQuantumModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="NPZ with rho[B,4,4], angles[B,2], optional target[B] or target[B,1]")
    parser.add_argument("--split", default="test_joint", choices=["test_joint", "test_state", "test_context", "bell"])
    parser.add_argument("--weights", type=Path, default=ROOT/"results/torch_quantum/weights.npz")
    parser.add_argument("--output", type=Path, default=Path("results/pretrained_inference"))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if args.input:
        data = np.load(args.input, allow_pickle=False)
        rho, angles = data["rho"], data["angles"]
        target = data["target"] if "target" in data else None
    else:
        data = np.load(ROOT/"data/benchmark.npz", allow_pickle=False)
        rho, angles, target = (data[f"{args.split}_{k}"] for k in ["rho", "angles", "y"])
    if rho.ndim != 3 or rho.shape[1:] != (4, 4) or angles.shape != (len(rho), 2):
        raise ValueError("The pretrained model expects two-qubit matrices and [theta, phi] angles.")
    if target is not None:
        if target.shape == (len(rho), 1):
            target = target[:, 0]
        if target.shape != (len(rho),) or not np.isrealobj(target) or not np.isfinite(target).all():
            raise ValueError("Targets must be finite real values with shape [batch] or [batch, 1].")
    weights = np.load(args.weights, allow_pickle=False)["joint_exchange__selected"]
    model = TorchQuantumModel("joint_exchange", initial_parameters=weights)
    with torch.no_grad():
        raw = model(torch.as_tensor(rho, dtype=torch.complex128), torch.as_tensor(angles))
    arrays = {key: value.numpy() for key, value in raw.items()}
    report = dict(examples=len(rho), source="custom NPZ" if args.input else args.split,
                  model="joint_exchange", source_copies_per_joint_shot=2,
                  mode="Exact simulator expectations with a frozen validation-selected checkpoint")
    if target is not None:
        arrays["target"] = target
        report["mse"] = float(np.mean((arrays["prediction"]-target)**2))
        report["max_abs_error"] = float(np.max(np.abs(arrays["prediction"]-target)))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output/"predictions.npz", **arrays)
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
