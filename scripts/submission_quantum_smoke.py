#!/usr/bin/env python3
"""Small development-only PennyLane/Torch fit before the large QIP rerun."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_torch import fit_one


def main():
    data_path = ROOT / "results/submission_revision/datasets/development.npz"
    manifest = json.loads(data_path.with_suffix(".json").read_text())
    rows = np.asarray(manifest["nested_subsets"]["8"]["train_rows"], dtype=int)
    with np.load(data_path, allow_pickle=False) as data:
        batches = {}
        for split, indices in (("train", rows), ("validation", slice(None))):
            y = data[f"{split}_y"][indices]
            batches[split] = {"rho": torch.as_tensor(data[f"{split}_rho"][indices], dtype=torch.complex128),
                              "angles": torch.as_tensor(data[f"{split}_angles"][indices], dtype=torch.float64),
                              "y": torch.as_tensor(y, dtype=torch.float64), "y_numpy": y.copy()}
    torch.set_num_threads(1)
    output = {}
    for name in ("joint_exchange", "exchange_matched_qnn"):
        _, _, fit = fit_one(name, batches["train"], batches["validation"], 17,
                            maxiter=3, batch_size=40, val_every=1)
        output[name] = {"elapsed_seconds": fit["elapsed_seconds"],
                        "best_validation_mse": fit["best_validation_mse"],
                        "gradient_evaluations": fit["counters"]["gradient_evaluations"],
                        "stop_reason": fit["stop_reason"]}
    target = ROOT / "results/submission_revision/quantum_smoke.json"
    target.write_text(json.dumps({"status": "development smoke only; not a reported learning result",
                                  "training_states": 8, "training_pairs": 40, "validation_pairs": 96,
                                  "models": output}, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
