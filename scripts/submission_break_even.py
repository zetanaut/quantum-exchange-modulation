#!/usr/bin/env python3
"""Exploratory two- and three-query copy-risk check using the frozen query pool."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from submission_amortization import query_directions, classical_trials
from qem.data import fragility, observable

RESULT = ROOT / "results/submission_revision"
plan = json.loads((RESULT / "amortization_plan.json").read_text())
copies = plan["total_source_copies_per_state"]
rows = []
for count in (2, 3):
    for i in range(5):
        dataset = f"confirmation_{i+1}"
        with np.load(RESULT / "datasets" / f"{dataset}.npz", allow_pickle=False) as data:
            ids, ix = np.unique(data["test_joint_state_id"], return_index=True)
            rho = data["test_joint_rho"][ix]
        n = query_directions(plan["context_seed"] + i, count, .55)
        flat_rho = np.repeat(rho, count, axis=0)
        flat_n = np.tile(n, (len(rho), 1))
        P = observable(flat_n)
        y = fragility(flat_rho, P).reshape(len(rho), count)
        m = np.einsum("nij,nji->n", flat_rho, P).real.reshape(len(rho), count)
        direct = float(np.mean((4 * (1 - m * m) - 2 * y * y) / (copies / count)))
        trials = classical_trials(rho, n, copies, plan["monte_carlo_replicates"],
                                  plan["context_seed"] + 1000 + i)
        replicate_risk = np.mean((trials - y[None]) ** 2, axis=(1, 2))
        rows.append({"dataset": dataset, "queries": count,
                     "direct_K_expected_mse": direct,
                     "reusable_Pauli_monte_carlo_mse": float(replicate_risk.mean()),
                     "reusable_monte_carlo_standard_error":
                         float(replicate_risk.std(ddof=1) / np.sqrt(len(replicate_risk)))})
summary = {str(count): {"direct_K_expected_mse": float(np.mean([
                       r["direct_K_expected_mse"] for r in rows if r["queries"] == count])),
                        "reusable_Pauli_monte_carlo_mse": float(np.mean([
                       r["reusable_Pauli_monte_carlo_mse"] for r in rows if r["queries"] == count]))}
           for count in (2, 3)}
result = {"scope": "exploratory integer break-even check after fixed 1/8/32 study",
          "copies_per_state": copies,
          "query_pool": "first 2 or 3 axes from the frozen 32-query pool in each dataset",
          "replicates_per_dataset": plan["monte_carlo_replicates"],
          "rows": rows, "five_dataset_means": summary}
destination = RESULT / "amortization_break_even.json"
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(summary, indent=2))
