#!/usr/bin/env python3
"""Check that the frozen QIP numerical record is complete before packaging."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("results/submission_revision")
DATASETS = [f"confirmation_{i}" for i in range(1, 6)]
BUDGETS = (8, 16, 32, 64)
STARTS = (17, 31, 73)
QUANTUM = ("joint_exchange", "exchange_matched_qnn", "no_commutator", "joint_exchange_neutral")


def load(path: Path):
    assert path.is_file(), path
    return json.loads(path.read_text())


def main():
    datasets = load(ROOT / "dataset_checks.json")
    checkpoints = load(ROOT / "classical_checkpoint_checks.json")
    generic_checks = load(ROOT / "generic_checks.json")
    summary = load(ROOT / "summary.json")
    assert checkpoints["status"] == "PASS" and checkpoints["checkpoints"] == 480
    assert generic_checks["status"] == "PASS"
    assert datasets["status"] == "PASS"
    assert len(DATASETS) == 5
    selected = load(ROOT / "classical_selection.json")["selected"]
    assert len(selected) == 8
    for dataset in DATASETS:
        load(ROOT / "datasets" / f"{dataset}.json")
        for folder in ("product_baselines", "analytic_head", "copy_risk", "copy_risk_ablations", "amortization"):
            load(ROOT / folder / f"{dataset}.json")
        for start in STARTS:
            for folder in ("generic_fits", "context_blind_fits"):
                load(ROOT / folder / dataset / f"lr0.01__seed{start}.json")
        for budget in BUDGETS:
            for model in QUANTUM:
                for start in STARTS:
                    record = load(ROOT / "quantum_confirmation" / dataset / str(budget) / f"{model}__seed{start}.json")
                    assert record["checkpoint_reload_max_abs"] < 1e-12
            for model in selected:
                for start in STARTS:
                    load(ROOT / "classical_confirmation" / dataset / str(budget) / f"{model}__seed{start}.json")
    assert summary["rows"] == 5 * (4 * (5 + 1 + 4 * 3 + 8 * 3) + 2 * 3)
    result = {"status": "PASS", "confirmation_datasets": 5, "quantum_fits": 240,
              "classical_confirmation_fits": 480, "generic_and_blind_fits": 30,
              "copy_risk_datasets": 5, "amortization_datasets": 5,
              "score_rows": summary["rows"]}
    (ROOT / "release_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
