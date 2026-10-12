#!/usr/bin/env python3
"""Check that the frozen QIP numerical record is complete before packaging."""
from __future__ import annotations

import csv
import json
import math
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
    assert math.isclose(summary["methods"]["64"]["joint_exchange"]["mean_of_dataset_means"],
                        1.563417611442566e-9, rel_tol=1e-10)
    assert summary["methods"]["8"]["gram_product_36"]["mean_of_dataset_means"] < 1e-25
    diagnostics = load(ROOT / "reviewer_diagnostics.json")
    aligned = diagnostics["initialization"]["joint_exchange"]
    neutral = diagnostics["initialization"]["joint_exchange_neutral"]
    assert aligned["starts"] == neutral["starts"] == 15
    assert aligned["validation_success_count"] == 15
    assert neutral["validation_success_count"] == 10
    risk = diagnostics["copy_risk_at_3840"]
    for record in risk.values():
        assert math.isclose(record["expected_mse"],
                            record["squared_bias"] + record["measurement_variance"],
                            rel_tol=1e-12, abs_tol=1e-12)
    assert risk["joint_exchange"]["measurement_variance"] > risk["analytic_exchange_direct_K"]["measurement_variance"]
    break_even = load(ROOT / "amortization_break_even.json")
    assert len(break_even["rows"]) == 10 and break_even["copies_per_state"] == 3840
    two, three = (break_even["five_dataset_means"][str(count)] for count in (2, 3))
    assert two["direct_K_expected_mse"] < two["reusable_Pauli_monte_carlo_mse"]
    assert three["direct_K_expected_mse"] > three["reusable_Pauli_monte_carlo_mse"]
    transfer = load(ROOT / "transfer_checks/finite_time.json")
    grid = load(ROOT / "transfer_checks/directional_response_map.json")
    selected = min((load(ROOT / "quantum_confirmation/confirmation_1/64" /
                         f"joint_exchange__seed{seed}.json") for seed in STARTS),
                   key=lambda item: item["fit"]["best_validation_mse"])
    assert transfer["checkpoint_seed"] == grid["checkpoint_seed"] == selected["seed"]
    assert transfer["dataset_sha256"] == load(ROOT / "datasets/confirmation_1.json")["dataset_sha256"]
    assert transfer["queries"] == 8064 and transfer["heldout_axis_min_abs_nx"] > .55
    with (ROOT / "transfer_checks/directional_response_map.csv").open(newline="") as source:
        grid_rows = list(csv.DictReader(source))
    assert len(grid_rows) == grid["grid_rows"] == 231
    grid_mse = sum(float(row["prediction_minus_exact"]) ** 2 for row in grid_rows) / len(grid_rows)
    assert math.isclose(grid_mse, grid["grid_mse"], rel_tol=1e-12)
    result = {"status": "PASS", "confirmation_datasets": 5, "quantum_fits": 240,
              "classical_confirmation_fits": 480, "generic_and_blind_fits": 30,
              "copy_risk_datasets": 5, "amortization_datasets": 5,
              "score_rows": summary["rows"], "new_protocol_transfer_queries": transfer["queries"],
              "neutral_validation_successes_of_15": neutral["validation_success_count"]}
    (ROOT / "release_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
