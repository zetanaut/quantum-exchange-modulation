#!/usr/bin/env python3
"""Aggregate the frozen QIP runs without selecting configurations by test error."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics


ROOT = Path("results/submission_revision")
DATASETS = [f"confirmation_{i}" for i in range(1, 6)]
BUDGETS = (8, 16, 32, 64)
STARTS = (17, 31, 73)


def load(path):
    return json.loads(path.read_text())


def run_rows():
    rows = []
    for dataset in DATASETS:
        products = load(ROOT / "product_baselines" / f"{dataset}.json")
        analytic_head = load(ROOT / "analytic_head" / f"{dataset}.json")
        for budget in BUDGETS:
            for name, details in products["budgets"][str(budget)]["models"].items():
                rows.append({"dataset": dataset, "budget": budget, "method": name,
                             "start": "analytic_or_linear_fit", "joint_mse": details["splits"]["test_joint"]["mse"],
                             "axis_only_mse": details["splits"]["test_context"]["mse"]})
            head = analytic_head["budgets"][str(budget)]["mse"]
            rows.append({"dataset": dataset, "budget": budget, "method": "analytic_port_head_3",
                         "start": "linear_fit", "joint_mse": head["test_joint"],
                         "axis_only_mse": head["test_context"]})
            for name in ("joint_exchange", "exchange_matched_qnn", "no_commutator", "joint_exchange_neutral"):
                for start in STARTS:
                    path = ROOT / "quantum_confirmation" / dataset / str(budget) / f"{name}__seed{start}.json"
                    if not path.exists():
                        continue
                    details = load(path)["metrics"]
                    rows.append({"dataset": dataset, "budget": budget, "method": name,
                                 "start": start, "joint_mse": details["test_joint"]["mse"],
                                 "axis_only_mse": details["test_context"]["mse"]})
            for name in load(ROOT / "classical_selection.json")["selected"]:
                for start in STARTS:
                    path = ROOT / "classical_confirmation" / dataset / str(budget) / f"{name}__seed{start}.json"
                    details = load(path)["metrics"]
                    rows.append({"dataset": dataset, "budget": budget, "method": name,
                                 "start": start, "joint_mse": details["test_joint"]["mse"],
                                 "axis_only_mse": details["test_context"]["mse"]})
        for start in STARTS:
            path = ROOT / "generic_fits" / dataset / f"lr0.01__seed{start}.json"
            details = load(path)["metrics"]
            rows.append({"dataset": dataset, "budget": 64, "method": "generic_source_conditioner_fixed_P_readout",
                         "start": start, "joint_mse": details["test_joint"]["mse"],
                         "axis_only_mse": details["test_context"]["mse"]})
            blind_path = ROOT / "context_blind_fits" / dataset / f"lr0.01__seed{start}.json"
            if blind_path.exists():
                blind = load(blind_path)["metrics"]
                rows.append({"dataset": dataset, "budget": 64, "method": "context_blind_source_fixed_P_readout",
                             "start": start, "joint_mse": blind["test_joint"]["mse"],
                             "axis_only_mse": blind["test_context"]["mse"]})
    return rows


def main():
    rows = run_rows()
    path = ROOT / "all_scores.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"scope": "Five fresh datasets; all optimizer starts retained; no test-based selection",
               "rows": len(rows), "methods": {}, "paired_exchange_minus_matched": {},
               "copy_risk_3840": {}, "multi_query_3840": {}}
    for budget in BUDGETS:
        names = sorted({r["method"] for r in rows if r["budget"] == budget})
        summary["methods"][str(budget)] = {}
        for name in names:
            dataset_scores = {dataset: statistics.mean(r["joint_mse"] for r in rows
                                  if r["budget"] == budget and r["dataset"] == dataset and r["method"] == name)
                              for dataset in DATASETS}
            values = list(dataset_scores.values())
            summary["methods"][str(budget)][name] = {"dataset_mean_joint_mse": dataset_scores,
                "mean_of_dataset_means": statistics.mean(values), "median_dataset_mean": statistics.median(values),
                "minimum_dataset_mean": min(values), "maximum_dataset_mean": max(values)}
        exchange = summary["methods"][str(budget)]["joint_exchange"]["dataset_mean_joint_mse"]
        matched = summary["methods"][str(budget)]["exchange_matched_qnn"]["dataset_mean_joint_mse"]
        paired = {dataset: exchange[dataset] - matched[dataset] for dataset in DATASETS}
        summary["paired_exchange_minus_matched"][str(budget)] = {"by_dataset": paired,
            "mean": statistics.mean(paired.values())}
    for dataset in DATASETS:
        risk = load(ROOT / "copy_risk" / f"{dataset}.json")
        summary["copy_risk_3840"][dataset] = {row["model"]: row["expected_mse"]
                                               for row in risk["rows"] if row["copies_per_query"] == 3840}
        amortized = load(ROOT / "amortization" / f"{dataset}.json")
        summary["multi_query_3840"][dataset] = {str(row["contexts"]) + "__" + row["method"]: row["expected_mse"]
                                                 for row in amortized["rows"]}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for method in ("joint_exchange", "exchange_matched_qnn", "gram_product_36",
                   "within_even__film", "generic_source_conditioner_fixed_P_readout"):
        item = summary["methods"]["64"][method]
        print(method, item["mean_of_dataset_means"], list(item["dataset_mean_joint_mse"].values()))


if __name__ == "__main__":
    main()
