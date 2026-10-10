#!/usr/bin/env python3
"""Train and evaluate the exact-density quantum benchmark reproducibly."""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qem.quantum_models import (QuantumBatch, make_model, metrics, gradient_check,
                                 analytic_exchange_reference)


def fit_one(model, train, validation, seed, maxiter):
    start = time.perf_counter()
    initial = model.initial_parameters(seed, float(np.mean(train.y)))
    evaluations = 0
    validation_evaluations = 0
    best = {"mse": float("inf"), "parameters": initial.copy(), "iteration": 0}
    trajectory = []
    latest = {}

    def objective(parameters):
        nonlocal evaluations
        evaluations += 1
        loss, gradient = model.loss_and_gradient(parameters, train)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(f"Nonfinite objective for {model.name}, seed {seed}")
        latest.update(parameters=parameters.copy(), loss=loss)
        return loss, gradient

    def checkpoint(parameters, iteration):
        nonlocal validation_evaluations
        validation_evaluations += 1
        pred = model.predict(parameters, validation)
        val_mse = float(np.mean((pred - validation.y) ** 2))
        train_mse = latest.get("loss") if np.array_equal(latest.get("parameters"), parameters) else None
        trajectory.append({"iteration": iteration, "train_mse": train_mse,
                           "validation_mse": val_mse, "objective_evaluations": evaluations,
                           "seconds": time.perf_counter() - start})
        if val_mse < best["mse"]:
            best.update(mse=val_mse, parameters=parameters.copy(), iteration=iteration)

    checkpoint(initial, 0)
    iteration = 0
    def callback(parameters):
        nonlocal iteration
        iteration += 1
        checkpoint(parameters, iteration)
        if iteration % 30 == 0:
            print(f"{model.name} seed={seed} iteration={iteration} "
                  f"best_val_mse={best['mse']:.6g} elapsed={time.perf_counter()-start:.1f}s",
                  flush=True)

    result = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=model.bounds(),
                      callback=callback, options={"maxiter": maxiter, "ftol": 1e-12,
                                                  "gtol": 1e-8, "maxls": 24, "maxcor": 12})
    checkpoint(result.x, iteration + 1)
    return best["parameters"], {
        "seed": seed, "optimizer": "L-BFGS-B", "max_iterations": maxiter,
        "iterations": int(result.nit), "objective_evaluations": evaluations,
        "gradient_evaluations": int(result.njev), "validation_evaluations": validation_evaluations,
        "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
        "optimizer_message": str(result.message), "best_validation_iteration": int(best["iteration"]),
        "best_validation_mse": best["mse"], "elapsed_seconds": time.perf_counter() - start,
        "selection_rule": "lowest validation MSE among iterates; no test-based selection",
        "trajectory": trajectory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "benchmark.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "quantum")
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--seeds", default="17,31,73")
    parser.add_argument("--models", default="joint_exchange,no_commutator,two_copy_qnn,exchange_matched_qnn")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Add new model runs while preserving completed models in the output directory")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    names = args.models.split(",")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with np.load(args.data) as source:
        split_names = [name for name in ("train", "validation", "test_state", "test_context",
                                         "test_joint", "bell") if f"{name}_rho" in source]
        batches = {name: QuantumBatch.from_npz(source, name) for name in split_names}
    small = batches["train"]
    check_batch = QuantumBatch.from_arrays(small.rho[:3], small.context[:3], small.angles[:3],
                                           small.P[:3], small.y[:3])
    checks = {name: gradient_check(make_model(name), check_batch) for name in names}
    print("Analytic gradient checks:", json.dumps(checks), flush=True)
    if args.check_only:
        (args.output / "gradient_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        return
    report = {
        "schema_version": 1, "dataset": str(args.data), "seeds": seeds,
        "training_regime": "exact density matrix; exact labels and analytic gradients",
        "finite_shot_training_performed": False,
        "compiled_gate_budgets_equalized": False,
        "context_prior": "Both learned quantum models receive P(c) and the same trainable Euler-frame family.",
        "gradient_checks": checks, "models": {}, "analytic_reference": {}}
    weights = {}
    predictions = {f"{split}_y": batch.y for split, batch in batches.items()}
    previous_elapsed = 0.
    if args.resume and (args.output / "metrics.json").exists():
        existing = json.loads((args.output / "metrics.json").read_text())
        if Path(existing["dataset"]).resolve() != args.data.resolve():
            raise ValueError("Resume dataset must match the existing result dataset")
        if set(existing["seeds"]) != set(seeds):
            raise ValueError("Resume seeds must match the existing result seeds")
        for name in names:
            if name in existing["models"]:
                raise ValueError(f"Refusing to overwrite completed model {name} when resuming")
        previous_elapsed = float(existing.get("elapsed_seconds", 0.))
        existing["gradient_checks"].update(checks)
        report = existing
        with np.load(args.output / "weights.npz") as z:
            weights = {key: z[key].copy() for key in z.files}
        with np.load(args.output / "predictions.npz") as z:
            predictions = {key: z[key].copy() for key in z.files}
    for split, batch in batches.items():
        raw = analytic_exchange_reference(batch)
        report["analytic_reference"][split] = metrics(batch.y, raw["prediction"])
        for key in ("prediction", "heads", "correlation"):
            predictions[f"analytic_reference__{split}__{key}"] = raw[key]
    for name in names:
        model = make_model(name)
        if hasattr(model, "analytic_parameters"):
            reference_error = max(float(np.max(np.abs(model.predict(model.analytic_parameters(), batch) - batch.y)))
                                  for batch in batches.values())
            if reference_error > 1e-10:
                raise AssertionError(f"Analytic solution is not represented in {name}: {reference_error}")
            report.setdefault("representation_checks", {})[name] = {
                "analytic_solution_max_abs_error": reference_error,
                "analytic_parameters_used_for_training_initialization": False}
        runs = []
        for seed in seeds:
            parameters, fit = fit_one(model, batches["train"], batches["validation"], seed, args.maxiter)
            fit["architecture"] = model.describe(parameters)
            fit["metrics"] = {}
            for split, batch in batches.items():
                raw = model.raw(parameters, batch)
                fit["metrics"][split] = metrics(batch.y, raw["prediction"])
                # Validity of two commuting +/-1 heads, including joint correlations.
                probs = np.stack([(1 + u * raw["heads"][:, 0] + v * raw["heads"][:, 1] +
                                   u * v * raw["correlation"]) / 4
                                  for u, v in ((-1, -1), (-1, 1), (1, -1), (1, 1))], axis=1)
                if np.min(probs) < -2e-9 or np.max(probs) > 1 + 2e-9:
                    raise AssertionError(f"Nonphysical joint probabilities: {name}/{split}")
                for key in ("prediction", "heads", "correlation"):
                    predictions[f"{name}__seed{seed}__{split}__{key}"] = raw[key]
            weights[f"{name}__seed{seed}"] = parameters
            runs.append(fit)
            print(f"Completed {name} seed={seed}: "
                  f"val={fit['metrics']['validation']['mse']:.6g}, "
                  f"joint={fit['metrics']['test_joint']['mse']:.6g}, "
                  f"seconds={fit['elapsed_seconds']:.1f}", flush=True)
            # Checkpoint after each finished seed; interrupted runs retain completed work.
            report["models"][name] = {"runs": runs}
            (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            np.savez_compressed(args.output / "weights.npz", **weights)
            np.savez_compressed(args.output / "predictions.npz", **predictions)
        selected = min(runs, key=lambda run: run["metrics"]["validation"]["mse"])
        summary = {}
        for split in split_names:
            values = np.array([run["metrics"][split]["mse"] for run in runs])
            summary[split] = {"mse_mean_over_seeds": float(np.mean(values)),
                              "mse_std_over_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                              "selected_seed_mse": selected["metrics"][split]["mse"]}
        report["models"][name] = {"runs": runs, "selected_seed": selected["seed"],
                                   "seed_summary": summary,
                                   "architecture": model.describe(weights[f"{name}__seed{selected['seed']}"])}
        weights[f"{name}__selected"] = weights[f"{name}__seed{selected['seed']}"]
    report["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "weights.npz", **weights)
    np.savez_compressed(args.output / "predictions.npz", **predictions)
    print(f"Saved quantum results to {args.output}; elapsed={report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
