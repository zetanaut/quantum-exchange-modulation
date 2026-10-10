#!/usr/bin/env python3
"""Primary quantum training with PyTorch and actual PennyLane QNodes.

NumPy-trained parameters are never used for initialization.  The separate
NumPy implementation supplies reproducible random starts and architecture
metadata only; all training predictions and derivatives use PennyLane/Torch.
"""
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
import torch
import pennylane as qml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qem.quantum_models import make_model, metrics
from qem.pennylane_models import TorchQuantumModel


def architecture_for_torch(name, parameters, model=None):
    if model is None:
        model = TorchQuantumModel(name, initial_parameters=parameters)
    architecture = model.describe(parameters)
    architecture["training"] = "actual PennyLane QNodes, Torch backpropagation, torch.optim.LBFGS (optional Adam warmup)"
    return architecture


def slices(n, batch_size):
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def predict_raw(model, batch, batch_size):
    parts = {"prediction": [], "heads": [], "correlation": []}
    with torch.no_grad():
        for selection in slices(len(batch["y"]), batch_size):
            raw = model(batch["rho"][selection], batch["angles"][selection])
            for key in parts:
                parts[key].append(raw[key].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in parts.items()}


def fit_one(name, train, validation, seed, maxiter, batch_size, val_every,
            adam_steps=0, learning_rate=.02):
    torch.manual_seed(seed)
    reference_class = make_model(name)
    initialization = reference_class.initial_parameters(seed, float(train["y"].mean()))
    model = TorchQuantumModel(name, initial_parameters=initialization)
    start = time.perf_counter()
    counters = {"objective_evaluations": 0, "gradient_evaluations": 0,
                "training_forward_batches": 0, "validation_forward_batches": 0}
    best = {"validation_mse": float("inf"), "parameters": initialization.copy(),
            "iteration": 0, "phase": "initial"}
    trajectory = []
    latest = {}
    n = len(train["y"])

    def loss_and_backward():
        model.zero_grad(set_to_none=True)
        total = 0.
        counters["objective_evaluations"] += 1
        counters["gradient_evaluations"] += 1
        for selection in slices(n, batch_size):
            raw = model(train["rho"][selection], train["angles"][selection])
            loss = torch.sum((raw["prediction"] - train["y"][selection]) ** 2) / n
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {name}, seed {seed}")
            loss.backward()
            total += float(loss.detach())
            counters["training_forward_batches"] += 1
        if not torch.all(torch.isfinite(model.parameters_vector.grad)):
            raise FloatingPointError(f"Nonfinite gradient in {name}, seed {seed}")
        latest.update(train_mse=total,
                      gradient_max_abs=float(model.parameters_vector.grad.abs().max()))
        return torch.tensor(total, dtype=torch.float64)

    def checkpoint(iteration, phase):
        raw = predict_raw(model, validation, batch_size)
        counters["validation_forward_batches"] += int(np.ceil(len(validation["y"]) / batch_size))
        value = float(np.mean((raw["prediction"] - validation["y_numpy"]) ** 2))
        trajectory.append({"phase": phase, "iteration": iteration,
                           "validation_mse": value, "train_mse": latest.get("train_mse"),
                           "gradient_max_abs": latest.get("gradient_max_abs"),
                           "objective_evaluations": counters["objective_evaluations"],
                           "seconds": time.perf_counter() - start})
        if value < best["validation_mse"]:
            best.update(validation_mse=value,
                        parameters=model.parameters_vector.detach().cpu().numpy().copy(),
                        iteration=iteration, phase=phase)

    checkpoint(0, "initial")
    if adam_steps:
        optimizer = torch.optim.Adam([model.parameters_vector], lr=learning_rate)
        for iteration in range(1, adam_steps + 1):
            loss_and_backward()
            optimizer.step()
            if iteration % val_every == 0 or iteration == adam_steps:
                checkpoint(iteration, "adam")
            if iteration % 20 == 0:
                print(f"torch {name} seed={seed} Adam={iteration} "
                      f"best_val={best['validation_mse']:.6g} "
                      f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
    optimizer = torch.optim.LBFGS([model.parameters_vector], lr=1., max_iter=1,
                                  max_eval=24, tolerance_grad=1e-8,
                                  tolerance_change=1e-12, history_size=12,
                                  line_search_fn="strong_wolfe")
    stop_reason = "maximum LBFGS iterations reached"
    completed_iterations = 0
    for iteration in range(1, maxiter + 1):
        before = model.parameters_vector.detach().clone()
        optimizer.step(loss_and_backward)
        completed_iterations = iteration
        step_max = float((model.parameters_vector.detach() - before).abs().max())
        if iteration % val_every == 0 or iteration == maxiter:
            checkpoint(iteration, "lbfgs")
        if iteration % 20 == 0:
            print(f"torch {name} seed={seed} LBFGS={iteration} "
                  f"best_val={best['validation_mse']:.6g} "
                  f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
        if iteration >= 5 and (latest["gradient_max_abs"] < 1e-8 or step_max < 1e-12):
            stop_reason = "gradient tolerance reached" if latest["gradient_max_abs"] < 1e-8 else "parameter step tolerance reached"
            checkpoint(iteration, "lbfgs")
            break
    with torch.no_grad():
        model.parameters_vector.copy_(torch.as_tensor(best["parameters"], dtype=torch.float64))
    fit = {"seed": seed, "optimizer": "torch.optim.Adam then LBFGS" if adam_steps else "torch.optim.LBFGS",
           "random_initialization": True, "NumPy_fitted_weights_loaded": False,
           "initial_parameters": initialization.tolist(),
           "Adam_steps": adam_steps, "LBFGS_max_iterations": maxiter,
           "LBFGS_completed_iterations": completed_iterations,
           "stop_reason": stop_reason, "batch_size": batch_size,
           "batching": "full-dataset exact objective; chunk gradients accumulated before each optimizer update",
           "validation_every_iterations": val_every, "best_validation_mse": best["validation_mse"],
           "best_validation_iteration": best["iteration"], "best_validation_phase": best["phase"],
           "selection_rule": "minimum validation MSE among checkpoints; no test-based selection",
           "parameter_domain": "real angles and affine-head parameters; physical eta/chi coefficients use trigonometric parametrization",
           "elapsed_seconds": time.perf_counter() - start, "counters": counters,
           "trajectory": trajectory, "architecture": architecture_for_torch(name, best["parameters"], model)}
    return model, best["parameters"], fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "benchmark.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "torch_quantum")
    parser.add_argument("--models", default="joint_exchange,no_commutator,exchange_matched_qnn")
    parser.add_argument("--seeds", default="17,31,73")
    parser.add_argument("--maxiter", type=int, default=120)
    parser.add_argument("--adam-steps", type=int, default=0)
    parser.add_argument("--adam-lr", type=float, default=.02)
    parser.add_argument("--batch-size", type=int, default=320)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    args.output.mkdir(parents=True, exist_ok=True)
    names = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    started = time.perf_counter()
    with np.load(args.data) as data:
        split_names = [name for name in ("train", "validation", "test_state", "test_context",
                                         "test_joint", "bell") if f"{name}_rho" in data]
        batches = {name: {"rho": torch.as_tensor(data[f"{name}_rho"], dtype=torch.complex128),
                          "angles": torch.as_tensor(data[f"{name}_angles"], dtype=torch.float64),
                          "y": torch.as_tensor(data[f"{name}_y"], dtype=torch.float64),
                          "y_numpy": data[f"{name}_y"].copy()} for name in split_names}
    report = {"schema_version": 1, "dataset": str(args.data), "seeds": seeds,
              "framework": {"torch": torch.__version__, "pennylane": qml.__version__,
                             "numpy": np.__version__},
              "training_regime": "actual PennyLane quantum circuits with Torch interface and exact backpropagation",
              "device": "CPU; PennyLane default.qubit with exact purification of mixed inputs",
              "simulation_representation": "4 data wires plus 4 untouched environment wires; all trainable gates and readouts act only on data wires",
              "environment_wire_scope": "simulator representation only, not an extra input-copy or processing resource",
              "torch_threads": args.threads,
              "finite_shot_training_performed": False,
              "initialization_policy": "fresh independent seeded random starts; no fitted NumPy weights",
              "NumPy_role": "independent verification and architecture metadata; not training forward or backward",
              "models": {}}
    weights, torch_weights = {}, {}
    predictions = {f"{split}_y": batch["y_numpy"] for split, batch in batches.items()}
    previous_elapsed = 0.
    if args.resume and (args.output / "metrics.json").exists():
        report = json.loads((args.output / "metrics.json").read_text())
        if Path(report["dataset"]).resolve() != args.data.resolve() or set(report["seeds"]) != set(seeds):
            raise ValueError("Resume must use the same dataset and seed set")
        if any(name in report["models"] for name in names):
            raise ValueError("Resume adds model names and does not overwrite completed models")
        previous_elapsed = report.get("elapsed_seconds", 0.)
        with np.load(args.output / "weights.npz") as data:
            weights = {key: data[key].copy() for key in data.files}
        torch_weights = {key: torch.as_tensor(value.copy()) for key, value in weights.items()}
        with np.load(args.output / "predictions.npz") as data:
            predictions = {key: data[key].copy() for key in data.files}

    def save():
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(args.output / "weights.npz", **weights)
        np.savez_compressed(args.output / "predictions.npz", **predictions)
        torch.save(torch_weights, args.output / "weights.pt")

    for name in names:
        runs = []
        for seed in seeds:
            model, parameters, fit = fit_one(name, batches["train"], batches["validation"], seed,
                                             args.maxiter, args.batch_size, args.val_every,
                                             args.adam_steps, args.adam_lr)
            fit["metrics"] = {}
            for split, batch in batches.items():
                raw = predict_raw(model, batch, args.batch_size)
                fit["metrics"][split] = metrics(batch["y_numpy"], raw["prediction"])
                probabilities = np.stack([(1 + u * raw["heads"][:, 0] + v * raw["heads"][:, 1] +
                                           u * v * raw["correlation"]) / 4
                                          for u, v in ((-1, -1), (-1, 1), (1, -1), (1, 1))], axis=1)
                if np.min(probabilities) < -2e-8 or np.max(probabilities) > 1 + 2e-8:
                    raise AssertionError(f"Invalid two-head joint probabilities: {name}/{split}")
                for key, value in raw.items():
                    predictions[f"{name}__seed{seed}__{split}__{key}"] = value
            key = f"{name}__seed{seed}"
            weights[key] = parameters
            torch_weights[key] = model.parameters_vector.detach().cpu().clone()
            runs.append(fit)
            report["models"][name] = {"runs": runs}
            report["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
            save()
            print(f"Completed TORCH {name} seed={seed}: val={fit['metrics']['validation']['mse']:.6g} "
                  f"joint={fit['metrics']['test_joint']['mse']:.6g} seconds={fit['elapsed_seconds']:.1f}", flush=True)
        selected = min(runs, key=lambda run: run["metrics"]["validation"]["mse"])
        summary = {}
        for split in split_names:
            values = np.array([run["metrics"][split]["mse"] for run in runs])
            summary[split] = {"mse_mean_over_seeds": float(np.mean(values)),
                              "mse_std_over_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                              "selected_seed_mse": selected["metrics"][split]["mse"]}
        selected_parameters = weights[f"{name}__seed{selected['seed']}"]
        report["models"][name] = {"runs": runs, "selected_seed": selected["seed"],
                                   "seed_summary": summary,
                                   "architecture": architecture_for_torch(name, selected_parameters)}
        weights[f"{name}__selected"] = selected_parameters
        torch_weights[f"{name}__selected"] = torch.as_tensor(selected_parameters.copy())
        save()
    report["elapsed_seconds"] = previous_elapsed + time.perf_counter() - started
    save()
    print(f"Torch/PennyLane results saved; elapsed={report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
