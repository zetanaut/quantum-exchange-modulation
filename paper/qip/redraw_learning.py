"""Regenerate Fig. 2 from five independent axis-holdout datasets."""
from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("QFILM_REPO", HERE.parents[1])) / "results/submission_revision"
BUDGETS = [8, 16, 32, 64]
DATASETS = [f"confirmation_{i}" for i in range(1, 6)]
SEEDS = [17, 31, 73]


def values(method, budget):
    rows = []
    for dataset in DATASETS:
        per_start = []
        for seed in SEEDS:
            group = "quantum_confirmation" if method in ("joint_exchange", "exchange_matched_qnn") else "classical_confirmation"
            path = ROOT / group / dataset / str(budget) / f"{method}__seed{seed}.json"
            report = json.loads(path.read_text())
            per_start.append(report["metrics"]["test_joint"]["mse"])
        rows.append(np.mean(per_start))
    return np.asarray(rows)


plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.65), layout="constrained")
panels = [
    (axes[0], [("joint_exchange", "Compact exchange", "#1565c0"),
               ("exchange_matched_qnn", "Matched exchange QNN", "#d35400")],
     "Quantum models", (1e-10, 3e-5)),
    (axes[1], [("within_even__mlp", "Product-feature MLP", "#248f72"),
               ("within_even__film", "Product-feature FiLM", "#7e57a5"),
               ("raw_signed__mlp", "Raw Pauli MLP", "#6d737d")],
     "Classical neural controls", (1e-3, .2)),
]
for ax, models, title, limits in panels:
    for offset, (key, label, color) in enumerate(models):
        array = np.stack([values(key, budget) for budget in BUDGETS])
        means = array.mean(axis=1)
        xs = np.asarray(BUDGETS, dtype=float) * (1 + (offset - (len(models)-1)/2) * .025)
        for x, outcomes in zip(xs, array):
            ax.scatter(np.full_like(outcomes, x), outcomes, color=color, alpha=.36, s=16, linewidths=0)
        ax.plot(xs, means, marker="o", markersize=5, linewidth=1.7, color=color, label=label)
    ax.set(xscale="log", yscale="log", xticks=BUDGETS, xlim=(6.8, 77), ylim=limits,
           xlabel="Training source states", title=title)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="major", alpha=.25)
    ax.legend(fontsize=8.2, loc="best", frameon=False)
axes[0].set_ylabel("Joint physical-axis test MSE")
axes[1].set_ylabel("Joint physical-axis test MSE")
for extension in ("pdf", "png"):
    fig.savefig(HERE / f"Fig2.{extension}", dpi=250)
plt.close(fig)
