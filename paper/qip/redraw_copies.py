"""Regenerate Fig. 4 from fresh copy-risk and amortization records."""
from pathlib import Path
import os
import json
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("QFILM_REPO", HERE.parents[1])) / "results/submission_revision"
DATASETS = [f"confirmation_{i}" for i in range(1, 6)]
COPIES = [240, 960, 3840, 15360]
QUERIES = [1, 8, 32]
COLORS = {"analytic_exchange_direct_K": "#202d3b", "joint_exchange": "#1565c0",
          "exchange_matched_qnn": "#d35400", "specified_eight_setting_Pauli": "#248f72",
          "reusable_12_Pauli_U_statistics": "#248f72"}
LABELS = {"analytic_exchange_direct_K": "Direct $K_P$ / analytic exchange",
          "joint_exchange": "Learned exchange", "exchange_matched_qnn": "Matched exchange QNN",
          "specified_eight_setting_Pauli": "Specified eight-setting Pauli",
          "reusable_12_Pauli_U_statistics": "Reusable 12-setting Pauli"}


def one_query(dataset, method, copies):
    rows = json.loads((ROOT / "copy_risk" / f"{dataset}.json").read_text())["rows"]
    if method in ("joint_exchange", "exchange_matched_qnn"):
        return np.mean([row["expected_mse"] for row in rows
                        if row["model"].startswith(method + "__seed") and row["copies_per_query"] == copies])
    return next(row["expected_mse"] for row in rows
                if row["model"] == method and row["copies_per_query"] == copies)


def multi_query(dataset, method, count):
    rows = json.loads((ROOT / "amortization" / f"{dataset}.json").read_text())["rows"]
    return next(row["expected_mse"] for row in rows if row["method"] == method and row["contexts"] == count)


plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), layout="constrained")
left_methods = ["analytic_exchange_direct_K", "joint_exchange", "exchange_matched_qnn",
                "specified_eight_setting_Pauli"]
for method in left_methods:
    matrix = np.asarray([[one_query(dataset, method, copies) for copies in COPIES] for dataset in DATASETS])
    mean = matrix.mean(axis=0)
    axes[0].plot(COPIES, mean, marker="o", linewidth=1.6, markersize=4.5,
                 color=COLORS[method], label=LABELS[method])
    axes[0].fill_between(COPIES, matrix.min(axis=0), matrix.max(axis=0),
                         color=COLORS[method], alpha=.09)
axes[0].set(xscale="log", yscale="log", xlim=(190, 19000), xlabel="Source copies per query",
            ylabel="Expected inference MSE", title="One query per state")
axes[0].set_xticks(COPIES, labels=["240", "960", "3,840", "15,360"])
axes[0].tick_params(axis="x", labelsize=8.5)
axes[0].grid(True, which="major", alpha=.25)
axes[0].legend(fontsize=7.1, frameon=False, loc="upper right")

right_methods = ["analytic_exchange_direct_K", "joint_exchange", "exchange_matched_qnn",
                 "reusable_12_Pauli_U_statistics"]
for method in right_methods:
    matrix = np.asarray([[multi_query(dataset, method, count) for count in QUERIES] for dataset in DATASETS])
    for x, outcomes in zip(QUERIES, matrix.T):
        axes[1].scatter(np.full_like(outcomes, x), outcomes, color=COLORS[method], alpha=.28, s=14)
    axes[1].plot(QUERIES, matrix.mean(axis=0), marker="o", linewidth=1.6, markersize=4.5,
                 color=COLORS[method], label=LABELS[method])
axes[1].set(xscale="log", yscale="log", xlim=(.8, 40), xticks=QUERIES,
            xlabel="Queries on the same source state", ylabel="Mean expected MSE per query",
            title="3,840 total copies per state")
axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[1].grid(True, which="major", alpha=.25)
axes[1].legend(fontsize=7.1, frameon=False, loc="upper left")
for extension in ("pdf", "png"):
    fig.savefig(HERE / f"Fig4.{extension}", dpi=250)
plt.close(fig)
