"""Regenerate Fig. 3 from the fresh physical-axis protocol direction map."""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("QFILM_REPO", ROOT.parents[1])) / "results/submission_revision/transfer_checks/directional_response_map.csv"
rows = np.genfromtxt(SOURCE, delimiter=",", names=True, dtype=None, encoding="utf-8")
theta = np.unique(rows["theta_rad"])
phi = np.unique(rows["phi_rad"])
assert len(theta) == 11 and len(phi) == 21 and len(rows) == 231
shape = (len(theta), len(phi))
exact = rows["D_exact"].reshape(shape)
prediction = rows["D_frozen_prediction"].reshape(shape)
error = rows["prediction_minus_exact"].reshape(shape)
assert np.max(np.abs(prediction - exact - error)) < 1e-12

plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
fig = plt.figure(figsize=(7.2, 6.5), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=[1, .9])
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
extent = [-180, 180, 0, 180]
vmin = min(exact.min(), prediction.min())
vmax = max(exact.max(), prediction.max())
limit = max(abs(error.min()), abs(error.max()))
for ax, values, title, cmap, low, high in zip(
    axes, [exact, prediction, error],
    [r"Exact $D_{P_c}$", "Frozen predictor", "Prediction minus exact"],
    ["viridis", "viridis", "coolwarm"], [vmin, vmin, -limit], [vmax, vmax, limit]
):
    image = ax.imshow(values, origin="lower", aspect="auto", extent=extent,
                      cmap=cmap, vmin=low, vmax=high)
    ax.set(title=title, xlabel=r"Azimuth $\varphi$ (degrees)", ylabel=r"Polar angle $\theta$ (degrees)")
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    fig.colorbar(image, ax=ax, shrink=.86)
for extension in ("pdf", "png"):
    fig.savefig(ROOT / f"Fig3.{extension}", dpi=250)
plt.close(fig)
