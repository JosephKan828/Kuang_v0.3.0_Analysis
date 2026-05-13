# ====================================================
# This script is to compare vertical motion in convection
# regions of waves
# ====================================================

# ====================================================
# Environment setup
# ====================================================

# limit the usage of CPU
import os
THREAD_VAR = [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
]

for var in THREAD_VAR:
    os.environ[var]="1"

# Import package
import numpy as np

from typing import cast
from pathlib import Path

from matplotlib import pyplot as plt

# Plot configuration
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Load data in NoRad and CLdRad experiments
# ====================================================

# file path
file_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Files")

# Load NoRad data
NoRad: dict[str, np.ndarray] = dict(np.load(file_path / "NoRad/2024-05-01_RUN01/Composite/w_westward.npz"))

# Load CldRad data
CldRad: dict[str, np.ndarray] = dict(np.load(file_path / "CldRad/2024-05-01_RUN01/Composite/w_westward.npz"))

# ====================================================
# Calculate ensemble mean and horizontal mean profile 
# ====================================================

# calculate horizontal mean
NoRad_phase_avg: np.ndarray = np.array([
    item[:, :item.shape[1]//2].mean(axis=1)
    for item in NoRad.values()
])

CldRad_phase_avg: np.ndarray = np.array([
    item[:, :item.shape[1]//2].mean(axis=1)
    for item in CldRad.values()
])

# Calculate statistics among ensemble
NoRad_mean: np.ndarray = NoRad_phase_avg.mean(axis=0)
NoRad_std : np.ndarray = NoRad_phase_avg.std(axis=0)

CldRad_mean: np.ndarray = CldRad_phase_avg.mean(axis=0)
CldRad_std : np.ndarray = CldRad_phase_avg.std(axis=0)

# ====================================================
# Visualization 
# ====================================================

# Design coordinate
z: np.ndarray = np.linspace(0, 14000, 71)

# Plot for comparing relative magnitude
fig, ax = plt.subplots(1, 1, figsize=(5, 9))

ax.plot(NoRad_mean, z, linewidth=4, label="No Rad.")
ax.fill_betweenx(
    z, NoRad_mean - NoRad_std, NoRad_mean + NoRad_std,
    alpha=0.3
)

ax.plot(CldRad_mean, z, linewidth=4, label="Cloud Rad.")
ax.fill_betweenx(
    z, CldRad_mean - CldRad_std, CldRad_mean + CldRad_std,
    alpha=0.3
)

ax.set_xlim(-2e-5, 8e-5)
ax.set_ylim(0, 14000)
ax.set_xlabel("Vertical Motion (m/s)")
ax.set_ylabel("Level [m]")
ax.legend(frameon=False, loc="best")
plt.savefig(
    "/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/vertical_motion_compare.png",
    dpi=300,
    bbox_inches="tight"
    )
plt.close(fig)

# Plot for bottom-heaviness
fig, ax = plt.subplots(1, 1, figsize=(5, 9))

ax.plot(NoRad_mean / NoRad_mean.max(), z, linewidth=4, label="No Rad.")
ax.plot(CldRad_mean / CldRad_mean.max(), z, linewidth=4, label="Cloud Rad.")
ax.hlines(z[np.nanargmax(NoRad_mean)], xmin=0, xmax=1, color="C0", linewidth=3, linestyles="--")
ax.hlines(z[np.nanargmax(CldRad_mean)], xmin=0, xmax=1, color="C1", linewidth=3, linestyles="--")
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 14000)
ax.set_xlabel("Relative Magnitude of Vertical Motion")
ax.set_ylabel("Level [m]")
ax.legend(frameon=False, loc="best")
plt.savefig(
    "/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/vertical_motion_norm_compare.png",
    dpi=300,
    bbox_inches="tight"
    )
plt.close(fig)