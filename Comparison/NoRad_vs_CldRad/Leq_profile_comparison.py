# ====================================================
# This script is to demonstrate difference in time tendency
# of convective heating between two experiments.
# ====================================================

# ====================================================
# Environment Setup
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
import h5py
import numpy as np
import tomllib as tl

from typing import cast
from pathlib import Path

from matplotlib import pyplot as plt

# Plot configuration
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Load data
# ====================================================

# file path
file_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Files")

Jeq: dict[str, np.ndarray] = {
    "NoRad" : np.load(file_path / "NoRad/2024-05-01_RUN01/Jeq.npz")["arr_0"],
    "CldRad": np.load(file_path / "CldRad/2024-05-01_RUN01/Jeq.npz")["arr_0"]
}

DJDt: dict[str, np.ndarray] = {
    "NoRad" : np.load(file_path / "NoRad/2024-05-01_RUN01/DJDt.npz")["arr_0"],
    "CldRad": np.load(file_path / "CldRad/2024-05-01_RUN01/DJDt.npz")["arr_0"]
}

# ====================================================
# Calculate statistics for equilibrium profile and 
# time tendency
# ====================================================

# Calculate ensemble statistics of J_eq for two experiments
Jeq_mean: dict[str, np.ndarray] = {
    key: value.mean(axis=1)
    for (key, value) in Jeq.items()
}

Jeq_std : dict[str, np.ndarray] = {
    key: value.std(axis=1)
    for (key, value) in Jeq.items()
}

# Calculate ensemble statistics of DJDt for two experiments
DJDt_mean: dict[str, np.ndarray] = {
    key: value.mean(axis=1)
    for (key, value) in DJDt.items()
}

DJDt_std : dict[str, np.ndarray] = {
    key: value.std(axis=1)
    for (key, value) in DJDt.items()
}

# ====================================================
# Visualization
# ====================================================

z: np.ndarray = np.linspace(0, 14000, 71)

# figure path
figure_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure")

# Plot for comparing equilibrium profile
fig, ax = plt.subplots(1, 1, figsize=(5, 9))

ax.plot(Jeq_mean["NoRad"], z, color="C0", linewidth=4, label="NoRad")
ax.fill_betweenx(
    z, Jeq_mean["NoRad"] - Jeq_std["NoRad"], Jeq_mean["NoRad"] + Jeq_std["NoRad"],
    color="C0", alpha=0.3
)

ax.plot(Jeq_mean["CldRad"], z, color="C2", linewidth=4, label="CldRad")
ax.fill_betweenx(
    z, Jeq_mean["CldRad"] - Jeq_std["CldRad"], Jeq_mean["CldRad"] + Jeq_std["CldRad"],
    color="C2", alpha=0.3
)

ax.set_xlim(-0.015, 0.015)
ax.set_ylim(0, 14000)
ax.set_xlabel("K/day")
ax.set_ylabel("Level [m]")
ax.set_title("Equilibrium profile comparison")
ax.legend(frameon=False, loc="best")

plt.savefig(
    figure_path / "NoRad_vs_CldRad/Jeq_compare.png",
    dpi = 300, bbox_inches="tight"
    )

plt.close(fig)

# Plot for time tendency of heating
fig, ax = plt.subplots(1, 1, figsize=(5, 9))

ax.plot(DJDt_mean["NoRad"], z, color="C0", linewidth=4, label="NoRad")
ax.fill_betweenx(
    z, DJDt_mean["NoRad"] - DJDt_std["NoRad"], DJDt_mean["NoRad"] + DJDt_std["NoRad"],
    color="C0", alpha=0.3
)

ax.plot(DJDt_mean["CldRad"], z, color="C2", linewidth=4, label="CldRad")
ax.fill_betweenx(
    z, DJDt_mean["CldRad"] - DJDt_std["CldRad"], DJDt_mean["CldRad"] + DJDt_std["CldRad"],
    color="C2", alpha=0.3
)

ax.set_xlim(-0.05, 0.05)
ax.set_ylim(0, 14000)
ax.set_xlabel("K/day")
ax.set_ylabel("Level [m]")
ax.set_title("Time Tendency profile comparison")
ax.legend(frameon=False, loc="best")

plt.savefig(
    figure_path / "NoRad_vs_CldRad/DJDt_compare.png",
    dpi = 300, bbox_inches="tight"
    )

plt.close(fig)