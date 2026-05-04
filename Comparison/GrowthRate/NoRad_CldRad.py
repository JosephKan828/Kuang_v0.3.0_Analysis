# ====================================================
# This file is to compare growth rate derived from 
# linear growth rate in NoRad and CldRad simulations
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

from pathlib import Path
from typing import cast

from matplotlib import pyplot as plt

style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Load data
# ====================================================
file_path: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")

# Load for NoRad simulations
with h5py.File(file_path / "Rad(0.0,0.0,0.0)/latest/EigenAnalysis.h5", "r") as file:

    k: np.ndarray = cast(h5py.Dataset, file["k"])[...]

    norad_growth: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[...]

# Load for CldRad simulations
with h5py.File(file_path / "Rad(0.0,0.0,0.1)/latest/EigenAnalysis.h5", "r") as file:

    cldrad_growth: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[...]

# Modify wavenumber
k_display: np.ndarray = 40000/(2*np.pi*4320) * k

# ====================================================
# Calculate maximum and decrease of growth rate
# ====================================================

# Acquire maximum growth rate of different simulation
norad_growth_max : np.ndarray = np.nanmax(norad_growth[:, 0])
cldrad_growth_max: np.ndarray = np.nanmax(cldrad_growth[:, 0])

# Relative difference
growth_rate_diff: np.ndarray = (cldrad_growth_max - norad_growth_max) / norad_growth_max

# ====================================================
# Visualization
# ====================================================

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

ax.plot(k_display, norad_growth[:, 0], linewidth=4, label=f"No Rad. max: {norad_growth_max:.3f}")
ax.plot(k_display, cldrad_growth[:, 0], linewidth=4, label=f"Cloud Rad. max: {cldrad_growth_max:.3f}")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlim(0, 30)
ax.set_ylim(0, 0.13)

ax.set_xlabel("Non-dimensional Wavenumber")
ax.set_title("Growth Rate [1/day]")

ax.legend(frameon=False)
plt.savefig(
    "/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/growthrate_diff.png",
    dpi=300, bbox_inches="tight"
    )
plt.close(fig)