# ====================================================
# This program is to calculate time evolution of 
# temperature and convective heating
# ====================================================

# ====================================================
# Import package
# ====================================================
import h5py
import numpy as np

from typing import cast
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# using style sheet
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Load data
# ====================================================

# Setup path for field data
HomePath: Path = Path("/home/b11209013/Kuang2008_v0.3.0/Config")
NoPath: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full/Rad(0.0,0.0,0.0)/2026-04-16_RUN05")
CldPath: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full/Rad(0.0,0.0,0.1)/2026-04-16_RUN05")
FigPath : Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad")

# Load growth rate data for identifying most-unstable mode
## Load growth rate
with h5py.File(NoPath / "EigenAnalysis.h5", "r") as file:
    No_GR: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]
    No_PS: np.ndarray = cast(h5py.Dataset, file["PhaseSpeeds"])[:, 0]
    k          : np.ndarray = cast(h5py.Dataset, file["k"])[...]

with h5py.File(CldPath / "EigenAnalysis.h5", "r") as file:
    Cld_GR: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]
    Cld_PS: np.ndarray = cast(h5py.Dataset, file["PhaseSpeeds"])[:, 0]

## identifying most unstable mode
Noidx       : int = np.argmax(No_GR).astype(int)
Cldidx      : int = np.argmax(Cld_GR).astype(int)
Nok         : float = k[Noidx]
Cldk        : float = k[Cldidx]

# Load Galerkin data
with h5py.File(NoPath / "State.h5", "r") as file:
    NoState: dict[str, np.ndarray] = {
        key: np.abs(cast(h5py.Dataset, file[key])[Noidx, ...])
        for key in file.keys()
        if key.startswith(("w"))
    }

Now1: np.ndarray = np.abs(NoState["w1"])
Now2: np.ndarray = np.abs(NoState["w2"])

No_ratio: np.ndarray = Now2 / Now1

with h5py.File(CldPath / "State.h5", "r") as file:
    CldState: dict[str, np.ndarray] = {
        key: np.abs(cast(h5py.Dataset, file[key])[Cldidx, ...])
        for key in file.keys()
        if key.startswith(("w"))
    }

Cldw1: np.ndarray = np.abs(CldState["w1"])
Cldw2: np.ndarray = np.abs(CldState["w2"])

Cld_ratio: np.ndarray = Cldw2 / Cldw1

# ====================================================
# Preprocessing 
# ====================================================

# Maximum at different time stamp

# Calculate mean and standard deviation
Now1_max_mean: np.ndarray = Now1.mean(axis=-1)
Now1_max_std : np.ndarray = Now1.std(axis=-1)

Now2_max_mean: np.ndarray = Now2.mean(axis=-1)
Now2_max_std : np.ndarray = Now2.std(axis=-1)

No_ratio_mean: np.ndarray = No_ratio.mean(axis=-1)
No_ratio_std : np.ndarray = No_ratio.std(axis=-1)

Cldw1_max_mean: np.ndarray = Cldw1.mean(axis=-1)
Cldw1_max_std : np.ndarray = Cldw1.std(axis=-1)

Cldw2_max_mean: np.ndarray = Cldw2.mean(axis=-1)
Cldw2_max_std : np.ndarray = Cldw2.std(axis=-1)

Cld_ratio_mean: np.ndarray = Cld_ratio.mean(axis=-1)
Cld_ratio_std : np.ndarray = Cld_ratio.std(axis=-1)

# ====================================================
# Visualization
# ====================================================
Time: np.ndarray = np.linspace(0, Now1_max_mean.size//2, Now1_max_mean.size)

# Temperature
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(Time, Now1_max_mean, label="No Rad.", linewidth=4)
ax.plot(Time, Cldw1_max_mean, label="Cld Rad.", color="deeppink", linewidth=4)
ax.fill_between(Time, Now1_max_mean-Now1_max_std, Now1_max_mean+Now1_max_std, alpha=0.3)
ax.fill_between(Time, Cldw1_max_mean-Cldw1_max_std, Cldw1_max_mean+Cldw1_max_std, alpha=0.3, color="deeppink")
# ax.fill_betweenx([0, 2000], 26, 28, color="red", alpha=0.2, zorder=0)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.minorticks_on()
ax.set_xlim(0, 30)
# ax.set_ylim(0, 10)
ax.set_xlabel("Time [day]")
ax.set_title(r"$w_1$ [K/day]")
ax.legend(frameon=False, loc="best")

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/w1_evo.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# Temperature
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(Time, Now2_max_mean, label="No Rad.", linewidth=4)
ax.plot(Time, Cldw2_max_mean, label="Cld Rad.", color="deeppink", linewidth=4)
ax.fill_between(Time, Now2_max_mean-Now2_max_std, Now2_max_mean+Now2_max_std, alpha=0.3)
ax.fill_between(Time, Cldw2_max_mean-Cldw2_max_std, Cldw2_max_mean+Cldw2_max_std, alpha=0.3, color="deeppink")
# ax.fill_betweenx([0, 2000], 26, 28, color="red", alpha=0.2, zorder=0)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.minorticks_on()
ax.set_xlim(0, 30)
# ax.set_ylim(0, 10)
ax.set_xlabel("Time [day]")
ax.set_title(r"$w_2$ [K/day]")
ax.legend(frameon=False, loc="best")

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/w2_evo.png", dpi=600, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(Time, No_ratio_mean, label="No Rad.", linewidth=4)
ax.plot(Time, Cld_ratio_mean, label="Cld Rad.", color="deeppink", linewidth=4)
ax.fill_between(Time, No_ratio_mean-No_ratio_std, No_ratio_mean+No_ratio_std, alpha=0.3)
ax.fill_between(Time, Cld_ratio_mean-Cld_ratio_std, Cld_ratio_mean+Cld_ratio_std, alpha=0.3, color="deeppink")
# ax.fill_betweenx([0, 2000], 26, 28, color="red", alpha=0.2, zorder=0)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.minorticks_on()
ax.set_xlim(0, 30)
ax.set_ylim(0, 7)
ax.set_xlabel("Time [day]")
ax.set_title(r"$w_2 / w_1$")
ax.legend(frameon=False, loc="best")

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/w2_w1_evo.png", dpi=600, bbox_inches="tight")
plt.close(fig)