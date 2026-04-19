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
with h5py.File(NoPath / "GalerkinState.h5", "r") as file:
    NoState: dict[str, np.ndarray] = {
        key: np.real(cast(h5py.Dataset, file[key])[Noidx, ...])
        for key in file.keys()
        if key.startswith(("w", "T", "J"))
    } # shape: (nz, nx, nt, nens)

    x: np.ndarray = cast(h5py.Dataset, file["x"])[...]
    z: np.ndarray = cast(h5py.Dataset, file["z"])[...]

with h5py.File(CldPath / "GalerkinState.h5", "r") as file:
    CldState: dict[str, np.ndarray] = {
        key: np.real(cast(h5py.Dataset, file[key])[Cldidx, ...])
        for key in file.keys()
        if key.startswith(("w", "T", "J"))
    } # shape: (nz, nx, nt, nens)

# ====================================================
# Preprocessing 
# ====================================================

# Combining data
NoT: np.ndarray = NoState["T1"] + NoState["T2"]
NoJ: np.ndarray = NoState["J1"] + NoState["J2"]
Now: np.ndarray = NoState["w1"] + NoState["w2"]

CldT: np.ndarray = CldState["T1"] + CldState["T2"]
CldJ: np.ndarray = CldState["J1"] + CldState["J2"]
Cldw: np.ndarray = CldState["w1"] + CldState["w2"]

# Calculate EAPE budget
NoGen: np.ndarray = NoT * NoJ
NoDis: np.ndarray = -Now * NoT * (9.8/1004.5 - 0.0065)

CldGen: np.ndarray = CldT * CldJ
CldDis: np.ndarray = -Cldw * CldT * (9.8/1004.5 - 0.0065)

# Maximum at different time stamp
NoGen_max: np.ndarray = np.nanmax(NoGen, axis=(0, 1))
NoDis_max: np.ndarray = np.nanmax(NoDis, axis=(0, 1))

CldGen_max: np.ndarray = np.nanmax(CldGen, axis=(0, 1))
CldDis_max: np.ndarray = np.nanmax(CldDis, axis=(0, 1))

# Calculate mean and standard deviation
NoGen_max_mean: np.ndarray = NoGen_max.mean(axis=-1)
NoGen_max_std : np.ndarray = NoGen_max.std(axis=-1)
NoDis_max_mean: np.ndarray = NoDis_max.mean(axis=-1)
NoDis_max_std : np.ndarray = NoDis_max.std(axis=-1)

CldGen_max_mean: np.ndarray = CldGen_max.mean(axis=-1)
CldGen_max_std : np.ndarray = CldGen_max.std(axis=-1)
CldDis_max_mean: np.ndarray = CldDis_max.mean(axis=-1)
CldDis_max_std : np.ndarray = CldDis_max.std(axis=-1)

# ====================================================
# Visualization
# ====================================================
Time: np.ndarray = np.linspace(0, NoGen_max_mean.size//2, NoGen_max_mean.size)

# Temperature
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(Time, NoGen_max_mean, label="No Rad.")
ax.plot(Time, CldGen_max_mean, label="Cld Rad.")
ax.fill_between(Time, NoGen_max_mean-NoGen_max_std, NoGen_max_mean+NoGen_max_std, alpha=0.3)
ax.fill_between(Time, CldGen_max_mean-CldGen_max_std, CldGen_max_mean+CldGen_max_std, alpha=0.3)
ax.fill_betweenx([0, 2000], 26, 28, color="red", alpha=0.2, zorder=0)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.minorticks_on()
ax.set_xlim(0, 30)
ax.set_ylim(0, 250)
ax.set_xlabel("Time [day]")
ax.set_title(r"Generation (Blue: NoRad; Green: CldRad; K$^2$/day)")
# ax.legend(frameon=False, loc="best")

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/Gen_evo.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# # Convective heating 
# fig, ax = plt.subplots(1, 1, figsize=(9, 4))

# ax.plot(Time, NoJ_max_mean, label="No Rad.")
# ax.plot(Time, CldJ_max_mean, label="Cld Rad.")

# ax.fill_between(Time, NoJ_max_mean-NoJ_max_std, NoJ_max_mean+NoJ_max_std, alpha=0.3)
# ax.fill_between(Time, CldJ_max_mean-CldJ_max_std, CldJ_max_mean+CldJ_max_std, alpha=0.3)
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.minorticks_on()
# ax.set_xlim(0, 30)
# ax.set_ylim(0, None)
# ax.set_xlabel("Time [day]")
# ax.set_ylabel("Heating anomaly [K/day]")
# ax.legend(frameon=False, loc="best")

# plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/J_evo.png", dpi=600, bbox_inches="tight")
# plt.close(fig)