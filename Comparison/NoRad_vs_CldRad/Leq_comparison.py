# ====================================================
# This program is to calculate time evolution of 
# Leq and L evolution
# ====================================================

# ====================================================
# Import package
# ====================================================
import h5py
import numpy as np
import tomllib as tl

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
## Load configuration parameters
with open(HomePath / "ModelParams.toml", "rb") as file:
    parameter = tl.load(file)

cqe_param = parameter["Convection"]["CQE"]
heating_params = parameter["Convection"]["Heating"]

print(parameter)
### Calculate parameters
A = 1 - 2*cqe_param["f"] + (cqe_param["b2"] - cqe_param["b1"])/cqe_param["F"]
B = 1 + (cqe_param["b2"] + cqe_param["b1"])/cqe_param["F"] - A * heating_params["r0"]

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
        if key.startswith(("T", "w", "q", "L"))
    }

with h5py.File(CldPath / "State.h5", "r") as file:
    CldState: dict[str, np.ndarray] = {
        key: np.abs(cast(h5py.Dataset, file[key])[Cldidx, ...])
        for key in file.keys()
        if key.startswith(("T", "w", "q", "L"))
    }

# ====================================================
# Preprocessing 
# ====================================================

# Calculate Leq in the two simulations
NoLeq: np.ndarray = (A*heating_params["rq"]*(NoState["q"]-1.5*NoState["T1"]) + cqe_param["f"]*NoState["w1"] + (1-cqe_param["f"])*NoState["w2"])/B
CldLeq: np.ndarray = (A*heating_params["rq"]*(CldState["q"]-1.5*CldState["T1"]) + cqe_param["f"]*CldState["w1"] + (1-cqe_param["f"])*CldState["w2"])/B

# Maximum at different time stamp

# Calculate mean and standard deviation
NoLeq_max_mean: np.ndarray = NoLeq.mean(axis=-1)
NoLeq_max_std : np.ndarray = NoLeq.std(axis=-1)

CldLeq_max_mean: np.ndarray = CldLeq.mean(axis=-1)
CldLeq_max_std : np.ndarray = CldLeq.std(axis=-1)

NoL_max_mean: np.ndarray = NoState["L"].mean(axis=-1)
NoL_max_std : np.ndarray = NoState["L"].std(axis=-1)

CldL_max_mean: np.ndarray = CldState["L"].mean(axis=-1)
CldL_max_std : np.ndarray = CldState["L"].std(axis=-1)


# ====================================================
# Visualization
# ====================================================
Time: np.ndarray = np.linspace(0, NoL_max_mean.size//2, NoL_max_mean.size)

# Temperature
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(Time, NoL_max_mean, label="No Rad.", linewidth=4)
ax.plot(Time, CldL_max_mean, label="Cld Rad.", color="deeppink", linewidth=4)
ax.fill_between(Time, NoL_max_mean-NoL_max_std, NoL_max_mean+NoL_max_std, alpha=0.3)
ax.fill_between(Time, CldL_max_mean-CldL_max_std, CldL_max_mean+CldL_max_std, alpha=0.3, color="deeppink")
# ax.fill_betweenx([0, 2000], 26, 28, color="red", alpha=0.2, zorder=0)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.minorticks_on()
ax.set_xlim(0, 30)
# ax.set_ylim(0, 10)
ax.set_xlabel("Time [day]")
ax.set_title("Convective heating (K/day)")
ax.legend(frameon=False, loc="best")

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/L_evo.png", dpi=600, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(Time, NoLeq_max_mean, label="No Rad.", linewidth=4)
ax.plot(Time, CldLeq_max_mean, label="Cld Rad.", color="deeppink", linewidth=4)
ax.fill_between(Time, NoLeq_max_mean-NoLeq_max_std, NoLeq_max_mean+NoLeq_max_std, alpha=0.3)
ax.fill_between(Time, CldLeq_max_mean-CldLeq_max_std, CldLeq_max_mean+CldLeq_max_std, alpha=0.3, color="deeppink")
# ax.fill_betweenx([0, 2000], 26, 28, color="red", alpha=0.2, zorder=0)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.minorticks_on()
ax.set_xlim(0, 30)
# ax.set_ylim(0, 10)
ax.set_xlabel("Time [day]")
ax.set_title(r"Equilibrium Convective Heating (K/day)")
ax.legend(frameon=False, loc="best")

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/Leq_evo.png", dpi=600, bbox_inches="tight")
plt.close(fig)