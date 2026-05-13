# ====================================================
# This script is to show the relation between convective
# heating and vertical motion
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

# Path setting# Setup path for field data
HomePath: Path = Path("/home/b11209013/Kuang2008_v0.3.0/Config")
NoPath: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full/Rad(0.0,0.0,0.0)/2026-05-08_RUN01")
CldPath: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full/Rad(0.0,0.0,0.1)/2026-05-08_RUN01")
FigPath : Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad")

## Load configuration parameters
with open(HomePath / "ModelParams.toml", "rb") as file:
    parameter = tl.load(file)

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

# Load State data
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
# Calculate convective heating
# ====================================================

# parameters
r0: float = parameter["Convection"]["Heating"]["r0"]
rq: float = parameter["Convection"]["Heating"]["rq"]

# Caculate for upper-level convective heating
NoRad_U : np.ndarray = r0 * NoState["L"] + rq * (NoState["q"] - 1.5*NoState["T1"])
CldRad_U: np.ndarray = r0 * CldState["L"] + rq * (CldState["q"] - 1.5*CldState["T1"])

# Calculate convective heating for 1st and 2nd modes
NoRad_J1: np.ndarray = NoState["L"] + NoRad_U
NoRad_J2: np.ndarray = NoState["L"] - NoRad_U

CldRad_J1: np.ndarray = CldState["L"] + CldRad_U
CldRad_J2: np.ndarray = CldState["L"] - CldRad_U

# ====================================================
# Normalize the convective heating and vertical motion
# ====================================================

NoRad_w1_norm: np.ndarray = NoState["w1"] / np.max(np.abs(NoState["w1"]))
NoRad_w2_norm: np.ndarray = NoState["w2"] / np.max(np.abs(NoState["w2"]))
CldRad_w1_norm: np.ndarray = CldState["w1"] / np.max(np.abs(CldState["w1"]))
CldRad_w2_norm: np.ndarray = CldState["w2"] / np.max(np.abs(CldState["w2"]))

NoRad_J1_norm: np.ndarray = NoRad_J1 / np.max(np.abs(NoRad_J1))
NoRad_J2_norm: np.ndarray = NoRad_J2 / np.max(np.abs(NoRad_J2))
CldRad_J1_norm: np.ndarray = CldRad_J1 / np.max(np.abs(CldRad_J1))
CldRad_J2_norm: np.ndarray = CldRad_J2 / np.max(np.abs(CldRad_J2))

# ====================================================
# Create scatter plot for the relations
# ====================================================

fig, ax = plt.subplots(figsize=(6, 6))

plt.scatter(NoRad_w1_norm, NoRad_J1_norm)
plt.scatter(CldRad_w1_norm, CldRad_J1_norm)

plt.savefig(FigPath / "J_w_Relation" / "w1_J1_relation.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))

plt.scatter(NoRad_w2_norm, NoRad_J2_norm)
plt.scatter(CldRad_w2_norm, CldRad_J2_norm)

plt.savefig(FigPath / "J_w_Relation" / "w2_J2_relation.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))

plt.scatter(NoRad_w1_norm, NoRad_J2_norm)
plt.scatter(CldRad_w1_norm, CldRad_J2_norm)

plt.savefig(FigPath / "J_w_Relation" / "w1_J2_relation.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))

plt.scatter(NoRad_w2_norm, NoRad_J1_norm)
plt.scatter(CldRad_w2_norm, CldRad_J1_norm)

plt.savefig(FigPath / "J_w_Relation" / "w2_J1_relation.png", dpi=300)
plt.close(fig)