# ====================================================
# This program is to compare EAPE generation and 
# dissipation between NoRad and CldRad
# ====================================================

# ====================================================
# Import package
# ====================================================
import h5py
import numpy as np
import pandas as pd

from glob import glob
from typing import cast
from pathlib import Path
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

# using style sheet
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Load data
# ====================================================

# Path for input file
NoPath: str = "/home/b11209013/Kuang2008_v0.3.0_Analysis/Files/NoRad/Rolled"
CldPath: str = "/home/b11209013/Kuang2008_v0.3.0_Analysis/Files/CldRad/Rolled"

NoEnsList = glob(NoPath +"/Ens*")
CldEnsList = glob(CldPath + "/Ens*")

# Load data
NoState = {}
CldState = {}

for EnsName in NoEnsList:
	NoState[EnsName.split("/")[-1]] = {}
	
	for file in (glob(EnsName+"/J_day2*.csv") + glob(EnsName+"/T_day2*.csv")):
		NoState[EnsName.split("/")[-1]][file.split("/")[-1].split(".")[0]] = np.array(pd.read_csv(file))[:, 1:]
	
for EnsName in CldEnsList:
	CldState[EnsName.split("/")[-1]] = {}
	
	for file in (glob(EnsName+"/J_day2*.csv") + glob(EnsName+"/T_day2*.csv")):
		CldState[EnsName.split("/")[-1]][file.split("/")[-1].split(".")[0]] = np.array(pd.read_csv(file))[:, 1:]

# ====================================================
# Collecting profile of different ensemble and time
# ====================================================

# pre-allocate the list 
NoJ: list = []; CldJ: list = []
NoT: list = []; CldT: list = []

# for loop for saving
for state in NoState.values():
	for key, val in state.items():
		if key.startswith("J"):
			NoJ.append(val)
		elif key.startswith("T"):
			NoT.append(val)

for state in CldState.values():
	for key, val in state.items():
		if key.startswith("J"):
			CldJ.append(val)
		elif key.startswith("T"):
			CldT.append(val)

# Mean over all the sampling
NoJ_Mean : np.ndarray = np.array(NoJ).mean(axis=0)
NoT_Mean : np.ndarray = np.array(NoT).mean(axis=0)
CldJ_Mean: np.ndarray = np.array(CldJ).mean(axis=0)
CldT_Mean: np.ndarray = np.array(CldT).mean(axis=0)

CldJ_Mean_Itp: np.ndarray = interp1d(np.arange(CldJ_Mean.shape[1]), CldJ_Mean, axis=1)(np.arange(NoJ_Mean.shape[1]))
CldT_Mean_Itp: np.ndarray = interp1d(np.arange(CldT_Mean.shape[1]), CldT_Mean, axis=1)(np.arange(NoT_Mean.shape[1]))

# ====================================================
# Plotting
# ====================================================

# Load vertical coordinate
WorkPath = Path("/work/b11209013/Kuang2008_v0.3.0/full/Rad(0.0,0.0,0.0)/2026-04-16_RUN05")
with h5py.File(WorkPath / "GalerkinState.h5", "r") as file:
    z: np.ndarray = cast(h5py.Dataset, file["z"])[...]

# Reconstruct phase
phase: np.ndarray = np.linspace(-np.pi, np.pi, NoJ_Mean.shape[1])

# Figure 1: Generation Overlay
fig1, ax1 = plt.subplots(figsize=(11, 4.5))

ctf1 = ax1.contourf(
    phase, z, CldJ_Mean_Itp,
    cmap="BrBG", extend="both", levels=np.linspace(-5, 5, 11)
)
ct1 = ax1.contour(
    phase, z, NoJ_Mean,
    colors="black", linewidths=3, alpha=0.7
)

ax1.set_xlim(-np.pi, np.pi)
ax1.set_ylim(0, 14000)
ax1.set_ylabel("Level [m]")
ax1.set_title(r"Convective Heating Overlay (shading: CldRad, contour: NoRad, K/day)")
ax1.set_xticks(
    np.linspace(-np.pi, np.pi, 5),
    [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"]
)
ax1.set_xlabel("Phase [rad]")
ax1.clabel(ct1, inline=True, fontsize=12)
fig1.colorbar(ctf1, ax=ax1)

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/J_compare.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

# Figure 2: Dissipation Overlay
fig2, ax2 = plt.subplots(figsize=(11, 4.5))

ctf2 = ax2.contourf(
    phase, z, CldT_Mean_Itp,
    cmap="BrBG", extend="both", levels=np.linspace(-2, 2, 11)
)
ct2 = ax2.contour(
    phase, z, NoT_Mean,
    colors="black", linewidths=3, alpha=0.7
)

ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(0, 14000)
ax2.set_ylabel("Level [m]")
ax2.set_title(r"Temperature Overlay (shading: CldRad, contour: NoRad, K)")
ax2.set_xticks(
    np.linspace(-np.pi, np.pi, 5),
    [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"]
)
ax2.set_xlabel("Phase [rad]")
ax2.clabel(ct2, inline=True, fontsize=12)
fig2.colorbar(ctf2, ax=ax2)

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/T_compare.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
