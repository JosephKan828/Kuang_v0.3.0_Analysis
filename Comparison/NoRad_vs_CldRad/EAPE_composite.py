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
	
	for file in (glob(EnsName+"/Gen_day2*.csv") + glob(EnsName+"/Dis_day2*.csv")):
		NoState[EnsName.split("/")[-1]][file.split("/")[-1].split(".")[0]] = np.array(pd.read_csv(file))[:, 1:]
	
for EnsName in CldEnsList:
	CldState[EnsName.split("/")[-1]] = {}
	
	for file in (glob(EnsName+"/Gen_day2*.csv") + glob(EnsName+"/Dis_day2*.csv")):
		CldState[EnsName.split("/")[-1]][file.split("/")[-1].split(".")[0]] = np.array(pd.read_csv(file))[:, 1:]

# ====================================================
# Collecting profile of different ensemble and time
# ====================================================

# pre-allocate the list 
NoGen: list = []; CldGen: list = []
NoDis: list = []; CldDis: list = []

# for loop for saving
for state in NoState.values():
	for key, val in state.items():
		if key.startswith("Gen"):
			NoGen.append(val)
		elif key.startswith("Dis"):
			NoDis.append(val)

for state in CldState.values():
	for key, val in state.items():
		if key.startswith("Gen"):
			CldGen.append(val)
		elif key.startswith("Dis"):
			CldDis.append(val)

# Mean over all the sampling
NoGen_Mean : np.ndarray = np.array(NoGen).mean(axis=0)
NoDis_Mean : np.ndarray = np.array(NoDis).mean(axis=0)
CldGen_Mean: np.ndarray = np.array(CldGen).mean(axis=0)
CldDis_Mean: np.ndarray = np.array(CldDis).mean(axis=0)

CldGen_Mean_Itp: np.ndarray = interp1d(np.arange(CldGen_Mean.shape[1]), CldGen_Mean, axis=1)(np.arange(NoGen_Mean.shape[1]))
CldDis_Mean_Itp: np.ndarray = interp1d(np.arange(CldDis_Mean.shape[1]), CldDis_Mean, axis=1)(np.arange(NoDis_Mean.shape[1]))

# ====================================================
# Plotting
# ====================================================

# Load vertical coordinate
WorkPath = Path("/work/b11209013/Kuang2008_v0.3.0/full/Rad(0.0,0.0,0.0)/2026-04-16_RUN05")
with h5py.File(WorkPath / "GalerkinState.h5", "r") as file:
    z: np.ndarray = cast(h5py.Dataset, file["z"])[...]

# Reconstruct phase
phase: np.ndarray = np.linspace(-np.pi, np.pi, NoGen_Mean.shape[1])

# Figure 1: Generation Overlay
fig1, ax1 = plt.subplots(figsize=(11, 4.5))

ctf1 = ax1.contourf(
    phase, z, CldGen_Mean_Itp,
    cmap="BrBG", extend="both", levels=np.linspace(-20, 20, 11)
)
ct1 = ax1.contour(
    phase, z, NoGen_Mean,
    colors="black", linewidths=3, alpha=0.7
)

ax1.set_xlim(-np.pi, np.pi)
ax1.set_ylim(0, 14000)
ax1.set_ylabel("Level [m]")
ax1.set_title(r"Generation Overlay (shading: CldRad, contour: NoRad, K$^2$/day)")
ax1.set_xticks(
    np.linspace(-np.pi, np.pi, 5),
    [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"]
)
ax1.set_xlabel("Phase [rad]")
ax1.clabel(ct1, inline=True, fontsize=12)
fig1.colorbar(ctf1, ax=ax1)

plt.savefig("EAPE_Gen.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

# Figure 2: Dissipation Overlay
fig2, ax2 = plt.subplots(figsize=(11, 4.5))

ctf2 = ax2.contourf(
    phase, z, CldDis_Mean_Itp,
    cmap="BrBG", extend="both", levels=np.linspace(-20, 20, 11)
)
ct2 = ax2.contour(
    phase, z, NoDis_Mean,
    colors="black", linewidths=3, alpha=0.7
)

ax2.set_xlim(-np.pi, np.pi)
ax2.set_ylim(0, 14000)
ax2.set_ylabel("Level [m]")
ax2.set_title(r"Dissipation Overlay (shading: CldRad, contour: NoRad, K$^2$/day)")
ax2.set_xticks(
    np.linspace(-np.pi, np.pi, 5),
    [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"]
)
ax2.set_xlabel("Phase [rad]")
ax2.clabel(ct2, inline=True, fontsize=12)
fig2.colorbar(ctf2, ax=ax2)

plt.savefig("EAPE_Dis.png", dpi=300, bbox_inches="tight")
plt.close(fig2)
