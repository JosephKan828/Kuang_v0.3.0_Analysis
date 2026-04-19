# ====================================================
# This program is to compare phase relation between 
# variables at different layers
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
from scipy.signal import correlate, correlation_lags

from matplotlib import pyplot as plt

# using style sheet
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Helper functions
# ====================================================

def phase_shifts(
        field1: np.ndarray,
        field2: np.ndarray,        
) -> np.ndarray:
    
	# dimension check
	if field1.shape != field2.shape:
		raise ValueError("Input signals must have  the same shape.")

	# shapes
	nz, nphase = field1.shape

	# calculate phase difference
	d_phase: np.ndarray = 2*np.pi / (nphase-1)

	# Calculate lages
	lags: np.ndarray = correlation_lags(nphase, nphase, mode="full")

	phase_shift: np.ndarray = np.zeros(nz)

	for i in range(nz):
		# caluclate cross-correlation
		corr: np.ndarray = correlate(field1[i], field2[i], mode="full")

		# Find index of maximum correlation
		max_corr_idx: int = np.nanargmax(corr).astype(int)

		# convert index shift to phase shift
		shift = lags[max_corr_idx] * d_phase

		# Wrap phase to [-pi, pi]
		phase_shift[i] = (shift + np.pi) % (2 * np.pi) - np.pi
	
	return phase_shift

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
	
	for file in glob(EnsName+"/*.csv"):
		NoState[EnsName.split("/")[-1]][file.split("/")[-1].split(".")[0]] = np.array(pd.read_csv(file))[:, 1:]
	
for EnsName in CldEnsList:
	CldState[EnsName.split("/")[-1]] = {}
	
	for file in glob(EnsName+"/*.csv"):
		CldState[EnsName.split("/")[-1]][file.split("/")[-1].split(".")[0]] = np.array(pd.read_csv(file))[:, 1:]
	
# ====================================================
# Collecting profile of different ensemble and time
# ====================================================

# pre-allocate the list 
NoJ: list = []; CldJ: list = []
NoT: list = []; CldT: list = []
Now: list = []; Cldw: list = []

# for loop for saving
for state in NoState.values():
	for key, val in state.items():
		if key.startswith("w"):
			Now.append(val)
		elif key.startswith("T"):
			NoT.append(val)
		elif key.startswith("J"):
			NoJ.append(val)

for state in CldState.values():
	for key, val in state.items():
		if key.startswith("w"):
			Cldw.append(val)
		elif key.startswith("T"):
			CldT.append(val)
		elif key.startswith("J"):
			CldJ.append(val)

# ====================================================
# Calculate phase lag beween signals
# ====================================================
No_QT: list = []; Cld_QT: list = []
No_wT: list = []; Cld_wT: list = []

for i in range(len(NoJ)):
	No_QT.append(phase_shifts(NoJ[i], NoT[i]))
	No_wT.append(phase_shifts(Now[i], NoT[i]))


for i in range(len(CldJ)):
	Cld_QT.append(phase_shifts(CldJ[i], CldT[i]))
	Cld_wT.append(phase_shifts(Cldw[i], CldT[i]))
plt.plot(phase_shifts(CldJ[0], CldT[0])[1:])
plt.savefig("test.png")
plt.close()

plt.plot()
# convert from list to array
No_QT_arr: np.ndarray = np.array(No_QT)
No_wT_arr: np.ndarray = np.array(No_wT)

Cld_QT_arr: np.ndarray = np.array(Cld_QT)
Cld_wT_arr: np.ndarray = np.array(Cld_wT)

# Calculate mean and standard deviation
No_QT_mean: np.ndarray = No_QT_arr.mean(axis=0)
No_QT_std : np.ndarray = No_QT_arr.std(axis=0)

No_wT_mean: np.ndarray = No_wT_arr.mean(axis=0)
No_wT_std : np.ndarray = No_wT_arr.std(axis=0)

Cld_QT_mean: np.ndarray = Cld_QT_arr.mean(axis=0)
Cld_QT_std : np.ndarray = Cld_QT_arr.std(axis=0)

Cld_wT_mean: np.ndarray = Cld_wT_arr.mean(axis=0)
Cld_wT_std : np.ndarray = Cld_wT_arr.std(axis=0)

# ====================================================
# Visualization
# ====================================================

nz = len(No_QT_mean)
z = np.linspace(0, 14000, nz)[1:]

fig, ax = plt.subplots(1, 1, figsize=(5, 9), sharey=True)

# Plot w-T phase lag
ax.plot(No_wT_mean[1:], z, label="NoRad", color="C0", lw=4)
ax.fill_betweenx(z, No_wT_mean[1:] - No_wT_std[1:], No_wT_mean[1:] + No_wT_std[1:], color="C0", alpha=0.2)

ax.plot(Cld_wT_mean[1:], z, label="CldRad", color="deeppink", lw=4)
ax.fill_betweenx(z, Cld_wT_mean[1:] - Cld_wT_std[1:], Cld_wT_mean[1:] + Cld_wT_std[1:], color="deeppink", alpha=0.2)

ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.set_xticks(np.linspace(-np.pi, np.pi, 5), [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"])
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(0, 14000)
ax.set_title("w-T Phase Lag")
ax.set_xlabel("Phase Shift (rad)")
ax.set_ylabel("Level [m]")
ax.legend()
plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/wT_phase.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Plot J-T (QT) phase lag
fig, ax = plt.subplots(1, 1, figsize=(5, 9), sharey=True)

ax.plot(No_QT_mean[1:], z, label="NoRad", color="C0", lw=4)
ax.fill_betweenx(z, No_QT_mean[1:] - No_QT_std[1:], No_QT_mean[1:] + No_QT_std[1:], color="C0", alpha=0.2)

ax.plot(Cld_QT_mean[1:], z, label="CldRad", color="deeppink", lw=4)
ax.fill_betweenx(z, Cld_QT_mean[1:] - Cld_QT_std[1:], Cld_QT_mean[1:] + Cld_QT_std[1:], color="deeppink", alpha=0.2)

ax.axvline(0, color="k", linestyle="--", alpha=0.5)
ax.set_xticks(np.linspace(-np.pi, np.pi, 5), [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"])
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(0, 14000)
ax.set_title("J-T Phase Lag")
ax.set_xlabel("Phase Shift (rad)")
ax.set_ylabel("Level [m]")
ax.legend()

plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/QT_phase.png", dpi=300, bbox_inches="tight")
plt.show()