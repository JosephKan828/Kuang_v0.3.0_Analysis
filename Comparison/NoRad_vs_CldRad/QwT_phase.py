# ====================================================
# This program is to compare phase relation between 
# variables at different layers
# ====================================================

# ====================================================
# Import package
# ====================================================
import os
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

def compute_ensemble_phase(field1, field2, avg_axes=(0, 1)):
    """
    Unifies phase detection and ensemble averaging.
    
    Parameters:
    -----------
    field1, field2 : np.ndarray
        Input data with shape (n_members, n_days, nz, n_phase).
    avg_axes : tuple
        Axes to average over (default is ensemble and days).
        
    Returns:
    --------
    dict containing:
        'mean_phase': The raw circular mean phase [-pi, pi].
        'unwrapped_phase': The continuous phase profile for plotting.
        'coherence': The R-value (0-1) indicating ensemble agreement.
    """
    
    # 1. Dimensions
    # Assumes last axis is the phase axis (n_phase)
    nz = field1.shape[-2]
    nphase = field1.shape[-1]
    d_phase = 2 * np.pi / nphase # Frequency spacing
    
    # 2. Calculate Cross-Correlation via FFT (more efficient for periodic phase)
    # We compute the correlation for all members, days, and levels at once
    f1 = np.fft.fft(field1, axis=-1)
    f2 = np.fft.fft(field2, axis=-1)
    
    # Cross-power spectrum
    cross_power = f1 * np.conj(f2)
    cross_corr = np.fft.ifft(cross_power, axis=-1).real
    
    # 3. Find the peak lag for every single realization
    # max_lags shape: (n_members, n_days, nz)
    max_lag_indices = np.argmax(cross_corr, axis=-1)
    
    # Adjust lags to center them (mapping indices to -nphase/2 : nphase/2)
    lags = np.where(max_lag_indices > nphase//2, 
                    max_lag_indices - nphase, 
                    max_lag_indices)
    
    phase_shifts = lags * d_phase
    
    # 4. Vector Averaging (Circular Statistics)
    # Convert phase shifts to complex unit vectors
    vectors = np.exp(1j * phase_shifts)
    
    # Average over the requested axes (e.g., members and days)
    mean_vector = np.mean(vectors, axis=avg_axes)
    
    # 5. Extract Results
    mean_phase = np.angle(mean_vector)
    coherence = np.abs(mean_vector)
    unwrapped_phase = np.unwrap(mean_phase)
    
    return {
        "mean_phase": mean_phase,
        "unwrapped_phase": unwrapped_phase,
        "coherence": coherence
    }
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
	NoState[EnsName.split("/")[-1]] = {"w": {}, "T": {}, "J": {}}
	
	for file in glob(EnsName+"/*.csv"):
		sub_key = file.split("/")[-1].split(".")[0]

		if sub_key.startswith("w"):
			NoState[EnsName.split("/")[-1]]["w"][f"{sub_key.split("_")[-1][3:]}"] = np.array(pd.read_csv(file))[:, 1:]
		elif sub_key.startswith("J"):
			NoState[EnsName.split("/")[-1]]["J"][f"{sub_key.split("_")[-1][3:]}"] = np.array(pd.read_csv(file))[:, 1:]
		elif sub_key.startswith("T"):
			NoState[EnsName.split("/")[-1]]["T"][f"{sub_key.split("_")[-1][3:]}"] = np.array(pd.read_csv(file))[:, 1:]
		else:
			continue
	
for EnsName in CldEnsList:
	CldState[EnsName.split("/")[-1]] = {"w": {}, "T": {}, "J": {}}
	
	for file in glob(EnsName+"/*.csv"):
		sub_key = file.split("/")[-1].split(".")[0]

		if sub_key.startswith("w"):
			CldState[EnsName.split("/")[-1]]["w"][f"{sub_key.split("_")[-1][3:]}"] = np.array(pd.read_csv(file))[:, 1:]
		elif sub_key.startswith("J"):
			CldState[EnsName.split("/")[-1]]["J"][f"{sub_key.split("_")[-1][3:]}"] = np.array(pd.read_csv(file))[:, 1:]
		elif sub_key.startswith("T"):
			CldState[EnsName.split("/")[-1]]["T"][f"{sub_key.split("_")[-1][3:]}"] = np.array(pd.read_csv(file))[:, 1:]
		else:
			continue
# ====================================================
# Collecting profile of different ensemble and time
# ====================================================

def collect_ensemble_data(state_dict):
    """
    Collects dictionary-based state data into 4D numpy arrays.
    Shape: (n_members, n_days, nz, n_phase)
    """
    all_w, all_T, all_J = [], [], []
    
    # Ensure consistent order of ensembles and days
    for ens_name in sorted(state_dict.keys()):
        ens_val = state_dict[ens_name]
        
        # Assume 'w', 'T', 'J' have the same day keys
        day_keys = sorted(ens_val['w'].keys(), key=int)
        
        all_w.append(np.array([ens_val['w'][d] for d in day_keys]))
        all_T.append(np.array([ens_val['T'][d] for d in day_keys]))
        all_J.append(np.array([ens_val['J'][d] for d in day_keys]))
        
    return np.array(all_w), np.array(all_T), np.array(all_J)

# Organize NoRad and CldRad data
Now, NoT, NoJ = collect_ensemble_data(NoState)
Cldw, CldT, CldJ = collect_ensemble_data(CldState)

print(f"NoRad shapes: w={Now.shape}, T={NoT.shape}, J={NoJ.shape}")
print(f"CldRad shapes: w={Cldw.shape}, T={CldT.shape}, J={CldJ.shape}")

# ====================================================
# Calculate phase lag beween signals
# ====================================================
No_QT: list = []; Cld_QT: list = []
No_wT: list = []; Cld_wT: list = []

# NoRad ensemble
for i in range(NoJ.shape[0]):
	res_QT = compute_ensemble_phase(NoJ[i], NoT[i], avg_axes=(0,))
	res_wT = compute_ensemble_phase(Now[i], NoT[i], avg_axes=(0,))
	No_QT.append(res_QT["mean_phase"])
	No_wT.append(res_wT["mean_phase"])

# CldRad ensemble
for i in range(CldJ.shape[0]):
	res_QT = compute_ensemble_phase(CldJ[i], CldT[i], avg_axes=(0,))
	res_wT = compute_ensemble_phase(Cldw[i], CldT[i], avg_axes=(0,))
	Cld_QT.append(res_QT["mean_phase"])
	Cld_wT.append(res_wT["mean_phase"])

# convert from list to array
No_QT_arr: np.ndarray = np.array(No_QT)
No_wT_arr: np.ndarray = np.array(No_wT)
print("No_QT_arr shape: ", No_QT_arr.shape)
Cld_QT_arr: np.ndarray = np.array(Cld_QT)
Cld_wT_arr: np.ndarray = np.array(Cld_wT)

# Calculate

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
plt.close()

# plot Q-T phase relation of all samples
PhasePath = "/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/NoRad/QTPhase/"

os.makedirs(PhasePath, exist_ok=True)

for i in range(No_QT_arr.shape[0]):
	print(f"Start sample {i+1}")
	fig, ax = plt.subplots(1, 1, figsize=(6, 9))

	ax.plot(No_QT_arr[i, 1:], z, label=f"Sample {i+1}")

	ax.set_ylim(0, 14000)
	plt.savefig(PhasePath+f"Sample{i+1}.png", dpi=300, bbox_inches="tight")
	plt.close(fig)