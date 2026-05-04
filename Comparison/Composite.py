# ====================================================
# This script is for composite analysis of different
# experiments and ensembles
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

from typing import cast
from pathlib import Path

from matplotlib import pyplot as plt

# Plot configuration
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Load data
# ====================================================

# Path setting
sim_name: str = "Rad(0.0,0.0,0.0)"
exp_name: str = "NoRad"
run_name: str = "2024-05-01_RUN01"

work_path: Path = Path(
    "/work/b11209013/Kuang2008_v0.3.0/full" +\
    f"/{sim_name}/{run_name}"
    )

# Select wavenumber for the most unstable mode
## Load growth rate data
with h5py.File(work_path / "EigenAnalysis.h5", "r") as file:
    wavenumber : np.ndarray = cast(h5py.Dataset, file["k"])[:]
    growth_rate: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0] 

## Identify most unstable index
k_unstable_idx: np.ndarray = np.nanargmax(growth_rate).astype(int)
k_unstable    : int        = int(wavenumber[k_unstable_idx])

# Load J1 Fourier state for identifying the convective center
## Setting time slice for composite
time_slice: list[int] = [26, 27, 28]

## Loading
with h5py.File(work_path / "FourierState.h5", "r") as file:
    J1: np.ndarray = cast(h5py.Dataset, file["J1"])[k_unstable_idx, :, time_slice, :]

# Load Galerkin state
with h5py.File(work_path / "GalerkinState.h5", "r") as file:

    ## Load coordinate information
    x: np.ndarray = cast(h5py.Dataset, file["x"])[...]
    z: np.ndarray = cast(h5py.Dataset, file["z"])[...]

    ## Load variables and store in dictionary
    GalerkinState: dict[str, np.ndarray] = {
        key: cast(h5py.Dataset, file[key])[k_unstable_idx, :, :, time_slice, :]
        for key in list(file.keys())
        if key.startswith(("w", "T", "J"))
    } # Shape: (nz, nx, nt, nens)

nz, nx, nt, nens = GalerkinState["w1"].shape

print("Finish loading all datasets")

# ====================================================
# Composite
# ====================================================

# Combine the first and the second mode 
w: np.ndarray = (GalerkinState["w1"] + GalerkinState["w2"]) / 86400.0
T: np.ndarray = GalerkinState["T1"] + GalerkinState["T2"]
J: np.ndarray = GalerkinState["J1"] + GalerkinState["J2"]

del GalerkinState # remove total dictionary

# Calculate average over time slice
w_mean : np.ndarray = np.real(np.nanmean(w, axis=-2))
T_mean : np.ndarray = np.real(np.nanmean(T, axis=-2))
J_mean : np.ndarray = np.real(np.nanmean(J, axis=-2))
J1_mean: np.ndarray = np.nanmean(J1, axis=-2)


# Roll data based on J1 maximum
## Calculate shifts
margin: int = nx//3 # distance between the indentifying section to the boarder

shifts: np.ndarray = nx//2 - np.argmax(J1_mean[margin:-margin, :], axis=0) - margin

w_rolled: np.ndarray = np.stack([
    np.roll(w_mean[..., i], shift=shifts[i], axis=1)
    for i in range(nens)
], axis=-1)

T_rolled: np.ndarray = np.stack([
    np.roll(T_mean[..., i], shift=shifts[i], axis=1)
    for i in range(nens)
], axis=-1)

J_rolled: np.ndarray = np.stack([
    np.roll(J_mean[..., i], shift=shifts[i], axis=1)
    for i in range(nens)
], axis=-1)

print("Finish rolling data")

# Calculate index for showing period in radian
## Calculate the number of grid in wave length
half_period     : float = (2*np.pi*4.32e6 / k_unstable) / 2
half_period_grid: int   = int(half_period / (x[1] - x[0]))

## calculate index of period
pi_neg: int = int(nx//2 - half_period_grid//2)
pi_pos: int = int(nx//2 + 3*half_period_grid//2)

phase: np.ndarray = np.linspace(-np.pi, np.pi, int(pi_pos-pi_neg))

# ====================================================
# Visualizing data
# ====================================================

# Save directory
fig_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure")
os.makedirs(fig_path / exp_name, exist_ok=True)
os.makedirs(fig_path / exp_name / run_name, exist_ok=True)
os.makedirs(fig_path / exp_name / run_name / "Composite", exist_ok=True)

# Pre-allocate array for phase-based data
J_phase: dict[str, np.ndarray] = {}
T_phase: dict[str, np.ndarray] = {}
w_phase: dict[str, np.ndarray] = {}

west_ens: list = []

# figure generate
for i in range(nens):

    print(f"Start processing Ens {i+1}")

    ## select slice for plotting
    J_plot: np.ndarray = J_rolled[:, pi_neg:pi_pos, i]
    T_plot: np.ndarray = T_rolled[:, pi_neg:pi_pos, i]
    w_plot: np.ndarray = w_rolled[:, pi_neg:pi_pos, i]

    ## Identifying the tilting of ensemble
    if np.nanargmax(J_plot[60, :phase.size//2]) < np.nanargmax(J_plot[20, :phase.size//2]):
        west_ens.append(i)
        J_phase[f"Ens{i}"] = J_plot
        T_phase[f"Ens{i}"] = T_plot
        w_phase[f"Ens{i}"] = w_plot
    else:
        continue
        
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    ax1, ax2 = axes.flatten()

    J_ctf = ax1.contourf(
        phase, z, J_rolled[:, pi_neg:pi_pos, i],
        cmap="BrBG", levels=11
    )
    w_ct = ax1.contour(
        phase, z, w_rolled[:, pi_neg:pi_pos, i],
        colors="black", linewidths=3
    )

    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(0, 14000)
    ax1.set_ylabel("Level [m]")
    ax1.set_title("J (shading, K/day) vs. w (contour, m/s)")
    ax1.clabel(w_ct, inline=True)
    fig.colorbar(J_ctf, ax=ax1)

    J_ctf = ax2.contourf(
        phase, z, J_rolled[:, pi_neg:pi_pos, i],
        cmap="BrBG", levels=11
    )
    T_ct = ax2.contour(
        phase, z, T_rolled[:, pi_neg:pi_pos, i],
        colors="black", linewidths=3
    )
    ax2.set_xticks(
        np.linspace(-np.pi, np.pi, 5),
        [r"-$\pi$", r"-$\pi$/2", "0", r"$\pi$/2", r"$\pi$"]
        )
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(0, 14000)
    ax2.set_xlabel("Phase [rad]")
    ax2.set_ylabel("Level [m]")
    ax2.set_title("J (shading, K/day) vs. T (contour, K)")
    ax2.clabel(T_ct, inline=True)
    fig.colorbar(J_ctf, ax=ax2)
    plt.savefig(fig_path / exp_name / run_name / "Composite" / f"Ens{i+1}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ====================================================
# Save file
# ====================================================
# Save directory
file_path : Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Files")
output_dir: Path = file_path / exp_name / run_name / "Composite"

os.makedirs(file_path / exp_name, exist_ok=True)
os.makedirs(file_path / exp_name / run_name, exist_ok=True)
os.makedirs(file_path / exp_name / run_name / "Composite", exist_ok=True)

np.savez(output_dir / "J_westward.npz", **J_phase) # type: ignore
np.savez(output_dir / "T_westward.npz", **T_phase) # type: ignore
np.savez(output_dir / "w_westward.npz", **w_phase) # type: ignore
np.savetxt(output_dir / "Ens.txt", np.array(west_ens))