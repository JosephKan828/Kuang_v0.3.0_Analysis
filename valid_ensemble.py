# ====================================================
# This run is to identify the ensemble that is westward
# tilting
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

# ====================================================
# Load data
# ====================================================

# Path setting
sim_name: str = "Rad(0.0,0.0,0.0)"
exp_name: str = "NoRad"
run_name: str = "2026-05-12_RUN01"

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

    ## Load J variables and store in dictionary
    GalerkinState: dict[str, np.ndarray] = {
        key: cast(h5py.Dataset, file[key])[k_unstable_idx, :, :, time_slice, :]
        for key in list(file.keys())
        if key.startswith("J")
    } # Shape: (nz, nx, nt, nens)

nz, nx, nt, nens = GalerkinState["J1"].shape

print("Finish loading datasets")

# ====================================================
# Identification Process
# ====================================================

# Combine heating modes
J: np.ndarray = GalerkinState["J1"] + GalerkinState["J2"]
del GalerkinState # remove total dictionary

# Calculate average over time slice and J1 mean for rolling
J_mean : np.ndarray = np.real(np.nanmean(J, axis=-2))
J1_mean: np.ndarray = np.nanmean(J1, axis=-2)

# Roll data based on J1 maximum to align convective centers
margin: int = nx//3 
shifts: np.ndarray = nx//2 - np.argmax(J1_mean[margin:-margin, :], axis=0) - margin

J_rolled: np.ndarray = np.stack([
    np.roll(J_mean[..., i], shift=shifts[i], axis=1)
    for i in range(nens)
], axis=-1)

# Calculate phase indices for the wave period
half_period     : float = (2*np.pi*4.32e6 / k_unstable) / 2
half_period_grid: int   = int(half_period / (x[1] - x[0]))
pi_neg: int = int(nx//2 - half_period_grid//2)
pi_pos: int = int(nx//2 + 3*half_period_grid//2)

west_ens: list[int] = []

# Identification loop
for i in range(nens):
    J_slice: np.ndarray = J_rolled[:, pi_neg:pi_pos, i]
    num_phase_grid = J_slice.shape[1]

    # Westward tilt criteria: Peak at level 60 is west of peak at level 20
    if np.nanargmax(J_slice[60, :num_phase_grid//2]) < np.nanargmax(J_slice[20, :num_phase_grid//2]):
        west_ens.append(i)

print(f"Identified {len(west_ens)} westward tilting ensembles: {west_ens}")

# ====================================================
# Save Identification Result
# ====================================================
file_path : Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Files")
output_dir: Path = file_path / "Ens" 

os.makedirs(output_dir, exist_ok=True)

np.savetxt(output_dir / f"{exp_name}_{run_name}.txt", np.array(west_ens), fmt="%d")

print(f"Ensemble list saved to {output_dir / f'{exp_name}_{run_name}.txt'}")
