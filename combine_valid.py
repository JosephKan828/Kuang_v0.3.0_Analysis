# ====================================================
# This file is to collect valid ensemble members into
# one file for each variable.
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
import xarray as xr

from typing import cast
from pathlib import Path

# ====================================================
# Load data
# ====================================================

# Path setting
## experiment configuration
sim_name: str = "Rad(0.0,0.0,0.1)"
exp_name: str = "CldRad"
run_name: list[str] = ["2026-05-08_RUN01", "2026-05-12_RUN01"]

## path configuration
home_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis")
work_path: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full" +\
    f"/{sim_name}"
    )

# Load file
## Load growth rate data
with h5py.File(work_path / run_name[0] / "EigenAnalysis.h5", "r") as file:
    wavenumber : np.ndarray = cast(h5py.Dataset, file["k"])[:]
    growth_rate: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]

## Identify most unstable index
k_unstable_idx: np.ndarray = np.nanargmax(growth_rate).astype(int)
k_unstable    : int        = int(wavenumber[k_unstable_idx])

## Load variables
State   : dict[str, dict[str, np.ndarray]] = {}
Fourier : dict[str, dict[str, np.ndarray]] = {}
Galerkin: dict[str, dict[str, np.ndarray]] = {}


for run in run_name:
    ### Load valid ensemble
    valid_ens: np.ndarray = np.array([i for i in np.loadtxt(home_path / "Files" / "Ens" / f"{exp_name}_{run}.txt", dtype=float)], dtype=int)

    with h5py.File(work_path / run / "State.h5", "r") as file:
        State[run] = {
            key: cast(h5py.Dataset, file[key])[k_unstable_idx, ...][..., valid_ens] for key in file.keys()
            if key.startswith(("w", "T", "J", "q"))
        }

    with h5py.File(work_path / run / "FourierState.h5", "r") as file:
        Fourier[run] = {
            key: cast(h5py.Dataset, file[key])[k_unstable_idx, ...][..., valid_ens] for key in file.keys()
            if key.startswith(("w", "T", "J"))
        }

    with h5py.File(work_path / run / "GalerkinState.h5", "r") as file:
        Galerkin[run] = {
            key: cast(h5py.Dataset, file[key])[k_unstable_idx, ...][..., valid_ens] for key in file.keys()
            if key.startswith(("w", "T", "J"))
        }

# Collect the key of variables
state_key   : list[str] = list(State[run_name[0]].keys())
Fourier_key : list[str] = list(Fourier[run_name[0]].keys())
Galerkin_key: list[str] = list(Galerkin[run_name[0]].keys())

# ====================================================
# Combining data across ensembles
# ====================================================

# Pre-allocate dictionary for data
state_total   : dict[str, np.ndarray] = {}
Fourier_total : dict[str, np.ndarray] = {}
Galerkin_total: dict[str, np.ndarray] = {}

# organising data
## State dictionary

for key in state_key:
    state_tmp    = []
    Fourier_tmp  = []
    Galerkin_tmp = []

    for run in run_name:
        state_tmp.append(State[run][key])

        if key in (Fourier_key or Galerkin_key):
            Fourier_tmp.append(Fourier[run][key])
            Galerkin_tmp.append(Galerkin[run][key])
        else:
            continue

    state_total[key]    = np.concatenate(state_tmp, axis=-1)

    if key in (Fourier_key or Galerkin_key):
        Fourier_total[key]  = np.concatenate(Fourier_tmp, axis=-1)
        Galerkin_total[key] = np.concatenate(Galerkin_tmp, axis=-1)

# ====================================================
# Output combined dictionary
# ====================================================

# output path
output_path: Path = Path(f"/work/b11209013/Kuang2008_v0.3.0/full/{exp_name}_combined")

os.makedirs(output_path, exist_ok=True)

# Save state
with h5py.File(output_path / "State.h5", "w") as f:
    for key in state_key:
        f.create_dataset(key, data=state_total[key])

# save Fourier
with h5py.File(output_path / "Fourier.h5", "w") as f:
    for key in Fourier_key:
        f.create_dataset(key, data=Fourier_total[key])

# save Galerkin
with h5py.File(output_path / "Galerkin.h5", "w") as f:
    for key in Galerkin_key:
        f.create_dataset(key, data=Galerkin_total[key])


