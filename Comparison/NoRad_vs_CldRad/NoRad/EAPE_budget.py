# ====================================================
# This script is to calculate budget terms of EAPE
# ====================================================

# %% =================================================
# Import packages
# ====================================================

import os
import h5py
import numpy as np
import pandas as pd
import tomllib as tl

from typing import cast
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

print("Finish importing packages")

# %% =================================================
# Functions
# ====================================================
def daily_mean(
        dt   : float,
        field: np.ndarray
) -> np.ndarray:
    
    # sample of day
    day_sample: int = int(1/dt)

    # daily mean
    field_daily: np.ndarray = np.stack([
        np.nanmean(field[..., day_sample*i:day_sample*(i+1), :], axis=-2)
        for i in range(field.shape[-2]//day_sample)
    ], axis=-2)

    return field_daily

# %% ==================================================
# Load data
# =====================================================

# File path
CfgPath : Path = Path("/home/b11209013/Kuang2008_v0.3.0/Config")
WorkPath: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full/Rad(0.0,0.0,0.0)/2026-04-16_RUN05")
FigPath : Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/NoRad")

# Load configuration
with open(CfgPath / "Domain.toml", "rb") as f:

    dt: float = float(tl.load(f)["Domain"]["dt"])

# Identifying most unstable mode
## Load growth rate data
with h5py.File(WorkPath / "EigenAnalysis.h5", "r") as file:
    GrowthRates: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]
    k          : np.ndarray = cast(h5py.Dataset, file["k"])[...]

## find the greatest growth rate
kidx     : int = np.argmax(GrowthRates).astype(int)
kunstable: float = k[kidx]

# Load Galerkin state
with h5py.File(WorkPath / "GalerkinState.h5", "r") as file:
    print(cast(h5py.Dataset, file["w1"])[kidx, ...].shape)

    GalerkinState: dict[str, np.ndarray] = {
        key: np.real(cast(h5py.Dataset, file[key])[kidx, ...])
        for key in file.keys()
        if key.startswith(("w", "T", "J"))
    } # shape: (nz, nx, nt, nens)

    x: np.ndarray = cast(h5py.Dataset, file["x"])[...]
    z: np.ndarray = cast(h5py.Dataset, file["z"])[...]

# Load J1 for identifying the peak of signal
with h5py.File(WorkPath / "FourierState.h5", "r") as file:
    J1: np.ndarray = daily_mean(dt, cast(h5py.Dataset, file["J1"])[kidx, ...])

# Load eastward propagating ensembles
ens_east = np.loadtxt("Ens.txt", delimiter=",", dtype=int)

print("Finish loading data")

# %% =================================================
# Preprocessing
# ====================================================

# combining field
w: np.ndarray = GalerkinState["w1"] + GalerkinState["w2"]
T: np.ndarray = GalerkinState["T1"] + GalerkinState["T2"]
J: np.ndarray = GalerkinState["J1"] + GalerkinState["J2"]

nz, nx, nt, nens = w.shape

# Calculate EAPE budget
Gen: np.ndarray = daily_mean(dt, J * T)
Dis: np.ndarray = daily_mean(dt, -w*T * (9.8/1004.5 - 0.0065))
J  : np.ndarray = daily_mean(dt, J) 

print("Finish calculating EAPE budget")

# %% =================================================
# Calculate composite
# ====================================================

target_days: list[int] = [26, 27, 28]

for ei in ens_east:
    # Select data for ensemble members
    GenSelectEns: np.ndarray = Gen[..., ei]
    DisSelectEns: np.ndarray = Dis[..., ei]
    JSelectEns  : np.ndarray = J[..., ei]
    J1SelectEns : np.ndarray = J1[..., ei]

    for day in target_days:
        print(f"Start processing Ens {ei} day {day}")

        # Select specific ensemble member and day
        GenSelect : np.ndarray = GenSelectEns[..., day]
        DisSelect : np.ndarray = DisSelectEns[..., day]
        JSelect   : np.ndarray = JSelectEns[..., day]
        J1Select  : np.ndarray = J1SelectEns[..., day]

        # Roll J1 maximum to the center of x domain
        ## Calculate shift
        shifts: int = nx//2 - np.argmax(np.array(J1Select[nx//3:-nx//3])) - nx//3

        ## Roll data
        GenRoll: np.ndarray = np.roll(GenSelect, shifts, axis=1)
        DisRoll: np.ndarray = np.roll(DisSelect, shifts, axis=1)
        JRoll: np.ndarray = np.roll(JSelect, shifts, axis=1)

        # Calculate index for period
        ## Half wavelength
        half_period: float = (2*np.pi*4.32e6 / kunstable) / 2
        half_period_grid: int = int(half_period / (x[1]-x[0]))

        ## period
        pi_neg: int = int(nx//2 - half_period_grid//2)
        pi_pos: int = int(nx//2 + 3*half_period_grid//2)

        phase: np.ndarray = np.linspace(-np.pi, np.pi, int(pi_pos-pi_neg))

        # Visualization

        os.makedirs(FigPath / "EAPE", exist_ok=True)
        os.makedirs(FigPath / "EAPE" / f"Ens{ei}", exist_ok=True)

        figfolder = FigPath / "EAPE" / f"Ens{ei}" / f"day{day+1}.png"

        fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        ax1, ax2 = axes.flatten()

        J_ctf = ax1.contourf(
            phase, z, GenRoll[:, pi_neg:pi_pos],
            cmap="BrBG", levels=np.linspace(-1, 1, 21)
        )
        w_ct = ax1.contour(
            phase, z, JRoll[:, pi_neg:pi_pos],
            colors="black", linewidths=3
        )

        ax1.set_xlim(-np.pi, np.pi)
        ax1.set_ylim(0, 14000)
        ax1.set_ylabel("Level [m]")
        ax1.set_title(r"Generation (shading, K$^2$/day) vs. J (contour, K/day)")
        ax1.clabel(w_ct, inline=True)
        fig.colorbar(J_ctf, ax=ax1)

        J_ctf = ax2.contourf(
            phase, z, DisRoll[:, pi_neg:pi_pos],
            cmap="BrBG", levels=np.linspace(-1, 1, 11)
        )
        T_ct = ax2.contour(
            phase, z, JRoll[:, pi_neg:pi_pos],
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
        ax2.set_title(r"Dissipation (shading, K$^2$/day) vs. J (contour, K/day)")
        ax2.clabel(T_ct, inline=True)
        fig.colorbar(J_ctf, ax=ax2)

        plt.savefig(figfolder, dpi=300, bbox_inches="tight")
        plt.close(fig)


        # save data
        os.makedirs(f"/home/b11209013/Kuang2008_v0.3.0_Analysis/Files/NoRad/Rolled/Ens{ei}", exist_ok=True)
        Output = Path(f"/home/b11209013/Kuang2008_v0.3.0_Analysis/Files/NoRad/Rolled/Ens{ei}")

        pd.DataFrame(GenRoll[:, pi_neg:pi_pos]).to_csv(Output / f"Gen_day{day+1}.csv")
        pd.DataFrame(DisRoll[:, pi_neg:pi_pos]).to_csv(Output / f"Dis_day{day+1}.csv")
