# ====================================================
# Composite for NoRad experiments
# ====================================================

# %%==================================================
# Import package
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

# %%==================================================
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

# %%==================================================
# Load data
# ====================================================

# File path
CfgPath : Path = Path("/home/b11209013/Kuang2008_v0.3.0/Config")
WorkPath: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full/Rad(0.0,0.0,0.1)/2026-04-16_RUN05")
FigPath : Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/NoRad_vs_CldRad/CldRad")

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
        key: np.real(daily_mean(dt, cast(h5py.Dataset, file[key])[kidx, ...]))
        for key in file.keys()
        if key.startswith(("w", "T", "J"))
    } # shape: (nt, nz, nx, nens)

    x: np.ndarray = cast(h5py.Dataset, file["x"])[...]
    z: np.ndarray = cast(h5py.Dataset, file["z"])[...]

# Load J1 for identifying the peak of signal
with h5py.File(WorkPath / "FourierState.h5", "r") as file:
    J1: np.ndarray = daily_mean(dt, cast(h5py.Dataset, file["J1"])[kidx, ...])

# %%==================================================
# Preprocessing
# ====================================================

# combining field
w: np.ndarray = (GalerkinState["w1"] + GalerkinState["w2"]) / 86400.0
T: np.ndarray = GalerkinState["T1"] + GalerkinState["T2"]
J: np.ndarray = GalerkinState["J1"] + GalerkinState["J2"]

nz, nx, nt, nens = w.shape

# %%==================================================
# Composite
# ====================================================

target_day: list[int] = [5, 6, 7]

for ei in range(nens):
    for day in target_day:
        print(f"Start processing Ens {ei} day {day}")

        # Select specific ensemble member and day
        wSelect : np.ndarray = w[..., day, :][..., ei]
        TSelect : np.ndarray = T[..., day, :][..., ei]
        JSelect : np.ndarray = J[..., day, :][..., ei]
        J1Select: np.ndarray = J1[..., day, :][..., ei]

        # Roll J1 maximum to the center of x domain
        ## Calculate shift
        shifts: int = nx//2 - np.argmax(np.array(J1Select[nx//3:-nx//3])) - nx//3

        ## Roll data
        wRoll: np.ndarray = np.roll(wSelect, shifts, axis=1)
        TRoll: np.ndarray = np.roll(TSelect, shifts, axis=1)
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
        os.makedirs(FigPath / "Composite", exist_ok=True)
        os.makedirs(FigPath / "Composite" / f"Ens{ei}", exist_ok=True)

        figfolder = FigPath / "Composite" / f"Ens{ei}" / f"day{day+1}.png"

        fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        ax1, ax2 = axes.flatten()

        J_ctf = ax1.contourf(
            phase, z, JRoll[:, pi_neg:pi_pos],
            cmap="BrBG", levels=np.linspace(-2, 2, 21)
        )
        w_ct = ax1.contour(
            phase, z, wRoll[:, pi_neg:pi_pos],
            colors="black", linewidths=3
        )

        ax1.set_xlim(-np.pi, np.pi)
        ax1.set_ylim(0, 14000)
        ax1.set_ylabel("Level [m]")
        ax1.set_title("J (shading, K/day) vs. w (contour, m/s)")
        ax1.clabel(w_ct, inline=True)
        fig.colorbar(J_ctf, ax=ax1)

        J_ctf = ax2.contourf(
            phase, z, JRoll[:, pi_neg:pi_pos],
            cmap="BrBG", levels=np.linspace(-2, 2, 21)
        )
        T_ct = ax2.contour(
            phase, z, TRoll[:, pi_neg:pi_pos],
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

        plt.savefig(figfolder, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Save fil for phase analysis
        os.makedirs(f"/home/b11209013/Kuang2008_v0.3.0_Analysis/Files/CldRad/Rolled/Ens{ei}", exist_ok=True)
        Output = Path(f"/home/b11209013/Kuang2008_v0.3.0_Analysis/Files/CldRad/Rolled/Ens{ei}")

        pd.DataFrame(wRoll[:, pi_neg:pi_pos]).to_csv(Output / f"w_day{day+1}.csv")
        pd.DataFrame(TRoll[:, pi_neg:pi_pos]).to_csv(Output / f"T_day{day+1}.csv")
        pd.DataFrame(JRoll[:, pi_neg:pi_pos]).to_csv(Output / f"J_day{day+1}.csv")

# %%
