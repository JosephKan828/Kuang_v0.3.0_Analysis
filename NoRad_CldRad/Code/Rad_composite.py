# ====================================================
# This script is to calculate and compare the equilibrium
# convective heating profile between experiments
# ====================================================

# ====================================================
# Environment setup
# ====================================================

import h5py
import numpy as np

from typing import cast
from pathlib import Path
from scipy.signal import find_peaks

from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Plot configuration
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Main function
# ====================================================

def main():
    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    # Path setup
    home_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad")
    work_path: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")

    cld_shifts: np.ndarray = np.load(home_path / "data/composite/CldRad/shifts.npy")
    cld_neg_pi: np.ndarray = np.load(home_path / "data/composite/CldRad/neg_pi.npy")
    cld_pos_pi: np.ndarray = np.load(home_path / "data/composite/CldRad/pos_pi.npy")


    with h5py.File(work_path / "Rad(0.0,0.0,0.1)" / "latest" / "EigenAnalysis.h5", "r") as file:
        wavenumber: np.ndarray = cast(h5py.Dataset, file["k"])[:]
        cld_growth: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]

    cld_k: float = wavenumber[np.argmax(cld_growth)]

    # Load state vector of both experiments
    ## Load CldRad experiment
    with h5py.File(work_path / "CldRad_combined" / "State.h5", "r") as file:
        Cld_state: dict[str, np.ndarray] = {
                key: cast(h5py.Dataset, value)[..., :46]
                for (key, value) in file.items()
                if key.startswith(("w", "q", "T"))
                }

    # ------------------------------------------------
    # Calculate radiative heating evolution
    # ------------------------------------------------

    Rw11LW: float = 0.0763 * 0.1 ; Rw11SW: float = 0.0569 * 0.1
    Rw12LW: float = 0.118 * 0.1  ; Rw12SW: float = -0.0571 * 0.1
    Rw21LW: float = -0.0585 * 0.1; Rw21SW: float = -0.0422 * 0.1
    Rw22LW: float = -0.141 * 0.1 ; Rw22SW: float = 0.0711 * 0.1

    # Calculate Leq evolution for the two experiments
    LW1: np.ndarray = Rw11LW * Cld_state["w1"] + Rw21LW * Cld_state["w2"]
    LW2: np.ndarray = Rw12LW * Cld_state["w1"] + Rw22LW * Cld_state["w2"]
    SW1: np.ndarray = Rw11SW * Cld_state["w1"] + Rw21SW * Cld_state["w2"]
    SW2: np.ndarray = Rw12SW * Cld_state["w1"] + Rw22SW * Cld_state["w2"]

    # ------------------------------------------------
    # Calculate mean and standard deviation of Qr
    # ------------------------------------------------

    LW1_tavg: np.ndarray = np.abs(LW1).mean(axis=0)
    SW1_tavg: np.ndarray = np.abs(SW1).mean(axis=0)

    LW2_tavg: np.ndarray = np.abs(LW2).mean(axis=0)
    SW2_tavg: np.ndarray = np.abs(SW2).mean(axis=0)

    LW1_tstd: np.ndarray = np.abs(LW1).std(axis=0)
    SW1_tstd: np.ndarray = np.abs(SW1).std(axis=0)

    LW2_tstd: np.ndarray = np.abs(LW2).std(axis=0)
    SW2_tstd: np.ndarray = np.abs(SW2).std(axis=0)

    # ------------------------------------------------
    # Calculate Vertical profiles of Qr
    # ------------------------------------------------

    # Calculate daily mean
    ## design time slice
    time_slice: list[int] = [26, 27, 28]

    LW1_daily: np.ndarray = np.array([
            LW1[i*4:(i+1)*4].mean(axis=0) for i in range(LW1.shape[0]//4-1)
            ])[time_slice, :]
    LW2_daily: np.ndarray = np.array([
            LW2[i*4:(i+1)*4].mean(axis=0) for i in range(LW2.shape[0]//4-1)
            ])[time_slice, :]
    SW1_daily: np.ndarray = np.array([
            SW1[i*4:(i+1)*4].mean(axis=0) for i in range(SW1.shape[0]//4-1)
            ])[time_slice, :]
    SW2_daily: np.ndarray = np.array([
            SW2[i*4:(i+1)*4].mean(axis=0) for i in range(SW2.shape[0]//4-1)
            ])[time_slice, :]

    # calculate horizontal structure of convective heating
    ## design basis
    x: np.ndarray = np.arange(-4e7, 4e7+5e4, 5e4)
    
    F_basis: np.ndarray = np.exp(1j * cld_k * x / 4.32e6)

    ## Multiply coefficient with basis
    LW1_2D : np.ndarray = np.einsum("te,x->txe", LW1_daily, F_basis)
    LW2_2D : np.ndarray = np.einsum("te,x->txe", LW2_daily, F_basis)
    SW1_2D : np.ndarray = np.einsum("te,x->txe", SW1_daily, F_basis)
    SW2_2D : np.ndarray = np.einsum("te,x->txe", SW2_daily, F_basis)

    # Calculate vertical structure
    ## design Galerkin basis
    z : np.ndarray = np.linspace(0, 14000, 71)
    G1: np.ndarray = np.pi/2 * np.sin(np.pi*z/z.max()) * (9.81/1004.5 - 0.0065)
    G2: np.ndarray = np.pi/2 * np.sin(2*np.pi*z/z.max()) * (9.81/1004.5 - 0.0065)

    ## Calculate mean density profile
    Tmean    : np.ndarray = 300.0 - 0.0065*z
    pmean    : np.ndarray = 1e5 * (1 - 0.0065*z/300.0) ** (9.81/0.0065/287.5)
    alphamean: np.ndarray = 1/(pmean / Tmean / 287.5)


    ## Calculate vertical profile
    LW1_3D : np.ndarray = np.einsum("txe,z->txze", LW1_2D, G1) * alphamean[None, None, :, None]
    LW2_3D : np.ndarray = np.einsum("txe,z->txze", LW2_2D, G2) * alphamean[None, None, :, None]
    SW1_3D : np.ndarray = np.einsum("txe,z->txze", SW1_2D, G1) * alphamean[None, None, :, None]
    SW2_3D : np.ndarray = np.einsum("txe,z->txze", SW2_2D, G2) * alphamean[None, None, :, None]

    # ------------------------------------------------
    # Composite Leq profile
    # ------------------------------------------------
    
    # Pre-allocate data
    LW1_rolled: np.ndarray = np.zeros_like(LW1_3D)
    LW2_rolled: np.ndarray = np.zeros_like(LW2_3D)
    SW1_rolled: np.ndarray = np.zeros_like(SW1_3D)
    SW2_rolled: np.ndarray = np.zeros_like(SW2_3D)

    # Roll data
    for j in range(cld_shifts.shape[0]):
        for e in range(cld_shifts.shape[-1]):
            LW1_rolled[j, ..., e] = np.roll(LW1_3D[j, ..., e], shift=cld_shifts[j, e], axis=0)
            LW2_rolled[j, ..., e] = np.roll(LW2_3D[j, ..., e], shift=cld_shifts[j, e], axis=0)
            SW1_rolled[j, ..., e] = np.roll(SW1_3D[j, ..., e], shift=cld_shifts[j, e], axis=0)
            SW2_rolled[j, ..., e] = np.roll(SW2_3D[j, ..., e], shift=cld_shifts[j, e], axis=0)

    # Composite and chunk the domain
    LW_chunked: np.ndarray = np.real(
            LW1_rolled.mean(axis=(0, -1)) + LW2_rolled.mean(axis=(0, -1))
            )[cld_neg_pi:cld_pos_pi+1]

    SW_chunked: np.ndarray = np.real(
            SW1_rolled.mean(axis=(0, -1)) + SW2_rolled.mean(axis=(0, -1))
            )[cld_neg_pi:cld_pos_pi+1]

    # Form data to save
    LW_chunked_wo_composite: np.ndarray = np.real(
            LW1_rolled[:, cld_neg_pi:cld_pos_pi+1, ...] + LW2_rolled[:, cld_neg_pi:cld_pos_pi+1, ...]
            ).transpose(0, 3, 1, 2)

    SW_chunked_wo_composite: np.ndarray = np.real(
            SW1_rolled[:, cld_neg_pi:cld_pos_pi+1, ...] + SW2_rolled[:, cld_neg_pi:cld_pos_pi+1, ...]
            ).transpose(0, 3, 1, 2)


    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    # setting time coordinate
    t_coords: np.ndarray = np.arange(LW1_tstd.shape[0])/4
    
    # setting figure path
    figure_path: Path = home_path / "Figure"

    # plot for LW heating evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, LW1_tavg,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"LW 1"
            )
    ax.plot(
            t_coords, LW2_tavg,
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"LW 2"
            )

    ax.fill_between(
            t_coords,
            LW1_tavg - LW1_tstd,
            LW1_tavg + LW1_tstd,
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            LW2_tavg - LW2_tstd,
            LW2_tavg + LW2_tstd,
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K/day")
    ax.set_title(r"LW heating Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "LW_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot for SW heating evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))


    ax.plot(
            t_coords, SW1_tavg,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"SW 1"
            )
    ax.plot(
            t_coords, SW2_tavg,
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"SW 2"
            )

    ax.fill_between(
            t_coords,
            SW1_tavg - SW1_tstd,
            SW1_tavg + SW1_tstd,
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            SW2_tavg - SW2_tstd,
            SW2_tavg + SW2_tstd,
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K/day")
    ax.set_title(r"SW heating Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "SW_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot overlay of LW profiles
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    lw_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, int(cld_pos_pi - cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            LW_chunked.T.real, 
            cmap="RdBu_r",
            levels=np.linspace(-0.01, 0.01, 11), extend="both"
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"LW Radiative Heating Rate (K/day)")
    cbar = fig.colorbar(lw_ctf, ax=ax, label="K/day")
    cbar.set_ticks([-0.01, -0.005, 0,00, 0.005, 0.01])

    plt.savefig(figure_path / "LW.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot overlay of SW profiles
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    sw_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, int(cld_pos_pi - cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            SW_chunked.T.real, 
            cmap="RdBu_r",
            levels=np.linspace(-0.01, 0.01, 11), extend="both"
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"SW Radiative Heating Rate (K/day)")
    cbar = fig.colorbar(sw_ctf, ax=ax, label="K/day")
    cbar.set_ticks([-0.01, -0.005, 0,00, 0.005, 0.01])

    plt.savefig(figure_path / "SW.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------
    # save files
    # ------------------------------------------------

    
    np.save(home_path / "data/composite/CldRad/LW.npy", LW_chunked_wo_composite)
    np.save(home_path / "data/composite/CldRad/SW.npy", SW_chunked_wo_composite)


# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
