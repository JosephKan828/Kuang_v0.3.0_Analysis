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

    # Load phase-related information
    no_shifts: np.ndarray = np.load(home_path / "data/composite/NoRad/shifts.npy")
    no_neg_pi: np.ndarray = np.load(home_path / "data/composite/NoRad/neg_pi.npy")
    no_pos_pi: np.ndarray = np.load(home_path / "data/composite/NoRad/pos_pi.npy")

    cld_shifts: np.ndarray = np.load(home_path / "data/composite/CldRad/shifts.npy")
    cld_neg_pi: np.ndarray = np.load(home_path / "data/composite/CldRad/neg_pi.npy")
    cld_pos_pi: np.ndarray = np.load(home_path / "data/composite/CldRad/pos_pi.npy")


    # Load eigen analysis for identify wavenumber
    with h5py.File(work_path / "Rad(0.0,0.0,0.0)" / "latest" / "EigenAnalysis.h5", "r") as file:
        wavenumber: np.ndarray = cast(h5py.Dataset, file["k"])[...]
        no_growth : np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]

    with h5py.File(work_path / "Rad(0.0,0.0,0.1)" / "latest" / "EigenAnalysis.h5", "r") as file:
        cld_growth: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]

    no_k : float = wavenumber[np.argmax(no_growth)]
    cld_k: float = wavenumber[np.argmax(cld_growth)]

    # Load state vector of both experiments
    ## Load NoRad experiment
    with h5py.File(work_path / "NoRad_combined" / "State.h5", "r") as file:
        No_state: dict[str, np.ndarray] = {
                key: cast(h5py.Dataset, value)[...]
                for (key, value) in file.items()
                if key.startswith(("w", "q", "T"))
                }

    ## Load CldRad experiment
    with h5py.File(work_path / "CldRad_combined" / "State.h5", "r") as file:
        Cld_state: dict[str, np.ndarray] = {
                key: cast(h5py.Dataset, value)[..., :46]
                for (key, value) in file.items()
                if key.startswith(("w", "q", "T"))
                }

    # ------------------------------------------------
    # Calculate Leq evolution
    # ------------------------------------------------

    # Calculate associated coefficients
    b1: float = 1.0
    b2: float = 2.0
    f : float = 0.5
    F : float = 4.0
    rq: float = 0.7
    r0: float = 1.0
    A : float = 1.0 - 2.0*f + (b2 - b1)/F 
    B : float = 1 + (b2+b1)/F - A*r0

    # Calculate Leq evolution for the two experiments
    No_Leq: np.ndarray = (
            A * rq * (No_state["q"] - 1.5*No_state["T1"]) + \
                    f * No_state["w1"] + \
                    (1-f) * No_state["w1"]
            ) / B

    Cld_Leq: np.ndarray = (
            A * rq * (Cld_state["q"] - 1.5*Cld_state["T1"]) + \
                    f * Cld_state["w1"] + \
                    (1-f) * Cld_state["w1"]
            ) / B

    # ------------------------------------------------
    # Calculate mean and standard deviation of Leq
    # ------------------------------------------------
    
    # NoRad
    No_mean: np.ndarray = np.abs(No_Leq).mean(axis=-1)
    No_std : np.ndarray = np.abs(No_Leq).mean(axis=-1)

    # CldRad
    Cld_mean: np.ndarray = np.abs(Cld_Leq).mean(axis=-1)
    Cld_std : np.ndarray = np.abs(Cld_Leq).mean(axis=-1)

    # ------------------------------------------------
    # Calculate Vertical profiles of J eq
    # ------------------------------------------------

    # Calculate upper-level convective heating
    No_Ueq : np.ndarray = r0 * No_Leq  + rq * (No_state["q"]-1.5*No_state["T1"])
    Cld_Ueq: np.ndarray = r0 * Cld_Leq + rq * (Cld_state["q"]-1.5*Cld_state["T1"])

    # Calculate convective heating associated with the first and second modes
    No_J1: np.ndarray = No_Leq + No_Ueq
    No_J2: np.ndarray = No_Leq - No_Ueq

    Cld_J1: np.ndarray = Cld_Leq + Cld_Ueq
    Cld_J2: np.ndarray = Cld_Leq - Cld_Ueq

    # Calculate daily mean
    ## design time slice
    time_slice: list[int] = [26, 27, 28]

    No_J1_daily: np.ndarray = np.array([
            No_J1[i*4:(i+1)*4].mean(axis=0) for i in range(No_J1.shape[0]//4-1)
            ])[time_slice, :]
    No_J2_daily: np.ndarray = np.array([
            No_J2[i*4:(i+1)*4].mean(axis=0) for i in range(No_J2.shape[0]//4-1)
            ])[time_slice, :]
    Cld_J1_daily: np.ndarray = np.array([
            Cld_J1[i*4:(i+1)*4].mean(axis=0) for i in range(Cld_J1.shape[0]//4-1)
            ])[time_slice, :]
    Cld_J2_daily: np.ndarray = np.array([
            Cld_J2[i*4:(i+1)*4].mean(axis=0) for i in range(Cld_J2.shape[0]//4-1)
            ])[time_slice, :]

    # calculate horizontal structure of convective heating
    ## design basis
    x: np.ndarray = np.arange(-4e7, 4e7+5e4, 5e4)
    
    No_F_basis : np.ndarray = np.exp(1j * no_k * x / 4.32e6)
    Cld_F_basis: np.ndarray = np.exp(1j * cld_k * x / 4.32e6)

    ## Multiply coefficient with basis
    No_J1_2D : np.ndarray = np.einsum("te,x->txe", No_J1_daily , No_F_basis)
    No_J2_2D : np.ndarray = np.einsum("te,x->txe", No_J2_daily , No_F_basis)
    Cld_J1_2D: np.ndarray = np.einsum("te,x->txe", Cld_J1_daily, Cld_F_basis)
    Cld_J2_2D: np.ndarray = np.einsum("te,x->txe", Cld_J2_daily, Cld_F_basis)

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
    No_J1_3D : np.ndarray = np.einsum("txe,z->txze", No_J1_2D, G1) * alphamean[None, None, :, None]
    No_J2_3D : np.ndarray = np.einsum("txe,z->txze", No_J2_2D, G2) * alphamean[None, None, :, None]
    Cld_J1_3D: np.ndarray = np.einsum("txe,z->txze", Cld_J1_2D, G1) * alphamean[None, None, :, None]
    Cld_J2_3D: np.ndarray = np.einsum("txe,z->txze", Cld_J2_2D, G2) * alphamean[None, None, :, None]

    # ------------------------------------------------
    # Composite Leq profile
    # ------------------------------------------------
    
    # Pre-allocate data
    No_J1_rolled: np.ndarray = np.zeros_like(No_J1_3D)
    No_J2_rolled: np.ndarray = np.zeros_like(No_J2_3D)
    Cld_J1_rolled: np.ndarray = np.zeros_like(Cld_J1_3D)
    Cld_J2_rolled: np.ndarray = np.zeros_like(Cld_J2_3D)

    # Roll data
    for j in range(no_shifts.shape[0]):
        for e in range(no_shifts.shape[-1]):
            No_J1_rolled[j, ..., e]  = np.roll(No_J1_3D[j, ..., e], shift=no_shifts[j, e], axis=0)
            No_J2_rolled[j, ..., e]  = np.roll(No_J2_3D[j, ..., e], shift=no_shifts[j, e], axis=0)
            Cld_J1_rolled[j, ..., e] = np.roll(Cld_J1_3D[j, ..., e], shift=cld_shifts[j, e], axis=0)
            Cld_J2_rolled[j, ..., e] = np.roll(Cld_J2_3D[j, ..., e], shift=cld_shifts[j, e], axis=0)

    # Composite and chunk the domain
    No_J_chunked: np.ndarray = np.real(
            No_J1_rolled.mean(axis=(0, -1)) + No_J2_rolled.mean(axis=(0, -1))
            )[no_neg_pi:no_pos_pi+1]

    Cld_J_chunked: np.ndarray = np.real(
            Cld_J1_rolled.mean(axis=(0, -1)) + Cld_J2_rolled.mean(axis=(0, -1))
            )[cld_neg_pi:cld_pos_pi+1]

    # Form data to save
    No_J_chunked_wo_composite: np.ndarray = np.real(
            No_J1_rolled[:, no_neg_pi:no_pos_pi+1, ...] + No_J2_rolled[:, no_neg_pi:no_pos_pi+1, ...]
            ).transpose(0, 3, 1, 2)

    Cld_J_chunked_wo_composite: np.ndarray = np.real(
            Cld_J1_rolled[:, cld_neg_pi:cld_pos_pi+1, ...] + Cld_J2_rolled[:, cld_neg_pi:cld_pos_pi+1, ...]
            ).transpose(0, 3, 1, 2)


    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    # setting time coordinate
    t_coords: np.ndarray = np.arange(No_std.shape[0])/4
    
    # setting figure path
    figure_path: Path = home_path / "Figure"

    # plot for vertical motion evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, No_mean,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"NoRad $L_{eq}$"
            )
    ax.plot(
            t_coords, Cld_mean,
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"CldRad $L_{eq}$"
            )

    ax.fill_between(
            t_coords,
            No_mean - No_std,
            No_mean + No_std,
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            Cld_mean - Cld_std,
            Cld_mean + Cld_std,
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K/day")
    ax.set_title(r"$L_{eq}$ Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "Leq_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
 
    # Plot overlay of Jeq profiles
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, int(no_pos_pi - no_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_J_chunked.T, 
            cmap="RdBu_r",
            levels=np.linspace(-0.2, 0.2, 9), extend="both"
            )

    cld_ct = ax.contour(
            np.linspace(-np.pi, np.pi, int(cld_pos_pi - cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_J_chunked.T,
            colors="k", levels=np.linspace(-0.2, 0.2, 9), linewidths=4
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"NoRad $J_{eq}$ (shading; K/day) vs. CldRad $J_{eq}$ (black contour; K/day)")
    ax.clabel(cld_ct, inline=True, fontsize=12)
    cbar = fig.colorbar(no_ctf, ax=ax, label="K/day")
    cbar.set_ticks([-0.2, -0.1, 0,00, 0.1, 0.2])

    plt.savefig(figure_path / "Jeq_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()


    # ------------------------------------------------
    # save files
    # ------------------------------------------------

    
    np.save(home_path / "data/composite/NoRad/Jeq.npy", No_J_chunked_wo_composite)
    np.save(home_path / "data/composite/CldRad/Jeq.npy", Cld_J_chunked_wo_composite)


# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
