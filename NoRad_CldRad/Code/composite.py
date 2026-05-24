# ====================================================
# This script is to apply composite analysis
# ====================================================

# ====================================================
# Environment Setup
# ====================================================

import os
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
# Helper function
# ====================================================

# Find minimum index
def _find_min(
        data   : np.ndarray,
        idx_max: int
        ) -> int:

    # invert data
    data_lim: np.ndarray = data[idx_max:]
    return np.argmin(data_lim[:100]) + idx_max


# ====================================================
# Main function
# ====================================================

def main():
    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    # path setting
    work_path  : Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")

    NoRad_path : Path = work_path / "NoRad_combined"
    CldRad_path: Path = work_path / "CldRad_combined"

    # Load Galerkin state data
    # define composite time slice
    time_slice: list[int] = [26, 27, 28]

    ## Load NoRad experiment
    with h5py.File(NoRad_path / "Galerkin.h5", "r") as no_file:
        No_state: dict[str, np.ndarray] = {
                key: cast(h5py.Dataset, value)[..., time_slice, :].real
                    for (key, value) in no_file.items()
                }

    ## Load CldRad experiment
    with h5py.File(CldRad_path / "Galerkin.h5", "r") as cld_file:
        Cld_state: dict[str, np.ndarray] = {
                key: cast(h5py.Dataset, value)[..., :46][..., time_slice, :].real
                    for (key, value) in cld_file.items()
                }

    # ------------------------------------------------
    # Combining field
    # ------------------------------------------------

    No_state_combined : dict[str, np.ndarray] = {
                "w": No_state["w1"] + No_state["w2"],
                "T": No_state["T1"] + No_state["T2"],
                "J": No_state["J1"] + No_state["J2"]
            }

    Cld_state_combined: dict[str, np.ndarray] = {
                "w": Cld_state["w1"] + Cld_state["w2"],
                "T": Cld_state["T1"] + Cld_state["T2"],
                "J": Cld_state["J1"] + Cld_state["J2"]
            }

    # ------------------------------------------------
    # Roll data for composite
    # ------------------------------------------------

    # Vertically averaging convective heating
    No_J_vavg : np.ndarray = No_state_combined["J"].mean(axis=0)
    Cld_J_vavg: np.ndarray = Cld_state_combined["J"].mean(axis=0)

    # Roll vertically averaged to centering
    nx    : int = No_J_vavg.shape[0] # size of x domain

    margin: int = nx // 3 # margin for rolling data

    ## Calculate shifts in x domain
    ### Calculate index for peak
    No_max_loc : np.ndarray = np.argmax(No_J_vavg[margin:-margin, ...] , axis=0) + margin
    Cld_max_loc: np.ndarray = np.argmax(Cld_J_vavg[margin:-margin, ...], axis=0) + margin

    ### Calculate shifts
    No_shifts : np.ndarray = nx // 2 - No_max_loc
    Cld_shifts: np.ndarray = nx // 2 - Cld_max_loc

    ## Roll data
    ### Pre-allocate dictionary
    No_rolled: dict[str, np.ndarray] = {}
    Cld_rolled: dict[str, np.ndarray] = {}

    for key in No_state_combined.keys():
        No_rolled[key] = np.array([
            np.roll(No_state_combined[key][:, :, j, i], shift=No_shifts[j, i], axis=1)
            for j in range(No_shifts.shape[0])
            for i in range(No_shifts.shape[1])
            ])

        Cld_rolled[key] = np.array([
            np.roll(Cld_state_combined[key][:, :, j, i], shift=Cld_shifts[j, i], axis=1)
            for j in range(Cld_shifts.shape[0])
            for i in range(Cld_shifts.shape[1])
            ])

    # ------------------------------------------------
    # Composite
    # ------------------------------------------------
    # Averaging over all the events
    No_composite: dict[str, np.ndarray] = {
            key: val.mean(axis=0)
            for (key, val) in No_rolled.items()
            }
    Cld_composite: dict[str, np.ndarray] = {
            key: val.mean(axis=0)
            for (key, val) in Cld_rolled.items()
            }

    # Identify the index for phase
    ## Identify -pi/2
    neg_pi_2: np.ndarray = nx // 2 # set peak as -pi/2

    ## Identify pi/2 for the two experiments
    No_composite_J_vavg : np.ndarray = No_composite["J"].mean(axis=0)
    Cld_composite_J_vavg: np.ndarray = Cld_composite["J"].mean(axis=0)

    No_pos_pi_2 : int = np.argmin(No_composite_J_vavg[neg_pi_2:neg_pi_2+100]) + neg_pi_2
    Cld_pos_pi_2: int = np.argmin(Cld_composite_J_vavg[neg_pi_2:neg_pi_2+100]) + neg_pi_2

    ## Calculate left and right boundary
    No_pi_grid : int = int((No_pos_pi_2 - neg_pi_2))
    Cld_pi_grid: int = int((Cld_pos_pi_2 - neg_pi_2))

    No_neg_pi: int = neg_pi_2 - int(No_pi_grid // 2)
    No_pos_pi: int = No_pos_pi_2 + int(No_pi_grid // 2)

    Cld_neg_pi: int = neg_pi_2 - int(Cld_pi_grid // 2)
    Cld_pos_pi: int = Cld_pos_pi_2 + int(Cld_pi_grid // 2)

    # Chunking compostie data
    No_chunked: dict[str, np.ndarray] = {
            key: val[:, No_neg_pi:No_pos_pi+1]
            for (key, val) in No_composite.items()
            }
    Cld_chunked: dict[str, np.ndarray] = {
            key: val[:, Cld_neg_pi:Cld_pos_pi+1]
            for (key, val) in Cld_composite.items()
            }

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # Setup path
    figure_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad/Figure")

    # Plot NoRad experiment
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex="col")

    ax1, ax2 = axes.flatten()

    J_ctf = ax1.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["J"],
            cmap="BrBG", levels=np.linspace(-0.15, 0.15, 11)
            )

    w_ct = ax1.contour(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["w"]/86400.0 * 1000.0, colors="k",
            levels=[i for i in np.linspace(-0.6, 0.6, 7) if i != 0],
            linewidths=4
            )

    ax1.minorticks_on()
    ax1.set_yticks(np.linspace(0, 12000, 7))
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(0, 14000)
    ax1.set_ylabel("Level [m]")
    ax1.set_title(r"$J$ (shading; K/day) vs. $w$ (black contour; $\times$ 1000 m/s)")
    ax1.clabel(w_ct, inline=True, fontsize=12)

    fig.colorbar(J_ctf, ax=ax1, label="K/day")


    J_ctf = ax2.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["J"],
            cmap="BrBG", levels=np.linspace(-0.15, 0.15, 11)
            )

    T_ct = ax2.contour(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["T"], colors="k",
            levels=np.linspace(-0.04, 0.04, 5),
            linewidths=4
            )

    ax2.minorticks_on()
    ax2.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax2.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax2.set_yticks(np.linspace(0, 12000, 7))
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(0, 14000)
    ax2.set_xlabel("Phase [rad]")
    ax2.set_ylabel("Level [m]")
    ax2.set_title(r"$J$ (shading; K/day) vs. $T$ (black contour; K)")
    ax2.clabel(T_ct, inline=True, fontsize=12)
    fig.colorbar(J_ctf, ax=ax2, label="K/day")

    plt.savefig(figure_path / "NoRad_composite.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot CldRad experiment
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex="col")

    ax1, ax2 = axes.flatten()

    J_ctf = ax1.contourf(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["J"],
            cmap="BrBG", levels=np.linspace(-0.15, 0.15, 11)
            )

    w_ct = ax1.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["w"]/86400.0 * 1000.0, colors="k",
            levels=[i for i in np.linspace(-0.6, 0.6, 7) if i != 0],
            linewidths=4
            )

    ax1.minorticks_on()
    ax1.set_yticks(np.linspace(0, 12000, 7))
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(0, 14000)
    ax1.set_ylabel("Level [m]")
    ax1.set_title(r"$J$ (shading; K/day) vs. $w$ (black contour; $\times$ 1000 m/s)")
    ax1.clabel(w_ct, inline=True, fontsize=12)

    fig.colorbar(J_ctf, ax=ax1, label="K/day")


    J_ctf = ax2.contourf(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["J"],
            cmap="BrBG", levels=np.linspace(-0.15, 0.15, 11)
            )

    T_ct = ax2.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["T"], colors="k",
            levels=np.linspace(-0.04, 0.04, 5),
            linewidths=4
            )

    ax2.minorticks_on()
    ax2.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax2.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax2.set_yticks(np.linspace(0, 12000, 7))
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(0, 14000)
    ax2.set_xlabel("Phase [rad]")
    ax2.set_ylabel("Level [m]")
    ax2.set_title(r"$J$ (shading; K/day) vs. $T$ (black contour; K)")
    ax2.clabel(T_ct, inline=True, fontsize=12)
    fig.colorbar(J_ctf, ax=ax2, label="K/day")

    plt.savefig(figure_path / "CldRad_composite.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot overlay of vertical motion
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["w"] / 86400 * 1000, 
            cmap="RdBu_r", levels=np.linspace(-0.8, 0.8, 9), extend="both"
            )

    cld_ct = ax.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["w"] / 86400 * 1000,
            colors="k", levels=np.linspace(-0.3, 0.3, 11), linewidths=4
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"NoRad $w$ (shading; $\times$ 1000 m/s) vs. CldRad $w$ (black contour; $\times$ 1000 m/s)")
    ax.clabel(cld_ct, inline=True, fontsize=12)
    cbar = fig.colorbar(no_ctf, ax=ax, label=r"$\times 1000$ m/s")
    cbar.set_ticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])

    plt.savefig(figure_path / "w_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()

     # Plot overlay of Convective heating
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["J"], 
            cmap="RdBu_r",
            levels=np.linspace(-0.15, 0.15, 11), extend="both"
            )

    cld_ct = ax.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["J"],
            colors="k", levels=np.linspace(-0.1, 0.1, 11), linewidths=4
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"NoRad $J$ (shading; K/day) vs. CldRad $J$ (black contour; K/day)")
    ax.clabel(cld_ct, inline=True, fontsize=12)
    cbar = fig.colorbar(no_ctf, ax=ax, label="K/day")
    cbar.set_ticks([-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])

    plt.savefig(figure_path / "J_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()


    # Plot overlay of temperature perturbation
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["T"], 
            cmap="RdBu_r",
            levels=np.linspace(-0.06, 0.06, 11), extend="both"
            )

    cld_ct = ax.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["T"],
            colors="k", levels=np.linspace(-0.06, 0.06, 11), linewidths=4
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"NoRad $T$ (shading; K) vs. CldRad $T$ (black contour; K)")
    ax.clabel(cld_ct, inline=True, fontsize=12)
    cbar = fig.colorbar(no_ctf, ax=ax, label="K")
    cbar.set_ticks([-0.06, -0.03, 0,00, 0.03, 0.06])

    plt.savefig(figure_path / "T_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()


    print("NoRad temperature max.: ", No_chunked["T"].max())
    print("CldRad temperature max.: ", Cld_chunked["T"].max())

    # ------------------------------------------------
    # Save file
    # ------------------------------------------------
    # setup path
    file_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad/data/composite")

    os.makedirs(file_path, exist_ok=True)

    np.save(file_path / "NoRad/J.npy", No_rolled["J"][..., No_neg_pi:(No_pos_pi+1)])
    np.save(file_path / "NoRad/T.npy", No_rolled["T"][..., No_neg_pi:(No_pos_pi+1)])
    np.save(file_path / "NoRad/w.npy", No_rolled["w"][..., No_neg_pi:(No_pos_pi+1)])

    np.save(file_path / "CldRad/J.npy", Cld_rolled["J"][..., Cld_neg_pi:(Cld_pos_pi+1)])
    np.save(file_path / "CldRad/T.npy", Cld_rolled["T"][..., Cld_neg_pi:(Cld_pos_pi+1)])
    np.save(file_path / "CldRad/w.npy", Cld_rolled["w"][..., Cld_neg_pi:(Cld_pos_pi+1)])

    np.save(file_path / "NoRad/shifts.npy", No_shifts)
    np.save(file_path / "NoRad/neg_pi.npy", No_neg_pi)
    np.save(file_path / "NoRad/pos_pi.npy", No_pos_pi)

    np.save(file_path / "CldRad/shifts.npy", Cld_shifts)
    np.save(file_path / "CldRad/neg_pi.npy", Cld_neg_pi)
    np.save(file_path / "CldRad/pos_pi.npy", Cld_pos_pi)


# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
