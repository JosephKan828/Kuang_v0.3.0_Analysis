# ====================================================
# This script is to compare EAPE bedget terms
# ====================================================

# ====================================================
# Environment Setup
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

    # path setup
    home_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad")
    work_path: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")

    # Design time slice
    time_slice: list[int] = [26, 27, 28]

    # Load Galerkin state
    ## Load NoRad experiment
    with h5py.File(work_path / "NoRad_combined" / "Galerkin.h5", "r") as file:
        No_state: dict[str, np.ndarray] = {
                key: cast(h5py.Dataset, value)[..., time_slice, :].real
                for (key, value) in file.items()
                }

    ## Load CldRad experiment
    with h5py.File(work_path / "CldRad_combined" / "Galerkin.h5", "r") as file:
        Cld_state: dict[str, np.ndarray] = {
                key: cast(h5py.Dataset, value)[..., time_slice, :][..., :46].real
                for (key, value) in file.items()
                }


    # ------------------------------------------------
    # Combining field
    # ------------------------------------------------

    No_state_combined : dict[str, np.ndarray] = {
                "w": (No_state["w1"] + No_state["w2"]),
                "T": No_state["T1"] + No_state["T2"],
                "J": No_state["J1"] + No_state["J2"]
            }

    Cld_state_combined: dict[str, np.ndarray] = {
                "w": (Cld_state["w1"] + Cld_state["w2"]),
                "T": Cld_state["T1"] + Cld_state["T2"],
                "J": Cld_state["J1"] + Cld_state["J2"]
            }

    # ------------------------------------------------
    # Calculate EAPE budget terms
    # ------------------------------------------------

    No_EAPE: dict[str, np.ndarray] = {
            "generation": No_state_combined["J"] * No_state_combined["T"],
            "conversion": No_state_combined["w"] * No_state_combined["T"] * (0.0065 - 9.81/1004.5)
            }
    Cld_EAPE: dict[str, np.ndarray] = {
            "generation": Cld_state_combined["J"] * Cld_state_combined["T"],
            "conversion": Cld_state_combined["w"] * Cld_state_combined["T"] * (0.0065 - 9.81/1004.5)
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

    for key in No_EAPE.keys():
        No_rolled[key] = np.array([
            np.roll(No_EAPE[key][:, :, j, i], shift=No_shifts[j, i], axis=1)
            for j in range(No_shifts.shape[0])
            for i in range(No_shifts.shape[1])
            ])

        Cld_rolled[key] = np.array([
            np.roll(Cld_EAPE[key][:, :, j, i], shift=Cld_shifts[j, i], axis=1)
            for j in range(Cld_shifts.shape[0])
            for i in range(Cld_shifts.shape[1])
            ])

    No_J_rolled: np.ndarray = np.array([
        np.roll(No_state_combined["J"][:, :, j, i], shift=No_shifts[j, i], axis=1)
        for j in range(No_shifts.shape[0])
        for i in range(No_shifts.shape[1])
        ])
    Cld_J_rolled: np.ndarray = np.array([
        np.roll(Cld_state_combined["J"][:, :, j, i], shift=Cld_shifts[j, i], axis=1)
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
    No_composite_J_vavg : np.ndarray = No_J_rolled.mean(axis=0).mean(axis=0)
    Cld_composite_J_vavg: np.ndarray = Cld_J_rolled.mean(axis=0).mean(axis=0)

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

    No_J_chunked : np.ndarray = No_J_rolled.mean(axis=0)[:, No_neg_pi:No_pos_pi+1]
    Cld_J_chunked: np.ndarray = Cld_J_rolled.mean(axis=0)[:, Cld_neg_pi:Cld_pos_pi+1]

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # Setup path
    figure_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad/Figure")

    # Plot NoRad experiment
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex="col")

    ax1, ax2 = axes.flatten()

    gen_ctf = ax1.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["generation"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    J_ct = ax1.contour(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_J_chunked, colors="k",
            levels=[i for i in np.linspace(-0.1, 0.1, 11) if np.abs(i) > 1e-5 ],
            linewidths=4
            )

    ax1.minorticks_on()
    ax1.set_yticks(np.linspace(0, 12000, 7))
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(0, 14000)
    ax1.set_ylabel("Level [m]")
    ax1.set_title(r"EAPE Generation (shading; $K^2$/day) vs. $J$ (black contour; K/day)")
    ax1.clabel(J_ct, inline=True, fontsize=12)

    fig.colorbar(gen_ctf, ax=ax1, label=r"$K^2$/day")


    conv_ctf = ax2.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["conversion"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    J_ct = ax2.contour(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_J_chunked, colors="k",
            levels=[i for i in np.linspace(-0.1, 0.1, 11) if np.abs(i) > 1e-5],
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
    ax2.set_title(r"EAPE Conversion (shading; $K^2$/day) vs. $J$ (black contour; K/day)")
    ax2.clabel(J_ct, inline=True, fontsize=12)
    fig.colorbar(conv_ctf, ax=ax2, label=r"$K^2$/day")

    plt.savefig(figure_path / "NoRad_EAPE.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot CldRad experiment
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex="col")

    ax1, ax2 = axes.flatten()

    gen_ctf = ax1.contourf(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["generation"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    J_ct = ax1.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_J_chunked, colors="k",
            levels=[i for i in np.linspace(-0.1, 0.1, 11) if np.abs(i) > 1e-5],
            linewidths=4
            )

    ax1.minorticks_on()
    ax1.set_yticks(np.linspace(0, 12000, 7))
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(0, 14000)
    ax1.set_ylabel("Level [m]")
    ax1.set_title(r"EAPE Generation (shading; $K^2$/day) vs. $J$ (black contour; K/day)")
    ax1.clabel(J_ct, inline=True, fontsize=12)

    fig.colorbar(gen_ctf, ax=ax1, label=r"$K^2$/day")


    conv_ctf = ax2.contourf(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["conversion"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    J_ct = ax2.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_J_chunked, colors="k",
            levels=[i for i in np.linspace(-0.1, 0.1, 11) if np.abs(i) > 1e-5],
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
    ax2.set_title(r"EAPE Conversion (shading; $K^2$/day) vs. $J$ (black contour; K/day)")
    ax2.clabel(J_ct, inline=True, fontsize=12)
    fig.colorbar(conv_ctf, ax=ax2, label=r"$K^2$/day")

    plt.savefig(figure_path / "CldRad_EAPE.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot the contrast in generation
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    cld_ct = ax.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["generation"],
            colors="k", linewidths=4,
            levels=[i for i in np.linspace(-5e-3, 5e-3, 11) if np.abs(i) > 1e-5]
            )

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["generation"],
            cmap="BrBG",
            levels = np.linspace(-1e-2, 1e-2, 11), extend="both"
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"EAPE Generation in NoRad (shading; $K^2$/day) vs. CldRad (black contour; $K^2$/day)")
    ax.clabel(cld_ct, inline=True, fontsize=12)
    cbar = fig.colorbar(no_ctf, ax=ax, label=r"$K^2$/day")
    cbar.set_ticks([-0.01, -0.008, -0.004, 0.000, 0.004, 0.008, 0.01])

    plt.savefig(figure_path / "generation_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot the contrast in conversion
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    cld_ct = ax.contour(
            np.linspace(-np.pi, np.pi, int(Cld_pos_pi - Cld_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            Cld_chunked["conversion"],
            colors="k", linewidths=4,
            levels=[i for i in np.linspace(-5e-3, 5e-3, 11) if np.abs(i) > 1e-5]
            )

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, int(No_pos_pi - No_neg_pi + 1)),
            np.linspace(0, 14000, 71),
            No_chunked["conversion"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11), extend="both"
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"EAPE Conversion in NoRad (shading; $K^2$/day) vs. CldRad (black contour; $K^2$/day)")
    ax.clabel(cld_ct, inline=True, fontsize=12)
    cbar = fig.colorbar(no_ctf, ax=ax, label=r"$K^2$/day")
    cbar.set_ticks([-0.010, -0.008, -0.004, 0.000, 0.004, 0.008, 0.010])

    plt.savefig(figure_path / "conversion_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()


# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
