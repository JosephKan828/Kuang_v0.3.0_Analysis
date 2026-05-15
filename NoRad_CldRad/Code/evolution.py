# ====================================================
# This script is for demonstrate the evolution of
# different variables
# ====================================================

# ====================================================
# Import package
# ====================================================

import h5py
import numpy as np

from typing import cast
from pathlib import Path

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

    # path setting
    work_path  : Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")

    NoRad_path : Path = work_path / "NoRad_combined"
    CldRad_path: Path = work_path / "CldRad_combined"

    # Load state data
    ## Load NoRad experiment
    with h5py.File(NoRad_path / "State.h5", "r") as no_file:
        No_state: dict[str, np.ndarray] = {
                key: np.abs(cast(h5py.Dataset, value)[...])
                    for (key, value) in no_file.items()
                }
    print(No_state.keys())

    ## Load CldRad experiment
    with h5py.File(CldRad_path / "State.h5", "r") as cld_file:
        Cld_state: dict[str, np.ndarray] = {
                key: np.abs(cast(h5py.Dataset, value)[..., :46][...])
                    for (key, value) in cld_file.items()
                }

    # ------------------------------------------------
    # Calculate ensemble mean and standard deviation
    # ------------------------------------------------

    # Calculate mean
    No_avg: dict[str, np.ndarray] = {
                key: np.nanmean(val, axis=-1)
                for (key, val) in No_state.items()
            }
    Cld_avg: dict[str, np.ndarray] = {
                key: np.nanmean(val, axis=-1)
                for (key, val) in Cld_state.items()
            }

    # Calculate standard deviation
    No_std: dict[str, np.ndarray] = {
                key: np.nanstd(val, axis=-1)
                for (key, val) in No_state.items()
            }
    Cld_std: dict[str, np.ndarray] = {
                key: np.nanstd(val, axis=-1)
                for (key, val) in Cld_state.items()
            }

    # ------------------------------------------------
    # visualization
    # ------------------------------------------------

    # setting time coordinate
    t_coords: np.ndarray = np.arange(No_std["w1"].shape[0])/4
    
    # setting figure path
    figure_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad/Figure")

    # plot for vertical motion evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, No_avg["w1"] ,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"NoRad $w_1$"
            )
    ax.plot(
            t_coords, Cld_avg["w1"],
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"CldRad $w_1$"
            )

    ax.fill_between(
            t_coords,
            No_avg["w1"] - No_std["w1"],
            No_avg["w1"] + No_std["w1"],
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            Cld_avg["w1"] - Cld_std["w1"],
            Cld_avg["w1"] + Cld_std["w1"],
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K/day")
    ax.set_title(r"$w_1$ Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "w1_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plot for vertical motion evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, No_avg["w2"] ,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"NoRad $w_2$"
            )
    ax.plot(
            t_coords, Cld_avg["w2"],
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"CldRad $w_2$"
            )

    ax.fill_between(
            t_coords,
            No_avg["w2"] - No_std["w2"],
            No_avg["w2"] + No_std["w2"],
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            Cld_avg["w2"] - Cld_std["w2"],
            Cld_avg["w2"] + Cld_std["w2"],
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K/day")
    ax.set_title(r"$w_2$ Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "w2_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plot for T1 evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, No_avg["T1"] ,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"NoRad $T_1$"
            )
    ax.plot(
            t_coords, Cld_avg["T1"],
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"CldRad $T_1$"
            )

    ax.fill_between(
            t_coords,
            No_avg["T1"] - No_std["T1"],
            No_avg["T1"] + No_std["T1"],
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            Cld_avg["T1"] - Cld_std["T1"],
            Cld_avg["T1"] + Cld_std["T1"],
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K")
    ax.set_title(r"$T_1$ Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "T1_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plot for T_2 evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, No_avg["T2"] ,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"NoRad $T_2$"
            )
    ax.plot(
            t_coords, Cld_avg["T2"],
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"CldRad $T_2$"
            )

    ax.fill_between(
            t_coords,
            No_avg["T2"] - No_std["T2"],
            No_avg["T2"] + No_std["T2"],
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            Cld_avg["T2"] - Cld_std["T2"],
            Cld_avg["T2"] + Cld_std["T2"],
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K")
    ax.set_title(r"$T_2$ Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "T2_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plot for J1 evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, No_avg["J1"] ,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"NoRad $J_1$"
            )
    ax.plot(
            t_coords, Cld_avg["J1"],
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"CldRad $J_1$"
            )

    ax.fill_between(
            t_coords,
            No_avg["J1"] - No_std["J1"],
            No_avg["J1"] + No_std["J1"],
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            Cld_avg["J1"] - Cld_std["J1"],
            Cld_avg["J1"] + Cld_std["J1"],
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K/day")
    ax.set_title(r"$J_1$ Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "J1_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plot for J2 evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, No_avg["J2"] ,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"NoRad $J2$"
            )
    ax.plot(
            t_coords, Cld_avg["J2"],
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"CldRad $J_2$"
            )

    ax.fill_between(
            t_coords,
            No_avg["J2"] - No_std["J2"],
            No_avg["J2"] + No_std["J2"],
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            Cld_avg["J2"] - Cld_std["J2"],
            Cld_avg["J2"] + Cld_std["J2"],
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K/day")
    ax.set_title(r"$J_2$ Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "J2_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # plot for q evolution
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    ax.plot(
            t_coords, No_avg["q"] ,
            color="C0",
            linewidth=4, linestyle="-",
            label=r"NoRad $q$"
            )
    ax.plot(
            t_coords, Cld_avg["q"],
            color="C2", 
            linewidth=4, linestyle="-",
            label=r"CldRad $q$"
            )

    ax.fill_between(
            t_coords,
            No_avg["q"] - No_std["q"],
            No_avg["q"] + No_std["q"],
            color="C0", alpha=0.2
            )
    ax.fill_between(
            t_coords,
            Cld_avg["q"] - Cld_std["q"],
            Cld_avg["q"] + Cld_std["q"],
            color="C2", alpha=0.2
            )

    ax.set_xticks(np.linspace(0, 30, 7))
    ax.set_xlim(0, t_coords.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("Time [day]")
    ax.set_ylabel("K")
    ax.set_title(r"$q$ Evolution")
    ax.legend(frameon=False, loc="best")
    
    plt.savefig(figure_path / "q_evo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()

