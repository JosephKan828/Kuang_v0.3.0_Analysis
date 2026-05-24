# ====================================================
# This script is to calculate the composite profile of 
# convection work
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

    # path setup
    data_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad/data/composite")

    No_J : np.ndarray = np.load(data_path / "NoRad" / "J.npy")
    Cld_J: np.ndarray = np.load(data_path / "CldRad" / "J.npy")

    No_Jeq : np.ndarray = np.load(data_path / "NoRad" / "Jeq.npy")
    Cld_Jeq: np.ndarray = np.load(data_path / "CldRad" / "Jeq.npy")
    
    nt, nens, No_nx , nz = No_Jeq.shape
    _, _, Cld_nx, _ = Cld_Jeq.shape
    
    # ------------------------------------------------
    # Calculate time tendency
    # ------------------------------------------------

    # Reshape data
    No_Jeq  = No_Jeq.transpose(0, 1, 3, 2).reshape(nt*nens, nz, No_nx)
    Cld_Jeq = Cld_Jeq.transpose(0, 1, 3, 2).reshape(nt*nens, nz, Cld_nx)


    # Calculate time tendency
    No_tendency : np.ndarray = (No_Jeq - No_J) * 12
    Cld_tendency: np.ndarray = (Cld_Jeq - Cld_J) * 12

    # calculate statistics
    No_tendency_mean: np.ndarray = No_tendency.mean(axis=0)
    Cld_tendency_mean: np.ndarray = Cld_tendency.mean(axis=0)

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    # figure path
    figure_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad/Figure")

    # Plot overlay of temperature perturbation
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, No_tendency_mean.shape[1]),
            np.linspace(0, 14000, 71),
            No_tendency_mean, 
            cmap="RdBu_r",
            levels=np.linspace(-1, 1, 11), extend="both"
            )

    cld_ct = ax.contour(
            np.linspace(-np.pi, np.pi, Cld_tendency_mean.shape[1]),
            np.linspace(0, 14000, 71),
            Cld_tendency_mean,
            colors="k", levels=np.linspace(-0.6, 0.6, 7), linewidths=4
            )

    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks(np.linspace(0, 12000, 7))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.set_title(r"NoRad $dL/dt$ (shading; K$^2$/day$^2$) vs. CldRad $dL/dt$ (black contour; K$^2$/day$^2$)")
    ax.clabel(cld_ct, inline=True, fontsize=12)
    cbar = fig.colorbar(no_ctf, ax=ax, label=r"K$^2$/day$^2$")
    cbar.set_ticks([-1.0, -0.8, -0.4, 0, 0.4, 0.8, 1.0])

    plt.savefig(figure_path / "dLdt_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()



# ====================================================
# Excute main function
# ====================================================

if __name__ == "__main__":
    main()
