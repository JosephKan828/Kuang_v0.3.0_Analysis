# ====================================================
# This script is to calculate the time scale of wave
# ====================================================

# ====================================================
# Environment Setup
# ====================================================

import h5py
import numpy as np

from typing import cast
from pathlib import Path

from matplotlib import pyplot as plt

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

    # Load eigenvalues
    ## Load NoRad
    with h5py.File(work_path / "Rad(0.0,0.0,0.0)/latest" / "EigenAnalysis.h5", "r") as file:
        k    : np.ndarray = cast(h5py.Dataset, file["k"])[...]
        no_gr: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]
        no_ps: np.ndarray = cast(h5py.Dataset, file["PhaseSpeeds"])[:, 0]

    ## Load CldRad
    with h5py.File(work_path / "Rad(0.0,0.0,0.1)/latest" / "EigenAnalysis.h5", "r") as file:
        cld_gr: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]
        cld_ps: np.ndarray = cast(h5py.Dataset, file["PhaseSpeeds"])[:, 0]


    # ------------------------------------------------
    # Calculate characteristic time scale
    # ------------------------------------------------
    # dimensionalize wavenumber (unit: 1/m)
    dim_k: np.ndarray = k / (2*np.pi*4.32e7)

    # calculate time scale
    no_tauw : np.ndarray = 1 / dim_k / no_ps / 86400.0
    cld_tauw: np.ndarray = 1 / dim_k / cld_ps / 86400.0

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # transform wavenumber
    k_display: np.ndarray = (40000.0 / (2*np.pi*4320.0)) * k

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))

    ax.plot(
            k_display[1:], no_tauw[1:],
            color="C0", linewidth=4, label="NoRad"
            )
    ax.plot(
            k_display[1:], cld_tauw[1:],
            color="C2", linewidth=4, label="CldRad"
            )
    ax.axhline(1/12, xmin=0, xmax=1, color="k", linestyle="--", linewidth=2)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.linspace(0, 30, 6))
#    ax.set_yticks(np.linspace(0, 0.12, 7))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Non-dimensional Wavenumber")
    ax.set_ylabel("Time Scale [day]")
    ax.legend(frameon=False, loc=(20/30, 0.8))
    ax.grid(False)

    plt.savefig(
            home_path / "Figure" / "wave_time_scale.png", 
            dpi=300, bbox_inches="tight"
            )
    plt.close(fig)



# ====================================================
# Execute main function
# ====================================================
if __name__ == "__main__":
    main()
