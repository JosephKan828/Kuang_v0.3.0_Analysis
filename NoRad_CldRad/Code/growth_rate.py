# ====================================================
# This script is to demonstrate growth rate difference
# between the two experiments
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
# define main function
# ====================================================

def main():
    # ------------------------------------------------
    # Load data 
    # ------------------------------------------------

    # Path setting
    home_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis")
    work_path: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")

    # Load eigendecomposition of the two experiments
    with h5py.File(work_path / "Rad(0.0,0.0,0.0)" / "latest" / "EigenAnalysis.h5", "r") as file:
        wavenumber    : np.ndarray = cast(h5py.Dataset, file["k"])[...]
        No_growth_rate: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]

    with h5py.File(work_path / "Rad(0.0,0.0,0.1)" / "latest" / "EigenAnalysis.h5", "r") as file:
        Cld_growth_rate: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # transform wavenumber
    k_display: np.ndarray = (40000.0 / (2*np.pi*4320.0)) * wavenumber

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))

    ax.plot(
            k_display, No_growth_rate,
            color="C0", linewidth=4, label="NoRad"
            )
    ax.plot(
            k_display, Cld_growth_rate,
            color="C2", linewidth=4, label="CldRad"
            )
    ax.text(
            20, 0.09,
            f"NoRad max.  : {No_growth_rate.max():.3f} 1/day\nCldRad max. : {Cld_growth_rate.max():.3f} 1/day"
            )
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(np.linspace(0, 30, 6))
    ax.set_yticks(np.linspace(0, 0.12, 7))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"Non-dimensional Wavenumber")
    ax.set_ylabel("1/day")
    ax.legend(frameon=False, loc=(20/30, 0.8))
    ax.grid(False)

    plt.savefig(
            home_path / "NoRad_CldRad" / "Figure" / "growth_rate.png", 
            dpi=300, bbox_inches="tight"
            )
    plt.close(fig)


# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()

