# ====================================================
# This script is to visualize difference in growth rate
# ====================================================

# ====================================================
# Import package
# ====================================================

import h5py
import numpy as np

from typing import cast
from pathlib import Path

from matplotlib import pyplot as plt

# use style sheet
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Main function
# ====================================================
def main():

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    # Experiment and name list
    ExpList: list[str] = [
        "Rad(0.0,0.0,0.0)", "Rad(0.001,0.0,0.0)",
        "Rad(0.0,0.001,0.0)", "Rad(0.0,0.0,0.1)"
        ]

    NameList: list[str] = [
        "No Rad.", "Moisture Rad.", "Temp. Rad.", "Cloud Rad."
    ]

    # Pre-allocate dictionary to save growth rate array
    GrowthRates: dict[str, np.ndarray] = {}


    # load growth rate
    ## File path
    WorkPath: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")

    for i, exp in enumerate(ExpList):

        with h5py.File(WorkPath / exp / "latest" / "EigenAnalysis.h5", "r") as file:
            GrowthRates[NameList[i]] = cast(h5py.Dataset, file["GrowthRates"])[:, 0]

            if i == 0:
                k: np.ndarray = cast(h5py.Dataset, file["k"])[...]
            else:
                continue

    # ------------------------------------------------
    # Visualizing 
    # ------------------------------------------------

    # Convert non-dimensional wavenumber
    k_nondim: np.ndarray = np.asarray(k * (40000.0 / (2*np.pi*4320.0)))

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    for i, key in enumerate(GrowthRates.keys()):
        ax.plot(
            k_nondim, GrowthRates[key],
            marker="o", markevery=1,
            markerfacecolor="white",
            markeredgewidth=1,
            linestyle="-", linewidth=2,
            label=key,
            zorder= 100 - i
        )

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 0.14)
    ax.set_xlabel("Non-dimensional Wavenumber")
    ax.set_ylabel(r"Growth Rate [ day$^{-1}$ ]")
    ax.tick_params(direction="in", length=6, width=1, top=True, right=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="best")
    plt.savefig("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure/GrowthRate.png", dpi=300, bbox_inches="tight")
    plt.close()


# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()