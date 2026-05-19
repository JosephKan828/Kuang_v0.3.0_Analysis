# -===================================================
# This script is to quantify the relation among
# convective heating, vertical motion, and temperature
# perturbations
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
# Helper function
# ====================================================

def phase_relation(
        data1: np.ndarray,
        data2: np.ndarray
        ) -> np.ndarray:

    # FFT on data
    fft1: np.ndarray = np.fft.fft(data1, axis=1)
    fft2: np.ndarray = np.fft.fft(data2, axis=1)

    # cross density
    cs: np.ndarray = fft1 * np.conj(fft2)

    phase_shift_rad: np.ndarray = np.angle (cs[:, 1])

    return phase_shift_rad


# ====================================================
# main function
# ====================================================

def main():
    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    # Path setting
    home_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad")

    # Load data
    ## Load NoRad experiment
    No_state: dict[str, np.ndarray] = {
            "J": np.load(home_path / "data" / "composite" / "NoRad" / "J.npy"),
            "T": np.load(home_path / "data" / "composite" / "NoRad" / "T.npy"),
            "w": np.load(home_path / "data" / "composite" / "NoRad" / "w.npy"),
            }

    ## Load NoRad experiment
    Cld_state: dict[str, np.ndarray] = {
            "J": np.load(home_path / "data" / "composite" / "CldRad" / "J.npy"),
            "T": np.load(home_path / "data" / "composite" / "CldRad" / "T.npy"),
            "w": np.load(home_path / "data" / "composite" / "CldRad" / "w.npy"),
            }

    # ------------------------------------------------
    # Calculate phase relation between variables
    # ------------------------------------------------

    No_phase: dict[str, np.ndarray] = {
            "J_T": np.array([
                phase_relation(No_state["J"][i], No_state["T"][i])
                for i in range(No_state["T"].shape[0])
                ]),
            "w_T": np.array([
                phase_relation(No_state["w"][i], No_state["T"][i])
                for i in range(No_state["T"].shape[0])
                ]),
            }

    Cld_phase: dict[str, np.ndarray] = {
            "J_T": np.array([
                phase_relation(Cld_state["J"][i], Cld_state["T"][i])
                for i in range(Cld_state["T"].shape[0])
                ]),
            "w_T": np.array([
                phase_relation(Cld_state["w"][i], Cld_state["T"][i])
                for i in range(Cld_state["T"].shape[0])
                ]),
            }

    # ------------------------------------------------
    # Calculate mean and stdev for phase relation
    # ------------------------------------------------

    No_mean: dict[str, np.ndarray] = {
            key: value.mean(axis=0)[1:]
            for (key, value) in No_phase.items()
            }

    No_std: dict[str, np.ndarray] = {
            key: value.std(axis=0)[1:]
            for (key, value) in No_phase.items()
            }

    Cld_mean: dict[str, np.ndarray] = {
            key: value.mean(axis=0)[1:]
            for (key, value) in Cld_phase.items()
            }

    Cld_std: dict[str, np.ndarray] = {
            key: value.std(axis=0)[1:]
            for (key, value) in Cld_phase.items()
            }

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------
    # path setup
    figure_path: Path = home_path / "Figure"

    # plot phase relation between Q and T
    fig, ax = plt.subplots(1, 1, figsize=(5, 9))

    ax.errorbar(
            No_mean["J_T"], np.linspace(0, 14000, 71)[1:],
                xerr = No_std["J_T"], color="C0", label="NoRad"
            )
    ax.errorbar(
            Cld_mean["J_T"], np.linspace(0, 14000, 71)[1:],
                xerr = Cld_std["J_T"], color="C2", label="CldRad"
            )
    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.legend(frameon=False, loc="best")

    plt.savefig(figure_path / "J_T.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


    # plot phase relation between w and T
    fig, ax = plt.subplots(1, 1, figsize=(5, 9))

    ax.errorbar(
            No_mean["w_T"], np.linspace(0, 14000, 71)[1:],
                xerr = No_std["w_T"], color="C0", label="NoRad"
            )
    ax.errorbar(
            Cld_mean["w_T"], np.linspace(0, 14000, 71)[1:],
                xerr = Cld_std["w_T"], color="C2", label="CldRad"
            )
    ax.minorticks_on()
    ax.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 14000)
    ax.set_xlabel("Phase [rad]")
    ax.set_ylabel("Level [m]")
    ax.legend(frameon=False, loc="best")

    plt.savefig(figure_path / "w_T.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()

