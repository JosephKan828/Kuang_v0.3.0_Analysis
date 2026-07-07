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
    data_path: Path = Path(home_path / "data" / "composite")
    work_path: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")


    # Design time slice
    time_slice: list[int] = [26, 27, 28]

    # Load composite state 
    ## Load CldRad experiment
    Cld_state:  dict[str, np.ndarray] = {
            "LW": np.load(data_path / "CldRad" / "LW.npy").reshape(-1, 113, 71).transpose(0, 2, 1),
            "SW": np.load(data_path / "CldRad" / "SW.npy").reshape(-1, 113, 71).transpose(0, 2, 1),
            "T": np.load(data_path / "CldRad" / "T.npy"),
            }
    
    # ------------------------------------------------
    # Convert temperature into specific volume
    # ------------------------------------------------
    
    z: np.ndarray = np.linspace(0, 14000, 71) # vertical layer

    # calculate for mean profile using EXACT SAME numerical scheme
    dz: float = z[1] - z[0]
    
    Tmean        : np.ndarray = 300.0 - 0.0065 * z
    integral_mean: np.ndarray = np.cumsum(1/Tmean) * dz
    integral_mean = integral_mean - integral_mean[0]
    pmean_num    : np.ndarray = 1e5 * np.exp(-9.81/287.5 * integral_mean)
    alphamean_num: np.ndarray = 287.5 * Tmean / pmean_num

    # calculate for full profile
    Ttot        : np.ndarray = Tmean[None, :, None] + Cld_state["T"]
    integral_tot: np.ndarray = np.cumsum(1/Ttot, axis=1) * dz
    integral_tot = integral_tot - integral_tot[:, 0:1, :]
    ptot        : np.ndarray = 1e5 * np.exp(-9.81/287.5 * integral_tot)
    alphatot    : np.ndarray = 287.5 * Ttot/ptot

    # calculate alpha deviation
    alpha_prime: np.ndarray = alphatot - alphamean_num[None, :, None]

    # ------------------------------------------------
    # Calculate EAPE budget terms
    # ------------------------------------------------
    Gamma_d: float = 9.81 / 1004.5
    Gamma: float = 0.0065
    gamma: float = Gamma_d / (Gamma_d - Gamma)

    coeff: np.ndarray = gamma / alphamean_num

    Cld_EAPE: dict[str, np.ndarray] = {
            "LW_gen": Cld_state["LW"] * alpha_prime * coeff[None, :, None],
            "SW_gen": Cld_state["SW"] * alpha_prime * coeff[None, :, None]
            }

    # ------------------------------------------------
    # Composite
    # ------------------------------------------------
    # Chunking compostie data
    Cld_chunked: dict[str, np.ndarray] = {
            key: val.mean(axis=0)
            for (key, val) in Cld_EAPE.items()
            }

    Cld_LW_chunked: np.ndarray = Cld_state["LW"].mean(axis=0)
    Cld_SW_chunked: np.ndarray = Cld_state["SW"].mean(axis=0)
    Cld_alpha_prime_chunked: np.ndarray = alpha_prime.mean(axis=0)

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # Setup path
    figure_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad/Figure")

    # Plot NoRad experiment
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex="col")

    ax1, ax2 = axes.flatten()

    LW_ctf = ax1.contourf(
            np.linspace(-np.pi, np.pi, Cld_chunked["LW_gen"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_chunked["LW_gen"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    LW_ct = ax1.contour(
            np.linspace(-np.pi, np.pi, Cld_chunked["LW_gen"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_LW_chunked,
            colors="k", linewidths=0.8, alpha=0.6, levels=np.linspace(-0.001, 0.001, 6)
            )
            
    alpha_ct1 = ax1.contour(
            np.linspace(-np.pi, np.pi, Cld_chunked["LW_gen"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_alpha_prime_chunked,
            colors="magenta", linewidths=0.8, alpha=0.8, levels=7
            )

    ax1.minorticks_on()
    ax1.set_yticks(np.linspace(0, 12000, 7))
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_ylim(0, 14000)
    ax1.set_ylabel("Level [m]")
    ax1.set_title(r"LW EAPE Generation (shading; $K$/day)")
    fig.colorbar(LW_ctf, ax=ax1, label=r"$K$/day")

    SW_ctf = ax2.contourf(
            np.linspace(-np.pi, np.pi, Cld_chunked["SW_gen"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_chunked["SW_gen"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    SW_ct = ax2.contour(
            np.linspace(-np.pi, np.pi, Cld_chunked["SW_gen"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_SW_chunked,
            colors="k", linewidths=0.8, alpha=0.6, levels=np.linspace(-0.001, 0.001, 6)
            )
            
    alpha_ct2 = ax2.contour(
            np.linspace(-np.pi, np.pi, Cld_chunked["SW_gen"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_alpha_prime_chunked,
            colors="magenta", linewidths=0.8, alpha=0.8, levels=7
            )

    ax2.minorticks_on()
    ax2.set_yticks(np.linspace(0, 12000, 7))
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(0, 14000)
    ax2.set_ylabel("Level [m]")
    ax2.set_title(r"SW EAPE Generation (shading; $K$/day)")
    fig.colorbar(SW_ctf, ax=ax2, label=r"$K$/day")



    plt.savefig(figure_path / "Rad_EAPE_gen.png", dpi=300, bbox_inches="tight")
    plt.close()
 
    # ------------------------------------------------
    # Save file
    # ------------------------------------------------

    # path setup
    data_path: Path = Path(home_path / "data" / "composite")

    np.save(data_path / "CldRad" / "LW_gen.npy", Cld_EAPE["LW_gen"])
    np.save(data_path / "CldRad" / "SW_gen.npy", Cld_EAPE["SW_gen"])



# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
