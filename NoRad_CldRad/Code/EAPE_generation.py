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
    data_path: Paht = Path(home_path / "data" / "composite")
    work_path: Path = Path("/work/b11209013/Kuang2008_v0.3.0/full")


    # Design time slice
    time_slice: list[int] = [26, 27, 28]

    # Load composite state
    ## Load NoRad experiment
    No_state:  dict[str, np.ndarray] = {
            "J": np.load(data_path / "NoRad" / "J.npy"),
            "T": np.load(data_path / "NoRad" / "T.npy"),
            "w": np.load(data_path / "NoRad" / "w.npy"),
            }
    
    ## Load NoRad experiment
    Cld_state:  dict[str, np.ndarray] = {
            "J": np.load(data_path / "CldRad" / "J.npy"),
            "T": np.load(data_path / "CldRad" / "T.npy"),
            "w": np.load(data_path / "CldRad" / "w.npy"),
            }

    # ------------------------------------------------
    # Calculate EAPE budget terms
    # ------------------------------------------------

    No_EAPE: dict[str, np.ndarray] = {
            "generation": No_state["J"] * No_state["T"],
            "conversion": No_state["w"] * No_state["T"] * (0.0065 - 9.81/1004.5)
            }
    Cld_EAPE: dict[str, np.ndarray] = {
            "generation": Cld_state["J"] * Cld_state["T"],
            "conversion": Cld_state["w"] * Cld_state["T"] * (0.0065 - 9.81/1004.5)
            }

    # ------------------------------------------------
    # Composite
    # ------------------------------------------------
    # Chunking compostie data
    No_chunked: dict[str, np.ndarray] = {
            key: val.mean(axis=0)
            for (key, val) in No_EAPE.items()
            }
    Cld_chunked: dict[str, np.ndarray] = {
            key: val.mean(axis=0)
            for (key, val) in Cld_EAPE.items()
            }

    No_J_chunked : np.ndarray = No_state["J"].mean(axis=0)
    Cld_J_chunked: np.ndarray = Cld_state["J"].mean(axis=0)

    print("NoRad max. generation: ", No_chunked["generation"].max())
    print("CldRad max. generation: ", Cld_chunked["generation"].max())
    print("NoRad max. J: ", No_J_chunked.max())
    print("CldRad max. J: ", Cld_J_chunked.max())
    print("Maximum generation ratio: ", Cld_chunked["generation"].max()/No_chunked["generation"].max())

    # ------------------------------------------------
    # Visualization
    # ------------------------------------------------

    # Setup path
    figure_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/NoRad_CldRad/Figure")

    # Plot NoRad experiment
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex="col")

    ax1, ax2 = axes.flatten()

    gen_ctf = ax1.contourf(
            np.linspace(-np.pi, np.pi, No_chunked["generation"].shape[1]),
            np.linspace(0, 14000, 71),
            No_chunked["generation"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    J_ct = ax1.contour(
            np.linspace(-np.pi, np.pi, No_J_chunked.shape[1]),
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
            np.linspace(-np.pi, np.pi, No_chunked["conversion"].shape[1]),
            np.linspace(0, 14000, 71),
            No_chunked["conversion"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    J_ct = ax2.contour(
            np.linspace(-np.pi, np.pi, No_J_chunked.shape[1]),
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
            np.linspace(-np.pi, np.pi, Cld_chunked["generation"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_chunked["generation"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    J_ct = ax1.contour(
            np.linspace(-np.pi, np.pi, Cld_J_chunked.shape[1]),
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
            np.linspace(-np.pi, np.pi, Cld_chunked["conversion"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_chunked["conversion"],
            cmap="BrBG", levels=np.linspace(-1e-2, 1e-2, 11)
            )

    J_ct = ax2.contour(
            np.linspace(-np.pi, np.pi, Cld_J_chunked.shape[1]),
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
            np.linspace(-np.pi, np.pi, Cld_chunked["generation"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_chunked["generation"],
            colors="k", linewidths=4,
            levels=[i for i in np.linspace(-5e-3, 5e-3, 11) if np.abs(i) > 1e-5]
            )

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, No_chunked["generation"].shape[1]),
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
            np.linspace(-np.pi, np.pi, Cld_chunked["conversion"].shape[1]),
            np.linspace(0, 14000, 71),
            Cld_chunked["conversion"],
            colors="k", linewidths=4,
            levels=[i for i in np.linspace(-5e-3, 5e-3, 11) if np.abs(i) > 1e-5]
            )

    no_ctf = ax.contourf(
            np.linspace(-np.pi, np.pi, No_chunked["conversion"].shape[1]),
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

    # ------------------------------------------------
    # Save file
    # ------------------------------------------------

    # path setup
    data_path: Path = Path(home_path / "data" / "composite")

    np.save(data_path / "NoRad" / "generation.npy", No_EAPE["generation"])
    np.save(data_path / "NoRad" / "conversion.npy", No_EAPE["conversion"])

    np.save(data_path / "CldRad" / "generation.npy", Cld_EAPE["generation"])
    np.save(data_path / "CldRad" / "conversion.npy", Cld_EAPE["conversion"])

# ====================================================
# Execute main function
# ====================================================

if __name__ == "__main__":
    main()
