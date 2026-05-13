# ====================================================
# This script is to calculate profile of Leq based on
# the definition of Kuang (2008)
# ====================================================

# ====================================================
# Environment setup
# ====================================================

# limit the usage of CPU
import os
THREAD_VAR = [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"
]

for var in THREAD_VAR:
    os.environ[var]="1"

# Import package
import h5py
import numpy as np
import tomllib as tl

from typing import cast
from pathlib import Path

from matplotlib import pyplot as plt

# Plot configuration
style_path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/style_sheet/SingleLine.mplstyle")
plt.style.use(["seaborn-v0_8-colorblind", str(style_path)])

# ====================================================
# Load data
# ====================================================

# Path setting
sim_name: str = "Rad(0.0,0.0,0.0)"
exp_name: str = "NoRad"
run_name: str = "2024-05-01_RUN01"

home_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0")
work_path: Path = Path(
    "/work/b11209013/Kuang2008_v0.3.0/full" +\
    f"/{sim_name}/{run_name}"
    )

# Load model parameter
with open(home_path / "Config/ModelParams.toml", "rb") as file:
    model_param: dict = dict(tl.load(file))

# Load domain parameter
with open(home_path / "Config/Domain.toml", "rb") as file:
    domain: dict = dict(tl.load(file))

# Load growth rate for identifying the most unstable mode
with h5py.File(work_path / "EigenAnalysis.h5", "r") as file:
    k         : np.ndarray = cast(h5py.Dataset, file["k"])[:]
    growthrate: np.ndarray = cast(h5py.Dataset, file["GrowthRates"])[:, 0]

## Identify the most unstable wavenumber
k_idx     : int   = int(np.nanargmax(growthrate))
k_unstable: float = k[k_idx]

# Load westward tilting ensemble
ens_txt: np.ndarray = np.loadtxt(
        f"/home/b11209013/Kuang2008_v0.3.0_Analysis/Files/{exp_name}/{run_name}/Composite/Ens.txt",
        dtype=np.float64
        )

ens_list: list = [int(e) for e in ens_txt]

# Load state variables for construction
with h5py.File(work_path / "State.h5", "r") as file:
    
    q : np.ndarray = cast(h5py.Dataset, file["q"])[k_idx, :, ens_list]

# Load Fourier state variable for construction
time_slice: list[int] = [26, 27, 28]

with h5py.File(work_path / "FourierState.h5", "r") as file:
    x : np.ndarray = cast(h5py.Dataset, file["x"])[:]

    w1: np.ndarray = cast(h5py.Dataset, file["w1"])[k_idx, ..., ens_list][:, time_slice, ...]
    w2: np.ndarray = cast(h5py.Dataset, file["w2"])[k_idx, ..., ens_list][:, time_slice, ...]
    T1: np.ndarray = cast(h5py.Dataset, file["T1"])[k_idx, ..., ens_list][:, time_slice, ...]
    J1: np.ndarray = cast(h5py.Dataset, file["J1"])[k_idx, ..., ens_list][:, time_slice, ...]
    J2: np.ndarray = cast(h5py.Dataset, file["J2"])[k_idx, ..., ens_list][:, time_slice, ...]

# ====================================================
# Project moisture to Fourier basis
# ====================================================

F_basis: np.ndarray = np.exp(1j*k_unstable*x/4.32e6)

q_Fourier: np.ndarray = np.einsum("te,x->tex", q, F_basis)
q_Fourier = np.array([
    q_Fourier[i*4:(i+1)*4].mean(axis=0) for i in range(q_Fourier.shape[0]//4)
]).transpose(2, 0, 1)[:, time_slice, :]

# ====================================================
# Calculate equilibrium profile for heating
# ====================================================

# Prepare coefficient for calculating Leq
f : float = model_param["Convection"]["CQE"]["f"]
F : float = model_param["Convection"]["CQE"]["F"]
b1: float = model_param["Convection"]["CQE"]["b1"]
b2: float = model_param["Convection"]["CQE"]["b2"]
r0: float = model_param["Convection"]["Heating"]["r0"]
rq: float = model_param["Convection"]["Heating"]["rq"]

A: float = 1 - 2*f + (b2 - b1)/F
B: float = 1 + (b2 + b1)/F - A*r0

q_plus: np.ndarray = q_Fourier - 1.5*T1

# Calculate Leq
Leq: np.ndarray = (A*rq*q_plus + f*w1 + (1-f)*w2)

# Calculate Ueq
Ueq: np.ndarray = r0*Leq + rq*q_plus

# calculate J1_eq and J2_eq
J1_eq: np.ndarray = Leq + Ueq
J2_eq: np.ndarray = Leq - Ueq

# ====================================================
# Project coefficient onto Galerkin space
# ====================================================

# Design Galerkin basis
Z : float = domain["Domain"]["Z"]
dz: float = domain["Domain"]["dz"]

z : np.ndarray = np.arange(0, Z+dz, dz)

G1_basis: np.ndarray = np.pi/2 * np.sin(np.pi*z/Z)
G2_basis: np.ndarray = np.pi/2 * np.sin(2.0*np.pi*z/Z)

## Calculate density profile
T: np.ndarray = 300 - 0.0065 * z
p: np.ndarray = 100000.0 * (1 + -0.0065*z/300)**(9.81/287.5/0.0065)
rho: np.ndarray = p / T / 287.5

## Galerkin basis
J1_eq_Galerkin: np.ndarray = np.einsum("txe,z->txze", J1_eq, G1_basis) / rho[None, None, :, None] * (9.81/1004.5 - 0.0065)
J2_eq_Galerkin: np.ndarray = np.einsum("txe,z->txze", J2_eq, G2_basis) / rho[None, None, :, None] * (9.81/1004.5 - 0.0065)
J1_Galerkin   : np.ndarray = np.einsum("txe,z->txze", J1, G1_basis) / rho[None, None, :, None] * (9.81/1004.5 - 0.0065)
J2_Galerkin   : np.ndarray = np.einsum("txe,z->txze", J2, G2_basis) / rho[None, None, :, None] * (9.81/1004.5 - 0.0065)

## Average over time slice
J1_eq_mean: np.ndarray = np.real(J1_eq_Galerkin.mean(axis=1))
J2_eq_mean: np.ndarray = np.real(J2_eq_Galerkin.mean(axis=1))
J1_mean   : np.ndarray = np.real(J1_Galerkin.mean(axis=1))
J2_mean   : np.ndarray = np.real(J2_Galerkin.mean(axis=1))

# ====================================================
# Apply composite on these fields
# ====================================================

# Rolling data
margin: int = x.size//3
shifts: np.ndarray = x.size//2 - np.argmax(np.mean(J1, axis=1)[margin:-margin, :], axis=0) - margin

nens: int = shifts.shape[-1]

J1_eq_rolled: np.ndarray = np.stack([
    np.roll(J1_eq_mean[..., i], shift=shifts[i], axis=1)
    for i in range(nens)
], axis=-1)

J2_eq_rolled: np.ndarray = np.stack([
    np.roll(J2_eq_mean[..., i], shift=shifts[i], axis=1)
    for i in range(nens)
], axis=-1)

J1_rolled: np.ndarray = np.stack([
    np.roll(J1_mean[..., i], shift=shifts[i], axis=1)
    for i in range(nens)
], axis=-1)

J2_rolled: np.ndarray = np.stack([
    np.roll(J2_mean[..., i], shift=shifts[i], axis=1)
    for i in range(nens)
], axis=-1)

# Calculate index for showing period in radian
## Calculate the number of grid in wave length
half_period     : float = (2*np.pi*4.32e6 / k_unstable) / 2
half_period_grid: int   = int(half_period / (x[1] - x[0]))

## calculate index of period
pi_neg: int = int(x.size//2 - half_period_grid//2)
pi_pos: int = int(x.size//2 + 3*half_period_grid//2)

phase: np.ndarray = np.linspace(-np.pi, np.pi, int(pi_pos-pi_neg))

# ====================================================
# Calculate phase average at positive phase
# ====================================================
# Average over positive phase
Jeq_phase_avg: np.ndarray = (J1_eq_rolled + J2_eq_rolled)[pi_neg:pi_pos, ...][:phase.size//2, :].mean(axis=0)
J_phase_avg  : np.ndarray = (J1_rolled + J2_rolled)[pi_neg:pi_pos, ...][:phase.size//2, :].mean(axis=0)

J_diff_phase_avg: np.ndarray = ((J1_eq_rolled + J2_eq_rolled) - (J1_rolled + J2_rolled))[pi_neg:pi_pos, ...][:phase.size//2, :].mean(axis=0)
J_time_tendency: np.ndarray = J_diff_phase_avg / (1/12)

# Calculate mean and standard deviation
Jeq_mean: np.ndarray = Jeq_phase_avg.mean(axis=-1)
Jeq_std : np.ndarray = Jeq_phase_avg.std(axis=-1)

J_mean: np.ndarray = J_phase_avg.mean(axis=-1)
J_std : np.ndarray = J_phase_avg.std(axis=-1)

J_time_tendency_mean: np.ndarray = J_time_tendency.mean(axis=-1)
J_time_tendency_std : np.ndarray = J_time_tendency.std(axis=-1)

# ====================================================
# Visualization
# ====================================================

# Figure path
fig_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Figure")
os.makedirs(fig_path / exp_name, exist_ok=True)
os.makedirs(fig_path / exp_name / run_name, exist_ok=True)
os.makedirs(fig_path / exp_name / run_name / "Composite", exist_ok=True)

# Calculate 

# Visualizing
fig, ax = plt.subplots(1, 1, figsize=(5, 9))

ax.plot(Jeq_mean, z, linewidth=4, color="C0", label="Jeq")
ax.fill_betweenx(
    z, Jeq_mean - Jeq_std, Jeq_mean + Jeq_std,
    alpha=0.3
)

ax.plot(J_mean, z, linewidth=4, color="C1", linestyle="--", label="J")
ax.fill_betweenx(
    z, J_mean - J_std, J_mean + J_std,
    alpha=0.3
)
ax.axvline(0, 0, 1, color="k", linewidth=3, linestyle="--")
ax.set_xlim(-0.01, 0.01)
ax.set_ylim(0, 14000)
ax.set_xlabel("K/day")
ax.set_ylabel("Level [m]")
ax.set_title("Convective Heating Profile")
ax.legend(frameon=False, loc="best")

plt.savefig(
    fig_path / exp_name / run_name / "Jeq_J.png",
    dpi=300,
    bbox_inches="tight"
)

fig, ax = plt.subplots(1, 1, figsize=(5, 9))

ax.plot(J_time_tendency_mean, z, linewidth=4, color="C0", label="Jeq")
ax.fill_betweenx(
    z, J_time_tendency_mean - J_time_tendency_std, J_time_tendency_mean + J_time_tendency_std,
    alpha=0.3
)
ax.axvline(0, 0, 1, color="k", linewidth=3, linestyle="--")
ax.set_xlim(-0.03, 0.03)
ax.set_ylim(0, 14000)
ax.set_xlabel(r"K/day$^2$")
ax.set_ylabel("Level [m]")
ax.set_title("Convective Heating Time Tendency")

plt.savefig(
    fig_path / exp_name / run_name / "Jeq_J_diff.png",
    dpi=300,
    bbox_inches="tight"
)

# ====================================================
# Save file
# ====================================================

# File path
file_path: Path = Path("/home/b11209013/Kuang2008_v0.3.0_Analysis/Files")

np.savez(file_path / exp_name / run_name / "Jeq.npz", Jeq_phase_avg)
np.savez(file_path / exp_name / run_name / "DJDt.npz", J_time_tendency)