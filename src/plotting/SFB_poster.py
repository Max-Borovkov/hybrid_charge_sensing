#%% Initialization — load the database via a RELATIVE path (no global/absolute path)
import os
from pathlib import Path

import numpy as np
import scipy.signal as scp
import matplotlib.pyplot as plt
import qcodes as qc
from qcodes import (
    initialise_or_create_database_at,
    load_by_id,
    load_by_run_spec,
)


def find_repo_root(marker="pyproject.toml"):
    """Locate the project root by walking up until `marker` is found.

    Works both as a script (uses __file__) and in interactive #%% cells
    (falls back to the current working directory).
    """
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path.cwd().resolve()
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    # Fallback: src/plotting -> repo root is two levels up
    return start.parents[2]


REPO_ROOT = find_repo_root()
# NOTE: runs 137/135/132/130 are NOT in *_db2.db (load_by_id fails there);
# they live in this DB. Switch the filename back to *_db2.db if needed.
DB_PATH = REPO_ROOT / "data" / "SQDv4_S10_grAl" / "SQDv4_S10_grAl_db2.db"

print("Repo root :", REPO_ROOT)
print("Database  :", DB_PATH.relative_to(REPO_ROOT))
print("Exists    :", DB_PATH.exists())

initialise_or_create_database_at(str(DB_PATH))


#%% SFB poster — 4-panel conductance (dI/dV) maps with a shared colorbar
g0 = 7.748091729e-5  # 2e^2/h in S

measurement_ids = [135, 133, 129, 130]  # left -> right

# Parameter names in the datasets (same as the "eye" block in Plotting.py).
# If a run errors or looks empty, check the names with: print(ds.data_vars)
current_name = "current_1"
bias_name = "bias_1"
gate_name = "Left_Dot_Plunger"

# If 137/135/132/130 are *captured* run ids rather than sequential run ids,
# switch USE_CAPTURED to True (uses load_by_run_spec instead of load_by_id).
USE_CAPTURED = False


def load_dataset(rid):
    if USE_CAPTURED:
        return load_by_run_spec(captured_run_id=rid).to_xarray_dataset()
    return load_by_id(rid).to_xarray_dataset()


# --- poster style (requires a working LaTeX install for text.usetex) ---
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "axes.unicode_minus": False,
})

# --- poster font sizes ---
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# --- load every panel first so a single color scale can be shared ---
panels = []
for rid in measurement_ids:
    ds = load_dataset(rid)
    current = -ds[current_name].values
    bias = ds[bias_name].values
    gate = ds[gate_name].values
    # conductance dI/dV in units of 2e^2/h, scaled x10^3 (i.e. value shown is G x 10^-3)
    cond = scp.savgol_filter(current, 3, 1, deriv=1, axis=1) \
        / ((bias[1] - bias[0]) * 1e-3) / g0 * 1e3
    panels.append((rid, gate, bias, cond))

# --- shared color limits across all four maps (in units of G x 10^-3) ---
all_vals = np.concatenate([cond.ravel() for *_, cond in panels])
vmin = 0
vmax = 5

# --- figure size (INCHES) ---
# figsize in inches == the physical size the figure occupies in PowerPoint (dpi only
# controls sharpness and cancels out of the physical size). Design at the FINAL on-slide
# size so you never drag-resize in PPT (dragging shrinks the fonts too).
FIG_W_IN = 20.0  # figure width  in inches
FIG_H_IN = 4.0   # figure height in inches

# x-axis ticks: 3 per panel at [center - frac*half, center, center + frac*half]
# of each panel's plunger range. Increase toward 1.0 to push outer ticks to the edges.
X_TICK_FRAC = 0.8

# --- plot: 1x4 row, one shared colorbar on the right ---
fig, axes = plt.subplots(
    1, 4,
    figsize=(FIG_W_IN, FIG_H_IN),
    constrained_layout=True,
    sharey=True,
)

# Bring panels closer together. In constrained_layout the real gap comes from
# w_pad (padding around each panel, in inches) -> DECREASE it to move panels closer.
# wspace adds *extra* relative space on top of w_pad, so keep it at 0 for tightest packing.
fig.get_layout_engine().set(w_pad=0.02, wspace=0.02)

im = None
for ax, (rid, gate, bias, cond) in zip(axes, panels):
    im = ax.pcolormesh(
        gate,
        bias * 1e3,
        cond.T,
        shading="auto",
        cmap="Blues_r",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_xlabel(r"$V_\mathrm{plunger}$ (mV)")
    ax.set_ylim(-290, 290)  # bias axis (µV) limits, shared across all panels

    # three x-ticks: center and +/- X_TICK_FRAC of the half-range
    g_center = 0.5 * (gate.min() + gate.max())
    g_half = 0.5 * (gate.max() - gate.min())
    ax.set_xticks([g_center - X_TICK_FRAC * g_half, g_center, g_center + X_TICK_FRAC * g_half])
    ax.set_xticklabels([f"{t:.0f}" for t in ax.get_xticks()])
axes[0].set_ylabel(r"$V_\mathrm{bias}$ ($\mu$V)")

# pad = gap between the last panel and the colorbar (smaller -> closer);
# shrink = colorbar height as a fraction of the panel height.
cbar = fig.colorbar(im, ax=axes, location="right", shrink=1.0, pad=0.01)
cbar.set_label(r"$G$ ($2e^2/h$) $\times 10^{-3}$")

# --- save the figure (relative path) ---
PLOTS_DIR = REPO_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
fig.savefig(PLOTS_DIR / "SFB_poster.png", dpi=300)

plt.show()


#%% Debug — panel shapes and value ranges
print(f"{'ID':>6} {'gate':>6} {'bias':>6} {'cond shape':>14} {'cond max':>12} {'cond min':>12}")
cond_shapes = []
for rid, gate, bias, cond in panels:
    cond_shapes.append(cond.shape)
    print(f"{rid:>6} {gate.shape[0]:>6} {bias.shape[0]:>6} {str(cond.shape):>14} "
          f"{np.nanmax(cond):>12.5g} {np.nanmin(cond):>12.5g}")

print("\nAll cond shapes identical:", len(set(cond_shapes)) == 1, cond_shapes)


#%% 3-panel: theory <n> charge diagram + measured I1 and I2 (id 278)
import numpy as np
import matplotlib.pyplot as plt
from qcodes import load_by_id


# === Theory: <n> charge diagram from the ZBW single-orbital model =============
# (copied from ZBW_single_orbital_model.py to avoid importing that module, whose
#  top-level #%% cells would otherwise run on import.)
def ZBW_single_orbital_hamiltonian(U, Delta, t, mu, E_Z=0.0):
    I4 = np.eye(4, dtype=complex)
    P = np.diag([1, -1, -1, 1]).astype(complex)
    f_up = np.array([[0, 1, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 0]], dtype=complex)
    f_down = np.array([[0, 0, 1, 0],
                       [0, 0, 0, -1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]], dtype=complex)
    d_up, d_down = np.kron(f_up, I4), np.kron(f_down, I4)
    c_up, c_down = np.kron(P, f_up), np.kron(P, f_down)
    n_du, n_dd = d_up.conj().T @ d_up, d_down.conj().T @ d_down
    epsilon_up, epsilon_down = mu + E_Z, mu - E_Z
    H_dot = epsilon_up * n_du + epsilon_down * n_dd + U * (n_du @ n_dd)
    H_sc = -Delta * (c_up.conj().T @ c_down.conj().T + c_down @ c_up)
    H_t = t * (d_up.conj().T @ c_up + c_up.conj().T @ d_up
               + d_down.conj().T @ c_down + c_down.conj().T @ d_down)
    return H_dot + H_sc + H_t, n_du, n_dd


U_th, Delta_th, E_Z_th = 3.0, 1.0, 0.0
mu_th = np.linspace(-4.5, 1.5, 400)
t_th = np.linspace(0.2, 1.2, 100)

charge_map = np.zeros((len(t_th), len(mu_th)))
for i, t in enumerate(t_th):
    for j, mu in enumerate(mu_th):
        H, n_du, n_dd = ZBW_single_orbital_hamiltonian(U_th, Delta_th, t, mu, E_Z_th)
        evals, evecs = np.linalg.eigh(H)
        gs = evecs[:, 0]
        charge_map[i, j] = (gs.conj() @ (n_du + n_dd) @ gs).real


# === Measurement: id 278, raw currents I1 and I2 =============================
# (copied from the "#%% plotting tracking the peaks" block in Plotting.py)
measurement_id = 280
bs_name = "QDL_BS_compensated"
plunger_name = "QDL_P_compensated"

qc_ds = load_by_id(measurement_id)
df = qc_ds.to_pandas_dataframe().reset_index()
# If a name errors, inspect once with: print(df.columns)

I1_df = df.pivot_table(index=bs_name, columns=plunger_name,
                       values="current_1").sort_index().sort_index(axis=1)
I2_df = df.pivot_table(index=bs_name, columns=plunger_name,
                       values="current_2").sort_index().sort_index(axis=1)

bs_meas = I1_df.index.values
plunger_meas = I1_df.columns.values
current_1_meas = I1_df.values
current_2_meas = I2_df.values


# === 3-panel figure ==========================================================
FIG3_W_IN = 15.0  # figure width  in inches (on-slide physical size)
FIG3_H_IN = 5.0   # figure height in inches

fig, axes = plt.subplots(1, 3, figsize=(FIG3_W_IN, FIG3_H_IN), constrained_layout=True)

# Panel 1: theory <n>  (colorbar on top, no title)
ax = axes[0]
pcm0 = ax.pcolormesh(mu_th, t_th, charge_map, shading="auto", cmap="RdBu_r")
fig.colorbar(pcm0, ax=ax, location="top", label=r"$\langle \hat{n} \rangle$")
ax.set_xlabel(r"$\mu\;/\;\Delta$")
ax.set_ylabel(r"$t\;/\;\Delta$")
ax.invert_yaxis()  # high t (strong hybridization) at the bottom, to match the measured panels

# Panel 2: measured transport current I1  (colorbar on top, no title)
ax = axes[1]
pcm1 = ax.pcolormesh(plunger_meas, bs_meas, current_1_meas * 1e9,
                     shading="auto", cmap="magma", rasterized=True)
fig.colorbar(pcm1, ax=ax, location="top", label=r"$I_1$ (nA)")
ax.set_xlabel(r"$V_\mathrm{QDL\_P}$ (mV)")
ax.set_ylabel(r"$V_\mathrm{QDL\_BS}$ (mV)")

# Panel 3: measured charge-sensor current I2  (colorbar on top, no title)
ax = axes[2]
pcm2 = ax.pcolormesh(plunger_meas, bs_meas, current_2_meas * 1e9,
                     shading="auto", cmap="cividis", rasterized=True)
fig.colorbar(pcm2, ax=ax, location="top", label=r"$I_2$ (nA)")
ax.set_xlabel(r"$V_\mathrm{QDL\_P}$ (mV)")
ax.set_ylabel(r"$V_\mathrm{QDL\_BS}$ (mV)")

fig.savefig(REPO_ROOT / "plots" / "SFB_poster_3panel.png", dpi=300)
plt.show()
