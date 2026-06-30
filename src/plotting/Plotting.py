#%% Initialization 
from qcodes import (
    Measurement,
    experiments,
    initialise_database,
    initialise_or_create_database_at,
    load_by_run_spec,
    load_or_create_experiment,
)
import os
import time
import numpy as np
import qcodes as qc
import scipy.signal as scp
import scipy.constants as cst
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from qcodes import load_by_run_spec, ScaledParameter, Parameter
from qcodes import Station, initialise_or_create_database_at, \
    load_or_create_experiment, Measurement
# from qcodes.tests.instrument_mocks import DummyInstrument, \
#    DummyInstrumentWithMeasurement
from qcodes.utils.dataset.doNd import do0d,do1d,do2d
from qcodes.dataset.plotting import plot_dataset, plot_by_id

from scipy.optimize import curve_fit

from scipy.signal import welch
from mpl_toolkits.axes_grid1 import make_axes_locatable

path = "~/gelab/data/hybrid_sensing/SQDv4_S10_grAl_db2/SQDv4_S10_grAl_db2.db"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed. Folder already exists?" % path)
else:
    print ("Successfully created the directory %s." % path)
db_file_path = os.path.join(path)
initialise_or_create_database_at(path)

# %% Linefit 
def line(x,m,q):
    return m*x+q
id = 46
dataset=load_by_run_spec(captured_run_id=id)

data = dataset.get_parameter_data()
I = data["current"]["current"]
V = data["current"]["voltage_bias"]*1e-3
popt, _ = curve_fit(line, V,I)
plt.figure(figsize=(3,3))
plt.plot(V*1e3, I*1e9)
plt.plot(V*1e3, line(V,*popt)*1e9)
plt.ylabel('I [nA]')
plt.xlabel('V [mV]')
plt.title('Run ID = '+ str(id) +', R = {} Ohm'.format(np.round(1/popt[0] ),1));

#%% parabola fit
def parabola(x,a,b,c):
    return a*x**2+b*x+c
id = 2366

dataset=load_by_run_spec(captured_run_id=id)

data = dataset.get_parameter_data()
I = data["current"]["current"]
B = data["current"]["Angle"]
popt, _ = curve_fit(parabola, B,I)
plt.figure()
plt.plot(B, I*1e9)
plt.plot(B, parabola(B,*popt)*1e9)
plt.ylabel('I [nA]')
plt.xlabel('V [mV]')
plt.title('Run ID = '+ str(id) +', TT 10mK, R = {} Ohm'.format(np.round(1/popt[0]),1));


# %% Plot pinchoffs up and down
run_id_1 = 363
run_id_2 = run_id_1 + 1

gate_swept_name = 'Right_Dot_Super_Barrier'

dataset1 = qc.load_by_id(run_id_1)
ds1 = dataset1.to_xarray_dataset()
dataset2 = qc.load_by_id(run_id_2)
ds2 = dataset2.to_xarray_dataset()

I_meas1 = ds1['current_2'].values
bias_meas1 = ds1[gate_swept_name].values
#deriv = scp.savgol_filter(current_meas,10,1 ,deriv=1, axis =0)/((bias_meas[1]-bias_meas[0])*1e-3)/(g0)

I_meas2 = ds2['current_2'].values
bias_meas2 = ds2[gate_swept_name].values


fig, ax = plt.subplots(figsize=(4,3))
ax.plot(bias_meas1, I_meas1*1e9, label = 'up')
ax.plot(bias_meas2, I_meas2*1e9, label = 'down')
ax.set_xlabel(gate_swept_name + '(mV)')
ax.set_ylabel('I (nA)')
ax.set_title(f"run_id {run_id_1},{run_id_2}")
ax.legend()

#%% Plot derivative IV (current 1)
import scipy.signal as scp
g0 =  7.748091729*1e-5
run_id = 369
dataset = qc.load_by_id(run_id)

df = dataset.to_pandas_dataframe().reset_index()
ds = dataset.to_xarray_dataset()
# ds = ds.isel(barrier_super=slice(-1))
#ds = ds.sel(Bias1=slice(-0.5,0.2))
#ds = ds.sel(P=slice(1200,1300))

current_meas = -1 * ds['current_1'].values#[18:32]#[len(bias_meas)//2-50: len(bias_meas)//2+50]
bias_meas= ds['bias_1'].values#[18:32]#[len(bias_meas)//2-50: len(bias_meas)//2+50]
B = ds['Left_Dot_Super_Barrier'].values
deriv = scp.savgol_filter(current_meas,3,1 ,deriv=1, axis =1)/((bias_meas[1]-bias_meas[0])*1e-3)/(g0)

fig, ax = plt.subplots(1,figsize = (8,3), dpi=400)
im = ax.pcolor(B, bias_meas*1e3, deriv.T ,vmin = -0.001, vmax=0.01, cmap='RdBu_r')#, norm=mpl.colors.LogNorm(clip=True, vmin=1e-5)) #vmin = 0,vmax = 0.002)
#im = ax.pcolor(B, bias_meas*1e3, current_meas.T*1e12,vmin = -100,vmax = 100)

cbar = fig.colorbar(im,ax=ax)
cbar.set_label("$G$ ($2e^2$/h)")
#cbar.set_label("$I$ ($pA$)")
ax.set_xlabel('$V_{super}$ (mV)')
ax.set_ylabel('$V_{bias}$ ($\mu$ V)')
ax.set_title(f"Measurement {run_id} (num. derivative)")
#ax.set_title(f"Measurement {run_id} ")
#cbar = fig.colorbar(im,ax=ax)
#%% Plot derivative IV (current 2)
import scipy.signal as scp
g0 =  7.748091729*1e-5
run_id = 370
dataset = qc.load_by_id(run_id)

df = dataset.to_pandas_dataframe().reset_index()
ds = dataset.to_xarray_dataset()
# ds = ds.isel(barrier_super=slice(-1))
#ds = ds.sel(Bias1=slice(-0.5,0.2))
#ds = ds.sel(P=slice(1200,1300))

current_meas = -1 * ds['current_2'].values#[18:32]#[len(bias_meas)//2-50: len(bias_meas)//2+50]
bias_meas= ds['bias_2'].values#[18:32]#[len(bias_meas)//2-50: len(bias_meas)//2+50]
B = ds['Right_Dot_Super_Barrier'].values
deriv = scp.savgol_filter(current_meas,9,1 ,deriv=1, axis =1)/((bias_meas[1]-bias_meas[0])*1e-3)/(g0)

fig, ax = plt.subplots(1,figsize = (10,4))
im = ax.pcolor(B, bias_meas*1e3, deriv.T ,vmin = 0.000)#, norm=mpl.colors.LogNorm(clip=True, vmin=1e-5)) #vmin = 0,vmax = 0.002)
#im = ax.pcolor(B, bias_meas*1e3, current_meas.T*1e12,vmin = -100,vmax = 100)

cbar = fig.colorbar(im,ax=ax)
cbar.set_label("$G$ ($2e^2$/h)")
#cbar.set_label("$I$ ($pA$)")
ax.set_xlabel('Plunger (mV)')
ax.set_ylabel('$V_{bias}$ ($\mu$ V)')
ax.set_title(f"Measurement {run_id} (num. derivative)")
#ax.set_title(f"Measurement {run_id} ")
#cbar = fig.colorbar(im,ax=ax)

#%% Coulomb diamonds with plunger fast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scp
from qcodes.dataset import load_by_id

# -------------------------
# User settings
# -------------------------

measurement_id = 270  # <-- change this

bias_name = 'bias_1'
gate_name = 'QDL_P_compensated'   # or 'QDL_P_compensated' depending on how it was saved

current_1_name = 'current_1'
current_2_name = 'current_2'

g0 = 7.748091729e-5   # conductance quantum, S


# -------------------------
# Load data
# -------------------------

qc_ds = load_by_id(measurement_id)
df = qc_ds.to_pandas_dataframe().reset_index()

# Optional: check column names once if needed
# print(df.columns)


# -------------------------
# Convert to 2D arrays
# -------------------------

I1_df = df.pivot_table(
    index=gate_name,
    columns=bias_name,
    values=current_1_name
).sort_index().sort_index(axis=1)

I2_df = df.pivot_table(
    index=gate_name,
    columns=bias_name,
    values=current_2_name
).sort_index().sort_index(axis=1)

gate_meas = I1_df.index.values
bias_meas = I1_df.columns.values

current_1_meas = I1_df.values
current_2_meas = I2_df.values


# -------------------------
# Calculate dI1/dVbias in units of g0
# -------------------------

deriv = -1 * scp.savgol_filter(
    current_1_meas,
    window_length=9,
    polyorder=1,
    deriv=1,
    axis=1
) / ((bias_meas[1] - bias_meas[0]) * 1e-3) / g0


# -------------------------
# Average current_2 over bias
# -------------------------

# current_2_avg = np.nanmean(current_2_meas, axis=1)
n_avg = 1
zero_bias = 0
bias_indices = np.argsort(np.abs(bias_meas - zero_bias))[:n_avg]
bias_indices = np.sort(bias_indices)

bias_used = bias_meas[bias_indices]

print("Averaging current_2 over these bias voltages:")
print(bias_used)

current_2_avg = np.nanmean(current_2_meas[:, bias_indices], axis=1)

# -------------------------
# Plot
# -------------------------

fig, axs = plt.subplots(
    1, 2,
    figsize=(10, 4),
    constrained_layout=True,
    dpi=300
)

# Left: dI1/dVbias map
im0 = axs[0].pcolormesh(
    gate_meas,
    bias_meas,
    deriv.T,
    shading='auto',
    rasterized=True,
    cmap='hot',
    vmax=0.005,
    vmin=0
)

axs[0].set_ylabel(r'$V_\mathrm{bias}$ (mV)')
axs[0].set_xlabel(r'$V_\mathrm{QDL\_P}$ (mV)')
axs[0].set_title(r'$dI_1/dV_\mathrm{bias}$')

cbar0 = fig.colorbar(im0, ax=axs[0])
cbar0.set_label(r'$dI_1/dV_\mathrm{bias}$ $(2e^2/h)$')

# Overlay average current_2 on left panel
ax_avg = axs[0].twinx()
ax_avg.plot(gate_meas, current_2_avg * 1e12, lw=2, color='white')
ax_avg.set_ylabel(r'$\langle I_2 \rangle_\mathrm{bias}$ (nA)')


# Right: current_2 map
im1 = axs[1].pcolormesh(
    gate_meas,
    bias_meas,
    current_2_meas.T * 1e12,
    shading='auto',
    rasterized=True,
    cmap='Greys_r'
)

axs[1].set_ylabel(r'$V_\mathrm{bias}$ (mV)')
axs[1].set_xlabel(r'$V_\mathrm{QDL\_P}$ (mV)')
axs[1].set_title(r'$I_2$')

cbar1 = fig.colorbar(im1, ax=axs[1])
cbar1.set_label(r'$I_2$ (nA)')

plt.show()

#%% Plot derivative of the sensor current
dV_gate = gate_meas[1] - gate_meas[0]

deriv_current_2_gate = scp.savgol_filter(
    current_2_meas,
    window_length=3,
    polyorder=1,
    deriv=1,
    axis=0
) / dV_gate

fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)

im = ax.pcolormesh(
    gate_meas,
    bias_meas,
    deriv_current_2_gate.T * 1e9,
    shading='auto',
    cmap='Reds_r',
    rasterized=True
)

ax.set_xlabel(r'$V_{\mathrm{QDL\_P}}$ (mV)')
ax.set_ylabel(r'$V_{\mathrm{bias}}$ (mV)')
ax.set_title(r'$dI_2/dV_{\mathrm{QDL\_P}}$')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$dI_2/dV_{\mathrm{QDL\_P}}$ (nA/mV)')

plt.show()

#%% plotting tracking the peaks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scp
from qcodes.dataset import load_by_id

# -------------------------
# User settings
# -------------------------

measurement_id = 280   # <-- change this

bs_name = 'QDL_BS_compensated'
plunger_name = 'QDL_P_compensated'

current_1_name = 'current_1'
current_2_name = 'current_2'

savgol_window = 12
savgol_polyorder = 3


# -------------------------
# Load data
# -------------------------

qc_ds = load_by_id(measurement_id)
df = qc_ds.to_pandas_dataframe().reset_index()

# Use this once if names are wrong:
# print(df.columns)


# -------------------------
# Convert to 2D arrays
# rows: QDL_BS
# columns: QDL_P
# -------------------------

I1_df = df.pivot_table(
    index=bs_name,
    columns=plunger_name,
    values=current_1_name
).sort_index().sort_index(axis=1)

I2_df = df.pivot_table(
    index=bs_name,
    columns=plunger_name,
    values=current_2_name
).sort_index().sort_index(axis=1)

bs_meas = I1_df.index.values
plunger_meas = I1_df.columns.values

current_1_meas = I1_df.values
current_2_meas = I2_df.values


# -------------------------
# Derivatives of current_2
# -------------------------

dBS = np.abs(bs_meas[1] - bs_meas[0])
dP = np.abs(plunger_meas[1] - plunger_meas[0])

# dI2 / dBS
deriv_I2_BS = scp.savgol_filter(
    current_2_meas,
    savgol_window,
    savgol_polyorder,
    deriv=1,
    axis=0
) / dBS

# dI2 / dQDL_P
deriv_I2_P = scp.savgol_filter(
    current_2_meas,
    savgol_window,
    savgol_polyorder,
    deriv=1,
    axis=1
) / dP


# -------------------------
# Plot
# -------------------------

fig, axs = plt.subplots(
    2, 2,
    figsize=(10, 8),
    constrained_layout=True,
    dpi=400
)

# Top left: current_1
im00 = axs[0, 0].pcolormesh(
    plunger_meas,
    bs_meas,
    current_1_meas * 1e9,
    shading='auto',
    rasterized=True
)
axs[0, 0].set_xlabel(r'$V_\mathrm{QDL\_P}$ (mV)')
axs[0, 0].set_ylabel(r'$V_\mathrm{QDL\_BS}$ (mV)')
axs[0, 0].set_title(r'$I_1$')
cbar00 = fig.colorbar(im00, ax=axs[0, 0])
cbar00.set_label(r'$I_1$ (nA)')


# Top right: current_2
im01 = axs[0, 1].pcolormesh(
    plunger_meas,
    bs_meas,
    current_2_meas * 1e9,
    shading='auto',
    rasterized=True
)
axs[0, 1].set_xlabel(r'$V_\mathrm{QDL\_P}$ (mV)')
axs[0, 1].set_ylabel(r'$V_\mathrm{QDL\_BS}$ (mV)')
axs[0, 1].set_title(r'$I_2$')
cbar01 = fig.colorbar(im01, ax=axs[0, 1])
cbar01.set_label(r'$I_2$ (nA)')


# Bottom left: derivative of current_2 vs BS
im10 = axs[1, 0].pcolormesh(
    plunger_meas,
    bs_meas,
    deriv_I2_BS * 1e9,
    shading='auto', vmin=-0.01, vmax=0,
    rasterized=True
)
axs[1, 0].set_xlabel(r'$V_\mathrm{QDL\_P}$ (mV)')
axs[1, 0].set_ylabel(r'$V_\mathrm{QDL\_BS}$ (mV)')
axs[1, 0].set_title(r'$dI_2/dV_\mathrm{QDL\_BS}$')
cbar10 = fig.colorbar(im10, ax=axs[1, 0])
cbar10.set_label(r'$dI_2/dV_\mathrm{QDL\_BS}$ (nA/mV)')


# Bottom right: derivative of current_2 vs plunger
im11 = axs[1, 1].pcolormesh(
    plunger_meas,
    bs_meas,
    deriv_I2_P * 1e9,
    shading='auto', vmin=-0.01, vmax=0,
    rasterized=True
)
axs[1, 1].set_xlabel(r'$V_\mathrm{QDL\_P}$ (mV)')
axs[1, 1].set_ylabel(r'$V_\mathrm{QDL\_BS}$ (mV)')
axs[1, 1].set_title(r'$dI_2/dV_\mathrm{QDL\_P}$')
cbar11 = fig.colorbar(im11, ax=axs[1, 1])
cbar11.set_label(r'$dI_2/dV_\mathrm{QDL\_P}$ (nA/mV)')

plt.show()

#%% Compensation checks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as scp
from qcodes.dataset import load_by_id

measurement_id = 437   # change this

alpha_name = "alpha_compensation"
gate_name = "QDL_P_compensated"
current_2_name = "current_2"

# Load dataset and convert to pandas
qc_ds = load_by_id(measurement_id)
df = qc_ds.to_pandas_dataframe().reset_index()

# Check names if necessary
# print(df.columns)

# Rows: alpha
# Columns: QDL_P
I2_df = df.pivot_table(
    index=alpha_name,
    columns=gate_name,
    values=current_2_name
).sort_index().sort_index(axis=1)

alpha_meas = I2_df.index.to_numpy()
gate_meas = I2_df.columns.to_numpy()
current_2_meas = I2_df.to_numpy()

# Derivative with respect to QDL_P
gate_step = gate_meas[1] - gate_meas[0]

deriv_current_2_gate = scp.savgol_filter(
    current_2_meas,
    window_length=5,
    polyorder=3,
    deriv=1,
    axis=1
) / gate_step

# Plot
fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

# Left: current_2
im0 = axs[0].pcolormesh(
    gate_meas,
    alpha_meas,
    current_2_meas * 1e9,
    shading="auto",
    rasterized=True
)

axs[0].set_xlabel(r"$V_{\mathrm{QDL\_P}}$ (mV)")
axs[0].set_ylabel(r"$\alpha$")
axs[0].set_title(r"$I_2$")

cbar0 = fig.colorbar(im0, ax=axs[0])
cbar0.set_label(r"$I_2$ (nA)")

# Right: derivative of current_2 with respect to QDL_P
im1 = axs[1].pcolormesh(
    gate_meas,
    alpha_meas,
    deriv_current_2_gate * 1e9,
    shading="auto",
    rasterized=True
)

axs[1].set_xlabel(r"$V_{\mathrm{QDL\_P}}$ (mV)")
axs[1].set_ylabel(r"$\alpha$")
axs[1].set_title(r"$dI_2/dV_{\mathrm{QDL\_P}}$")

cbar1 = fig.colorbar(im1, ax=axs[1])
cbar1.set_label(r"$dI_2/dV_{\mathrm{QDL\_P}}$ (nA/mV)")

plt.show()

# %% Plot Coulomb diamonds for different barrier voltages
from matplotlib.colors import TwoSlopeNorm

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
from qcodes.dataset import load_by_id

measurement_ids = [477, 478, 479, 480, 481, 482, 483]   # change these
barrier_voltages = [320, 310, 300, 290, 280, 270, 260]

gate_name = "Left_Dot_Plunger"
bias_name = "bias_1"
current_name = "current_1"

g0 = 7.748091729e-5   # 2e^2/h in S

ncols = 2
nrows = int(np.ceil(len(measurement_ids) / ncols))

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(10, 3.2 * nrows),
    constrained_layout=True,
    squeeze=False
)

axs = axs.ravel()

for ax, measurement_id, barrier_voltage in zip(
    axs,
    measurement_ids,
    barrier_voltages
):
    qc_ds = load_by_id(measurement_id)
    df = qc_ds.to_pandas_dataframe().reset_index()

    current_df = df.pivot_table(
        index=gate_name,
        columns=bias_name,
        values=current_name
    ).sort_index().sort_index(axis=1)

    gate_meas = current_df.index.to_numpy()
    bias_meas = current_df.columns.to_numpy()
    current_meas = current_df.to_numpy()

    bias_step_V = (bias_meas[1] - bias_meas[0]) * 1e-3

    conductance = scp.savgol_filter(-1 * current_meas, window_length=3, polyorder=1, deriv=1, axis=1) / bias_step_V / g0 * 1e3
    max_abs = np.nanmax(conductance)
    min_abs = np.nanmin(conductance)
    norm = TwoSlopeNorm(
        vmax=max_abs,
        vcenter=0,
        vmin=min_abs
        )
    im = ax.pcolormesh(
        gate_meas,
        bias_meas,
        conductance.T,
        shading="auto",
        cmap='RdBu_r',
        norm=norm,
        rasterized=True
    )

    ax.set_xlabel(r"$V_{\mathrm{QDL\_P}}$ (mV)")
    ax.set_ylabel(r"$V_{\mathrm{bias}}$ (mV)")
    ax.set_title(
        rf"$V_{{\mathrm{{BS}}}}={barrier_voltage}$ mV, ID {measurement_id}"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$G_0\times 10^3$ $(2e^2/h)$")

# Hide unused panels
for ax in axs[len(measurement_ids):]:
    ax.set_visible(False)

plt.show()


#%% Plotting the conductance of the eye
def find_snapshot_parameter(snapshot, parameter_name):
    if not isinstance(snapshot, dict):
        return None

    if parameter_name in snapshot and isinstance(snapshot[parameter_name], dict):
        parameter = snapshot[parameter_name]
        if 'value' in parameter:
            return parameter['value'], parameter.get('unit', '')

    for value in snapshot.values():
        result = find_snapshot_parameter(value, parameter_name)
        if result is not None:
            return result

    return None

import scipy.signal as scp

g0 = 7.748091729e-5
run_id = 462
dataset = qc.load_by_id(run_id)

df = dataset.to_pandas_dataframe().reset_index()
ds = dataset.to_xarray_dataset()

current_meas = -ds['current_1'].values
bias_meas = ds['bias_1'].values
B = ds['Left_Dot_Plunger'].values
deriv = scp.savgol_filter(current_meas, 3, 1, deriv=1, axis=1) / ((bias_meas[1] - bias_meas[0]) * 1e-3) / g0

gate_names = ['Backbone', 'Left_Dot_Normal_Barrier', 'Left_Dot_Super_Barrier', 'Right_Dot_Normal_Barrier', 
              'Right_Dot_Super_Barrier', 'Right_Dot_Plunger']

gate_lines = []
for gate in gate_names:
    result = find_snapshot_parameter(dataset.snapshot, gate)
    if result is None:
        gate_lines.append(f'{gate}:\n  not found')
    else:
        value, unit = result
        gate_lines.append(f'{gate}:\n  {value:.1f} {unit}')

fig, (ax_text, ax) = plt.subplots(1, 2, figsize=(7, 3), dpi=400, gridspec_kw={'width_ratios': [1, 5]})

ax_text.axis('off')
ax_text.text(0, 0.5, '\n\n'.join(gate_lines), ha='left', va='center', fontsize=8, family='monospace')

im = ax.pcolor(B, bias_meas * 1e3, deriv.T * 1e3, vmin=-1, vmax=10, cmap='RdBu_r')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$G$ ($2e^2/h$) $\times 10^{-3}$')

ax.set_xlabel('$V_{plunger}$ (mV)')
ax.set_ylabel('$V_{bias}$ ($\mu$V)')
ax.set_title(f'Measurement {run_id} (num. derivative)', fontsize=9)

fig.tight_layout()

#%% the bias search
import scipy.signal as scp

run_id = 239
ds = qc.load_by_id(run_id).to_xarray_dataset()

gate = ds['QDL_P_compensated'].values
bias = ds['bias_2'].values
current = ds['current_2'].values

dIdP = scp.savgol_filter(current, 3, 1, deriv=1, delta=gate[1] - gate[0], axis=1)

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
im = ax.pcolormesh(gate, bias, dIdP, shading='auto', cmap='RdBu_r')
fig.colorbar(im, ax=ax, label=r'$dI/dV_P$')
ax.set(xlabel='QDL_P compensated (mV)', ylabel='Bias 2 (mV)', title=f'Measurement {run_id}')