#%% imports
from qcodes.dataset import (
    Measurement,
    load_or_create_experiment,
)

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from qcodes import ScaledParameter, validators
from qcodes.utils.dataset.doNd import do1d, do2d

#%%
os.chdir("C://Users//Nanoelectronics.ad//src")

sample_name='SQDv1_S1_R2C1'
from gelab import connect_to_database, live_plot
exp, station = connect_to_database(\
    db_name='SQDv1_S1ltALD_cooldown2',\
    sample_name=sample_name, station_file='C:/Users/Nanoelectronics.ad/src/hybrid_dots/sqd_v1_S1/smurf_station.yml')


def snapshot():
    for component in station.components:
        if type(station.components[component]) == ScaledParameter:
            print(component + ': %.3f' % station.components[component]())
    return 0


def add_or_replace_component(component):
    if component.name in station.components:
        station.remove_component(component.name)
        station.add_component(component)
        print("Component replaced!")

    else:
        station.add_component(component)
        print("Success!")
#%% open plottr
os.environ['PATH'] = r'C:\Users\Nanoelectronics.ad\src\environment\gelab_env\Scripts' + os.pathsep + os.environ['PATH']
live_plot()
#%% Initialize the station
dac = station.load_instrument('ivvi')
dac.get_all()
dac.print_readable_snapshot()

dmm_1 = station.load_instrument('dmm_1')
# dmm_2 = station.load_instrument('dmm_2')
#%% Initialize the magnet
magnet_y = station.load_instrument('By')
magnet_y.current_ramp_limit(0.01)
magnet_y.ramp_rate(0.0001)
magnet_y.field()
magnet_y.is_quenched()
magnet_y.ramping_state()
magnet_y.field_limit()
magnet_y.current_limit()
magnet_y.coil_constant()
magnet_y.setpoint()
magnet_y.print_readable_snapshot()

magnet_z = station.load_instrument('Bz')
magnet_z.current_ramp_limit(0.01)
magnet_z.ramp_rate(0.0005)
magnet_z.field()
magnet_z.is_quenched()
magnet_z.ramping_state()
magnet_z.field_limit()
magnet_z.current_limit()
magnet_z.coil_constant()
magnet_z.setpoint()
magnet_z.print_readable_snapshot()
#%% Define measurement parameters
bias_1 = ScaledParameter(dac.dac2, name='Bias_1', gain=1e-2, unit='mV') #10mV/V = 1e-2
bias_2 = ScaledParameter(dac.dac3, name='Bias_2', gain=1e-2, unit='mV')

add_or_replace_component(bias_1)
add_or_replace_component(bias_2)

current_1 = ScaledParameter(dmm_1.volt, name='Current_1', gain=1e-7, unit='A') #1e-7 = 10M 1e-9 1G
# current_2 = ScaledParameter(dmm_2.volt, name='Current_2', gain=1e-9, unit='A')

bias_offset = 0.0
#%%
BB = ScaledParameter(dac.dac5, name = 'Backbone', gain = 1, unit = 'mV')

BL =  ScaledParameter(dac.dac6, name = 'Barrier_Left', gain = 1, unit = 'mV')
P = ScaledParameter(dac.dac7, name = 'Plunger', gain = 1, unit = 'mV')
BR = ScaledParameter(dac.dac8, name = 'Barrier_Right', gain = 1, unit = 'mV')

SBL = ScaledParameter(dac.dac9, name = 'QDS_Barrier_Left', gain = 1, unit = 'mV')
SP = ScaledParameter(dac.dac10, name = 'QDS_Plunger', gain = 1, unit = 'mV')
SBR = ScaledParameter(dac.dac11, name = 'QDS_Barrier_Right', gain = 1, unit = 'mV')

Hook = ScaledParameter(dac.dac12, name = 'Hook', gain = 5, unit = 'mV')
TG = ScaledParameter(dac.dac16, name = 'JJ_Gate', gain = 5, unit = 'mV')

add_or_replace_component(BB)
add_or_replace_component(Hook)
add_or_replace_component(TG)

add_or_replace_component(BL)
add_or_replace_component(P)
add_or_replace_component(BR)

add_or_replace_component(SBL)
add_or_replace_component(SP)
add_or_replace_component(SBR)
#%% IV DQD
dmm_1.NPLC(1.0)
bias_swept = bias_1
bias_start = -1 + bias_offset
bias_stop =  1 + bias_offset
bias_num = 101
#bias_num = int(abs(bias_start-bias_stop)/20e-3)

bias_swept(bias_start)

meas_params = [current_1]

time.sleep(1)
exp = load_or_create_experiment(experiment_name = 'ohmics 09-11', sample_name = sample_name)
start_time = time.time()
data=do1d(bias_swept, bias_start,bias_stop, bias_num, 0.001, *meas_params, write_period=2, do_plot=False, measurement_name='Ohmics', show_progress=True,use_threads=True)
print(time.time() - start_time)
bias_swept(0)

#%%%% Plot IV Bias_1(Left)
from qcodes import load_by_id
i = 8
dataset = load_by_id(i)
ds = dataset.to_xarray_dataset()

y = ds['Current_1'] #A
x = ds['Bias_1'] * 1e-3 #V

m, b = np.polyfit(x, y, 1) 
print('Resistance: %.2f Ohm' % (1/m))
print('Offset: %.2f nA' %(b * 1e9))
#plt.plot(x*1e3, y*1e9, 'k-')
plt.plot(x*1e3, (b+m*x)*1e9, 'g-')
plt.xlabel('Bias [mV]')
plt.ylabel('Current [nA]')
plt.title( ds.exp_name + ': %.2f kOhm' % ((1/m) / 1e3) + f', pid: {ds.run_id}')

#%%% IV dac2
dmm_1.NPLC(1)
bias_swept = bias_2
bias_start = -0.5
bias_stop =  0.5
bias_num = 201
#bias_num = int(abs(bias_start-bias_stop)/20e-3)

bias_swept(bias_start)

meas_params = [current_1]

time.sleep(1)
exp = load_or_create_experiment(experiment_name = 'ohmics_33-27', sample_name = sample_name)
data=do1d(bias_swept, bias_start,bias_stop, bias_num, 0.02, *meas_params, write_period=2, do_plot=False, measurement_name='Ohmics', show_progress=True,use_threads=True)
bias_swept(0)

#%%%% Plot IV Bias_2(Right)
from qcodes import load_by_id
i = 98
dataset = load_by_id(i)
ds = dataset.to_xarray_dataset()

y = ds['Current_2'] #A
x = ds['Bias_2'] * 1e-3 #V

m, b = np. polyfit(x, y, 1) 
print('Resistance: %.2f Ohm' % (1/m))
print('Offset: %.2f nA' %(b * 1e9))
plt.plot(x*1e3, y*1e9, 'k-')
plt.plot(x*1e3, (b+m*x)*1e9, 'g-')
plt.xlabel('bias [mV]')
plt.ylabel('Current [nA]')
plt.title( ds.exp_name + ': %.2f kOhm' % (1/m / 1e3) + f', pid: {ds.run_id}')
#%%%% Plot deriv Bias_2(Right)
import scipy.signal as scp
from qcodes import load_by_id
i = 222
dataset = load_by_id(i)
ds = dataset.to_xarray_dataset()

y = ds['Current_1'] #A
x = ds['Bias_1'] * 1e-3 #V
deriv = scp.savgol_filter(y, 3, 1, deriv=1, axis=0)

plt.plot(x*1e3, deriv, 'g-')
plt.xlabel('bias [mV]')
plt.ylabel('dI/dV')
plt.title( ds.exp_name + f' pid: {ds.run_id}')
#%% Pinchoffs BB
bias_1(0.67)

gate_swept = BR
gate_start = 1800
gate_stop = 2300
gate_num = 201

meas_params = [current_1]
gate_swept(gate_start)

time.sleep(2)

exp = load_or_create_experiment(experiment_name = f'Pinchoff {gate_swept.name}', sample_name = sample_name)
data=do1d(gate_swept, gate_start, gate_stop, gate_num, 0.1, *meas_params, write_period=2, do_plot=False, measurement_name='Pinchoffs', show_progress=True,use_threads=True)
# bias_1(0)


#%% Pinchoffs SBL
bias_2(0.6)

gate_swept = Hook
gate_start = 8000
gate_stop = 0
gate_num = 401

meas_params = [current_1]
gate_swept(gate_start)

time.sleep(2)

exp = load_or_create_experiment(experiment_name = f'Pinchoff {gate_swept.name}', sample_name = sample_name)
data=do1d(gate_swept, gate_start, gate_stop, gate_num, 0.4, *meas_params, write_period=2, do_plot=False, measurement_name='Pinchoff', show_progress=True,use_threads=True)
# bias_1(0)

#%%%% Plot pinch-off curves
from qcodes import load_by_id
gate_name='Hook'
id = 189
dataset = load_by_id(id)
ds = dataset.to_xarray_dataset()
y_forward = ds['Current_1'] #A
x = ds[gate_name] * 1e-3 #V

plt.plot(x, y_forward*1e9, color='orange', label=f'pid: {ds.run_id}, forward')

dataset = load_by_id(id+1)
ds = dataset.to_xarray_dataset()
y_backwards = ds['Current_1'] #A
x = ds[gate_name] * 1e-3 #V
plt.plot(x, y_backwards*1e9, color='green', label=f'pid: {ds.run_id}, backward')

plt.xlabel(gate_name + ' [V]')
plt.ylabel('Current [nA]')
plt.legend()
plt.title(ds.exp_name)

#%%%Pinchoffs QDS
bias_2(0.5)

gate_swept = Hook
gate_start = 0
gate_stop = 3000
gate_num = 301

meas_params = [current_2]
gate_swept(gate_start)

time.sleep(1)

exp = load_or_create_experiment(experiment_name = f'Pinchoff {gate_swept.name}', sample_name = sample_name)
data=do1d(gate_swept, gate_start, gate_stop, gate_num, 0.01, *meas_params, write_period=2, do_plot=False, measurement_name='Pinchoffs', show_progress=True,use_threads=True)
#bias_2(0)

#%%%% Plot pinch-off curves
from qcodes import load_by_id
gate_name='QDS_Barrier_Left'
id = 176
dataset = load_by_id(id)
ds = dataset.to_xarray_dataset()
y_forward = ds['Current_2'] #A
x = ds[gate_name] * 1e-3 #V

plt.plot(x, y_forward*1e9, color='orange', label=f'pid: {ds.run_id}, forward')

dataset = load_by_id(id+1)
ds = dataset.to_xarray_dataset()
y_backwards = ds['Current_2'] #A
x = ds[gate_name] * 1e-3 #V
plt.plot(x, y_backwards*1e9, color='green', label=f'pid: {ds.run_id}, backward')

plt.xlabel(gate_name+' [V]')
plt.ylabel('Current [nA]')
plt.legend()
plt.title(ds.exp_name)

#%% sweep two parameters

class sweep_two(Parameter):
    """Apply assymetric voltage to b"""
    def __init__(self, plunger_x, plunger_y, scale_x, label):
        # only name is required
        super().__init__(name = "sweep_two", label=label,     
                         docstring='Sweeping the plungers 1 and 2 together, with different scaling factors')
        self.plunger_x = plunger_x
        self.plunger_y = plunger_y
        self.scale_x = scale_x
    
    # you must provide a get method, a set method, or both
    def get_raw(self):
        return None
    
    def set_raw(self, value):
        self.plunger_x.set(value/self.scale_x)
        self.plunger_y.set(value)


BB_BL = sweep_two(BL, BB, 1, "BB_BL")
BB_BR = sweep_two(BR, BB, 1, "BB_BR")
#%% pinch together
bias_1(0.57)

gate_swept = BB_BR
gate_start = 0
gate_stop = 2400
gate_num = 301

meas_params = [current_1]
gate_swept(gate_start)

time.sleep(2)

exp = load_or_create_experiment(experiment_name = f'Pinchoff {gate_swept.name}', sample_name = sample_name)
data=do1d(gate_swept, gate_start, gate_stop, gate_num, 0.1, *meas_params, write_period=2, do_plot=False, measurement_name='Pinchoffs', show_progress=True,use_threads=True)
#%% corner plot
bias_2(0.67)

gate_swept_fast = SBL
gate_fast_start = 1800
gate_fast_stop = 2200
gate_fast_num = 51

gate_swept_slow = SBR
gate_slow_start = 2250
gate_slow_stop = 2650
gate_slow_num = 51

meas_params = [current_1]

gate_swept_fast(gate_fast_start)
gate_swept_slow(gate_slow_start)

time.sleep(3)
dac.dac9.inter_delay = 0.01
dac.dac11.inter_delay = 0.01
exp = load_or_create_experiment(experiment_name = f'{gate_swept_slow.name} vs {gate_swept_fast.name}', sample_name = sample_name)
data=do2d(gate_swept_slow, gate_slow_start, gate_slow_stop, gate_slow_num, 0.1,gate_swept_fast, gate_fast_start, gate_fast_stop,gate_fast_num,0.05, *meas_params, write_period=2, do_plot=False, measurement_name='CornerPlot', show_progress=False,use_threads=True)
dac.dac9.inter_delay = 0.1
dac.dac11.inter_delay = 0.1

#%% spectroscopy
dmm_1.NPLC(0.2)

gate_swept_slow = Hook
gate_slow_start = -1000
gate_slow_stop = 700
gate_slow_num = int(abs(gate_slow_start-gate_slow_stop)/40) #0.8

bias_swept_fast = bias_2
bias_fast_start = -0.8
bias_fast_stop = 0.8
bias_fast_num = 251 #201

bias_swept_fast(bias_fast_start)
gate_swept_slow(gate_slow_start)

meas_params = [current_1]

time.sleep(5)

dac.dac1.inter_delay = 0.0
dac.dac12.inter_delay = 0.01
exp = load_or_create_experiment(experiment_name = 'spectroscopy', sample_name = sample_name)
data=do2d(gate_swept_slow, gate_slow_start, gate_slow_stop, gate_slow_num, 0.01, bias_swept_fast,bias_fast_start, bias_fast_stop,bias_fast_num, 0.001,
           *meas_params, write_period=2, do_plot=False, measurement_name=f'BB@{np.round(BB(),1)}',
             show_progress=False,use_threads=True)
dac.dac12.inter_delay = 0.1
dac.dac1.inter_delay = 0.01
#%% sensor plunger trace
bias_2(0.15)
dmm_1.NPLC(1.0)

gate_swept = SP
gate_start = 1600
gate_stop = 2000
gate_num = int(abs(gate_start-gate_stop)/1) 

gate_swept(gate_start)

time.sleep(5)
meas_params = [current_1]

exp = load_or_create_experiment(experiment_name = f'{gate_swept.name} Sweep, bias@{np.round(bias_2(),2)}', sample_name = sample_name)
start_time = time.time()
data=do1d(gate_swept, gate_start, gate_stop, gate_num, 0.05, *meas_params, write_period=2, do_plot=False, measurement_name=f'SBL@{np.round(SBL(),1)},SBR@{np.round(SBR(),1)},BB@{np.round(BB(),1)}', show_progress=True,use_threads=True)
print(time.time() - start_time)
#%% Coulomb Diamonds sensor
dmm_1.NPLC(0.02)

gate_swept_slow = SP
gate_slow_start = 1700
gate_slow_stop = 2000
gate_slow_num = int(abs(gate_slow_start-gate_slow_stop)/0.8) #0.4

bias_swept_fast = bias_2
bias_fast_start = -1
bias_fast_stop = 1
bias_fast_num = 401 #300

bias_swept_fast(bias_fast_start)
gate_swept_slow(gate_slow_start)

meas_params = [current_1]

time.sleep(5)
dac.dac2.inter_delay = 0.0
dac.dac10.inter_delay = 0.01
exp = load_or_create_experiment(experiment_name = 'SQD stability', sample_name = sample_name)
data=do2d(gate_swept_slow, gate_slow_start, gate_slow_stop, gate_slow_num, 0.05, bias_swept_fast,bias_fast_start, bias_fast_stop,bias_fast_num, 0.002, *meas_params,
           write_period=2, do_plot=False, measurement_name=f'SBL@{np.round(SBL(),1)},SBR@{np.round(SBR(),1)},BB@{np.round(BB(),1)}', 
           show_progress=False,use_threads=True)
dac.dac10.inter_delay = 0.1
dac.dac2.inter_delay = 0.01
#%% corner plot barrier vs plunger
bias_2(0.3)

gate_swept_fast = SP
gate_fast_start = 2000
gate_fast_stop = 2400
gate_fast_num = 401

gate_swept_slow = SBL
gate_slow_start = 2200
gate_slow_stop = 2000
gate_slow_num = 81

meas_params = [current_1]

gate_swept_fast(gate_fast_start)
gate_swept_slow(gate_slow_start)

time.sleep(3)
dac.dac9.inter_delay = 0.01
dac.dac10.inter_delay = 0.01
exp = load_or_create_experiment(experiment_name = f'{gate_swept_slow.name} vs {gate_swept_fast.name}', sample_name = sample_name)
data=do2d(gate_swept_slow, gate_slow_start, gate_slow_stop, gate_slow_num, 0.1,gate_swept_fast, gate_fast_start, gate_fast_stop,gate_fast_num,0.01, *meas_params, write_period=2, do_plot=False, measurement_name='CornerPlot', show_progress=False,use_threads=True)
dac.dac9.inter_delay = 0.1
dac.dac10.inter_delay = 0.1

#%% spectroscopy
dmm_1.NPLC(1.0)

gate_swept_slow = magnet_y.field
gate_slow_start = -0.03
gate_slow_stop = 0.03
gate_slow_num = 60

bias_swept_fast = bias_1
bias_fast_start = -0.3 + bias_offset
bias_fast_stop = 0.3 + bias_offset
bias_fast_num = 301 #201

bias_swept_fast(bias_fast_start)
gate_swept_slow(gate_slow_start)

meas_params = [current_1]

time.sleep(5)

exp = load_or_create_experiment(experiment_name = 'Y-field spectroscopy', sample_name = sample_name)
data=do2d(gate_swept_slow, gate_slow_start, gate_slow_stop, gate_slow_num, 5, bias_swept_fast, bias_fast_start, bias_fast_stop,bias_fast_num, 0.001,
           *meas_params, write_period=2, do_plot=False, measurement_name=f'perpendicular',
             show_progress=False,use_threads=True)
