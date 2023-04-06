# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 12:45:41 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.signal as scsig
from matplotlib.animation import FuncAnimation

import src.ultrasound as US


#%%
########################################################
# Paths and file names to use
########################################################
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\Methacrylate'
Experiment_folder_name = 'focal_width' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_PEref_file_name = 'PEref.bin'
Experiment_WP_file_name = 'WP.bin'
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_scanpath_file_name = 'scanpath.txt'
Experiment_description_file_name = 'Experiment_description.txt'
Experiment_PEforCW_file_name = 'PEat0_PEat10mm.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
PEref_path = os.path.join(MyDir, Experiment_PEref_file_name)
WP_path = os.path.join(MyDir, Experiment_WP_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Experiment_description_path = os.path.join(MyDir, Experiment_description_file_name)
Scanpath_path = os.path.join(MyDir, Experiment_scanpath_file_name)
PEforCW_path = os.path.join(MyDir, Experiment_PEforCW_file_name)


#%%
########################################################
# Load data
########################################################
# Config
config_dict = US.load_config(Config_path)
N_acqs = config_dict['N_acqs']
Fs = config_dict['Fs']

# Data
TT, PE = US.load_bin_acqs(Acqdata_path, N_acqs)

# Temperature and CW
temperature_dict = US.load_columnvectors_fromtxt(Temperature_path)
temperature = temperature_dict['Inside']
Cw = temperature_dict['Cw']

# Scan pattern
scanpattern = US.load_columnvectors_fromtxt(Scanpath_path, delimiter=',', header=False, dtype=str)

# WP
with open(WP_path, 'rb') as f:
    WP = np.fromfile(f)

# PE ref
with open(PEref_path, 'rb') as f:
    PEref = np.fromfile(f)


#%% Plot temperature
lpf_order = 2
lpf_fc = 100e3 # Hz

if Fs < 2*lpf_fc:
    print(f'Signal does not have frequency components beyond {lpf_fc} Hz, therefore it is not filtered.')
else:
    # Create filter:
    # Calculate the coefficients
    b_IIR, a_IIR = scsig.iirfilter(lpf_order, 2*lpf_fc/Fs, btype='lowpass')
       
    # Filter signal
    temperature_lpf = scsig.filtfilt(b_IIR, a_IIR, temperature)
    
    Cw2 = US.speedofsound_in_water(temperature_lpf)
    
    ax1, ax2 = plt.subplots(2)[1]
    ax1.scatter(np.arange(N_acqs), temperature, marker='.', color='k')
    ax1.plot(temperature_lpf, 'r', lw=3)
    ax1.set_ylabel('Temperature (\u2103)')
    
    ax2.scatter(np.arange(N_acqs), Cw, marker='.', color='k')
    ax2.plot(Cw2, 'r', lw=3)
    ax2.set_ylabel('Cw (m/s)')
    
    plt.tight_layout()


#%% Parameters
step = float(scanpattern[0][1:])
x = np.arange(N_acqs) * step # mm


#%% Computations
def TOF(x, y):
    return US.CalcToFAscanCosine_XCRFFT(x,y)[0]

def ID(x, y):
    return US.deconvolution(x, y)[0]

maxs = np.max(PE, axis=0)
ToF_TW = np.apply_along_axis(TOF, 0, TT, WP)
ToF_RW = np.apply_along_axis(ID, 0, PE, PEref)
ToF_R21 = ToF_RW[1] - ToF_RW[0]


#%% Plot results
ax1, ax2 = plt.subplots(2)[1]
ax2_2 = ax2.twinx()

ax1.plot(x, maxs, 'k')
ax1.set_ylabel('Amplitude (V)')

ax2.plot(x, ToF_TW, 'b')
ax2.set_ylabel('ToF_TW (samples)', color='b')
ax2.set_xlabel('Scanner Position (mm)')

ax2_2.plot(x, ToF_R21)
ax2_2.plot(x, ToF_R21, 'r')
ax2_2.set_ylabel('ToF_R21 (samples)', color='r')

plt.tight_layout()

#%% Find on and off values
on_time = 6 # mm
off_time = 20 # mm
on_time_start = 37 # mm

on_idx = np.where(x <= on_time)[0][-1]
off_idx = np.where(x <= off_time)[0][-1]
on_start_idx = US.find_nearest(x, on_time_start)[0]
on_end_idx = US.find_nearest(x, on_time_start + on_time)[0]

off_val = np.mean(np.r_[maxs[:off_idx], maxs[len(maxs)-off_idx:]])
on_val = np.mean(maxs[on_start_idx:on_end_idx])

print(f'{off_val = } V')
print(f'{on_val = } V')

on_val10 = (on_val - off_val) * 0.1 # 10%
on_val90 = (on_val - off_val) * 0.9 # 90%


#%% Find rise time and fall time
# Rise time is from 10% to 90%
on_idx10 = US.find_nearest(maxs[:on_start_idx], on_val10)[0]
on_idx90 = US.find_nearest(maxs[:on_start_idx], on_val90)[0]

on_time10_rise = x[on_idx10]
on_time90_rise = x[on_idx90]
risetime = on_time90_rise - on_time10_rise

print(f'{on_time10_rise = } mm')
print(f'{on_time90_rise = } mm')
print(f'{risetime = } mm')

print('-----------------')

# Fall time is from 90% to 10%
on_idx10 = US.find_nearest(maxs[on_end_idx:], on_val10)[0] + on_end_idx
on_idx90 = US.find_nearest(maxs[on_end_idx:], on_val90)[0] + on_end_idx

on_time10_fall = x[on_idx10]
on_time90_fall = x[on_idx90]
falltime = on_time10_fall - on_time90_fall

print(f'{on_time10_fall = } mm')
print(f'{on_time90_fall = } mm')
print(f'{falltime = } mm')

time = (risetime + falltime) / 2
print(f'Average fall and rise times: {time} mm')


#%% Plot results with risetime and falltime
ax1, ax2 = plt.subplots(2)[1]
ax2_2 = ax2.twinx()

ax1.plot(x, maxs, 'k')
ax1.axvline(on_time10_rise, ls='--', color='grey')
ax1.axvline(on_time90_rise, ls='--', color='grey')
ax1.axvline(on_time10_fall, ls='--', color='grey')
ax1.axvline(on_time90_fall, ls='--', color='grey')
ax1.axhline(on_val10, ls='--', color='grey')
ax1.axhline(on_val90, ls='--', color='grey')
ax1.set_ylabel('Amplitude (V)')
ax1.set_xlim([25,55])

ax2.plot(x, ToF_TW, 'b')
ax2.set_ylabel('ToF_TW (samples)', color='b')
ax2.set_xlabel('Scanner Position (mm)')
ax2.set_xlim([25,55])

ax2_2.plot(x, ToF_R21)
ax2_2.plot(x, ToF_R21, 'r')
ax2_2.set_ylabel('ToF_R21 (samples)', color='r')
ax2_2.set_xlim([25,55])


# ax1_2 = ax1.twinx()
# ax1_2.plot(x, ToF_R21, 'b')
# ax1_2.set_ylabel('ToF_R21 (samples)', color='b')
# ax1_2.set_xlabel('Scanner Position (mm)')
# ax1_2.set_xlim([25,55])

plt.tight_layout()


#%%
N, M = TT.shape
xaxis = np.arange(N)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10,8))
ax1.set_ylim(-0.5,0.5)
ax1.set_xlim(0, N)
ax2.set_ylim(-0.5,0.5)
ax2.set_xlim(0, N)
ax3.set_ylim(-5000,5000)
ax3.set_xlim(0, x[-1])
ax4.set_ylim(-150,5)
ax4.set_xlim(0, x[-1])

PE_line, = ax1.plot([], [], lw=2, c='b')
TT_line, = ax2.plot([], [], lw=2, c='r')
ax3.plot(x, ToF_R21, lw=2, c='b')
ax4.plot(x, ToF_TW, lw=2, c='r')
ToF_R21_vline = ax3.axvline(0, lw=2, c='k', zorder=10)
ToF_TW_vline = ax4.axvline(0, lw=2, c='k', zorder=10)

line = [PE_line, TT_line, ToF_R21_vline, ToF_TW_vline]
ax1.set_ylabel('PE')
ax2.set_ylabel('TT')
ax3.set_ylabel('ToF_R21')
ax4.set_ylabel('ToF_TW')

def init():
    line[0].set_data([], [])
    line[1].set_data([], [])
    line[2].set_xdata(0)
    line[2].set_xdata(0)
    return line

def animate(frame):
    print(f'x = {round(x[frame], 2)} mm')
    line[0].set_data(xaxis, PE[:,frame])
    line[1].set_data(xaxis, TT[:,frame])
    line[2].set_xdata(x[frame])
    line[3].set_xdata(x[frame])
    return line

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(fig, animate, init_func=init, frames=M, interval=0, blit=True, cache_frame_data=False, repeat=False)

plt.show()