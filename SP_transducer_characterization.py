# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:58:53 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from numpy.polynomial import polynomial as P

import src.ultrasound as US

#%% Load the data
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\transducer_characterization\stainless_steel-50mm'
Experiment_folder_name = 'F' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_acqdata_file_basename = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_PulseFrequencies_file_name = 'pulseFrequencies.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_basename)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
PulseFrequencies_path = os.path.join(MyDir, Experiment_PulseFrequencies_file_name)

Config_dict = US.load_config(Config_path)
pulse_freqs = US.load_columnvectors_fromtxt(PulseFrequencies_path, header=False)
Temperature_dict = US.load_columnvectors_fromtxt(Temperature_path)
PE_Ascans = US.load_bin_acqs(Acqdata_path, Config_dict['N_acqs'], TT_and_PE=False)
PE_Ascans = PE_Ascans.T

Fs = Config_dict['Fs']
Smax2 = Config_dict['Smax2']
Smin2 = Config_dict['Smin2']
N_acqs = Config_dict['N_acqs']
temperatures = Temperature_dict['Inside']
Cw_vector = Temperature_dict['Cw']

#%% Material properties
L = 50e-3 # stainless steel -> 50 mm
v = 5790 # stainless steel -> 5790 m/s
tof = L/v # ToF in seconds
t = 2*tof # time to receive eco

#%% Windowing
time_axis = np.arange(Smin2, Smax2)/Fs
relative_time_axis = np.zeros_like(time_axis)

trigger_idx = np.where(PE_Ascans[-1,:] >= np.max(PE_Ascans[-1,:])/2)[0][0]

relative_time_axis[trigger_idx:] = np.arange(len(time_axis)-trigger_idx)/Fs
eco_idx = np.where(relative_time_axis >= t-1e-6)[0][0]

PEs = np.copy(PE_Ascans)
PEs[:,eco_idx:] = 0

#%% Plot windowing
ax1, ax2, ax3 = plt.subplots(3)[1]
ax1.plot(PE_Ascans.T)
ax1.set_title('Original')

ax2.plot(PEs.T)
ax2.set_title('Windowed')

ax3.plot(PE_Ascans.T - PEs.T)
ax3.set_title('Difference')
plt.tight_layout()

#%% Correct gain
true_gain = Config_dict['Gain_Ch2'] - Config_dict['Attenuation_Ch2']
PEs = PEs * 10**(-true_gain/10)

#%% Compute FFT
ScanLen = Smax2 - Smin2
nfft = 2**(int(np.ceil(np.log2(np.abs(ScanLen)))) + 3) # Number of FFT points (power of 2)
freq_axis = np.linspace(0, Fs/2, nfft//2)

PEs_FFT = np.fft.fft(PEs, nfft, axis=1, norm='forward')[:,:nfft//2] # norm='forward' is the same as dividing by nfft

#%% Find maximum of FFT
MaxLocs = np.zeros(N_acqs, dtype=int)
MaxVals = np.zeros(N_acqs)

width = (pulse_freqs[1] - pulse_freqs[0])*3 # 150% more just in case
for i, PE in enumerate(PEs_FFT):
    lims = (pulse_freqs[i] - width/2, pulse_freqs[i] + width/2)
    MaxLocs[i], MaxVals[i] = US.max_in_range(np.abs(PE), lims, indep=freq_axis, axis=None)

#%% Plot
plt.figure()
plt.plot(time_axis*1e6, PEs.T);
plt.xlabel('Time (us)')

plt.figure()
plt.plot(freq_axis*1e-6, np.abs(PEs_FFT.T), zorder=1)
plt.plot(freq_axis*1e-6, np.mean(np.abs(PEs_FFT), axis=0), lw=2, c='k', zorder=2) # Average spectrum
plt.scatter(freq_axis[MaxLocs]*1e-6, MaxVals, c='k', s=100, zorder=3)
plt.xlabel('Frequency (MHz)')
plt.xlim([0,10])
if Experiment_folder_name=='D': plt.xlim([0,20])


#%%
# plt.figure()
# plt.xlabel('Frequency (MHz)')
# plt.xlim([0,10])

# for trace in np.abs(PEs_FFT):
#     plt.plot(freq_axis*1e-6, trace, zorder=1)
#     plt.pause(2)








#%% Best method for each trace
if Experiment_folder_name=='A':
    poly = True
    deg = 2
elif Experiment_folder_name=='B':
    poly = False
elif Experiment_folder_name=='C':
    poly = False
elif Experiment_folder_name=='D':
    poly = True
    deg = 14
elif Experiment_folder_name=='E':
    poly = False
elif Experiment_folder_name=='F':
    poly = False

new_freq_axis = np.linspace(freq_axis[MaxLocs[0]], freq_axis[MaxLocs[-1]], 2048)

#%% Interpolation
if not poly:   
    interpolator = interp1d(freq_axis[MaxLocs], MaxVals, kind=2)
    Maxs_full = interpolator(new_freq_axis)

#%% Polynomial fit
if poly:
    c = P.polyfit(freq_axis[MaxLocs], MaxVals, deg=deg)
    Maxs_full = P.polyval(new_freq_axis, c)

#%% Bandwidth
peak_idx = np.argmax(Maxs_full)
peak = np.max(Maxs_full)

bw_sup_idx, bw_sup = US.find_nearest(Maxs_full[peak_idx:], peak/np.sqrt(2))
bw_sup_idx = bw_sup_idx + peak_idx
bw_inf_idx, bw_inf = US.find_nearest(Maxs_full[:peak_idx], peak/np.sqrt(2))

BW = new_freq_axis[bw_sup_idx] - new_freq_axis[bw_inf_idx]
print(f'BW = {BW*1e-6} MHz')
print(f'Center = {new_freq_axis[peak_idx]*1e-6} MHz')
print(f'Center = {np.sqrt(new_freq_axis[bw_sup_idx] * new_freq_axis[bw_inf_idx])*1e-6} MHz')

#%% Plot fit (or interpolation) and bandwidth
max_val_A = 4.557009808215583e-05
max_val_B = 3.224393828340578e-05
max_val_F = 0.0008946725327523992
max_val_chosen = max_val_B

dashedlinescolor = 'gray'
linecolor = 'k'
markerfacecolor = 'w'
markeredgecolor = 'k'

plt.figure()
# plt.plot(new_freq_axis*1e-6, Maxs_full)
plt.plot(new_freq_axis*1e-6, Maxs_full/max_val_chosen, c=linecolor)
plt.scatter(freq_axis[MaxLocs]*1e-6, MaxVals/max_val_chosen, facecolor=markerfacecolor, edgecolor=markeredgecolor, marker='o', s=80, zorder=3)
# plt.scatter(freq_axis[MaxLocs]*1e-6, MaxVals, c='k', s=100, zorder=3)
plt.xlabel('Frequency (MHz)')
plt.xlim([0,10])
if Experiment_folder_name=='D': plt.xlim([0,20])

plt.axhline(peak/np.sqrt(2)/max_val_chosen, ls='--', color=dashedlinescolor)
plt.axvline(new_freq_axis[bw_inf_idx]*1e-6, ls='--', color=dashedlinescolor)
plt.axvline(new_freq_axis[bw_sup_idx]*1e-6, ls='--', color=dashedlinescolor);

# plt.yticks([])

'''
    Gains (dB)
A   20 - 10
B   20 - 10
C   20 - 10
D   15 - 10
E   10 - 10
F   15 - 2*10
'''

'''
Results found (MHz):
    Center    BW
A   3.94    3.39
B   1.64    2.61
C   3.64    1.78
D   8.80    5.25
E   3.56    1.63
F   4.13    1.15

'''