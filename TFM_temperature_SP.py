# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:15:56 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.signal as scsig

import src.ultrasound as US


#%%
########################################################
# Paths and file names to use
########################################################
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\TFM'
Experiment_folder_name = 'temperature3Dprintedvessel' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_description_file_name = 'Experiment_description.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Experiment_description_path = os.path.join(MyDir, Experiment_description_file_name)


#%%
########################################################
# Load data
########################################################
# Config
config_dict = US.load_config(Config_path)
N_acqs = config_dict['N_acqs']
Fs = config_dict['Fs']
Smin1 = config_dict['Smin1']
# Smin1 = 0

# Data
TT = US.load_bin_acqs(Acqdata_path, N_acqs, TT_and_PE=False)

# Temperature and CW
temperature_dict = US.load_columnvectors_fromtxt(Temperature_path)
temperature = temperature_dict['Inside']
Cw = temperature_dict['Cw']

# Results
results_dict = US.load_columnvectors_fromtxt(Results_path)
t = results_dict['t']
ToF = results_dict['ToF']
d = results_dict['d']
corrected_d = results_dict['corrected_d']
good_d = results_dict['good_d']


#%%
ax1, ax2 = plt.subplots(2)[1]
ax1.scatter(t/3600, d*1e3, marker='.', c='k')
ax1.scatter(t/3600, good_d*1e3, marker='.', c='r')
ax1.set_ylabel('d (mm)')
ax1.set_xlabel('Time (h)')

ax2.scatter(t/3600, temperature, marker='.', c='k')
ax2.set_ylabel('Temperature (celsius)')
ax2.set_xlabel('Time (h)')
plt.tight_layout()


#%%
def TTargmax(x):
    return US.CosineInterpMax(x, xcor=False)

TTmax = np.apply_along_axis(TTargmax, 0, TT)
L = (TTmax + Smin1) * Cw / Fs
L0 = (TTmax + Smin1) * Cw[0] / Fs


#%%
target_L = L0[0]
target_Cw = target_L * Fs / (TTmax + Smin1)

found_temperatures = np.zeros(N_acqs)
for i in range(N_acqs):
    found_temperatures[i] = US.speedofsound2temperature(target_Cw[i])

corrected_L = (TTmax + Smin1) * US.speedofsound_in_water(found_temperatures, 'Abdessamad', 148) / Fs
uncorrected_temperature = np.ones(N_acqs) * temperature[0]


#%%
ax1, ax2 = plt.subplots(2)[1]
# ax1, ax2 = plt.subplots(2, figsize=(10,8))[1]
ax1.scatter(t/3600, uncorrected_temperature, marker='.', c='k')
ax1.scatter(t/3600, temperature, marker='.', c='r')
ax1.scatter(t/3600, found_temperatures, marker='.', c='b')
ax1.set_ylabel('Temperature (celsius)')
ax1.set_xlabel('Time (h)')
ax1.set_ylim([18.5, 22.5])

# inset
axins1 = ax1.inset_axes([0.68, 0.63, 0.3, 0.3])
axins1.set_xlim(0, 0.08)
axins1.set_ylim(19.65, 19.85)
ax1.indicate_inset_zoom(axins1, edgecolor="black")
axins1.scatter(t/3600, uncorrected_temperature, marker='.', c='k')
axins1.scatter(t/3600, temperature, marker='.', c='r')
axins1.scatter(t/3600, found_temperatures, marker='.', c='b')
axins1.get_xaxis().set_ticks([])
axins1.get_yaxis().set_ticks([])



ax2.scatter(t/3600, L0*1e3, marker='.', c='k')
ax2.scatter(t/3600, L*1e3, marker='.', c='r')
ax2.scatter(t/3600, corrected_L*1e3, marker='.', c='b')
ax2.set_ylabel('L (mm)')
ax2.set_xlabel('Time (h)')
ax2.set_ylim([99.2, 100])

# inset
axins2 = ax2.inset_axes([0.68, 0.07, 0.3, 0.3])
axins2.set_xlim(0, 0.08)
axins2.set_ylim(99.65, 99.675)
ax2.indicate_inset_zoom(axins2, edgecolor="black")
axins2.scatter(t/3600, L0*1e3, marker='.', c='k')
axins2.scatter(t/3600, L*1e3, marker='.', c='r')
axins2.scatter(t/3600, corrected_L*1e3, marker='.', c='b')
axins2.get_xaxis().set_ticks([])
axins2.get_yaxis().set_ticks([])

plt.tight_layout()

#%%
# new_data, outliers, outliers_indexes = US.reject_outliers(data, m=0.6745)


#%%
idx = int(N_acqs//2)
print((corrected_L[idx] - L[idx])*1e6)
print((L[idx] - L0[idx])*1e6)

print(max(abs(temperature - found_temperatures)))