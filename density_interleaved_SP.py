# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:33:03 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.signal as scsig

import src.ultrasound as US

# --------------------------------
# Expected values for methacrylate
# --------------------------------
# Density: PMMA -> 1.18 g/cm^3 


#%%
########################################################
# Paths and file names to use
########################################################
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\09-03-23'
Experiment_folder_name = 'density50us18dB_metals' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_description_file_name = 'Experiment_description.txt'
Experiment_weights_file_name = 'weights.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Experiment_description_path = os.path.join(MyDir, Experiment_description_file_name)
Weights_path = os.path.join(MyDir, Experiment_weights_file_name)


#%%
########################################################
# Load data
########################################################
# Config
config_dict = US.load_config(Config_path)
N_acqs = config_dict['N_acqs']
Fs = config_dict['Fs']
Gain = config_dict['Gain_Ch2']

# Data
PE = US.load_bin_acqs(Acqdata_path, N_acqs, TT_and_PE=False)

# Temperature and CW
if os.path.exists(Temperature_path):
    temperature_dict = US.load_columnvectors_fromtxt(Temperature_path)
    temperature = temperature_dict['Inside']
    Cw = temperature_dict['Cw']
else:
    temperature = np.ones(N_acqs) * 20.5
    Cw = US.speedofsound_in_water(temperature)
# Densities
m = US.load_columnvectors_fromtxt(Weights_path, header=False) # weights (g)


#%% Parameters
d = 18.97 # diameter of the cilinder (cm)
r = d/2


#%% TOF computations
ToF = np.zeros(int(N_acqs//2))
for i in range(len(ToF)):
    ToF[i] = US.CalcToFAscanCosine_XCRFFT(PE[:,2*i+1], PE[:,2*i])[0]


#%% Results
Dh = Cw[::2]*1e2 * ToF / Fs / 2 # cm
V = np.pi * (r**2) * Dh # cm^3
densities = m / V # g/cm^3

print('Volumes (cm^3):')
for i,vol in enumerate(V):
    print(f'{i+1} --> {vol}')

print('Densities (g/cm^3):')
for i,density in enumerate(densities):
    print(f'{i+1} --> {density}')


#%% Aref
Arefs = np.max(np.abs(PE[::2]), axis=0)
Arefs_noGain = Arefs * (10**(-Gain/20))

Mean_Arefs = np.mean(Arefs)
print(f'{Mean_Arefs = } V')

Mean_Arefs_noGain = np.mean(Arefs_noGain)
print(f'{Mean_Arefs_noGain = } V')