# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:05:22 2023
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
# Longitudinal velocity: 2730 m/s
# Shear velocity: 1430 m/s
# Density: PMMA -> 1.18 g/cm^3 
#           MMA -> 0.94 g/cm^3


#%%
########################################################
# Paths and file names to use
########################################################
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\EpoxyResin'
Experiment_folder_name = 'test3' # Without Backslashes
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

# Data
PE = US.load_bin_acqs(Acqdata_path, N_acqs)

# Temperature and CW
temperature_dict = US.load_columnvectors_fromtxt(Temperature_path)
temperature = temperature_dict['Inside']
Cw = temperature_dict['Cw']

# Densities
m = US.load_columnvectors_fromtxt(Weights_path, header=False) # weights (g)


#%% Parameters
d = 10 # diameter of the cilinder (cm)
r = d/2


#%% TOF computations
def TOF(x, y):
    return US.CalcToFAscanCosine_XCRFFT(x,y)[0]

ToF = np.apply_along_axis(TOF, 0, PE[:,1:], PE[:,0])


#%%
Aref = np.max(np.abs(PE[:,0]))
print(f'{Aref = }')

Dh = Cw*1e2 * ToF / Fs # cm
V = np.pi * (r**2) * Dh # cm^3
densities = m / V # g/cm^3

print('Densities (g/cm^3):')
for i,density in enumerate(densities):
    print(f'{i+1} --> {density}')


#%% Plot
