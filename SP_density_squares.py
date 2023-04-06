# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:53:27 2023
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
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data'
Experiment_folder_name = 'density_squares_50us15dB' # Without Backslashes
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
PE = US.load_bin_acqs(Acqdata_path, N_acqs, TT_and_PE=False)

# Temperature and CW
if os.path.exists(Temperature_path):
    temperature_dict = US.load_columnvectors_fromtxt(Temperature_path)
    temperature = temperature_dict['Inside']
    Cw = temperature_dict['Cw']
    
    if len(temperature) == int(N_acqs//2 + 1):
        old_temp = temperature
        new_temp = np.zeros(int(N_acqs//2))
        for i in range(len(new_temp)):
            new_temp[i] = (temperature[i] + temperature[i+1]) / 2
        temperature = new_temp
        del new_temp
        Cw = US.speedofsound_in_water(temperature)
else:
    temperature = np.ones(N_acqs) * 20.5
    Cw = US.speedofsound_in_water(temperature)

# Densities
m = US.load_columnvectors_fromtxt(Weights_path, header=False) # weights (g)


#%% Parameters
# d = 19.1685 # diameter 2 of the cilinder (cm)
# d = 18.59 # diameter 1 of the cilinder (cm)
# d = 18.88 # Average diameter
# d = 18.97 # measured with caliber
# d = 18.84 # measured with a piece of paper (measuring the perimeter)
# d = 19.12 # compromise
d = 18.763910369683458 # avg
r = d/2


#%% TOF computations
# ToF1 = US.CalcToFAscanCosine_XCRFFT(PE[:,1], PE[:,0])[0]
# ToF2 = US.CalcToFAscanCosine_XCRFFT(PE[:,4], PE[:,3])[0]
# ToF = np.array([ToF1, ToF2])

ToF = np.zeros(int(N_acqs//2))
for i in range(len(ToF)):
    ToF[i] = US.CalcToFAscanCosine_XCRFFT(PE[:,2*i+1], PE[:,2*i])[0]


#%% Results
# Dh = np.array([Cw[0], Cw[3]])*1e2 * ToF / Fs / 2 # cm
Dh = Cw*1e2 * ToF / Fs / 2 # cm
V = np.pi * (r**2) * Dh # cm^3
densities = m / V # g/cm^3


#%% caliber
Vcal1 = 0.98 * 0.99 * 0.385
Vcal2 = 1.82 * 2.07 * 0.385
Vcal3 = 2.99 * 2.99 * 0.385
Vcal4 = 4.06 * 4.06 * 0.385

Vcal = np.array([Vcal1, Vcal2, Vcal3, Vcal4])
dcal = m / Vcal


#%% Compute diameter
r2 = np.sqrt(Vcal / np.pi / Dh)
d2 = 2 * r2

Dh2 = Vcal / (np.pi * ((d/2)**2))
print(Dh)
print(Dh2)



#%% Print
for diam in d2:
    print(f'diameter = {round(diam, 2)} cm')
print(f'Average diameter = {round(np.mean(d2), 2)} cm')
print()

# print('\t\t1x1\t\t\t3x3')
# print(f'Vcal\t\t{round(Vcal1, 5)}\t\t{round(Vcal2, 5)}\t\tcm^3')
# print(f'V\t\t{round(V[0], 5)}\t\t{round(V[1], 5)}\t\tcm^3')
# print(f'dcal\t\t{round(dcal1, 5)}\t\t{round(dcal2, 5)}\t\tg/cm^3')
# print(f'd\t\t{round(densities[0], 5)}\t\t{round(densities[1], 5)}\t\tg/cm^')


s1 = ''
s2 = 'Vcal'
s3 = 'V'
s4 = 'dcal'
s5 = 'd'
for i in range(len(V)):
    s1 += f'\t\t\t{i+1}x{i+1}'
    s2 += f'\t\t{round(Vcal[i], 5)}'
    s3 += f'\t\t{round(V[i], 5)}'
    s4 += f'\t\t{round(dcal[i], 5)}'
    s5 += f'\t\t{round(densities[i], 5)}'
print(f'{s1}')
print(f'{s2}\t\tcm^3')
print(f'{s3}\t\tcm^3')
print(f'{s4}\t\tg/cm^3')
print(f'{s5}\t\tg/cm^3')