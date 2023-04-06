# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:33:39 2023
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
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data'
Experiment_folder_name = 'Aref50us_0dB' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_description_file_name = 'Experiment_description.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
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
Gain = config_dict['Gain_Ch2']

# Data
PE = US.load_bin_acqs(Acqdata_path, N_acqs, TT_and_PE=False)

# Temperature and CW
if os.path.exists(Temperature_path):
    temperature_dict = US.load_columnvectors_fromtxt(Temperature_path, header=True)
    temperature = temperature_dict['Inside']
    Cw = temperature_dict['Cw']


#%%
plt.plot(PE)


#%% Aref
Aref = np.max(np.abs(PE[:,0]))
print(f'{Aref = } V')

Aref_noGain = Aref * (10**(-Gain/20))
print(f'{Aref_noGain = } V')



# 0.07990465875665427 - 10dB
# 0.05770434591711495 - 15dB
# 0.04539084517045455 - 20dB
# 0.02769173372686966 - 25dB
# 0.015579696436806524 - 30dB
# 0.008759385695141203 - 35dB


# ----
# 50us
# ----
# 0.23279786931818175
# 0.06713618336900547 - 11dB
# 0.05273746501890211 - 15dB


# ----------------------------
# Dogbone density measurements
# ----------------------------
# 0.056061742405956315 - 15dB