# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:18:00 2023
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
Experiment_folder_name = 'Aref50us' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_acqdata_file_name = 'acqdata.bin'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)


#%%
########################################################
# Load data
########################################################
# Config
config_dict = US.load_config(Config_path)
N_acqs = config_dict['N_acqs']
Fs = config_dict['Fs']
Gains = np.array([15, 16, 17, 18, 19, 20])

# Data
PE = US.load_bin_acqs(Acqdata_path, N_acqs, TT_and_PE=False)


#%%
plt.plot(PE);


#%%
Arefs = np.max(np.abs(PE), axis=0)
ArefsMean = np.mean(Arefs.reshape([6,10]), axis=1)
ArefsMean_noGain = ArefsMean * (10**(-Gains/20))

print(ArefsMean)
print(ArefsMean_noGain)
print(np.mean(ArefsMean_noGain[:4]))


plt.plot(ArefsMean_noGain);
plt.scatter(np.arange(len(ArefsMean_noGain)), ArefsMean_noGain);