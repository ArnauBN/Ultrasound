# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:05:46 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import os
import matplotlib.pylab as plt

import src.ultrasound as US


#%% Load data
# Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\EpoxyResin\Anosonicreversed'
Path = r'..\Data\Scanner\EpoxyResin\Anosonicreversed'
# Path = r'..\Data\Scanner\EpoxyResin\Anosonic'
# Path = r'..\Data\Scanner\EpoxyResin\DogboneScan\A'
# Path = r'..\Data\Scanner\EpoxyResin\DogboneScan\Areversed'
Nspecimens = 10
UseHilbEnv = False
WindowTTshear = False

experiments = {}
for i in range(Nspecimens):
    name = os.path.basename(Path)+str(i+1)
    fullPath = os.path.join(Path, name)
    experiments[name] = US.DogboneSP(fullPath, compute=True, UseHilbEnv=UseHilbEnv, WindowTTshear=WindowTTshear)
    print(f'Specimen {name} done.')
N_acqs = len(experiments[list(experiments.keys())[0]].scanpos)


#%% Plot
ax1 = plt.subplots(1)[1]
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Longitudinal velocity (m/s)')

for e in experiments.values():
    ax1.plot(e.scanpos, e.CL)