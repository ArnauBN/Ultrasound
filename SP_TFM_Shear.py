# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:33:49 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import os
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse

import src.ultrasound as US

#%%
Path = r'..\Data\Scanner\EpoxyResin\DogboneScan\A'
Nspecimens = 10
UseHilbEnv = False

experiments = {}
for i in range(Nspecimens):
    name = os.path.basename(Path)+str(i+1)
    fullPath = os.path.join(Path, name)
    experiments[name] = US.DogboneSP(fullPath, compute=True, UseHilbEnv=UseHilbEnv, WindowTTshear=False)
    print(f'Specimen {name} done.')
scanpos = experiments[list(experiments.keys())[0]].scanpos
N_acqs = len(scanpos)

#%% Plot
ax1 = plt.subplots(1)[1]
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Shear velocity (m/s)')
ax1.set_ylim([1100, 1550])
plt.plot(scanpos, np.array([v.Cs for v in experiments.values()]).T)

#%% Plot TT
TTgood = experiments['A1'].TT[:,640]
TTbad = experiments['A1'].TT[:,800]
Time_axis = np.arange(len(TTgood))/experiments['A1'].Fs

ellipse1 = Ellipse((23.35,-0.005), 0.8, 0.06, edgecolor='r', facecolor='w', lw=2)
ellipse2 = Ellipse((24.2,0.008), 0.8, 0.06, edgecolor='C0', facecolor='w', lw=2)

ax2 = plt.subplots(1)[1]
ax2.set_xlabel('Time ($\mu$s)')
ax2.set_ylabel('Amplitude (V)')
ax2.set_ylim([-0.04, 0.04])
ax2.set_xlim([22.5, 25])
ax2.add_patch(ellipse1)
ax2.add_patch(ellipse2)
ax2.text(x=23.35, y=0.028, s='Creep wave', horizontalalignment='center', c='r', fontsize=14, fontweight=500)
ax2.text(x=24.2, y=-0.028, s='Shear wave', horizontalalignment='center', c='C0', fontsize=14, fontweight=500)
# plt.plot(Time_axis*1e6, TTgood)
ax2.plot(Time_axis*1e6, TTbad, c='k')
plt.tight_layout()

#%% Compute ToFs
position_limit = 20 # mm
idx = US.find_nearest(experiments[list(experiments.keys())[0]].scanpos, position_limit)[0]

tofs = np.zeros(len(experiments))
wp_tofs = np.zeros(len(experiments))
for i,v in enumerate(experiments.values()):
    tofs[i] = np.mean(v.ToF_TW[::-1][:idx])
    wp_tofs[i] = US.CosineInterpMax(v.WP, xcor=False)
Loc_TTs = wp_tofs + tofs


#%% Compute Results
for i,v in enumerate(experiments.values()):
    v.computeTOF(UseHilbEnv=UseHilbEnv, WindowTTshear=True, Loc_TT=Loc_TTs[i]) # Compute Time-of-Flights
    v.computeResults() # Compute results
    v.computeDensity() # Compute density
    v.computeModuli() # Compute mechanical properties
    
#%% Plot
ax3 = plt.subplots(1)[1]
ax3.set_xlabel('Position (mm)')
ax3.set_ylabel('Shear velocity (m/s)')
ax3.set_ylim(ax1.get_ylim())
ax3.plot(scanpos, np.array([v.Cs for v in experiments.values()]).T)