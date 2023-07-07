# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:03:33 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt

import src.ultrasound as US


#%% Load data
Experiment_Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Deposition\W07_0'
e = US.RealtimeSP(Experiment_Path)
# e.windowAscans()
Smin1 = e.config_dict['Smin1']
Smax1 = e.config_dict['Smax1']
Fs = e.Fs

tt = e.TTraw[:, int(e.N_acqs//2)]
wp = e.WPraw
xcorr = US.fastxcorr(tt, wp)
xcorr = np.roll(xcorr, len(wp))

time = np.arange(0, len(wp))/Fs
tof_axis = np.hstack([-time[::-1], time[1:]])
time += Smin1/Fs


#%% Plotting
ax1, ax2, ax3 = plt.subplots(3)[1]

# Display
ax1.set_ylim([-0.05, 0.035])
ax1.set_xlabel('Time ($\mu$s)')
ax1.set_ylabel('TT signal')
ax1.set_xlim([78,88])
ax1.set_yticks([])

ax2.set_ylim([-0.5, 0.35])
ax2.set_xlabel('Time ($\mu$s)')
ax2.set_ylabel('WP signal')
ax2.set_xlim([78,88])
ax2.set_yticks([])

ax3.set_xlabel('Time of Flight ($\mu$s)')
ax3.set_ylabel('Correlation')
ax3.set_xlim([-5,5])
ax3.set_yticks([])

# Plots
ax1.plot(time*1e6, tt, c='k')
ax2.plot(time*1e6, wp, c='k')
ax3.plot(tof_axis*1e6, np.abs(xcorr), c='gray')
ax3.plot(tof_axis*1e6, np.abs(US.envelope(xcorr)), c='k')

plt.tight_layout()

