# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 09:26:58 2023
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
Experiment_PEref_file_name = 'PEref.bin'
Experiment_WP_file_name = 'WP.bin'
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_scanpath_file_name = 'scanpath.txt'
Experiment_description_file_name = 'Experiment_description.txt'
Experiment_PEforCW_file_name = 'PEat0_PEat10mm.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
PEref_path = os.path.join(MyDir, Experiment_PEref_file_name)
WP_path = os.path.join(MyDir, Experiment_WP_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Experiment_description_path = os.path.join(MyDir, Experiment_description_file_name)
Scanpath_path = os.path.join(MyDir, Experiment_scanpath_file_name)
PEforCW_path = os.path.join(MyDir, Experiment_PEforCW_file_name)


#%%
########################################################
# Load data
########################################################
# Config
config_dict = US.load_config(Config_path)
N_acqs = config_dict['N_acqs']
Fs = config_dict['Fs']

# Data
TT, PE = US.load_bin_acqs(Acqdata_path, N_acqs)

# Temperature and CW
temperature_dict = US.load_columnvectors_fromtxt(Temperature_path)
temperature = temperature_dict['Inside']
Cw = temperature_dict['Cw']

# Scan pattern
scanpattern = US.load_columnvectors_fromtxt(Scanpath_path, delimiter=',', header=False, dtype=str)

# WP
with open(WP_path, 'rb') as f:
    WP = np.fromfile(f)

# PE ref
with open(PEref_path, 'rb') as f:
    PEref = np.fromfile(f)


#%% Parameters
Ridx = [np.where(scanpattern == s)[0][0] for s in scanpattern if 'R' in s][0] + 1
theta_deg = float(scanpattern[Ridx-1][1:])
theta = theta_deg * np.pi / 180

step = float(scanpattern[0][1:])

x = np.arange(Ridx)*step # mm


#%% Plot temperature
lpf_order = 2
lpf_fc = 100e3 # Hz

if Fs < 2*lpf_fc:
    print(f'Signal does not have frequency components beyond {lpf_fc} Hz, therefore it is not filtered.')
else:
    # Create filter:
    # Calculate the coefficients
    b_IIR, a_IIR = scsig.iirfilter(lpf_order, 2*lpf_fc/Fs, btype='lowpass')
       
    # Filter signal
    temperature_lpf = scsig.filtfilt(b_IIR, a_IIR, temperature)
    
    Cw2 = US.speedofsound_in_water(temperature_lpf)
    
    ax1, ax2 = plt.subplots(2)[1]
    ax1.scatter(np.arange(N_acqs), temperature, marker='.', color='k')
    ax1.plot(temperature_lpf, 'r', lw=3)
    ax1.set_ylabel('Temperature (\u2103)')
    
    ax2.scatter(np.arange(N_acqs), Cw, marker='.', color='k')
    ax2.plot(Cw2, 'r', lw=3)
    ax2.set_ylabel('Cw (m/s)')
    
    plt.tight_layout()


#%% First plot
nfft = 2**(int(np.ceil(np.log2(np.abs(len(WP))))) + 1) # Number of FFT points (power of 2)

US.multiplot_tf(np.column_stack((PE[:,0], TT[:,0], WP, PEref)).T, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 20]), f_ylims=None, f_units='MHz', f_Norm=False,
            PSD=False, dB=False, label=['PE', 'TT', 'WP', 'PEref'], Independent=True, FigNum='Signals', FgSize=(6.4,4.8))


#%% TOF computations
def TOF(x, y):
    return US.CalcToFAscanCosine_XCRFFT(x,y)[0]

def ID(x, y):
    return US.deconvolution(x, y)[0]

ToF_TW = np.apply_along_axis(TOF, 0, TT, WP)
ToF_RW = np.apply_along_axis(ID, 0, PE, PEref)
ToF_R21 = ToF_RW[1] - ToF_RW[0]


#%% Velocity and thickness computations
cw = np.mean(Cw)
# cw = config_dict['Cw']
# cw = Cw
# cw = Cw2

L = cw/2*(2*np.abs(ToF_TW) + ToF_R21)/Fs # thickness - m    
L = L[:Ridx]

CL = cw*(2*np.abs(ToF_TW)/ToF_R21 + 1) # longitudinal velocity - m/s
CL = CL[:Ridx]

cw_aux = np.asarray([cw]).flatten()[::-1]
Cs = cw_aux / np.sqrt(np.sin(theta)**2 + (cw_aux * ToF_TW[::-1] / (np.mean(L) * Fs) + np.cos(theta))**2)
Cs = Cs[:Ridx]


#%% Plot results
ax1, ax2 = plt.subplots(2)[1]

ax1.set_ylabel('Longitudinal velocity (m/s)', color='b')
ax1.plot(x, CL, 'b')

ax1twin = ax1.twinx()
ax1twin.set_ylabel('Shear velocity (m/s)', color='r')
ax1twin.plot(x, Cs, 'r')

ax2.set_ylabel('Thickness (mm)')
ax2.plot(x, L*1e3, 'k')
ax2.set_xlabel('Position (mm)')

plt.tight_layout()


#%% Density
# AR1 = np.max(PE, axis=0)[:Ridx]
# Aref = 1
# d = 0.94*1e3 # kg/m^3
# Zw = d * CL * (Aref - AR1) / (Aref + AR1) # acoustic impedance of water (Pa.s/m)

# plt.figure()
# plt.plot(Zw*1e-6)
# plt.ylabel('Zw (MNs/m)')
# plt.tight_layout()


#%%
from pltGUI import pltGUI
import matplotlib.image as mpimg

img = mpimg.imread(r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\Methacrylate\methacrylate_photo.jpg')

Smin = config_dict['Smin1']
Smax = config_dict['Smax1']
t = np.arange(Smin, Smax) / Fs * 1e6 # us

pltGUI(x, t, CL, Cs, L*1e3, PE, TT, img)