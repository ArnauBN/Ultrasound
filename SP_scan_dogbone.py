# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:04:41 2023

@author: arnau
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scsig

import src.ultrasound as US

def loadDogBoneExperiments(Path: str, Verbose: bool=False, **kwargs) -> dict[US.DogboneSP]:
    ExperimentNames = US.get_dir_names(Path)
    experiments = {}
    for n in ExperimentNames:
        experiments[n] = US.DogboneSP(os.path.join(Path, n), **kwargs)
        if Verbose: print(f'Specimen {n} done.')
    return experiments

def LPF(x, Fs):
    lpf_order = 2
    lpf_fc = 2e-2 # Hz
    b_IIR, a_IIR = scsig.iirfilter(lpf_order, 2*lpf_fc/Fs, btype='lowpass')
    x_lpf = scsig.filtfilt(b_IIR, a_IIR, x)
    return x_lpf
        
#%%
Path = r'..\Data\Scanner'
Path = r'..\Data\Scanner\EpoxyResin\DogboneScan'
Batch = r'Ens'
Full_path = os.path.join(Path, Batch)

experiments = loadDogBoneExperiments(Full_path, 
                                     Verbose=False, 
                                     compute=False, 
                                     UseHilbEnv=False, 
                                     WindowTTshear=False, 
                                     Loc_TT=None)

for e in experiments.values():
#     e.computeTOFFinal2(UseHilbEnv=False, UseCentroid=False, WindowTTshear=False, Loc_TT=None) # Compute Time-of-Flights
#     e.computeResultsFinal2() # Compute results
    
    # e.computeTOFFinal(UseHilbEnv=False, UseCentroid=False, WindowTTshear=False, Loc_TT=None) # Compute Time-of-Flights
    # e.computeResultsFinal(lpf_temperature=True) # Compute results
    
    e.computeTOF(UseHilbEnv=False, UseCentroid=False, WindowTTshear=True, Loc_TT=None) # Compute Time-of-Flights
    e.computeResults() # Compute results
    
    print(f'Specimen {e.name} done.')


#%% Cs
plt.figure()
for e in experiments.values():
    p = plt.plot(LPF(e.Cs, 1/e.Ts), zorder=3)
    plt.plot(e.Cs, c=p[-1].get_color(), alpha=0.5)


#%% CL
plt.figure()
for e in experiments.values():
    p = plt.plot(LPF(e.CL, 1/e.Ts), zorder=3)
    plt.plot(e.CL, c=p[-1].get_color(), alpha=0.5)


#%% Temperature
plt.figure()
for e in experiments.values():
    p = plt.plot(LPF(e.temperature, 1/e.Ts), zorder=3)
    plt.plot(e.temperature, c=p[-1].get_color(), alpha=0.5)


#%% Variance
vars_Cs = [np.var(e.Cs) for e in experiments.values()]
means_Cs = [np.mean(e.Cs) for e in experiments.values()]

Css = np.array([e.Cs for e in experiments.values()])
varTotal_Cs = np.var(Css.flatten())
meanTotal_Cs = np.mean(Css.flatten())

print('Mean (m/s)\tVar (m/s)')
print('--------------------')
for m,v in zip(means_Cs, vars_Cs):
    print(f'{m:.4f}\t{v:.4f}')
print('--------------------')
print(f'{meanTotal_Cs:.4f}\t{varTotal_Cs:.4f}\t<-- Total')

#%%
plt.figure()
for e in experiments.values():
    plt.plot(LPF(e.Cs, 1/e.Ts), zorder=3)
plt.ylim([3110, 3200])

plt.figure()
for e in experiments.values():
    plt.plot(LPF(e.CL, 1/e.Ts), zorder=3)
plt.ylim([6660, 6760])