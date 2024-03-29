# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:13:53 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
import os

import src.ultrasound as US
from src.ultrasound import BasicSP


#%%
Names = ['W0_0', 'W0_0_2', 'W0_0_3', 'W0_0_4', 'W0_0_5', 'W0_0_6', 
         'W0_0_7a', 'W0_0_7b', 'W0_0_7c', 'W0_0_7d', 'W0_0_7e', 'W0_0_7f', 'W0_0_7g', 'W0_0_7h', 'W0_0_7i', 
         'W01_0', 'W02_0', 'W03_0', 'W05_0', 'W06_0', 'W07_0', 'W08_0', 'W09_0', 'W10_0',
         'W0_0_M', 'W01_0_M', 'W02_0_M', 'W03_0_M', 'W04_0_M', 'W05_0_M', 'W06_0_M', 'W07_0_M', 'W08_0_M']
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Deposition'

FullPath = os.path.join(Path, Names[0])
WP_path = os.path.join(FullPath, 'WP.bin')
with open(WP_path, 'rb') as f:
    WP = np.fromfile(f)
WPs = np.zeros([len(Names), len(WP)])

for i, Name in enumerate(Names[1:], 1):
    FullPath = os.path.join(Path, Name)
    WP_path = os.path.join(FullPath, 'WP.bin')
    with open(WP_path, 'rb') as f:
        WP2 = np.fromfile(f)
    WPs[i] = WP2

Loc = np.argmax(np.abs(WP))
WinLen = 1500
Win = US.makeWindow(SortofWin='tukey', WinLen=WinLen,
                    param1=0.25, param2=1, Span=len(WP), Delay=Loc - int(WinLen/2))

idx1 = Loc - int(WinLen/2)
idx2 = Loc + int(WinLen/2)
sn = np.r_[WP[:idx1], WP[idx2:]]
M = len(sn)
M_s = M
s = WP[:M]
sR = sn
plt.figure()
plt.plot(sn)
plt.plot(s)

plt.figure()
plt.plot(WP)
plt.plot(Win*np.max(WP))

# WP = WP * Win

# sR = WP[Loc-WinLen//2:Loc+WinLen//2]
# s = WP[Loc-WinLen//2:Loc+WinLen//2]
# sR = WP
# s = WP
# s = s - np.mean(s)
# sR = sR - np.mean(sR)

AvgSamplesNumber = 25
# M = len(sR)
# M_s = len(s)
Fs = 100e6

HalfM_s = np.floor(M_s / 2)  # length of the semi-frequency axis in frequency domain
FAxis1 = np.arange(HalfM_s + 1) / M_s  # Positive semi-frequency axis
FAxis2 = (np.arange(HalfM_s + 2, M_s + 1, 1) - (M_s + 1)) / M_s  # Negative semi-frequency axis
f = np.concatenate((FAxis1, FAxis2))*Fs  # Full reordered frequency axis

dt = 1/Fs
df = f[1]-f[0]



#%% Computation
# ---
# SNR
# ---
E = BasicSP.energy(s[Loc-WinLen//2:Loc+WinLen//2])
N0 = BasicSP.power(sn)
SNR = E/N0
SNR_dB = 10*np.log10(SNR)


# --
# BW
# --
S = BasicSP.PSD(s)
uniS = S[:len(S)//2]
unif = f[:len(f)//2]

peak_idx = np.argmax(uniS)
peak = np.max(uniS)

bw_sup_idx, bw_sup = US.find_nearest(uniS[peak_idx:], peak/2)
bw_sup_idx = bw_sup_idx + peak_idx
bw_inf_idx, bw_inf = US.find_nearest(uniS[:peak_idx], peak/2)

BW = unif[bw_sup_idx] - unif[bw_inf_idx]
f0 = np.sqrt(unif[bw_sup_idx] * unif[bw_inf_idx]) # geometric mean
# f0 = unif[peak_idx] # center
sigma_ToF = 1 / (2 * np.pi * f0 * np.sqrt(BW*M/Fs) * np.sqrt(SNR) * np.sqrt(1 + BW**2/12/f0))


# --
# Fe
# --
f0_Fe = np.sum(f * S) * df / E # center frequency
beta = np.sqrt(np.sum((f-f0_Fe)**2 * S) * df / E) # envelope bandwidth
Fe = np.sqrt(beta**2 + f0_Fe**2) # effective signal bandwidth

sigma_ToF_Fe = 1 / (2 * np.pi * Fe * np.sqrt(SNR))
eps_ToF = np.sqrt(np.pi) * (beta**2) / (Fs**3)


# --------
# Envelope
# --------
env_s = US.envelope(s)
fft_env_s = np.fft.fft(env_s)
envS2 = (np.abs(fft_env_s)/Fs)**2 # divide by Fs to obtain spectral density (bilateral)
uni_envS2 = envS2[:len(envS2)//2]

peak_idx = np.argmax(uni_envS2)
peak = np.max(uni_envS2)
bw_sup_idx = US.find_nearest(uni_envS2, peak/2)[0]

beta_env = unif[bw_sup_idx]
Fe_env = np.sqrt(beta_env**2 + f0_Fe**2) # effective signal bandwidth
sigma_ToF_env = 1 / (2 * np.pi * Fe_env * np.sqrt(SNR))
eps_ToF_env = np.sqrt(np.pi) * (beta_env**2) / (Fs**3)


# -----
# print
# -----
print(f'E             = {E} J')
print(f'N0            = {N0} W')
print(f'SNR_dB        = {SNR_dB} dB')
print(f'BW            = {BW*1e-6 :.4f} MHz')
print(f'Center        = {unif[peak_idx]*1e-6 :.4f} MHz')
print(f'f0            = {f0*1e-6} MHz')
print(f'f0_Fe         = {f0_Fe*1e-6} MHz')
print(f'beta          = {beta*1e-6} MHz')
print(f'Fe            = {Fe*1e-6} MHz')
print(f'sigma_ToF     = {sigma_ToF*1e9} ns')
print(f'sigma_ToF_Fe  = {sigma_ToF_Fe*1e9} ns')
print(f'eps_ToF       = {eps_ToF*1e9} ns')
print(f'beta_env      = {beta_env*1e-6} MHz')
print(f'Fe_env        = {Fe_env*1e-6} MHz')
print(f'sigma_ToF_env = {sigma_ToF_env*1e9} ns')
print(f'eps_ToF_env   = {eps_ToF_env*1e9} ns')
