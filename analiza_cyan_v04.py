# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:04:23 2022
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""

import sys
sys.path.insert(0, r"G:\Unidades compartidas\Proyecto Cianocrilatos")

from scipy import signal
import numpy as np
import matplotlib.pylab as plt
import US_Loaders as USL
import US_LoaderUI
import US_Graphics as USG
import US_Functions as USF
import os
from tkinter import Tk, ttk, filedialog
from scipy.signal import find_peaks

from pathlib import Path


#%%
##############################################################################
# Load data
##############################################################################
# This will open a window to load the specified data. You may keep the window open.
# After pressing the 'Load' button, execute the next cell to get the variables loaded
# to the environment.
_, ui = US_LoaderUI.main()
#%%
PE_Ascan,  TT_Ascan, WP_Ascan, ScanLen, stdVar = US_LoaderUI.getVars(ui)







#%%
##############################################################################
# Constants and variables
##############################################################################
Fs = 100e6 # sampling frequency, in MHz
nfft = 2**USF.nextpow2(ScanLen)
Freq_axis = np.arange(nfft) * Fs/nfft
Time_axis = np.arange(ScanLen)/Fs
Cw = 1498 # speed of sound in water m/s
Cc = 2300 # speed of sound in the plastic container m/s

#%%
USG.plot_tf(PE_Ascan, Data2=TT_Ascan, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 10]), f_ylims=None, f_units='MHz', f_Norm=False,
            PSD=False, dB=False, Phase=False, D1label='PE Ascan',
            D2label='TT_Ascan', FigNum='PE vs TT')

USG.plot_tf(WP_Ascan, Data2=TT_Ascan, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 10]), f_ylims=None, f_units='MHz', f_Norm=False,
            PSD=False, dB=False, Phase=False, D1label='WP Ascan',
            D2label='TT_Ascan', FigNum='WP vs TT')

#%% Zero padding
# From PE, we can remove the echoes after the backsurface reflection
# we will remove the samples and replace with zeros
PE_Ascan = USF.zeroPadding(PE_Ascan[0:6000], ScanLen)
TT_Ascan = USF.zeroPadding(TT_Ascan[0:6000], ScanLen)
WP_Ascan = USF.zeroPadding(WP_Ascan[0:6000], ScanLen)

#%%
USG.plot_tf(PE_Ascan, Data2=TT_Ascan, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 10]), f_ylims=None, f_units='MHz', f_Norm=False,
            PSD=False, dB=False, Phase=False, D1label='PE Ascan',
            D2label='TT_Ascan', FigNum='PE vs TT')

USG.plot_tf(WP_Ascan, Data2=TT_Ascan, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 10]), f_ylims=None, f_units='MHz', f_Norm=False,
            PSD=False, dB=False, Phase=False, D1label='WP Ascan',
            D2label='TT_Ascan', FigNum='WP vs TT')


#%% extract reflections by manual windowing
##############################################################################
'''
Extract reflection signals from PE

Select window length and location manually, as echoes are always more or less 
at the same location and for every sample. It should be the longest possible
to be sure it takes echoes when using long excitations, as chirps. Therefore,
we will select two times the location of the first echo, which will be 
approximated according to the longest signal.
We will select tukey window as it produces the lowest side lobes when used
for cross correlation. As we require that pulses are not distorted, we select 
0.25 for the parameter of the window
'''
##############################################################################
Loc_echo1 = 1100 # position in samples of echo from front surface, approximation
Loc_echo2 = 4200 # position in samples of echo from back surface, approximation
WinLen = Loc_echo1 * 2 # window length, approximation

# windows are centered at approximated surfaces location
MyWin1 = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
               param1=0.25, param2=1, Span=ScanLen, Delay=0)
MyWin2 = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_echo2 - int(WinLen/2))

USG.multiplot_tf(np.column_stack((PE_Ascan, MyWin1, MyWin2)).T, Fs=Fs,
                  nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
                  t_Norm=True, t_xlims=None, t_ylims=None, f_xlims=([0, 20]),
                  f_ylims=None, f_units='MHz', f_Norm=True, PSD=False, dB=False,
                  label=['PE', 'Win1', 'Win2'], Independent=False, FigNum='PE windowing')

PE_R = PE_Ascan * MyWin1 # extract front surface reflection
PE_TR = PE_Ascan * MyWin2 # extract back surface reflection

# We can see the high frequency attenuation in back surface echoes, as well as 
# the resonant spectrum
USG.multiplot_tf(np.column_stack((PE_R, PE_TR)).T, Fs=Fs,
                  nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
                  t_Norm=True, t_xlims=None, t_ylims=None, f_xlims=([0, 20]),
                  f_ylims=None, f_units='MHz', f_Norm=False, PSD=False, dB=False,
                  label=['Front surface PE_1', 'Back surfacePE_2'], 
                  Independent=True, FigNum='PE windowed')

##############################################################################
'''
Extract reflection signals from TT and WP

Select window length and location manually, as echoes are always more or less 
at the same location and for every sample. It should be the longest possible
to be sure it takes echoes when using long excitations, as chirps. Therefore,
we will the same window parameters as the ones used for PE. Windows should
be located at pulse locations aprox
'''
##############################################################################

TT = TT_Ascan * MyWin1 # extract echo from TT
WP = WP_Ascan * MyWin1 # extract echo from WP

USG.multiplot_tf(np.column_stack((TT, WP)).T, Fs=Fs,
                  nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
                  t_Norm=False, t_xlims=None, t_ylims=None, f_xlims=([0, 20]),
                  f_ylims=None, f_units='MHz', f_Norm=False, PSD=False, dB=False,
                  label=['TT', 'WP'], 
                  Independent=True, FigNum='TT & WP windowed')

#%% Find ToF_TW
ToF_TW, Aligned_TW, _ = USF.CalcToFAscanCosine_XCRFFT(TT_Ascan, WP_Ascan, UseCentroid=False, UseHilbEnv=False, Extend=False)
print(f'ToF_TW = {ToF_TW}')


#%% Iterative Deconvolution: first face
ToF_RW, StrMat = USF.deconvolution(PE_R, WP_Ascan, stripIterNo=2, UseHilbEnv=False)
ToF_R21 = ToF_RW[1] - ToF_RW[0]
print(f'ToF_RW = {ToF_RW}')
print(f'ToF_R21 = {ToF_R21}')

#%%
_, (ax1, ax2, ax3) = plt.subplots(3)
ax1.set_title('Iterative Deconvolution')

ax1.plot(StrMat[0,:])
ax1.set_ylim([-0.12, 0.12])

ax2.plot(StrMat[1,:])
ax2.set_ylim([-0.12, 0.12])

ax3.plot(StrMat[2,:])
ax3.set_ylim([-0.12, 0.12])

plt.tight_layout()

#%% Iterative Deconvolution: second face
ToF_TRW, StrMat = USF.deconvolution(PE_TR, WP_Ascan, stripIterNo=2, UseHilbEnv=False)
ToF_TR21 = ToF_TRW[1] - ToF_TRW[0]
print(f'ToF_TRW = {ToF_TRW}')
print(f'ToF_TR21 = {ToF_TR21}')

#%%
_, (ax1, ax2, ax3) = plt.subplots(3)
ax1.set_title('Iterative Deconvolution')

ax1.plot(StrMat[0,:])
ax1.set_ylim([-0.03, 0.025])

ax2.plot(StrMat[1,:])
ax2.set_ylim([-0.03, 0.025])

ax3.plot(StrMat[2,:])
ax3.set_ylim([-0.03, 0.025])

plt.tight_layout()

#%%
ToF_TR1 = np.min(ToF_TRW)
ToF_R2 = np.max(ToF_RW)
ToF_TR1R2 = ToF_TR1 - ToF_R2
print(f'ToF_TR1 = {ToF_TR1}')
print(f'ToF_R2 = {ToF_R2}')
print(f'ToF_TR1R2 = {ToF_TR1R2}')

#%% Compute velocity and thickness
Lc = Cc*np.abs(ToF_R21)/2/Fs
LM = (np.abs(ToF_R21) + ToF_TW + ToF_TR1R2/2)*Cw/Fs - 2*Lc
CM = 2*LM/ToF_TR1R2*Fs

print(f'Lc = {Lc*1e3} mm')
print(f'LM = {LM*1e3} mm')
print(f'CM = {CM} m/s')





#%% Without iterative deconvolution
ToF_PE, AlignedData1, MyXcor_PE = USF.CalcToFAscanCosine_XCRFFT(PE_Ascan, WP_Ascan, UseCentroid=True, UseHilbEnv=False, Extend=True)

USG.multiplot_tf(np.column_stack((PE_Ascan, MyXcor_PE)).T, Fs=Fs,
                  nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
                  t_Norm=False, t_xlims=None, t_ylims=None, f_xlims=([0, 7.5]),
                  f_ylims=None, f_units='MHz', f_Norm=False, PSD=False, dB=False,
                  label=['PE_Ascan', 'XC PE & WP'], 
                  Independent=True, FigNum='XC PE & WP')


Env_PE = USF.envelope(MyXcor_PE)

USG.multiplot_tf(np.column_stack((Env_PE, MyXcor_PE)).T, Fs=Fs,
                  nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
                  t_Norm=False, t_xlims=None, t_ylims=None, f_xlims=([0, 7.5]),
                  f_ylims=None, f_units='MHz', f_Norm=False, PSD=False, dB=False,
                  label=['Envelope', 'XC PE & WP'], 
                  Independent=False, FigNum='XC PE & WP and its envelope')

#%%
# From a visual analysis we can see that between maximums there are around 100 samples
# we have to find maxima and isolate them

# we will use the function peaks from scipy.signal. Beware that this function
# is a bit simplistic in the choosing of the maxima. We restrict maxima that
# are at least above the 10% of the maxima (main peak) of the envelope
peaks, _ = find_peaks(Env_PE, prominence=0.1*np.max(Env_PE), width=20)

# now we use cosine interpolation to be more precise... although not really important
Real_peaks = np.zeros((4))
for i in np.arange(4):
    A = peaks[i]-1
    B = peaks[i]+2
    MaxLoc = peaks[i]
    Alpha = np.arccos((Env_PE[A] + Env_PE[B]) / (2 * Env_PE[MaxLoc]))
    Beta = np.arctan((Env_PE[A] - Env_PE[B]) / (2 * Env_PE[MaxLoc] * np.sin(Alpha)))
    Px = Beta / Alpha
    print(Px)
    # recalculate peak location in samples adding the interpolated value
    Real_peaks[i] = peaks[i] - Px
    
ToF_R21 = Real_peaks[1]-Real_peaks[0]
ToF_TR21 = Real_peaks[3]-Real_peaks[2]

print(f'ToF_R21 = {ToF_R21}')
print(f'ToF_TR21 = {ToF_R21}')

