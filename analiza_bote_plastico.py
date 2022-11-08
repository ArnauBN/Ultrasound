# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:49:11 2022

@author: arnau
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
from scipy.signal import find_peaks


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
# Time_axis = np.arange(ScanLen)/Fs
Cw = 1498 # speed of sound in water m/s

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

USG.multiplot_tf(np.column_stack((PE_Ascan, TT_Ascan, WP_Ascan)).T, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 20]), f_ylims=None, f_units='MHz', f_Norm=False,
            PSD=False, dB=False, label=['PE', 'TT', 'WP'], FigNum='Signals')



#%% Find ToF_TW
ToF_TW, Aligned_TW, _ = USF.CalcToFAscanCosine_XCRFFT(TT_Ascan, WP_Ascan, UseCentroid=False, UseHilbEnv=False, Extend=False)
print(f'ToF_TW = {ToF_TW}')

#%% Window PE
# Loc_echo = 700 # position in samples of echos, approximation
# WinLen = 700 # window length, approximation

# # windows are centered at approximated surfaces location
# MyWin = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
#                param1=0.25, param2=1, Span=ScanLen, Delay=Loc_echo - int(WinLen/2))

# USG.multiplot_tf(np.column_stack((PE_Ascan, MyWin, WP_Ascan)).T, Fs=Fs,
#                   nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
#                   t_Norm=True, t_xlims=None, t_ylims=None, f_xlims=([0, 20]),
#                   f_ylims=None, f_units='MHz', f_Norm=True, PSD=False, dB=False,
#                   label=['PE', 'Win'], Independent=False, FigNum='PE windowing')

# PE = PE_Ascan * MyWin # extract echoes
PE_Ascan = USF.zeroPadding(PE_Ascan[0:3000], ScanLen)
plt.plot(PE_Ascan)

#%% Iterative Deconvolution
ToF_RW, StrMat = USF.deconvolution(PE_Ascan, WP_Ascan, stripIterNo=2, UseHilbEnv=False)
ToF21 = ToF_RW[1] - ToF_RW[0]
print(f'ToF_RW = {ToF_RW}')
print(f'ToF21 = {ToF21}')

#%%
plt.figure()
plt.title('Iterative Deconvolution')
plt.subplot(3,1,1)
plt.plot(StrMat[0,:])
plt.ylim([-0.12, 0.12])

plt.subplot(3,1,2)
plt.plot(StrMat[1,:])
plt.ylim([-0.12, 0.12])

plt.subplot(3,1,3)
plt.plot(StrMat[2,:])
plt.ylim([-0.12, 0.12])



# #%% Extract PE manually
# Loc_echo_TTWP = 1050
# WinLen_TTWP = 300
# Loc_echo1 = 380 # position in samples of echo from front surface, approximation
# Loc_echo2 = 460 # position in samples of echo from back surface, approximation
# WinLen = 90 # window length, approximation

# # windows are centered at approximated surfaces location
# MyWin1 = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
#                param1=0.25, param2=1, Span=ScanLen, Delay=Loc_echo1 - int(WinLen/2))
# MyWin2 = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
#                param1=0.25, param2=1, Span=ScanLen, Delay=Loc_echo2 - int(WinLen/2))

# USG.multiplot_tf(np.column_stack((PE_Ascan, MyWin1, MyWin2)).T, Fs=Fs,
#                   nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
#                   t_Norm=True, t_xlims=[300,550], t_ylims=None, f_xlims=([0, 20]),
#                   f_ylims=None, f_units='MHz', f_Norm=True, PSD=False, dB=False,
#                   label=['PE', 'Win1', 'Win2'], Independent=False, FigNum='PE windowing')

# PE1 = PE_Ascan * MyWin1 # extract front surface reflection
# PE2 = PE_Ascan * MyWin2 # extract back surface reflection

# USG.multiplot_tf(np.column_stack((PE1, PE2)).T, Fs=Fs,
#                   nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
#                   t_Norm=True, t_xlims=None, t_ylims=None, f_xlims=([0, 20]),
#                   f_ylims=None, f_units='MHz', f_Norm=False, PSD=False, dB=False,
#                   label=['Front surface PE_1', 'Back surface PE_2'], 
#                   Independent=True, FigNum='PE windowed')

# #%% Window TT and WP manually
# MyWin_TTWP = USF.makeWindow(SortofWin='tukey', WinLen=WinLen_TTWP,
#                param1=0.25, param2=1, Span=ScanLen, Delay=Loc_echo_TTWP - int(WinLen_TTWP/2))
# TT = TT_Ascan * MyWin_TTWP # extract echo from TT
# WP = WP_Ascan * MyWin_TTWP # extract echo from WP

# USG.multiplot_tf(np.column_stack((TT_Ascan, WP_Ascan, MyWin_TTWP)).T, Fs=Fs,
#                   nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
#                   t_Norm=True, t_xlims=None, t_ylims=None, f_xlims=([0, 20]),
#                   f_ylims=None, f_units='MHz', f_Norm=True, PSD=False, dB=False,
#                   label=['TT', 'WP', 'Win_TTWP'], Independent=False, FigNum='TT & WP windowing')
# USG.multiplot_tf(np.column_stack((TT, WP)).T, Fs=Fs,
#                   nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
#                   t_Norm=False, t_xlims=None, t_ylims=None, f_xlims=([0, 20]),
#                   f_ylims=None, f_units='MHz', f_Norm=False, PSD=False, dB=False,
#                   label=['TT', 'WP'], 
#                   Independent=True, FigNum='TT & WP windowed')
# # phase for TT_WP
# ToF_TT, AlignedData3, MyXcor_TT = USF.CalcToFAscanCosine_XCRFFT(TT, WP, UseCentroid=False, UseHilbEnv=False, Extend=False)
# print(f'ToF_TT = {ToF_TT}')
# # ToF_TT should be equal to ToF_TW from before

# #%%
# ToF21, _, Xcor21 = USF.CalcToFAscanCosine_XCRFFT(PE2, PE1, UseCentroid=False, UseHilbEnv=False, Extend=False)
# print(f'ToF21 = {ToF21}')

# USG.multiplot_tf(np.column_stack((PE_Ascan, Xcor21)).T, Fs=Fs,
#                   nfft=nfft, Cs=343, t_units='samples', t_ylabel='amplitude',
#                   t_Norm=False, t_xlims=None, t_ylims=None, f_xlims=([0, 7.5]),
#                   f_ylims=None, f_units='MHz', f_Norm=False, PSD=False, dB=False,
#                   label=['PE_Ascan', 'Front surface PE XC', 'Back surface PE XC'], 
#                   Independent=True, FigNum='Pulses correlation')

#%% Compute velocity and thickness
# Cl = Cw * ( 2*ToF_TW/ToF21 + 1) # experimental speed in m/s
# L = Cw/2 * (2*ToF_TW + ToF21)/Fs # experimental thickness in m

Cl = Cw * ( 2*np.abs(ToF_TW/ToF21) + 1) # experimental speed in m/s
L = Cw/2 * (2*np.abs(ToF_TW) + np.abs(ToF21))/Fs # experimental thickness in m

print(f'Cl = {Cl} m/s')
print(f'L = {L*1e3} mm')



#%% Frequency dependent parameters
TT_Ascan = USF.zeroPadding(TT_Ascan[0:3000], ScanLen)
WP_Ascan = USF.zeroPadding(WP_Ascan[0:3000], ScanLen)
xTW = USF.fastxcorr(TT_Ascan, WP_Ascan, Extend=True, Same=True)
Aligned_xTW = USF.ShiftSubsampleByfft(xTW, ToF_TW)
FT_xTW = np.fft.fft(Aligned_xTW, nfft)
phi_TW = np.angle(FT_xTW)

plt.figure()
plt.plot(Aligned_TW)

plt.figure()
plt.plot(phi_TW)


# USG.plot_tf(Aligned_xTW, Data2=None, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
#             t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
#             f_xlims=([0, 10]), f_ylims=None, f_units='MHz', f_Norm=False,
#             PSD=False, dB=False, Phase=True, D1label='Aligned_xTW',
#             D2label='xTW', FigNum='Aligned_xTW phase')
