# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:57:09 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import os.path
import sys
# sys.path.insert(0, r"D:\Dropbox\00 INVESTIGACION\30 CODIGO\PYTHON_CODE\PROYECTOS\PROYECTO_US")
sys.path.insert(0, r"G:\Unidades compartidas\Proyecto Cianocrilatos")

from scipy import signal
import numpy as np
import matplotlib.pylab as plt
import US_Loaders as USL
import US_Graphics as USG
import US_Functions as USF
import os
from tkinter import Tk, ttk, filedialog

from pathlib import Path

#%%
##############################################################################
# Define experiment parameters for data loading
##############################################################################

# DataPath = r'D:\Dropbox\00 INVESTIGACION\30 CODIGO\PYTHON_CODE\PROYECTOS\PROYECTO_US\P07_CIANOCRILATOS'
DataPath = r'G:\Unidades compartidas\Proyecto Cianocrilatos'

#             'Name of batch folder'     : Number of experiments in batch
BATCH_DICT = {r'\CNC3-LOTE 310322-01'    : 12,
              r'\CNC30-LOTE 010322-01'   : 12,
              r'\CNC120-LOTE 210322-01'  : 12,
              r'\CNC1100-LOTE 290322-01' : 11,
              r'\CNC1500-LOTE 290322-01' : 12,}

lettersList = USF.generateLettersList(N=max(BATCH_DICT.values()), backslash=True)
Max_NSpecimens = len(lettersList)
NLotes = len(BATCH_DICT)

# Assuming the same for all experiments
stdVar = USL.StdVar(DataPath + list(BATCH_DICT.keys())[0] + '\A' + '_Experiment' + r"\standard.var")
ScanLen = int(stdVar.Smax-stdVar.Smin) # 7855
Avg = 1 # number of Ascans to be read and averaged
GenCode = 1 # 1:pulse, 2:burst, 3:chirp

PE_R = np.empty((NLotes, Max_NSpecimens, ScanLen))*np.nan
PE_TR = PE_R.copy()

#%% Create windows
Loc_echo1 = 1100 # position in samples of echo from front surface, approximation
Loc_echo2 = 4200 # position in samples of echo from back surface, approximation
WinLen = Loc_echo1 * 2 # window length, approximation

# windows are centered at approximated surfaces location
MyWin1 = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
               param1=0.25, param2=1, Span=ScanLen, Delay=0)
MyWin2 = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_echo2 - int(WinLen/2))

#%% 
##############################################################################
# Load and window data
##############################################################################
PE, TT, WP = USL.loadAscanBatches(BATCH_DICT, DataPath, lettersList, GenCode, Avg)
for i, (batch, NSpecimens) in enumerate(BATCH_DICT.items()):
    for j in range(NSpecimens):
        PE[i, j, :] = USF.zeroPadding(PE[i, j, :6000], ScanLen)
        PE_R[i, j, :] = PE[i, j, :] * MyWin1 # extract front surface reflection
        PE_TR[i, j, :] = PE[i, j, :] * MyWin2 # extract back surface reflection
        
        TT[i, j, :] = TT[i, j, :] * MyWin1
        
        WP[i, :] = WP[i, :] * MyWin1
        

#%%
##############################################################################
# Constants and variables
##############################################################################
Fs = 100e6 # sampling frequency, in MHz
nfft = 2**USF.nextpow2(ScanLen)
Freq_axis = np.arange(nfft) * Fs/nfft
Cw = 1498 # speed of sound in water m/2
Cc = 2300 # speed of sound in the plastic container m/s
ID = True # use Iterative deconvolution

#%% Optional plot
Nlot = 4
Spec = 4
USG.multiplot_tf(np.column_stack((PE[Nlot, Spec, :], TT[Nlot, Spec, :], WP[Nlot, :])).T, 
                 Fs=Fs, nfft=nfft, Cs=343, t_units='samples',t_ylabel='amplitude',
                 t_Norm=False, t_xlims=None, t_ylims=None, f_xlims=([0, 20]),
                 f_ylims=None, f_units='MHz', f_Norm=False, PSD=False, dB=False,
                 label=['PE', 'TT', 'WP'], Independent=True, FigNum='Signals')

#%%
ToF_RW = np.zeros((NLotes, NSpecimens, 2))
ToF_TRW = np.zeros_like(ToF_RW)

ToF_TW = np.zeros((NLotes, NSpecimens))
ToF_R21 = np.zeros_like(ToF_TW)
ToF_TR21 = np.zeros_like(ToF_TW)
ToF_TR1R2 = np.zeros_like(ToF_TW)
Lc = np.zeros_like(ToF_TW)
LM = np.zeros_like(ToF_TW)
CM = np.zeros_like(ToF_TW)

MyXcor_PE = np.zeros_like(PE)
Env = np.zeros_like(PE)

for i in np.arange(NLotes):
    for j in np.arange(NSpecimens):
        # Find ToF_TW
        ToF_TW[i, j], Aligned_TW, _ = USF.CalcToFAscanCosine_XCRFFT(TT[i, j, :], WP[i, :], UseCentroid=False, UseHilbEnv=False, Extend=False)
        
        if ID:
            # Iterative Deconvolution: first face
            ToF_RW[i, j, :], StrMat = USF.deconvolution(PE_R[i, j, :], WP[i, :], stripIterNo=2, UseHilbEnv=False)
            ToF_R21[i, j] = ToF_RW[i, j, 1] - ToF_RW[i, j, 0]              
            
            # Iterative Deconvolution: second face
            ToF_TRW[i, j, :], StrMat = USF.deconvolution(PE_TR[i, j, :], WP[i, :], stripIterNo=2, UseHilbEnv=False)
            ToF_TR21[i, j] = ToF_TRW[i, j, 1] - ToF_TRW[i, j, 0]

        else:
            # find_peaks instead of ID is worse (higher variance)
            MyXcor_PE[i, j, :] = USF.fastxcorr(PE[i, j, :], WP[i, :], Extend=True)
            Env[i, j, :] = USF.envelope(MyXcor_PE[i, j, :])
            Real_peaks = USF.find_Subsampled_peaks(Env[i, j, :], prominence=0.07*np.max(Env[i, j, :]), width=20)
            ToF_R21[i, j] = Real_peaks[1] - Real_peaks[0]
            ToF_TR21[i, j] = Real_peaks[3] - Real_peaks[2]
            ToF_RW[i, j, :] = Real_peaks[:1]
            ToF_TRW[i, j, :] = Real_peaks[2:]
        
        ToF_TR1 = np.min(ToF_TRW[i, j, :])
        ToF_R2 = np.max(ToF_RW[i, j, :])
        ToF_TR1R2[i, j] = ToF_TR1 - ToF_R2
        
        # Compute velocity and thickness
        Lc[i, j] = Cc*np.abs(ToF_R21[i, j])/2/Fs
        LM[i, j] = (np.abs(ToF_R21[i, j]) + ToF_TW[i, j] + ToF_TR1R2[i, j]/2)*Cw/Fs - 2*Lc[i, j]
        CM[i, j] = 2*LM[i, j]/ToF_TR1R2[i, j]*Fs


CM_mean = np.mean(CM, where=~np.isnan(Lc), axis=1)
CM_std = np.std(CM, where=~np.isnan(Lc), axis=1)
Lc_std = np.std(Lc, where=~np.isnan(Lc))
Lc_mean = np.mean(Lc, where=~np.isnan(Lc))
print(f'CM_mean = {CM_mean} m/s')
print(f'CM_std = {CM_std} m/s')
print('-----------------------------')
print(f'Lc_mean = {Lc_mean*1e6} um')
print(f'Lc_std = {Lc_std*1e6} um')
print(f'Lc = {np.round(Lc_mean*1e6)} \u2a72 {np.round(3*Lc_std*1e6)} um')

#%%
import seaborn as sns
import pandas as pd

fig = plt.figure(num='Cl & L', clear=True)
axs1 = fig.add_subplot(211)
axs1.plot(CM, '.')
axs1.set_ylabel('Speed of sound (m/s)')
axs2 = fig.add_subplot(212)
axs2.plot(LM, '.')


df_CM = pd.DataFrame(CM.T, columns=['CNC3', 'CNC30', 'CNC120', 'CNC1100', 'CNC1500'])
df_LM = pd.DataFrame(LM.T, columns=['CNC3', 'CNC30', 'CNC120', 'CNC1100', 'CNC1500'])
fig = plt.figure(num='Boxplot seaborn', clear=True)
axs1 = fig.add_subplot(221)
sns.boxplot(data=df_CM) # Add in points to show each observation
sns.stripplot(data=df_CM, size=4, palette="dark:.3", linewidth=0)
axs2 = fig.add_subplot(222)
sns.violinplot(data=df_CM) # Add in points to show each observation
axs3 = fig.add_subplot(223)
sns.kdeplot(df_CM['CNC3'], fill=True)
sns.kdeplot(df_CM['CNC30'], fill=True)
sns.kdeplot(df_CM['CNC120'], fill=True)
sns.kdeplot(df_CM['CNC1100'], fill=True)
sns.kdeplot(df_CM['CNC1500'], fill=True)

axs1.set_ylabel('Speed of sound (m/s)')
axs2.set_ylabel('Speed of sound (m/s)')
axs3.set_xlabel('Speed of sound (m/s)')
axs4 = fig.add_subplot(224)
sns.boxplot(data=df_LM)# Add in points to show each observation
sns.stripplot(data=df_LM, size=4, palette="dark:.3", linewidth=0)
axs4.set_ylabel('Thickness (cm)')

plt.tight_layout()
