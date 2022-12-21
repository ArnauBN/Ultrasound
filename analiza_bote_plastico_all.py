# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:49:11 2022

@author: arnau
"""
import numpy as np
import sys
sys.path.insert(0, r"G:\Unidades compartidas\Proyecto Cianocrilatos")

import src.ultrasound as US


#%%
##############################################################################
# Load data
##############################################################################

DataPath = r"G:\Unidades compartidas\Proyecto Cianocrilatos\Plastic_Bottle"

ExperimentName = "_Plastic_Bottle"
Experiments = [r"\Center"+ExperimentName, 
               r"\Center"+ExperimentName+"_G2-31dB", 
               r"\Center"+ExperimentName+"_G2-35dB", 
               r"\Left"+ExperimentName, 
               r"\Left"+ExperimentName+"_G2-31dB", 
               r"\Left"+ExperimentName+"_G2-35dB", 
               r"\Right"+ExperimentName, 
               r"\Right"+ExperimentName+"_G2-31dB", 
               r"\Right"+ExperimentName+"_G2-35dB"]
WaterPaths = [r"\WP"+ExperimentName,
              r"\WP"+ExperimentName+"_G2-31dB",
              r"\WP"+ExperimentName+"_G2-35dB"]

N_Specimens = len(Experiments)
N_WP = len(WaterPaths)


stdVar = US.StdVar(DataPath + Experiments[0] + "_Experiment" + r"\standard.var")
ScanLen = int(stdVar.Smax-stdVar.Smin)
AvgSamplesNumber = stdVar.AvgSamplesNumber


PE_Ascan = np.zeros((N_Specimens, ScanLen)) # matrix with all data for PE
TT_Ascan = np.zeros_like(PE_Ascan) # matrix with all data for TT
WP_Ascan = np.zeros((N_WP, ScanLen)) # matrix with all data for WP
Cl = np.zeros(N_Specimens)
L = np.zeros_like(Cl)
ToF_TW = np.zeros_like(Cl)
ToF_RW = np.zeros((N_Specimens,2))
ToF21 = np.zeros_like(Cl)

GenCode = 0 # 0:pulse, 1:burst, 2:chirp
Avg = 1 # number of Ascans to be read and averaged
for i in range(N_Specimens):
    Specimen_Path = DataPath + Experiments[i] + "_Experiment"
    
    # beware!! in this experiment, Ascans where accumulated in acq. stage, but
    # not divided by the number of averages, that must be extracted from stdVar
    
    # Load PE
    filename_Ch2 = Specimen_Path + r'\ScanADCgenCod' + str(GenCode+1) + 'Ch2.bin' # load PE
    data = US.LoadBinAscan(filename_Ch2, Avg, ScanLen, N1=0, N2=0) / AvgSamplesNumber
    PE_Ascan[i, :] = US.zeroPadding(data[0:3000], ScanLen)

    # Load TT
    filename_Ch1 = Specimen_Path + r'\ScanADCgenCod' + str(GenCode+1) + 'Ch1.bin' # load TT 
    data = US.LoadBinAscan(filename_Ch1, Avg, ScanLen, N1=0, N2=0) / AvgSamplesNumber
    TT_Ascan[i, :] = data

    # Load WP
    if i<=2:
        WP_Path = DataPath + WaterPaths[i] + '_Experiment'
        filename_Ch1 = WP_Path + r'\ScanADCgenCod' + str(GenCode+1) + 'Ch1.bin' # load WP 
        data = US.LoadBinAscan(filename_Ch1, Avg, ScanLen, N1=0, N2=0) / AvgSamplesNumber
        WP_Ascan[i, :] = data

#%%
##############################################################################
# Constants and variables
##############################################################################
Fs = 100e6 # sampling frequency, in MHz
nfft = 2**US.nextpow2(ScanLen)
Freq_axis = np.arange(nfft) * Fs/nfft
# Time_axis = np.arange(ScanLen)/Fs
Cw = 1498 # speed of sound in water m/s

#%%
##############################################################################
# Computation
##############################################################################
cont = 0
for i in range(N_Specimens):
    if i==3 or i==6:
        cont = cont + 1
    # Find ToF_TW
    ToF_TW[i], _, _ = US.CalcToFAscanCosine_XCRFFT(TT_Ascan[i,:], WP_Ascan[cont,:], UseCentroid=False, UseHilbEnv=False, Extend=False)
    
    # Iterative Deconvolution
    ToF_RW[i,:], StrMat = US.deconvolution(PE_Ascan[i,:], WP_Ascan[cont,:], stripIterNo=2, UseHilbEnv=False)
    ToF21[i] = ToF_RW[i,1] - ToF_RW[i,0]
    
    # Compute velocity and thickness
    Cl[i] = Cw * ( 2*np.abs(ToF_TW[i]/ToF21[i]) + 1) # experimental speed in m/s
    L[i] = Cw/2 * (2*np.abs(ToF_TW[i]) + np.abs(ToF21[i]))/Fs # experimental thickness in m

Cl_mean = np.mean(Cl)
L_mean = np.mean(L)
ToF_TW_mean = np.mean(ToF_TW)
ToF21_mean = np.mean(ToF21)

Cl_std = np.std(Cl)
L_std = np.std(L)
ToF_TW_std = np.std(ToF_TW)
ToF21_std = np.std(ToF21)

#%%
##############################################################################
# Print results
##############################################################################
e = 'Center'
print('\n')
print('       | Cl (m/s)  | L (mm) | ToF_TW   | ToF21')
print('------------------------------------------------')
for i in range(N_Specimens):
    if i == 3: e = 'Left  '
    if i == 6: e = 'Right '
    print(f'{e} | {Cl[i]:.4f} | {L[i]*1e3:.4f} | {ToF_TW[i]:.4f} | {ToF21[i]:.4f}\n')
print('------------------------------------------------')
print(f'Mean   | {Cl_mean:.4f} | {L_mean*1e3:.4f} | {ToF_TW_mean:.4f} | {ToF21_mean:.4f}')
print(f'Std    | {Cl_std:.4f}  | {L_std*1e3:.4f}  | {ToF_TW_std:.4f}  | {ToF21_std:.4f}')

print(f'\nCenter mean: L = {np.mean(L[:3])*1e3:.4f} mm')
print(f'Left mean:   L = {np.mean(L[3:6])*1e3:.4f} mm')
print(f'Right mean:  L = {np.mean(L[6:])*1e3:.4f} mm')