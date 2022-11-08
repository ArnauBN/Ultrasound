# -*- coding: utf-8 -*-
"""
Based on APWPOptimisationTest.py from Linas

This version uses self.UpdateGenCode method to update gencodes
Includes many control variables to check performance
Also includes many graphics to check performance

@author: Alberto
"""

from scipy import signal
from scipy import interpolate
import time
import numpy as np
import matplotlib.pylab as plt
import SeDaq as SD
import US_Functions as uf
import US_ACQ as ACQ
import US_GenCode as gc
import US_Graphics as gp
import os


#%% 
########################################################
# constants
########################################################

Fs = 100.0e6 #D sampling frequency
ProgGenCLKfreqMHz=200.0 # CLK of the gencodes generator (internal CLOK, my guess)
ADC_CLKfreqMHz=100.0 # sampling CLOK (sampling frequency of the ACQ, my guess)


# Make dir to save data of the experiment
CurrentDir = r'D:\Data'
RootDir = r"\pruebas_acq"
#SaveDir = "/Experiment_"+time.strftime("%Y_%m_%d")+"_"+time.strftime("%H_%M_%S")
Experiment = r"\test"

MyDir = CurrentDir + RootDir + Experiment
if not os.path.exists(MyDir):
    os.makedirs(MyDir)

# save Ascan length    
#file_handle = file(MyDir + "/ADClength.txt", "w")
#np.savetxt(file_handle, [Smax-Smin], fmt='%i') #save Ascan length
#file_handle.close()    
########################################################

#%%
########################################################
# Initialize ACQ equipmen
########################################################
SeDaq = SD.SeDaqDLL() # connect ACQ
time.sleep(1) # wait to be sure
print("============================================================================")
print("============================================================================")

#RecLen = 32*1024 # max range of ACQ
RecLen = 32*1024 # max range of ACQ
SeDaq.SetRecLen(RecLen) # initialize record length



#SeDaq.SetRelay(1)
# SeDaq.SetRelay(0)

# SeDaq.SetRelay(1)


Gain_Ch2 = 50 #gain of channel 2 in dB
Gain_Ch1 = 50 #gain of channel 1 in dB
SeDaq.SetGain1(Gain_Ch1) #set gain of CH1
SeDaq.SetGain2(Gain_Ch2) #set gain of CH2



########################################################

#%%
########################################################
# initialize GenCode to use
########################################################
# SeDaq.SetRelay(0)

Gain_Ch2 = 20 #gain of channel 2 in dB
Gain_Ch1 = 20 #gain of channel 1 in dB
SeDaq.SetGain1(Gain_Ch1) #set gain of CH1
SeDaq.SetGain2(Gain_Ch2) #set gain of CH2

GNo = 1 # select GenCode to use
Fc=5*1e6 # Pulse frequency, Hz
GenCode = gc.MakeGenCode(Excitation='Pulse', ParamVal = Fc) #crear  

SeDaq.UpdateGenCode(GenCode)

Smin1, Smin2 = 8_000, 16_000
Smax1, Smax2 = 10_000, 18_000
Smin = (Smin1, Smin2) # starting point of the analysis (ACQ range)
Smax = (Smax1, Smax2)  # Last point of the analysis (ACQ range)
Ascan_Ch1, Ascan_Ch2 = ACQ.GetAscan_Ch1_Ch2(Smin, Smax, AvgSamplesNumber = 25, Quantiz_Levels = 1024) #acq Ascan

# Ascan_Ch1 = ACQ.GetAscan_Ch1(Smin, Smax, AvgSamplesNumber = 1, Quantiz_Levels = 1024) #acq Ascan

N = int(np.ceil(np.log2(np.abs(len(Ascan_Ch2)))))+1
nfft = 2**N

gp.plot_tf(Ascan_Ch2, Data2=Ascan_Ch1, Fs=Fs, nfft=None, Cs=1498, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 0]), f_ylims=None, f_units='Hz', f_Norm=False,
            PSD=False, dB=False, Phase=False, D1label='Ascan_ch2',
            D2label='Ascan_ch1', FigNum=None)






