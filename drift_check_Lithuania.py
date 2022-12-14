# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:33:44 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pyplot as plt
import time
import serial

import SeDaq as SD
import US_Functions as USF
import US_GenCode as USGC
import US_Graphics as USG
import US_ACQ as ACQ

def time2str(seconds) -> str:
    hours = seconds//3600
    minutes = seconds%3600//60
    seconds = seconds - hours*3600 - minutes*60
    s = f'{hours} h, {minutes} min, {seconds} s'
    return s

#%% Parameters
Fs = 100e6                  # Desired sampling frequency (Hz) - float
Ts_acq = 4
N_acqs = 2000
Fc = 5*1e6                      # Pulse frequency - Hz
RecLen = 32*1024                # Maximum range of ACQ - samples (max=32*1024)
Gain_Ch1 = 70                   # Gain of channel 1 - dB
Gain_Ch2 = 25                   # Gain of channel 2 - dB
Excitation = 'Pulse'            # Excitation to use ('Pulse, 'Chirp', 'Burst') - string
Excitation_params = Fc          # All excitation params - list or float
Smin1 = 3600                    # starting point of the scan of each channel - samples
Smax1 = 7400                    # last point of the scan of each channel - samples
AvgSamplesNumber = 25           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels

# ---------------------
# Arduino (temperature)
# ---------------------
Temperature = True
board = 'Arduino UNO'           # Board type
baudrate = 9600                 # Baudrate (symbols/s)
port = 'COM3'                   # Port to connect to
N_avg = 1                       # Number of temperature measurements to be averaged - int

print(f'The experiment will take {time2str(N_acqs*Ts_acq)}.')

#%% Start serial communication
if Temperature:
    ser = serial.Serial(port, baudrate, timeout=None)  # open comms
    
#%%
########################################################################
# Initialize ACQ equipment, GenCode to use, and set all parameters
########################################################################
SeDaq = SD.SeDaqDLL() # connect ACQ (32-bit architecture only)
_sleep_time = 1
print('Connected.')
print(f'Sleeping for {_sleep_time} s...')
time.sleep(_sleep_time) # wait to be sure
print("---------------------------------------------------")
SeDaq.SetRecLen(RecLen) # initialize record length
SeDaq.SetGain1(Gain_Ch1)
SeDaq.SetGain2(Gain_Ch2)
print(f'Gain of channel 1 set to {SeDaq.GetGain(1)} dB') # return gain of channel 1
print(f'Gain of channel 2 set to {SeDaq.GetGain(2)} dB') # return gain of channel 2
print("---------------------------------------------------")
GenCode = USGC.MakeGenCode(Excitation=Excitation, ParamVal=Excitation_params)
SeDaq.UpdateGenCode(GenCode)
print('Generator code created and updated.')
print("===================================================\n")


#%% Capture rapid data: nSegments
Cw = 1480 # speed of sound in water (m/s) - float
ToF2dist = Cw*1e6/Fs # conversion factor (um) - float
ToF = np.zeros(N_acqs)

if Temperature:
    temp1 = np.zeros(N_acqs)
    temp2 = np.zeros(N_acqs)
    Cw_vector = np.zeros(N_acqs)
    
for i in range(N_acqs):
    TT_Ascan = ACQ.GetAscan_Ch1(Smin1, Smax1, AvgSamplesNumber=AvgSamplesNumber, Quantiz_Levels=Quantiz_Levels) #acq Ascan
    
    start_time = time.time()

    if Temperature:
        temp1[i], temp2[i] = ACQ.getTemperature(ser, N_avg, f'Warning: wrong temperature data at Acq. #{i+1}/{N_acqs}. Retrying...', f'Warning: could not parse temperature data to float at Acq. #{i+1}/{N_acqs}. Retrying...')
        
        Cw = USF.speedofsound_in_water(temp1[i], method='Abdessamad', method_param=148)
        Cw_vector[i] = Cw
        ToF2dist = Cw*1e6/Fs
        
    # Create time data
    t = np.arange(0, len(TT_Ascan))/Fs

    TT = TT_Ascan
    if i==0 or i==1: TT0 = TT_Ascan

    fig, axs = plt.subplots(2, num='Signal', clear=True)
    USG.movefig(location='southeast')
    axs[0].plot(t*1e-3, TT0, lw=2)
    axs[1].plot(t*1e-3, TT, lw=2)
    axs[1].set_xlabel('Time (us)')
    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Voltage (mV)')
    axs[0].set_title('A')
    axs[1].set_title('B')
    plt.pause(0.05)
    
    ToF[i] = USF.CalcToFAscanCosine_XCRFFT(TT, TT0, UseCentroid=False, UseHilbEnv=False, Extend=False)[0]
    
    fig, ax = plt.subplots(1, num='ToF', clear=True)
    USG.movefig(location='northeast')
    ax.set_ylabel('ToF (samples)')
    ax.set_xlabel('Sample')
    ax.scatter(np.arange(i), ToF[:i], color='white', marker='o', edgecolors='black')
    
    # Secondary y-axis with distance
    ax2 = ax.twinx()
    mn, mx = ax.get_ylim()
    ax2.set_ylim([mn*ToF2dist, mx*ToF2dist])
    ax2.set_ylabel('Distance (um)')
    plt.pause(0.05)

    if Temperature:
        fig, axs = plt.subplots(2, num='Temperature', clear=True)
        USG.movefig(location='south')
        axs[0].set_ylabel('Temperature 1 (\u2103)')
        axs[1].set_ylabel('Temperature 2 (\u2103)')
        axs[1].set_xlabel('Sample')
        axs[0].scatter(np.arange(i), temp1[:i], color='white', marker='o', edgecolors='black')
        axs[1].scatter(np.arange(i), temp2[:i], color='white', marker='o', edgecolors='black')
        plt.tight_layout()
        plt.pause(0.05)

    elapsed_time = time.time() - start_time
    time_to_wait = Ts_acq - elapsed_time # time until next acquisition
    print(f'Acquisition #{i+1}/{N_acqs} done.')
    if time_to_wait < 0:
        print(f'Code is slower than Ts_acq = {Ts_acq} s at Acq #{i+1}. Elapsed time is {elapsed_time} s.')
        time_to_wait = 0
    time.sleep(time_to_wait)

#%% Save data
if Temperature:
    try:
        ser.close()
        print(f'Serial communication with {board} at port {port} closed successfully.')
    except Exception as e:
        print(e)
        
path = r'D:\Data\Arnau\ToF_drift\fishtank_lithuania\ToF.txt'
with open(path, 'w') as f:
    ToF.tofile(f, sep='\n')

path = r'D:\Data\Arnau\ToF_drift\fishtank_lithuania\temperature1.txt'
with open(path, 'w') as f:
    temp1.tofile(f, sep='\n')

path = r'D:\Data\Arnau\ToF_drift\fishtank_lithuania\temperature2.txt'
with open(path, 'w') as f:
    temp2.tofile(f, sep='\n')

#%%
Time_axis = np.arange(N_acqs) * Ts_acq
ToF2dist = Cw_vector*1e6/Fs
plt.figure()
plt.plot(ToF*ToF2dist)
# plt.plot(Time_axis/60, ToF*ToF2dist)
# plt.xlabel('Time (min)')
plt.ylabel('Distance (um)')