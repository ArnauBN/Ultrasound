# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:33:44 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pyplot as plt
import time

import src.ultrasound as US
from src.devices import SeDaq as SD
from src.devices import Arduino


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
Temperature = False
twoSensors = False              # If True, assume we have two temperature sensors, one inside and one outside - bool
board = 'Arduino UNO'           # Board type
baudrate = 9600                 # Baudrate (symbols/s)
port = 'COM3'                   # Port to connect to
N_avg = 1                       # Number of temperature measurements to be averaged - int

print(f'The experiment will take {US.time2str(N_acqs*Ts_acq)}.')

#%% Start serial communication
if Temperature:
    arduino = Arduino.Arduino(board, baudrate, port, twoSensors, N_avg)  # open comms

    
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
GenCode = US.MakeGenCode(Excitation=Excitation, ParamVal=Excitation_params)
SeDaq.UpdateGenCode(GenCode)
print('Generator code created and updated.')
print("===================================================\n")
SeDaq.AvgSamplesNumber = AvgSamplesNumber
SeDaq.Quantiz_Levels = Quantiz_Levels


#%% Capture rapid data: nSegments
Cw = 1480 # speed of sound in water (m/s) - float
ToF2dist = Cw*1e6/Fs # conversion factor (um) - float
ToF = np.zeros(N_acqs)

if Temperature:
    temp1 = np.zeros(N_acqs)
    temp2 = np.zeros(N_acqs)
    Cw_vector = np.zeros(N_acqs)
    
for i in range(N_acqs):
    start_time = time.time()
    
    TT_Ascan = SeDaq.GetAscan_Ch1(Smin1, Smax1) # acq Ascan
    
    if Temperature:
        tmp = arduino.getTemperature(error_msg=f'Warning: wrong temperature data at Acq. #{i+1}/{N_acqs}. Retrying...', 
                                                    exception_msg=f'Warning: could not parse temperature data to float at Acq. #{i+1}/{N_acqs}. Retrying...')
        if twoSensors:
            temperature2 = tmp[1]
            temperature1 = tmp[0]
        else:
            temperature1 = tmp
    
        Cw = US.speedofsound_in_water(temperature1, method='Abdessamad', method_param=148)
        Cw_vector[i] = Cw
        ToF2dist = Cw*1e6/Fs
    
    # Create time data
    t = np.arange(0, len(TT_Ascan))/Fs

    TT = TT_Ascan
    if i==0 or i==1: TT0 = TT_Ascan

    fig, axs = plt.subplots(2, num='Signal', clear=True)
    US.movefig(location='southeast')
    axs[0].plot(t*1e-3, TT0, lw=2)
    axs[1].plot(t*1e-3, TT, lw=2)
    axs[1].set_xlabel('Time (us)')
    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Voltage (mV)')
    axs[0].set_title('A')
    axs[1].set_title('B')
    plt.pause(0.05)
    
    ToF[i] = US.CalcToFAscanCosine_XCRFFT(TT, TT0, UseCentroid=False, UseHilbEnv=False, Extend=False)[0]
    
    fig, ax = plt.subplots(1, num='ToF', clear=True)
    US.movefig(location='northeast')
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
        US.movefig(location='south')
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
        arduino.close()
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