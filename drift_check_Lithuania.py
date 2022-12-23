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
import ultrasound as US
# import US_Functions as USF
# import US_GenCode as USGC
# import US_Graphics as USG
# import US_ACQ as ACQ

def time2str(seconds) -> str:
    hours = seconds//3600
    minutes = seconds%3600//60
    seconds = seconds - hours*3600 - minutes*60
    s = f'{hours} h, {minutes} min, {seconds} s'
    return s

def find_subsampled_max(x):
    MaxLoc = np.argmax(np.abs(x))  # find index of maximum
    N = x.size  # signal length
    A = MaxLoc - 1  # left proxima
    B = MaxLoc + 1  # Right proxima
    if MaxLoc == 0:  # Check if maxima is in the first or the last sample
        A = N - 1
    elif MaxLoc == N - 1:
        B = 0
    
    # calculate interpolation maxima according to cosine interpolation
    Alpha = np.arccos((x[A] + x[B]) / (2 * x[MaxLoc]))
    Beta = np.arctan((x[A] - x[B]) / (2 * x[MaxLoc] * np.sin(Alpha)))
    Px = Beta / Alpha

    # Calculate ToF in samples
    Real_max = MaxLoc - Px
    return Real_max

def DistToF(x, y, Cw_x, Cw_y, Fs):
    Real_max_x = find_subsampled_max(x)
    Real_max_y = find_subsampled_max(y)
    
    dist_x = Real_max_x * Cw_x/Fs
    dist_y = Real_max_y * Cw_y/Fs
    
    dist = dist_x - dist_y
    return dist

#%% Parameters
Experiment_description = "Drift check." \
                        " Only TT is recorded." \
                        " Focused tx." \
                        " Excitation_params: Pulse frequency (Hz)."
Fs = 100e6                  # Desired sampling frequency (Hz) - float
Ts_acq = 6
N_acqs = 10_000
Fc = 5*1e6                      # Pulse frequency - Hz
RecLen = 32*1024                # Maximum range of ACQ - samples (max=32*1024)
Gain_Ch1 = 75                   # Gain of channel 1 - dB
Gain_Ch2 = 30                   # Gain of channel 2 - dB
Excitation = 'Pulse'            # Excitation to use ('Pulse, 'Chirp', 'Burst') - string
Excitation_params = Fc          # All excitation params - list or float
Smin1 = 5600                    # starting point of the scan of each channel - samples
Smax1 = 7500                    # last point of the scan of each channel - samples
AvgSamplesNumber = 25           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels

# ---------------------
# Arduino (temperature)
# ---------------------
Temperature = True
board = 'Arduino UNO'           # Board type
baudrate = 9600                 # Baudrate (symbols/s)
port = 'COM4'                   # Port to connect to
N_avg = 1                       # Number of temperature measurements to be averaged - int

print(f'The experiment will take {time2str(N_acqs*Ts_acq)}.')

config_dict = {'Fs': Fs,
               'Gain_Ch1': Gain_Ch1,
               'Gain_Ch2': Gain_Ch2,
               'Attenuation_Ch1': 0,
               'Attenuation_Ch2': 10,
               'Excitation_voltage': 60,
               'Excitation': Excitation,
               'Excitation_params': Excitation_params,
               'Smin1': Smin1,
               'Smax1': Smax1,
               'AvgSamplesNumber': AvgSamplesNumber,
               'Quantiz_Levels': Quantiz_Levels,
               'Ts_acq': Ts_acq,
               'N_acqs': N_acqs,
               'WP_temperature' : None,
               'Outside_temperature': None,
               'N_avg' : N_avg,
               'Start_date': '',
               'End_date': '',
               'Experiment_description': Experiment_description}
path = r'D:\Data\measurements_21-12-22\config.txt'
US.saveDict2txt(Path=path, d=config_dict, mode='w', delimiter=',')
print(f'Configuration parameters saved to {path}.')
print("===================================================\n")

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
GenCode = US.MakeGenCode(Excitation=Excitation, ParamVal=Excitation_params)
SeDaq.UpdateGenCode(GenCode)
print('Generator code created and updated.')
print("===================================================\n")


#%% Capture rapid data: nSegments
Cw = 1480 # speed of sound in water (m/s) - float
ToF2dist = Cw*1e6/Fs # conversion factor (um) - float
ToF = np.zeros(N_acqs)
distance = np.zeros(N_acqs)
distance_2 = np.zeros(N_acqs)
distance_0 = np.zeros(N_acqs)
Acqdata_path = r'D:\Data\measurements_21-12-22\acqdata.bin'

if Temperature:
    temp1 = np.zeros(N_acqs)
    # temp2 = np.zeros(N_acqs)
    Cw_vector = np.zeros(N_acqs)

# -------------------------------
# Write start time to config file
# -------------------------------
_start_time = time.strftime("%Y/%m/%d - %H:%M:%S")
config_dict['Start_date'] = _start_time
US.saveDict2txt(Path=path, d=config_dict, mode='w', delimiter=',')
print(f'Experiment started at {_start_time}.')
print("===================================================\n")

for i in range(N_acqs):
    TT_Ascan = US.GetAscan_Ch1(Smin1, Smax1, AvgSamplesNumber=AvgSamplesNumber, Quantiz_Levels=Quantiz_Levels) #acq Ascan
    
    start_time = time.time()

    if Temperature:
        temp1[i] = US.getTemperature(ser, N_avg, twoSensors=False)
        
        Cw = US.speedofsound_in_water(temp1[i], method='Abdessamad', method_param=148)
        Cw_vector[i] = Cw
        ToF2dist = Cw*1e6/Fs # um
    
    _mode = 'wb' if i==0 else 'ab' # clear data from previous experiment before writing
    with open(Acqdata_path, _mode) as f:
        TT_Ascan.tofile(f)
    
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
    distance[i] = ToF[i]*ToF2dist
    if i==0:
        Cw0 = Cw_vector[0]
    else:
        Cw0 = Cw_vector[1]
    # distance_2[i] = DistToF(TT, TT0, Cw, Cw0, Fs)*1e6
    distance_2[i] = (find_subsampled_max(TT)+Smin1) / Fs * 1e3 * Cw
    distance_0[i] = (find_subsampled_max(TT0)+Smin1) / Fs * 1e3 * Cw0
    
    fig, ax = plt.subplots(1, num='ToF', clear=True)
    US.movefig(location='northeast')
    ax.set_ylabel('ToF (samples)')
    ax.set_xlabel('Sample')
    ax.scatter(np.arange(i), ToF[:i], color='k', marker='.')#, edgecolors='black')
    
    # Secondary y-axis with distance
    # ax2 = ax.twinx()
    # mn, mx = ax.get_ylim()
    # ax2.set_ylim([mn*ToF2dist, mx*ToF2dist])
    # ax2.set_ylabel('Distance (um)')
    plt.pause(0.05)

    fig, ax = plt.subplots(1, num='distance', clear=True)
    US.movefig(location='north')
    ax.set_ylabel('Distance (mm)')
    ax.set_xlabel('Sample')
    # ax.scatter(np.arange(i), distance[:i], color='k', marker='.')#color='white', marker='o', edgecolors='black')
    ax.scatter(np.arange(i), distance_2[:i], color='k', marker='.')#color='red', marker='o', edgecolors='red')
    ax.scatter(np.arange(i), distance_0[:i], color='r', marker='.')#color='red', marker='o', edgecolors='red')
    
    plt.pause(0.05)
    
    if Temperature:
        fig, axs = plt.subplots(2, num='Temperature', clear=True)
        US.movefig(location='south')
        axs[0].set_ylabel('Temperature 1 (\u2103)')
        axs[1].set_ylabel('Cw (m/s)')
        axs[1].set_xlabel('Sample')
        axs[0].scatter(np.arange(i), temp1[:i], color='k', marker='.')#color='white', marker='o', edgecolors='black')
        axs[1].scatter(np.arange(i), Cw_vector[:i], color='k', marker='.')#color='white', marker='o', edgecolors='black')
        plt.tight_layout()
        plt.pause(0.05)

    elapsed_time = time.time() - start_time
    time_to_wait = Ts_acq - elapsed_time # time until next acquisition
    print(f'Acquisition #{i+1}/{N_acqs} done.')
    if time_to_wait < 0:
        print(f'Code is slower than Ts_acq = {Ts_acq} s at Acq #{i+1}. Elapsed time is {elapsed_time} s.')
        time_to_wait = 0
    time.sleep(time_to_wait)

# -----------------------------
# Write end time to config file
# -----------------------------
_end_time = time.strftime("%Y/%m/%d - %H:%M:%S")
config_dict['End_date'] = _end_time
US.saveDict2txt(Path=path, d=config_dict, mode='w', delimiter=',')
print(f'Experiment ended at {_end_time}.')
print("===================================================\n")


# Save data     
path = r'D:\Data\measurements_21-12-22\ToF.txt'
with open(path, 'w') as f:
    ToF.tofile(f, sep='\n')

path = r'D:\Data\measurements_21-12-22\temperature.txt'
with open(path, 'w') as f:
    temp1.tofile(f, sep='\n')

path = r'D:\Data\measurements_21-12-22\distance.txt'
with open(path, 'w') as f:
    distance_2.tofile(f, sep='\n')

if Temperature:
    try:
        ser.close()
        print(f'Serial communication with {board} at port {port} closed successfully.')
    except Exception as e:
        print(e)
