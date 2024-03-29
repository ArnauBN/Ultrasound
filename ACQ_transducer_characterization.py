# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:29:26 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pyplot as plt
import serial
import time
import os

import src.ultrasound as US
from src.devices import SeDaq as SD
from src.devices import Arduino

#%%
Path = r'D:\Data\transducer_characterization\stainless_steel-50mm'
Experiment_folder_name = 'A' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_acqdata_file_basename = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_PulseFrequencies_file_name = 'pulseFrequencies.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_basename)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
PulseFrequencies_path = os.path.join(MyDir, Experiment_PulseFrequencies_file_name)
if not os.path.exists(MyDir):
    os.makedirs(MyDir)
    print(f'Created new path: {MyDir}')
print(f'Experiment path set to {MyDir}')


#%% Parameters
Experiment_description = "Transducer characterization." \
                        " Transducer A." \
                        " Focused 5 MHz." \
                        " Excitation_params: Number of cycles."
Fs = 100e6                  # Sampling frequency (Hz) - float
RecLen = 32*1024                # Maximum range of ACQ - samples (max=32*1024)
Gain_Ch1 = 65                   # Gain of channel 1 - dB
Gain_Ch2 = 15                   # Gain of channel 2 - dB
Attenuation_Ch1 = 0             # Attenuation of channel 1 - dB
Attenuation_Ch2 = 20            # Attenuation of channel 2 - dB
Smin2 = 4300                    # starting point of the scan of each channel - samples
Smax2 = 9200                    # last point of the scan of each channel - samples
AvgSamplesNumber = 25           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels
Excitation_voltage = 60         # Excitation voltage (min=20V) - V -- DOESN'T WORK
Excitation = 'Burst'            # Excitation to use ('Pulse, 'Chirp', 'Burst') - string
pulse_freqs = np.arange(0.5, 7.5, 0.5)*1e6   # Pulse frequency - Hz
Num_cycles = 5
Excitation_params = Num_cycles     # All excitation params - list or float

Cw = 1480                       # guess for speed of sound in water - m/s

# ---------------------
# Arduino (temperature)
# ---------------------
Temperature = True
board = 'Arduino UNO'           # Board type
baudrate = 9600                 # Baudrate (symbols/s)
port = 'COM3'                   # Port to connect to
N_avg = 10                       # Number of temperature measurements to be averaged - int


#%% Start serial communication with Arduino
if Temperature:
    arduino = Arduino.Arduino(board, baudrate, port, twoSensors=False, N_avg=N_avg)  # open comms


#%%
N_acqs = len(pulse_freqs)
config_dict = {'Fs': Fs,
               'Fs_Gencode_Generator': 200e6,
               'Gain_Ch1': Gain_Ch1,
               'Gain_Ch2': Gain_Ch2,
               'Attenuation_Ch1': Attenuation_Ch1,
               'Attenuation_Ch2': Attenuation_Ch2,
               'Excitation_voltage': Excitation_voltage,
               'Excitation': Excitation,
               'Excitation_params': Excitation_params,
               'Smin2': Smin2,
               'Smax2': Smax2,
               'AvgSamplesNumber': AvgSamplesNumber,
               'Quantiz_Levels': Quantiz_Levels,
               'N_avg' : N_avg,
               'N_acqs' : N_acqs,
               'Experiment_description': Experiment_description}

US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Configuration parameters saved to {Config_path}.')
print("===================================================\n")

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
GenCode = [US.MakeGenCode(Excitation=Excitation, ParamVal=[Fo, Num_cycles]) for Fo in pulse_freqs] # Create all GenCodes
SeDaq.UpdateGenCode(GenCode[0]) # set first GenCode
print('Generator code created and updated.')
print("===================================================\n")
SeDaq.AvgSamplesNumber = AvgSamplesNumber
SeDaq.Quantiz_Levels = Quantiz_Levels

#%% Acq
if Temperature:
    arduino.open()

temperatures = np.zeros(N_acqs)
Cw_vector = np.zeros(N_acqs)
PE_Ascans = np.zeros([N_acqs, Smax2 - Smin2])
for i, g in enumerate(GenCode):
    SeDaq.UpdateGenCode(g)
    print(f'Gencode updated at acq. {i+1}/{N_acqs}.')
    time.sleep(0.2) # just in case
    PE_Ascans[i] = SeDaq.GetAscan_Ch2(Smin2, Smax2) #acq Ascan
    if Temperature:
        temperatures[i] = arduino.getTemperature(error_msg=f'Warning: wrong temperature data at Acq. #{i+1}/{N_acqs}. Retrying...', 
                                     exception_msg=f'Warning: could not parse temperature data to float at Acq. #{i+1}/{N_acqs}. Retrying...')
        Cw_vector[i] = US.speedofsound_in_water(temperatures[i], method='Abdessamad', method_param=148)    
    print(f'Acq. {i+1}/{N_acqs} done.')
    
# ------------------------------------------------
# Save pulse frequencies, temperature and acq data
# ------------------------------------------------
with open(Acqdata_path, 'wb') as f:
    for PE_Ascan in PE_Ascans:
        PE_Ascan.tofile(f)

with open(PulseFrequencies_path, 'w') as f:
    for Fo in pulse_freqs:
        f.write(f'{Fo}\n')

if Temperature:
    with open(Temperature_path, 'w') as f:
        f.write('Inside,Cw\n')
        for t, c in zip(temperatures, Cw_vector):
            f.write(f'{t},{c}\n')

# -----------------
# Close serial comm
# -----------------
if Temperature:
    arduino.close()


#%% Plotting the data
ScanLen = Smax2 - Smin2
nfft = 2**(int(np.ceil(np.log2(np.abs(ScanLen)))) + 3) # Number of FFT points (power of 2)
freq_axis = np.linspace(0, Fs/2, nfft//2)
time_axis = np.arange(Smin2, Smax2)/Fs

PE_Ascans_FFT = np.fft.fft(PE_Ascans, nfft, axis=1)[:,:nfft//2]

plt.figure()
plt.plot(time_axis*1e6, PE_Ascans.T)
plt.xlabel('Time (us)')

plt.figure()
plt.plot(freq_axis*1e-6, np.abs(PE_Ascans_FFT.T))
plt.xlabel('Frequency (MHz)')