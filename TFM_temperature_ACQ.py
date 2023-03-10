# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:51:35 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import time
import numpy as np
import matplotlib.pylab as plt
import os

import src.ultrasound as US
from src.devices import SeDaq as SD
from src.devices import Arduino


#%%
########################################################
# Paths and file names to use
########################################################
Path = r'D:\Data\pruebas_acq\TFM'
Experiment_folder_name = 'temperature3Dvase' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_description_file_name = 'Experiment_description.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Experiment_description_path = os.path.join(MyDir, Experiment_description_file_name)
if not os.path.exists(MyDir):
    os.makedirs(MyDir)
    print(f'Created new path: {MyDir}')
print(f'Experiment path set to {MyDir}')


#%% 
########################################################
# Parameters and constants
########################################################
Experiment_description = """This data was generated by TFM_temperature.py.
Focused tx.
Only the TT is captured.
3Dprinted vase.
Excitation_params: Pulse frequency (Hz)."""

Fs = 100.0e6                    # Sampling frequency - Hz
Fs_Gencode_Generator = 200.0e6  # Sampling frequency for the gencodes generator - Hz
RecLen = 32*1024                # Maximum range of ACQ - samples (max=32*1024)
Gain_Ch1 = 70                   # Gain of channel 1 - dB
Gain_Ch2 = 20                   # Gain of channel 2 - dB
Attenuation_Ch1 = 0             # Attenuation of channel 1 - dB
Attenuation_Ch2 = 10            # Attenuation of channel 2 - dB
Excitation_voltage = 60         # Excitation voltage (min=20V) - V -- DOESN'T WORK
Fc = 5e6                        # Pulse frequency - Hz
Excitation = 'Pulse'            # Excitation to use ('Pulse, 'Chirp', 'Burst') - string
Excitation_params = Fc          # All excitation params - list or float
Smin1, Smin2 = 5600, 5600       # starting point of the scan of each channel - samples
Smax1, Smax2 = 7700, 7700       # last point of the scan of each channel - samples
AvgSamplesNumber = 25           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels
Ts_acq = 4                      # Time between acquisitions (if None, script waits for user input). Coding time is about 1.5s (so Ts_acq must be >1.5s) - seconds
N_acqs = 10_000                 # Total number of acquisitions
Save_acq_data = True            # If True, save all acq. data to {Acqdata_path} - bool
Plot_all_acq = False             # If True, plot every acquisition - bool
twoSensors = False              # If True, assume we have two temperature sensors, one inside and one outside - bool
Plot_temperature = True         # If True, plots temperature measuements at each acq. (has no effect if Temperature==False) - bool

board = 'Arduino UNO'           # Board type
baudrate = 9600                 # Baudrate (symbols/s)
port = 'COM3'                   # Port to connect to
N_avg = 1                       # Number of temperature measurements to be averaged - int

print(f'The experiment will take {US.time2str(N_acqs*Ts_acq)}.')


#%% Start serial communication
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

#%%
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


#%% 
########################################################################
# Save config
########################################################################
config_dict = {'Fs': Fs,
               'Fs_Gencode_Generator': Fs_Gencode_Generator,
               'Gain_Ch1': Gain_Ch1,
               'Gain_Ch2': Gain_Ch2,
               'Attenuation_Ch1': Attenuation_Ch1,
               'Attenuation_Ch2': Attenuation_Ch2,
               'Excitation_voltage': Excitation_voltage,
               'Excitation': Excitation,
               'Excitation_params': Excitation_params,
               'Smin1': Smin1,
               'Smin2': Smin2,
               'Smax1': Smax1,
               'Smax2': Smax2,
               'AvgSamplesNumber': AvgSamplesNumber,
               'Quantiz_Levels': Quantiz_Levels,
               'Ts_acq': Ts_acq,
               'N_acqs': N_acqs,
               'N_avg' : N_avg,
               'Start_date': '',
               'End_date': '',
               'Experiment_description': Experiment_description}

US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Configuration parameters saved to {Config_path}.')

with open(Experiment_description_path, 'w') as f:
    f.write(Experiment_description)
print(f'Experiment description saved to {Experiment_description_path}.')
print("===================================================\n")


#%% 
########################################################################
# Run Acquisition and Computations
########################################################################
# --------------------
# Initialize variables
# --------------------
_plt_pause_time = 0.01

Cw_vector = np.zeros(N_acqs)
ToF = np.zeros(N_acqs)
d = np.zeros(N_acqs)
corrected_d = np.zeros(N_acqs)
good_d = np.zeros(N_acqs)

Smin = (Smin1, Smin2)                   # starting points - samples
Smax = (Smax1, Smax2)                   # last points - samples
ScanLen1 = Smax1 - Smin1                # Total scan length for channel 1 - samples
ScanLen2 = Smax2 - Smin2                # Total scan length for channel 2 - samples
ScanLen = np.max([ScanLen1, ScanLen2])  # Total scan length for computations (zero padding is used) - samples

N = int(np.ceil(np.log2(np.abs(ScanLen)))) + 1 # next power of 2 (+1)
nfft = 2**N # Number of FFT points (power of 2)

Time_axis = np.arange(N_acqs) * Ts_acq # time vector (one acq every Ts_acq seconds) - s
_xlabel = 'Time (s)'
_factor = 1
if Time_axis[-1] > 120:
    _xlabel = 'Time (min)'
    _factor = 60
elif Time_axis[-1] > 7200:
    _xlabel = 'Time (h)'
    _factor = 3600


# ---------------------------------
# Write results header to text file
# ---------------------------------
header = 't,ToF,d,corrected_d,good_d'
with open(Results_path, 'w') as f:
    f.write(header+'\n')


# -------------------------------------
# Write temperature header to text file
# -------------------------------------
header = 'Inside,Outside,Cw' if twoSensors else 'Inside,Cw'
temperature1 = np.zeros(N_acqs)
temperature2 = np.zeros(N_acqs)

with open(Temperature_path, 'w') as f:
    f.write(header+'\n')

arduino.open()


# -------------------------------
# Write start time to config file
# -------------------------------
_start_time = time.strftime("%Y/%m/%d - %H:%M:%S")
config_dict['Start_date'] = _start_time
US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Experiment started at {_start_time}.')
print("===================================================\n")


# ---------
# Sart loop
# ---------
for i in range(N_acqs):
    # -------------------------------------------
    # Acquire signal, temperature and start timer
    # -------------------------------------------
    start_time = time.time() # start timer
    
    TT = SeDaq.GetAscan_Ch1(Smin1, Smax1) # acq Ascan

    tmp = arduino.getTemperature(error_msg=f'Warning: wrong temperature data at Acq. #{i+1}/{N_acqs}. Retrying...', 
                                 exception_msg=f'Warning: could not parse temperature data to float at Acq. #{i+1}/{N_acqs}. Retrying...')
    if twoSensors:
        temperature2[i] = tmp[1]
        temperature1[i] = tmp[0]
    else:
        temperature1[i] = tmp
    Cw = US.speedofsound_in_water(temperature1[i], method='Abdessamad', method_param=148)
    Cw_vector[i] = Cw
    
    if i==0:
        TT0 = TT
        Cw0 = Cw
        TT0_argmax = US.CosineInterpMax(TT0, xcor=False) + Smin1
    
    
    # -----------------------------
    # Save temperature and acq data
    # -----------------------------
    with open(Temperature_path, 'a') as f:
        row = f'{temperature1[i]},{temperature2[i]},{Cw}' if twoSensors else f'{temperature1[i]},{Cw}'
        f.write(row+'\n')
    
    if Save_acq_data:
        _mode = 'wb' if i==0 else 'ab' # clear data from previous experiment before writing
        with open(Acqdata_path, _mode) as f:
            TT.tofile(f)
    
    
    # -------------
    # Control plots
    # -------------
    # Plot every acquisition if Plot_all_acq==True
    # Plot one acquisition in the middle of experiment to see how things are going
    if Plot_all_acq or i==N_acqs//2:
        US.multiplot_tf(np.column_stack((TT, TT0)).T, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
                    t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
                    f_xlims=([0, 20]), f_ylims=None, f_units='MHz', f_Norm=False,
                    PSD=False, dB=False, label=['TT', 'TT0'], Independent=True, FigNum='Signals', FgSize=(6.4,4.8))
        US.movefig(location='southeast')
        plt.pause(_plt_pause_time)


    # ---------------
    # TOF computation
    # ---------------
    ToF[i] = US.CalcToFAscanCosine_XCRFFT(TT, TT0)[0]
    TT_argmax = US.CosineInterpMax(TT, xcor=False) + Smin1

    
    # --------------------
    # Distance computation
    # --------------------
    d[i] = ToF[i] * Cw0 / Fs
    corrected_d[i] = ToF[i] * Cw / Fs
    good_d[i] = (Cw*TT_argmax - Cw0*TT0_argmax) / Fs
    
    
    # ----------------------------------
    # Save results to text file as we go 
    # ----------------------------------
    with open(Results_path, 'a') as f:
        row = f'{Time_axis[i]},{ToF[i]},{d[i]},{corrected_d[i]},{good_d[i]}'
        f.write(row+'\n')


    # -----
    # Plots
    # -----
    _xdata = Time_axis[:i]/_factor

    # Plot temperature
    if Plot_temperature:
        # Plot Cw    
        fig, ax = plt.subplots(1, num='Cw', clear=True)
        US.movefig(location='north')
        ax.set_ylabel('Cw (m/s)')
        ax.set_xlabel(_xlabel)
        ax.scatter(_xdata, Cw_vector[:i], color='white', marker='o', edgecolors='black')
        plt.tight_layout()
        plt.pause(_plt_pause_time)
        
        if twoSensors:
            fig, axs = plt.subplots(2, num='Temperature', clear=True)
            US.movefig(location='south')
            axs[0].set_ylabel('Temperature 1 (\u2103)')
            axs[1].set_ylabel('Temperature 2 (\u2103)')
            axs[1].set_xlabel(_xlabel)
            axs[0].scatter(_xdata, temperature1[:i], color='white', marker='o', edgecolors='black')
            axs[1].scatter(_xdata, temperature2[:i], color='white', marker='o', edgecolors='black')
        else:
            fig, ax = plt.subplots(1, num='Temperature', clear=True)
            US.movefig(location='south')
            ax.set_ylabel('Temperature (\u2103)')
            ax.set_xlabel(_xlabel)
            ax.scatter(_xdata, temperature1[:i], color='white', marker='o', edgecolors='black')
        plt.tight_layout()
        plt.pause(_plt_pause_time)
    
    # Plot results
    fig, axs = plt.subplots(2,2, num='Results', clear=True, figsize=(10,8))
    US.movefig('northeast')
    axs[0,0].set_ylabel('ToF (samples)')
    axs[0,1].set_ylabel('d (mm)')
    axs[1,0].set_ylabel('corrected d (mm)')
    axs[1,1].set_ylabel('good d (mm)')

    axs[0,0].set_xlabel(_xlabel)
    axs[0,1].set_xlabel(_xlabel)
    axs[1,0].set_xlabel(_xlabel)
    axs[1,1].set_xlabel(_xlabel)
    
    axs[0,0].scatter(_xdata, ToF[:i], color='k', marker='.', edgecolors='k')
    axs[0,1].scatter(_xdata, d[:i]*1e3, color='k', marker='.', edgecolors='k')
    axs[1,0].scatter(_xdata, corrected_d[:i]*1e3, color='k', marker='.', edgecolors='k')
    axs[1,1].scatter(_xdata, good_d[:i]*1e3, color='k', marker='.', edgecolors='k')
    
    plt.tight_layout()
    plt.pause(_plt_pause_time)
    
    
    # ---------
    # End timer
    # ---------
    elapsed_time = time.time() - start_time
    time_to_wait = Ts_acq - elapsed_time # time until next acquisition
    print(f'Acquisition #{i+1}/{N_acqs} done.')
    if time_to_wait < 0:
        print(f'Code is slower than Ts_acq = {Ts_acq} s at Acq #{i+1}. Elapsed time is {elapsed_time} s.')
        time_to_wait = 0
    time.sleep(time_to_wait)
    

# -----------
# End of loop
# -----------
plt.tight_layout()
arduino.close()


# -----------------------------
# Write end time to config file
# -----------------------------
_end_time = time.strftime("%Y/%m/%d - %H:%M:%S")
config_dict['End_date'] = _end_time
US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Experiment ended at {_end_time}.')
print("===================================================\n")
