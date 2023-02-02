# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:24:00 2023
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
from src.devices import Scanner


#%%
########################################################
# Paths and file names to use
########################################################
Path = r'D:\Data\pruebas_acq'
Experiment_folder_name = 'test' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_PEref_file_name = 'PEref.bin'
Experiment_WP_file_name = 'WP.bin'
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_scanpath_file_name = 'scanpath.txt'
Experiment_description_file_name = 'Experiment_description.txt'
Experiment_PEforCW_file_name = 'PEat0_PEat10mm.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
PEref_path = os.path.join(MyDir, Experiment_PEref_file_name)
WP_path = os.path.join(MyDir, Experiment_WP_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Experiment_description_path = os.path.join(MyDir, 'Experiment_description.txt')
Scanpath_path = os.path.join(MyDir, Experiment_scanpath_file_name)
PEforCW_path = os.path.join(MyDir, Experiment_PEforCW_file_name)
if not os.path.exists(MyDir):
    os.makedirs(MyDir)
    print(f'Created new path: {MyDir}')
print(f'Experiment path set to {MyDir}')


#%% 
########################################################
# Parameters and constants
########################################################
# For Experiment_description do NOT use '\n'.
# Suggestion: write material brand, model, dopong, etc. in Experiment_description
Experiment_description = """Scanner test.
Epoxy resin dog-bone.
Focused tx.
Excitation_params: Pulse frequency (Hz).
"""

Fs = 100.0e6                    # Sampling frequency - Hz
Fs_Gencode_Generator = 200.0e6  # Sampling frequency for the gencodes generator - Hz
RecLen = 32*1024                # Maximum range of ACQ - samples (max=32*1024)
Gain_Ch1 = 70                   # Gain of channel 1 - dB
Gain_Ch2 = 25                   # Gain of channel 2 - dB
Attenuation_Ch1 = 0             # Attenuation of channel 1 - dB
Attenuation_Ch2 = 10            # Attenuation of channel 2 - dB
Excitation_voltage = 60         # Excitation voltage (min=20V) - V -- DOESN'T WORK
Fc = 5e6                        # Pulse frequency - Hz
Excitation = 'Pulse'            # Excitation to use ('Pulse, 'Chirp', 'Burst') - string
Excitation_params = Fc          # All excitation params - list or float
Smin1, Smin2 = 3900, 3900       # starting point of the scan of each channel - samples
Smax1, Smax2 = 7500, 7500       # last point of the scan of each channel - samples
AvgSamplesNumber = 25           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels
Reset_Relay = False             # Reset delay: ON>OFF>ON - bool
Save_acq_data = True            # If True, save all acq. data to {Acqdata_path} - bool
Temperature = True              # If True, take temperature measurements at each acq. (temperature data is always saved to file) and plot Cw - bool
twoSensors = False              # If True, assume we have two temperature sensors, one inside and one outside - bool
PE_as_ref = True                # If True, both a WP and a PE traces are acquired. The resulting ref. signal has the PE pulse aligned at WP - str
align_PEref = True              # If True, align PEref to zero - bool


# -------
# Arduino
# -------
board = 'Arduino UNO'           # Board type - str
baudrate = 9600                 # Baudrate (symbols/s) - int
port = 'COM3'                   # Port to connect to - str
N_avg = 1                       # Number of temperature measurements to be averaged - int


# -------
# Scanner
# -------
MeasureCW = True                # If True, measure de speed of sound in water by taking PE at 0 and PE at 1cm - bool
WP_axis = 'X'                   # Axis of the water path - str
baudrate_scanner = 19200        # Baudrate (symbols/s) - int
port_scanner = 'COM4'           # Port to connect to - str
timeout_scanner = 0.1           # Serial comm timeout (seconds) - float
pattern = 'line+turn on Z'      # Available patterns:  - str
                                # 'line on X', 'line on Y', 'line on Z',
                                # 'line+turn on X', 'line+turn on Y', 'line+turn on Z',
                                # 'zigzag XY', 'zigzag XZ',
                                # 'zigzag YX', 'zigzag YZ',
                                # 'zigzag ZX', 'zigzag ZY'
                                # 'zigzag+turn XY', 'zigzag+turn XZ',
                                # 'zigzag+turn YX', 'zigzag+turn YZ',
                                # 'zigzag+turn ZX', 'zigzag+turn ZY'
                                # The first axis is the first to move (the long one)
X_step = 0                      # smallest step to move in the X axis (mm), min is 0.01 - float
Y_step = 0                      # smallest step to move in the Y axis (mm), min is 0.01 - float
Z_step = 1                      # smallest step to move in the Z axis (mm), min is 0.005 - float
R_step = 0                      # smallest step to move in the R axis (deg), min is 1.8  - float
# If the step is zero, do not move on that axis

X_end = 0                       # last X coordinate of the experiment (mm) - float
Y_end = 0                       # last Y coordinate of the experiment (mm) - float
Z_end = 80                      # last Z coordinate of the experiment (mm) - float
R_end = 0                       # last R coordinate of the experiment (deg) - float

scanpatter = Scanner.makeScanPattern(pattern, [X_step, Y_step, Z_step, R_step], [X_end, Y_end, Z_end, R_end])
N_acqs = len(scanpatter)
print(f'The experiment will take {US.time2str(N_acqs*0.1)} at best.')


#%% Start serial communication
if Temperature:
    arduino = Arduino(board, baudrate, port, twoSensors, N_avg)  # open comms


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
if Reset_Relay:
    print('Resetting relay...')
    SeDaq.SetRelay(1)
    time.sleep(1) # wait to be sure
    SeDaq.SetRelay(0)
    time.sleep(1) # wait to be sure
    SeDaq.SetRelay(1)
    time.sleep(1) # wait to be sure
    print("---------------------------------------------------")
SeDaq.SetRecLen(RecLen) # initialize record length
# SeDaq.SetExtVoltage(Excitation_voltage) - DOESN'T WORK
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
# Initialize Scanner
########################################################################
scanner = Scanner.Scanner(port=port_scanner, baudrate=baudrate_scanner, timeout=timeout_scanner)
scanner.XLimit = X_end + scanner.uStepX
scanner.YLimit = Y_end + scanner.uStepY
scanner.ZLimit = Z_end + scanner.uStepZ
scanner.RLimit = R_end + scanner.uStepR


#%% 
########################################################################
# Save config, experiment description and scanpath
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
               'N_acqs': N_acqs,
               'WP_temperature' : None,
               'Outside_temperature': None,
               'N_avg' : N_avg,
               'Start_date': '',
               'End_date': ''}

US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Configuration parameters saved to {Config_path}.')
print("===================================================\n")

with open(Experiment_description_path, 'w') as f:
    f.write(Experiment_description)
print(f'Experiment description saved to {Experiment_description_path}.')
print("===================================================\n")

np.savetxt(Scanpath_path, np.c_[scanpatter], fmt='%s')
print(f'Scanpatter saved to {Scanpath_path}.')
print("===================================================\n")


#%% 
########################################################################
# Water Path acquisition
########################################################################
WP_Ascan = SeDaq.GetAscan_Ch1(Smin1, Smax1)
print('Water path acquired.')
if PE_as_ref:
    input("Press any key to acquire the pulse echo.")
    scanner.goHome()
    PEref_Ascan = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    MyWin_PEref = US.SliderWindow(PEref_Ascan, SortofWin='tukey', param1=0.25, param2=1)
    PEref_Ascan = PEref_Ascan * MyWin_PEref
    
    if align_PEref:
        US.align2zero(PEref_Ascan, UseCentroid=False, UseHilbEnv=False)
    print('Pulse echo as reference acquired.')
else:
    PEref_Ascan = WP_Ascan
    print('PE_as_ref is False. Setting PEref_Ascan = WP_Ascan.')
print("===================================================\n")

if Temperature:
    arduino.open()
    
    tmp = arduino.getTemperature(error_msg='Warning: wrong temperature data for Water Path.', 
                                 exception_msg='Warning: could not parse temperature data to float for Water Path.')
    
    if twoSensors:
        mean2 = tmp[1]
        mean1 = tmp[0]
    else:
        mean2 = None
        mean1 = tmp
    
    config_dict['WP_temperature'] = mean1
    config_dict['Outside_temperature'] = mean2
    US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
    print(f'Reference signal temperature is {mean1} \u00b0C.')
    print(f'Outside temperature is {mean2} \u00b0C.')
    print("===================================================\n")
    
plt.figure()
plt.plot(WP_Ascan)
plt.title('WP')
US.movefig(location='n')
plt.pause(0.05)

ScanLen1 = Smax1 - Smin1                # Total scan length for channel 1 - samples
ScanLen2 = Smax2 - Smin2                # Total scan length for channel 2 - samples
ScanLen = np.max([ScanLen1, ScanLen2])  # Total scan length for computations (zero padding is used) - samples

# Zero pad WP in case each channel has different scan length
if ScanLen1 < ScanLen:
    WP_Ascan = US.zeroPadding(WP_Ascan, ScanLen)
elif ScanLen2 < ScanLen:
    PEref_Ascan = US.zeroPadding(PEref_Ascan, ScanLen)

if Save_acq_data:
    with open(WP_path, 'wb') as f:
        WP_Ascan.tofile(f)
    with open(PEref_path, 'wb') as f:
        PEref_Ascan.tofile(f)

input("Press any key to start the experiment.")
print("===================================================\n")


#%% 
########################################################################
# Measurement of speed of sound in water
########################################################################
if MeasureCW:
    x = 10 # mm
    scanner.goHome()
    PE0 = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    
    if WP_axis=='X':
        scanner.diffMoveX(x)
    elif WP_axis=='Y':
        scanner.diffMoveY(x)
    elif WP_axis=='Z':
        scanner.diffMoveZ(x)
    
    PE10 = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    scanner.goHome()
    
    # Find ToF
    ToF = US.CalcToFAscanCosine_XCRFFT(PE0, PE10, UseCentroid=False, UseHilbEnv=False, Extend=True, Same=False)[0]
    Cw = 2 * Fs * x*1e-3 / ToF
    print(f'The speed of sound in the water is {Cw} m/s.')
    
    config_dict['Cw'] = Cw
    US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
    print("===================================================\n")
    
    np.savetxt(PEforCW_path, np.c_[PE0,PE10], header='PE0,PE10', comments='', delimiter=',')
    print(f'PEs for Cw saved to {PEforCW_path}.')
    print("===================================================\n")


#%% 
########################################################################
# Run Acquisition and Computations
########################################################################
# --------------------
# Initialize variables
# --------------------
Smin = (Smin1, Smin2)                   # starting points - samples
Smax = (Smax1, Smax2)                   # last points - samples
ScanLen1 = Smax1 - Smin1                # Total scan length for channel 1 - samples
ScanLen2 = Smax2 - Smin2                # Total scan length for channel 2 - samples
ScanLen = np.max([ScanLen1, ScanLen2])  # Total scan length for computations (zero padding is used) - samples


# -------------------------------------
# Write temperature header to text file
# -------------------------------------
if Temperature:
    if twoSensors:
        header = 'Inside,Outside,Cw'
        means2 = np.zeros(N_acqs)
    else:
        header = 'Inside,Cw'
    means1 = np.zeros(N_acqs)
    
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


scanner.goHome()
# ---------
# Sart loop
# ---------
try:
    for i, sp in enumerate(scanpatter):    
        # -------------------------------------------
        # Acquire signal, temperature and start timer
        # -------------------------------------------
        start_time = time.time() # start timer
        TT_Ascan, PE_Ascan = SeDaq.GetAscan_Ch1_Ch2(Smin, Smax) #acq Ascan
        if Temperature:
            tmp = arduino.getTemperature(error_msg=f'Warning: wrong temperature data at Acq. #{i+1}/{N_acqs}. Retrying...', 
                                         exception_msg=f'Warning: could not parse temperature data to float at Acq. #{i+1}/{N_acqs}. Retrying...')
            
            if twoSensors:
                means2[i] = tmp[1]
                means1[i] = tmp[0]
            else:
                means1[i] = tmp
        
        # -----------------------------------------------------------
        # Zero padding in case each channel has different scan length
        # -----------------------------------------------------------
        if ScanLen1 < ScanLen:
            TT_Ascan = US.zeroPadding(TT_Ascan, ScanLen)
        elif ScanLen2 < ScanLen:
            PE_Ascan = US.zeroPadding(PE_Ascan, ScanLen)
        
        
        # -----------------------------
        # Save temperature and acq data
        # -----------------------------  
        with open(Temperature_path, 'a') as f:
            if twoSensors:
                row = f'{means1[i]},{means2[i]},{Cw}'
            else:
                row = f'{means1[i]},{Cw}'
            f.write(row+'\n')
        
        _mode = 'wb' if i==0 else 'ab' # clear data from previous experiment before writing
        if Save_acq_data:
            with open(Acqdata_path, _mode) as f:
                TT_Ascan.tofile(f)
                PE_Ascan.tofile(f)


        # ------------
        # Move scanner
        # ------------
        ax = sp[0]
        val = sp[1:]
        if ax=='X':
            scanner.diffMoveX(val)
        elif ax=='Y':
            scanner.diffMoveY(val)
        elif ax=='Z':
            scanner.diffMoveZ(val)
        elif ax=='R':
            scanner.diffMoveR(val)
        
        
        # ---------
        # End timer
        # ---------
        elapsed_time = time.time() - start_time
        print(f'Acquisition #{i+1}/{N_acqs} done in {elapsed_time} s.')
        
except KeyboardInterrupt:
    scanner.stop()
    print('Scanner stopped succesfully.')
# -----------
# End of loop
# -----------
if Temperature:
    arduino.close()


# -----------------------------
# Write end time to config file
# -----------------------------
_end_time = time.strftime("%Y/%m/%d - %H:%M:%S")
config_dict['End_date'] = _end_time
US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Experiment ended at {_end_time}.')
print("===================================================\n")
