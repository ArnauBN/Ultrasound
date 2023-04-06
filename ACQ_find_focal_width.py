# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:36:37 2023
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
Experiment_folder_name = 'focal_width' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_PEref_file_name = 'PEref.bin'
Experiment_WP_file_name = 'WP.bin'
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_scanpath_file_name = 'scanpath.txt'
Experiment_description_file_name = 'Experiment_description.txt'
Experiment_PEforCW_file_name = 'PEat0_PEat10mm.txt'
Experiment_PEforCW2_file_name = 'PEat0_PEat10mm2.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
PEref_path = os.path.join(MyDir, Experiment_PEref_file_name)
WP_path = os.path.join(MyDir, Experiment_WP_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Experiment_description_path = os.path.join(MyDir, Experiment_description_file_name)
Scanpath_path = os.path.join(MyDir, Experiment_scanpath_file_name)
PEforCW_path = os.path.join(MyDir, Experiment_PEforCW_file_name)
PEforCW2_path = os.path.join(MyDir, Experiment_PEforCW2_file_name)
if not os.path.exists(MyDir):
    os.makedirs(MyDir)
    print(f'Created new path: {MyDir}')
print(f'Experiment path set to {MyDir}')

#%% 
########################################################
# Parameters and constants
########################################################
# Suggestion: write material brand, model, dopong, etc. in Experiment_description
Experiment_description = """Scanner test.
Methacrylate dog-bone.
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
Smin1, Smin2 = 3400, 3400       # starting point of the scan of each channel - samples
Smax1, Smax2 = 8300, 8300       # last point of the scan of each channel - samples
AvgSamplesNumber = 25           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels
Reset_Relay = False             # Reset delay: ON>OFF>ON - bool
Save_acq_data = True            # If True, save all acq. data to {Acqdata_path} - bool
Temperature = True              # If True, take temperature measurements at each acq. (temperature data is always saved to file) and plot Cw - bool
twoSensors = False              # If True, assume we have two temperature sensors, one inside and one outside - bool
PE_as_ref = True                # If True, both a WP and a PE traces are acquired. The resulting ref. signal has the PE pulse aligned at WP - str
align_PEref = False             # If True, align PEref to zero - bool


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
MeasureCW = True                # If True, measure de speed of sound in water by taking PE at 0 and PE at {MeasureCW_dist} mm - bool
MeasureCW_dist = 10             # Distance to move to measure Cw (mm) - float
sidetilt_dist = 10              # Distance to move to determine the side tilt (mm).
                                # Checks the width of the DUT at {-sidetilt_dist}, 0 and {sidetilt_dist} - float
tilt_step = 10                  # Step used to check the tilt (mm) - float
WP_axis = 'Y'                   # Axis of the water path - str
scan_axis = 'X'                 # Axis of the scan patter. Can be an axis (e.g. line) or a plane (e.g. zigzag). The first axis is the first to move (the long one) - str
water_plane = 'XY'              # Plane of the water (the DUT is assumed to be perpendicular to this plane) - str
baudrate_scanner = 19200        # Baudrate (symbols/s) - int
port_scanner = 'COM4'           # Port to connect to - str
timeout_scanner = 0.1           # Serial comm timeout (seconds) - float
pattern = 'line'                # Available patterns: 'line', 'line+turn', 'zigzag', 'zigzag+turn' - str
X_step = 0.01                   # smallest step to move in the X axis (mm), min is 0.01 - float
Y_step = 0                      # smallest step to move in the Y axis (mm), min is 0.01 - float
Z_step = 0                      # smallest step to move in the Z axis (mm), min is 0.005 - float
R_step = 0                      # smallest step to move in the R axis (deg), min is 1.8  - float
# If the step is zero, do not move on that axis

X_end = 90                     # last X coordinate of the experiment (mm) - float
Y_end = 20                      # last Y coordinate of the experiment (mm) - float
Z_end = 120                    # last Z coordinate of the experiment (mm) - float
R_end = 0                      # last R coordinate of the experiment (deg) - float

init_pos = 10
center_pos = 50
Z_center_pos = 60

# ------------------------------------
# Initialize variables - Do not modify
# ------------------------------------
if WP_axis not in water_plane or len(water_plane) != 2 or len(WP_axis) != 1:
    print('The WP axis must be contained in the water plane.')
    time.sleep(3)
    exit()

scanpatter = Scanner.makeScanPattern(pattern, scan_axis, [X_step, Y_step, Z_step, R_step], [X_end, Y_end, Z_end, R_end])
N_acqs = len(scanpatter) + 1
print(f'The experiment will take around {US.time2str(N_acqs*0.5)} plus the calibration.')


if scan_axis == 'X':
    end = X_end
    home = [init_pos, 0, Z_center_pos, 0]
    center = [center_pos, 0, Z_center_pos, 0]
elif scan_axis == 'Y':
    end = Y_end
    home = [0, init_pos, Z_center_pos, 0]
    center = [0, center_pos, Z_center_pos, 0]
elif scan_axis == 'Z':
    end = Z_end
    home = [0, 0, init_pos, 0]
    center = [0, 0, center_pos, 0]


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
scanner.setLimits(X_end + 1, Y_end + 1, Z_end + 1, R_end + 1)
scanner.Rspeedtype = 'gaussian'
scanner.home = home


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

with open(Experiment_description_path, 'w') as f:
    f.write(Experiment_description)
print(f'Experiment description saved to {Experiment_description_path}.')

np.savetxt(Scanpath_path, np.c_[scanpatter], fmt='%s')
print(f'Scanpatter saved to {Scanpath_path}.')
print("===================================================\n")


#%% 
########################################################################
# Measurement of speed of sound in water
########################################################################
if MeasureCW:
    scanner.move(*center)
    PE0 = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    
    scanner.diffMoveAxis(WP_axis, MeasureCW_dist)
    
    PE10 = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    scanner.move(*center)
    
    # Find ToF
    ToF = US.CalcToFAscanCosine_XCRFFT(PE0, PE10, UseCentroid=False, UseHilbEnv=False, Extend=True, Same=False)[0]
    Cw = 2 * Fs * MeasureCW_dist*1e-3 / abs(ToF)
    print(f'The speed of sound in the water is {Cw} m/s.')
    
    config_dict['Cw'] = Cw
    US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
    print("===================================================\n")
    
    np.savetxt(PEforCW_path, np.c_[PE0,PE10], header='PE0,PE10', comments='', delimiter=',')
    print(f'PEs for Cw saved to {PEforCW_path}.')
    print("===================================================\n")


#%% 
########################################################################
# Water Path acquisition
########################################################################
scanner.moveAxis(scan_axis, end-1)
WP_Ascan = SeDaq.GetAscan_Ch1(Smin1, Smax1)
print('Water path acquired.')
scanner.move(*center)

if PE_as_ref:
    input("Press any key to acquire the pulse echo.")
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
    header = 'Inside,Outside,Cw' if twoSensors else 'Inside,Cw'
    with open(Temperature_path, 'w') as f:
        f.write(header+'\n')

    arduino.open()

scanner.goHome()

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
                temperature2 = tmp[1]
                temperature1 = tmp[0]
            else:
                temperature1 = tmp
        
            Cwt = US.speedofsound_in_water(temperature1, method='Abdessamad', method_param=148)
        
        
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
        if Temperature:
            with open(Temperature_path, 'a') as f:
                row = f'{temperature1},{temperature2},{Cwt}' if twoSensors else f'{temperature1},{Cwt}'
                f.write(row+'\n')
        
        if Save_acq_data:
            _mode = 'wb' if i==0 else 'ab' # clear data from previous experiment before writing
            with open(Acqdata_path, _mode) as f:
                TT_Ascan.tofile(f)
                PE_Ascan.tofile(f)
    
    
        # ------------
        # Move scanner
        # ------------
        ax = sp[0]
        val = float(sp[1:])
        scanner.diffMoveAxis(ax, val)
        
        
        # ---------
        # End timer
        # ---------
        elapsed_time = time.time() - start_time
        print(f'Acquisition #{i+1}/{N_acqs} done in {elapsed_time} s.')
    
    
    
    # =========================
    # TAKE ONE LAST MEASUREMENT
    # =========================
    start_time = time.time() # start timer
    TT_Ascan, PE_Ascan = SeDaq.GetAscan_Ch1_Ch2(Smin, Smax) #acq Ascan
    if Temperature:
        tmp = arduino.getTemperature(error_msg=f'Warning: wrong temperature data at Acq. #{N_acqs}/{N_acqs}. Retrying...', 
                                     exception_msg=f'Warning: could not parse temperature data to float at Acq. #{N_acqs}/{N_acqs}. Retrying...')
        
        if twoSensors:
            temperature2 = tmp[1]
            temperature1 = tmp[0]
        else:
            temperature1 = tmp
    
        Cwt = US.speedofsound_in_water(temperature1, method='Abdessamad', method_param=148)

    if ScanLen1 < ScanLen:
        TT_Ascan = US.zeroPadding(TT_Ascan, ScanLen)
    elif ScanLen2 < ScanLen:
        PE_Ascan = US.zeroPadding(PE_Ascan, ScanLen)

    if Temperature:
        with open(Temperature_path, 'a') as f:
            row = f'{temperature1},{temperature2},{Cwt}' if twoSensors else f'{temperature1},{Cwt}'
            f.write(row+'\n')
    
    if Save_acq_data:
        with open(Acqdata_path, 'ab') as f:
            TT_Ascan.tofile(f)
            PE_Ascan.tofile(f)
    elapsed_time = time.time() - start_time
    print(f'Acquisition #{N_acqs}/{N_acqs} done in {elapsed_time} s.')
    
except KeyboardInterrupt:
    scanner.stop()
    print('Scanner successfully stopped.')
    raise
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


########################################################################
# Measurement of speed of sound in water
########################################################################
if MeasureCW:
    scanner.move(*center)
    PE0 = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    
    scanner.diffMoveAxis(WP_axis, MeasureCW_dist)
    
    PE10 = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    scanner.goHome()
    
    # Find ToF
    ToF = US.CalcToFAscanCosine_XCRFFT(PE0, PE10, UseCentroid=False, UseHilbEnv=False, Extend=True, Same=False)[0]
    Cw2 = 2 * Fs * MeasureCW_dist*1e-3 / abs(ToF)
    print(f'The speed of sound in the water is {Cw2} m/s.')
    
    config_dict['Cw'] = (Cw + Cw2) / 2
    US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
    print("===================================================\n")
    
    np.savetxt(PEforCW2_path, np.c_[PE0,PE10], header='PE0,PE10', comments='', delimiter=',')
    print(f'PEs for Cw2 saved to {PEforCW2_path}.')
    print("===================================================\n")