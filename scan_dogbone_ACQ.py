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
MeasureCW = True                # If True, measure de speed of sound in water by taking PE at 0 and PE at {MeasureCW_dist} mm - bool
MeasureCW_dist = 10             # Distance to move to measure Cw (mm) - float
sidetilt_dist = 10              # Distance to move to determine the side tilt (mm).
                                # Checks the width of the DUT at {-sidetilt_dist}, 0 and {sidetilt_dist} - float
tilt_step = 5                   # Step used to check the tilt (mm) - float
WP_axis = 'Y'                   # Axis of the water path - str
scan_axis = 'Z'                 # Axis of the scan patter. Can be an axis (e.g. line) or a plane (e.g. zigzag). The first axis is the first to move (the long one) - str
water_plane = 'XY'              # Plane of the water (the DUT is assumed to be perpendicular to this plane) - str
baudrate_scanner = 19200        # Baudrate (symbols/s) - int
port_scanner = 'COM4'           # Port to connect to - str
timeout_scanner = 0.1           # Serial comm timeout (seconds) - float
pattern = 'line+turn'           # Available patterns: 'line', 'line+turn', 'zigzag', 'zigzag+turn' - str
X_step = 0                      # smallest step to move in the X axis (mm), min is 0.01 - float
Y_step = 0                      # smallest step to move in the Y axis (mm), min is 0.01 - float
Z_step = 0.2                    # smallest step to move in the Z axis (mm), min is 0.005 - float
R_step = 30                     # smallest step to move in the R axis (deg), min is 1.8  - float
# If the step is zero, do not move on that axis

X_end = 30                      # last X coordinate of the experiment (mm) - float
Y_end = 0                       # last Y coordinate of the experiment (mm) - float
Z_end = 120                     # last Z coordinate of the experiment (mm) - float
R_end = 30                      # last R coordinate of the experiment (deg) - float




# ------------------------------------
# Initialize variables - Do not modify
# ------------------------------------
if WP_axis not in water_plane or len(water_plane) != 2 or len(WP_axis) != 1:
    print('The WP axis must be contained in the water plane.')
    time.sleep(3)
    exit()

scanpatter = Scanner.makeScanPattern(pattern, scan_axis, [X_step, Y_step, Z_step, R_step], [X_end, Y_end, Z_end, R_end])
N_acqs = len(scanpatter)
print(f'The experiment will take {US.time2str(N_acqs*0.1)} at best.')


# Axis specification
if len(scan_axis) != 1:
    longaxis = scan_axis[0]
    shortaxis = scan_axis[1]
else:
    longaxis = scan_axis
    
    _temp = ['X', 'Y', 'Z']
    _temp.remove(longaxis)
    _temp.remove(WP_axis)
    shortaxis = _temp[0]
    del _temp

if longaxis == 'X':
    longend = X_end
    longstep = X_step
elif longaxis == 'Y':
    longend = Y_end
    longstep = Y_step
elif longaxis == 'Z':
    longend = Z_end
    longstep = Z_step




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
scanner.setLimits(X_end + 1, Y_end + 1, Z_end + 1, R_end + 1)


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
# Determination of Center axis and Sidetilt
########################################################################
step = 0.01
step1 = 0.01 # mm - MUST be positive
step2 = -0.01 # mm - MUST be negative
Maxtol = 0.1
init_step = 5 # mm

try:
    # ------------
    # Check center
    # ------------
    scanner.moveAxis(longaxis, int(longend//2))
    PEc = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    Maxc = np.max(np.abs(PEc))
    x1_0 = scanner.findEdge(SeDaq, Smin2, Smax2, shortaxis, init_step, Maxtol)
    x2_0 = scanner.findEdge(SeDaq, Smin2, Smax2, shortaxis, -init_step, Maxtol)
    # x1_0 = scanner.checkside(SeDaq, Smin2, Smax2, shortaxis, step1, Maxc, Maxtol)
    # x2_0 = scanner.checkside(SeDaq, Smin2, Smax2, shortaxis, step2, Maxc, Maxtol)
    width0 = x1_0 - x2_0
    dif0 = x1_0 + x2_0
    scanner.diffMoveAxis(shortaxis, dif0)
    scanner.setAxis(shortaxis, 0)
    
    
    # ----------------------------
    # Check center + sidetilt_dist
    # ----------------------------
    scanner.diffMoveAxis(shortaxis, sidetilt_dist)
    PEc = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    Maxc = np.max(np.abs(PEc))
    x1_1 = scanner.findEdge(SeDaq, Smin2, Smax2, shortaxis, init_step, Maxtol)
    x2_1 = scanner.findEdge(SeDaq, Smin2, Smax2, shortaxis, -init_step, Maxtol)
    # x1_1 = scanner.checkside(SeDaq, Smin2, Smax2, shortaxis, step1, Maxc, Maxtol)
    # x2_1 = scanner.checkside(SeDaq, Smin2, Smax2, shortaxis, step2, Maxc, Maxtol)
    width1 = x1_1 - x2_1
    dif1 = x1_1 + x2_1
    scanner.diffMoveAxis(shortaxis, dif1)
    
    
    # ----------------------------
    # Check center - sidetilt_dist
    # ----------------------------
    scanner.diffMoveAxis(shortaxis, -2*sidetilt_dist)
    PEc = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    Maxc = np.max(np.abs(PEc))
    x1_2 = scanner.findEdge(SeDaq, Smin2, Smax2, shortaxis, init_step, Maxtol)
    x2_2 = scanner.findEdge(SeDaq, Smin2, Smax2, shortaxis, -init_step, Maxtol)
    # x1_2 = scanner.checkside(SeDaq, Smin2, Smax2, shortaxis, step1, Maxc, Maxtol)
    # x2_2 = scanner.checkside(SeDaq, Smin2, Smax2, shortaxis, step2, Maxc, Maxtol)
    width2 = x1_2 - x2_2
    dif2 = x1_2 + x2_2
    scanner.diffMoveAxis(shortaxis, dif2)
except KeyboardInterrupt:
    scanner.stop()
    print('Scanner successfully stopped.')


# ----------------
# Compute sidetilt
# ----------------
tg_theta = abs(x1_1 - x1_2) / (2*sidetilt_dist)
theta = np.arctan(tg_theta) * 180 / np.pi
print(f'Sidetilt is {theta = } deg')

scanner.goHome()


#%%
########################################################################
# Determination of Tilt
########################################################################
scanpatter_tilt_test = Scanner.makeScanPattern('line', shortaxis, [tilt_step, tilt_step, tilt_step, tilt_step], [X_end, Y_end, Z_end, R_end])
ToFs = np.zeros(len(scanpatter_tilt_test))
dists = np.arange(len(scanpatter_tilt_test))*tilt_step

scanner.goHome()
try:
    for i, sp in enumerate(scanpatter_tilt_test):
        ToFs[i] = US.CosineInterpMax(SeDaq.GetAscan_Ch2(Smin2, Smax2), xcor=False)
        ax = sp[0]
        val = float(sp[1:])
        scanner.diffMoveAxis(ax, val)
except KeyboardInterrupt:
    scanner.stop()
    print('Scanner successfully stopped.')


# ------------
# Compute tilt
# ------------
A = np.vstack([dists, np.ones(len(dists))]).T
tg_phi, c = np.linalg.lstsq(A, ToFs, rcond=None)[0]
phi = np.arctan(tg_phi) * 180 / np.pi
print(f'Tilt is {phi = } deg')


#%%
########################################################################
# Determination of Rotation
########################################################################
ToFs = []
dists = []

scanner.moveAxis(longaxis, int(longend//2))
scanner.moveAxis(shortaxis, x1_0)
try:
    i = 0
    while scanner.getAxis(shortaxis) <= x2_0:
        ToFs.append(US.CosineInterpMax(SeDaq.GetAscan_Ch2(Smin2, Smax2), xcor=False))
        scanner.diffMoveAxis(shortaxis, step)
        dists.append(step*i)
        i += 1
except KeyboardInterrupt:
    scanner.stop()
    print('Scanner successfully stopped.')


# ----------------
# Compute rotation
# ----------------
A = np.vstack([dists, np.ones(len(dists))]).T
tg_rho, c = np.linalg.lstsq(A, ToFs, rcond=None)[0]
rho = np.arctan(tg_rho) * 180 / np.pi
print(f'Rotation is {rho = } deg')


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
    scanner.goHome()
    PE0 = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    
    scanner.diffMoveAxis(WP_axis, MeasureCW_dist)
    
    PE10 = SeDaq.GetAscan_Ch2(Smin2, Smax2)
    scanner.goHome()
    
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
        val = float(sp[1:])
        scanner.diffMoveAxis(ax, val)
        
        
        # ---------
        # End timer
        # ---------
        elapsed_time = time.time() - start_time
        print(f'Acquisition #{i+1}/{N_acqs} done in {elapsed_time} s.')
except KeyboardInterrupt:
    scanner.stop()
    print('Scanner successfully stopped.')
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
