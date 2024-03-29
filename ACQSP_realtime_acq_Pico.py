# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:47:13 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

This script uses the PicoScope to both generate and acquire the signals.
Last updated: 31/01/2023.

"""

import numpy as np
import matplotlib.pylab as plt
import os
import time

import src.ultrasound as US
from src.devices import pico5000a
from src.devices import Arduino

#%% Check Pico drivers
pico5000a.check_drivers()

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
Experiment_waveform_file_name = 'waveform.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
PEref_path = os.path.join(MyDir, Experiment_PEref_file_name)
WP_path = os.path.join(MyDir, Experiment_WP_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Waveform_path = os.path.join(MyDir, Experiment_waveform_file_name)
if not os.path.exists(MyDir):
    os.makedirs(MyDir)
    print(f'Created new path: {MyDir}')
print(f'Experiment path set to {MyDir}')


#%% 
########################################################
# Parameters and constants
########################################################

# ----------
# Experiment
# ----------
# For Experiment_description do NOT use '\n'.
# Suggestion: write material brand, model, dopong, etc. in Experiment_description
Experiment_description = "First test with Pico." \
                        " Test no container." \
                        " Solid metacrlylate." \
                        " Focused tx." \
                        " Excitation_params: Pulse frequency (Hz)."
Ts_acq = 4                  # Time between acquisitions (if None, script waits for user input). Coding time is about 1.5s (so Ts_acq must be >1.5s) - seconds
N_acqs = 500                # Total number of acquisitions
Charac_container = False    # If True, then the material inside the container is assumed to be water (Cc=Cw) - bool
no_container = True         # If True, the material is held by itself, without a container (results are Lc and Cc) (has priority over Charac_container) - bool
Attenuation_ChA = 0         # Attenuation of channel A - dB
Attenuation_ChB = 10        # Attenuation of channel B - dB
Temperature = True          # If True, take temperature measurements at each acq. (temperature data is always saved to file) and plot Cw - bool


# ---------
# PicoScope
# ---------
num_bits = 12               # Number of bits to use (8, 12, 14, 15 or 16) - int
Fs = 125e6                  # Desired sampling frequency (Hz) - float
AvgSamplesNumber = 25       # Number of traces to average to improve SNR (up to 32) - int
TT_channel = 0              # Which channel is the TT connected (0='A' or 1='B'). The PE will be the other one - str
PE_channel = int(not TT_channel) # PE channel (oposite of TT_channel) - DO NOT MODIFY


# Channel A setup
coupling_A = 'DC'           # Coupling of channel A ('AC' or 'DC') - str
voltage_range_A = '2V'      # Voltage range of channel A ('10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V', '50V' or 'MAX') - str
offset_A = 0                # Analog offset of channel A (in volts) - float
enabled_A = 1               # Enable (1) or disable (0) channel A - int

# Channel B setup
coupling_B = 'DC'           # Coupling of channel B ('AC' or 'DC') - str
voltage_range_B = '2V'      # Voltage range of channel B ('10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V', '50V' or 'MAX') - str
offset_B = 0                # Analog offset of channel B (in volts) - float
enabled_B = 1               # Enable (1) or disable (0) channel B - int

# Capture options
channels = 'BOTH'           # 'A', 'B' or 'BOTH' - str
downsampling_ratio_mode = 0 # Downsampling ratio mode - int
downsampling_ratio = 0      # Downsampling ratio - int

# Trigger options
triggerChannel = 'B'        # 'A', 'B' or 'EXTERNAL' - str
triggerThreshold = 500      # Trigger threshold in mV - float
enabled_trigger = 1         # Enable (1) or disable (0) trigger - int
direction = 2               # Check API (2=rising) - int
delay = 0                   # time between trigger and first sample (s) - float
auto_Trigger = 1000         # starts a capture if no trigger event occurs within the specified ms - float
preTriggerSamples = 1000    # Number of samples to capture before the trigger - int
postTriggerSamples = 10_000 # Number of samples to capture after the trigger - int
ScanLen = preTriggerSamples + postTriggerSamples # total number of samples of every acquisition - int


# -------------------------
# Excitation (square pulse)
# -------------------------
Fc = 5e6                        # Frequency (bandwdth) of waveform (Hz) - float
waveformSize = 2**11            # Waveform length (power of 2, max=2**15) - int
Excitation_params = [Fc, waveformSize] # All excitation params - list or float

# Waveform computation (square pulse)
pulse = US.GC_MakePulse(Param='frequency', ParamVal=Fc, SignalPolarity=2, Fs=Fs)
pulse = pulse[1:-1]*32767
waveform = US.zeroPadding(pulse, waveformSize)
waveform_t = np.arange(0,waveformSize)/Fs


# ---------------------
# Pico Signal Generator (See Pico5000alib.py doc for codes)
# ---------------------
generate_builtin_signal = False             # If True, generate builtin signal - bool
generate_arbitrary_signal = True            # If True, generate arbitrary signal (has priority over builtin) - bool
gate_time = 1000                            # Gate time in milliseconds (only used for gated triggers) - float
BUILTIN_SIGNAL_GENERATOR_DICT = {
    'offsetVoltage'         : 0,            # Offset Voltage (microvolts, uV) - int
    'pkToPk'                : 2_000_000,    # Peak-to-peak voltage (uV) - int
    'wavetype'              : 0,            # Builtin type of waveform - int
    'startFrequency'        : 5e6,          # Frequency (Hz) - float
    'stopFrequency'         : 5e6,          # Stop Frequency of the sweep (Hz) - float
    'increment'             : 0,            # Freq. increment of the sweep (Hz) - float
    'dwellTime'             : 0,            # Time for which the sweep stays at each frequency (s) - float
    'sweepType'             : 0,            # Type of sweep - int
    'shots'                 : 1,            # Number of cycles per trigger. If 0, do sweeps - int
    'sweeps'                : 0,            # Number of sweeps per trigger. If 0, do shots - int
    'triggertype'           : 0,            # Type of trigger - int
    'triggerSource'         : 4,            # Source of trigger - int
    'extInThreshold'        : 0             # Trigger level for EXTERNAL trigger - int
}
ARBITRARY_SIGNAL_GENERATOR_DICT = {
    'offsetVoltage'         : 0,            # Offset Voltage (microvolts, uV) - int
    'pkToPk'                : 2_000_000,    # Peak-to-peak voltage (uV) - int
    'startFrequency'        : Fs/waveformSize, # Initial frequency of waveform - float
    'stopFrequency'         : Fs/waveformSize, # Final frequency before restarting or reversing sweep - float
    'increment'             : 0,            # Amount of sweep in each dwell period - float
    'dwellCount'            : 0,            # Number of 50 ns steps. Determines the rate of sweep - int
    'arbitraryWaveform'     : waveform,     # The signal - array of np.int16
    'arbitraryWaveformSize' : waveformSize, # Waveform size in samples - int
    'sweepType'             : 0,            # Type of sweep - int
    'indexMode'             : 0,            # Single (0) or Dual (1) mode - int
    'shots'                 : 1,            # Number of cycles per trigger. If 0, do sweeps - int
    'sweeps'                : 0,            # Number of sweeps per trigger. If 0, do shots - int
    'triggertype'           : 0,            # Type of trigger - int
    'triggerSource'         : 4,            # Source of trigger - int
    'extInThreshold'        : 0             # Trigger level for EXTERNAL trigger - int
}


# ------------------------
# Files and data managment
# ------------------------
save_waveform = True            # If True, save arbitrary waveform to {Waveform_path} - bool
Save_acq_data = True            # If True, save all acq. data to {Acqdata_path} - bool
Load_refs_from_bin = False      # If True, load reference signals from {WP_path} and {Ref_path} instead of making an acquisiton - bool


# -----
# Plots
# -----
Plot_all_acq = True             # If True, plot every acquisition - bool
Plot_temperature = True         # If True, plots temperature measuements at each acq. (has no effect if Temperature==False) - bool


# -----------------------
# Iterative Deconvolution
# -----------------------
ID = True                       # use Iterative Deconvolution or find_peaks - bool
PE_as_ref = True                # If True, both a WP and a PE traces are acquired. The resulting ref. signal has the PE pulse aligned at WP - str
align_PEref = True              # If True, align PEref to zero - bool
stripIterNo = 2                 # If 2, PER and PETR windows are used. If 4, only one deconvolution is used for all 4 echoes - int
Cw = 1498                       # speed of sound in water - m/s
Cc = 2300                       # speed of sound in the container - m/s


# -------
# Windows
# -------
# Loc_PER, Loc_PETR, WinLen_PER and WinLen_PETR are not used if stripIterNo==4.
windowPE = True                 # If True, apply windowing to the echo signal specified by Loc_PER, Loc_PETR, WinLen_PER and WinLen_PETR - bool
Loc_TT = 2800                   # position of Through Transmission, approximation - samples
Loc_WP = 2900                   # position of Water Path, approximation - samples
Loc_PER = 1300                  # position of echo from front surface, approximation - samples
Loc_PETR = 2700                 # position of echo from back surface, approximation (only used with container) - samples
WinLen_TT = 300                 # window length of Through Transmission, approximation - samples
WinLen_WP = 300                 # window length of Water Path, approximation - samples
WinLen_PER = 1500               # window length echo from front surface, approximation - samples
WinLen_PETR = 1500              # window length of echo from back surface, approximation (only used with container) - sample


# ---------------------
# Arduino (temperature)
# ---------------------
twoSensors = False              # If True, assume we have two temperature sensors, one inside and one outside - bool
board = 'Arduino UNO'           # Board type
baudrate = 9600                 # Baudrate (symbols/s)
port = 'COM3'                   # Port to connect to
N_avg = 1                       # Number of temperature measurements to be averaged - int


# Print experiment time
if Ts_acq is not None:
    print(f'The experiment will take {US.time2str(N_acqs*Ts_acq)}.')


#%% Plot and Save arbitrary Waveform
US.plot_tf(waveform, Data2=None, Fs=Fs, nfft=None, Cs=Cw, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 0]), f_ylims=None, f_units='Hz', f_Norm=False,
            PSD=False, dB=False, Phase=False, D1label='Excitation',
            D2label='Ascan2', FigNum='Excitation', FgSize=None)

if save_waveform:
    with open(Waveform_path, 'w') as f:
        waveform.tofile(f, sep=',')
    print(f'Waveform saved to {Waveform_path}.')
    print("===================================================\n")

#%% Start serial communication
if Temperature:
    arduino = Arduino(board, baudrate, port, twoSensors, N_avg)  # open comms









#%%
########################################################################
# Start Pico
########################################################################
# Start pico
print('Starting Pico...')
pico = pico5000a.Pico(num_bits)


# -------
# Setup
# -------
print('Setting up channels and trigger...')
# Set up channel A
pico.setup_channel('A', coupling_A, voltage_range_A, offset_A, enabled_A)

# Set up channel B
pico.setup_channel('B', coupling_B, voltage_range_B, offset_B, enabled_B)

# Set up simple trigger
voltage_range = voltage_range_B if triggerChannel=='B' else voltage_range_A
pico.set_simpleTrigger(enabled_trigger, triggerChannel, voltage_range, triggerThreshold, direction, delay, auto_Trigger)


# ------------
# Get timebase
# ------------
print('Getting timebase...')
timebase, timeIntervalns, maxSamples = pico.get_timebase(Fs, preTriggerSamples + postTriggerSamples, segmentIndex=0)
Real_Fs = 1e9/(2**timebase) # Hz

if generate_arbitrary_signal and ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency'] == ARBITRARY_SIGNAL_GENERATOR_DICT['stopFrequency']:
    pulse_freq = Real_Fs/waveformSize
    if pulse_freq != ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency']:
        ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency'] = pulse_freq
        ARBITRARY_SIGNAL_GENERATOR_DICT['stopFrequency'] = pulse_freq
        print('Frequency of the arbitrary waveform changed to {pulse_freq} Hz.')


# ---------------
# Generate signal
# ---------------
print('Setting up signal generator...')
if generate_arbitrary_signal:
    pico.generate_arbitrary_signal(**ARBITRARY_SIGNAL_GENERATOR_DICT)
    triggertype = ARBITRARY_SIGNAL_GENERATOR_DICT['triggertype']
    triggerSource = ARBITRARY_SIGNAL_GENERATOR_DICT['triggerSource']
elif generate_builtin_signal:
    pico.generate_builtin_signal(**BUILTIN_SIGNAL_GENERATOR_DICT)
    triggertype = BUILTIN_SIGNAL_GENERATOR_DICT['triggertype']
    triggerSource = BUILTIN_SIGNAL_GENERATOR_DICT['triggerSource']
trigger_sigGen = True if triggerSource==4 else False
print('Done!')
print("===================================================\n")


#%% 
########################################################################
# Save config
########################################################################
config_dict = {'Fs': Fs,
               'num_bits' : num_bits,
               'Attenuation_ChA': Attenuation_ChA,
               'Attenuation_ChB': Attenuation_ChB,
               'Excitation_params': Excitation_params,
               'AvgSamplesNumber': AvgSamplesNumber,
               'ScanLen' : ScanLen,
               'Ts_acq': Ts_acq,
               'N_acqs': N_acqs,
               'WP_temperature' : None,
               'Outside_temperature': None,
               'N_avg' : N_avg,
               'ID' : ID,
               'stripIterNo' : stripIterNo,
               'Start_date': '',
               'End_date': '',
               'Experiment_description': Experiment_description}

US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Configuration parameters saved to {Config_path}.')
print("===================================================\n")


#%% 
########################################################################
# Reference acquisition (Water Path and PEref)
########################################################################
dch = 0 if downsampling_ratio_mode == 0 else 2
if Load_refs_from_bin:
    with open(WP_path, 'rb') as f:
        WP_Ascan = np.fromfile(f)
    print(f'Water path signal loaded from {WP_path}.')
    
    if os.path.isfile(PEref_path):
        with open(PEref_path, 'rb') as f:
            PEref_Ascan = np.fromfile(f)
        print(f'PE reference signal loaded from {PEref_path}.')
    else:
        PEref_Ascan = WP_Ascan
        print(f'No {Experiment_PEref_file_name} found. Setting PEref_Ascan = WP_Ascan.')
else:
    BUFFERS_DICT, maxSamples, triggerTimeOffsets, triggerTimeOffsetUnits, time_indisposed, triggerInfo = pico.rapid_capture(
        channels, (preTriggerSamples, postTriggerSamples), timebase,
        AvgSamplesNumber, trigger_sigGen, triggertype, gate_time,
        downsampling=(downsampling_ratio_mode, downsampling_ratio))
        
    ACQmeans = pico.get_data_from_buffersdict(voltage_range_A, voltage_range_B, BUFFERS_DICT)[4]
    WP_Ascan = ACQmeans[TT_channel + dch]
    # You could create a time axis for each trace like this:
    # t = np.linspace(0, (maxSamples - 1) * timeIntervalns, maxSamples)
    print('Water path acquired.')
    if PE_as_ref:
        input("Press any key to acquire the pulse echo.")
        BUFFERS_DICT, maxSamples, triggerTimeOffsets, triggerTimeOffsetUnits, time_indisposed, triggerInfo = pico.rapid_capture(
            channels, (preTriggerSamples, postTriggerSamples), timebase,
            AvgSamplesNumber, trigger_sigGen, triggertype, gate_time,
            downsampling=(downsampling_ratio_mode, downsampling_ratio))
        ACQmeans = pico.get_data_from_buffersdict(voltage_range_A, voltage_range_B, BUFFERS_DICT)[4]
        PEref_Ascan = ACQmeans[PE_channel + dch]
        
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
# ----------------------------------------
# Initialize variables and window WP_Ascan
# ----------------------------------------
dch = 0 if downsampling_ratio_mode == 0 else 2
_plt_pause_time = 0.01
if Charac_container or no_container:
    Cc = np.zeros(N_acqs)
if Temperature:
    Cw_vector = np.zeros(N_acqs)
Lc = np.zeros(N_acqs)
LM = np.zeros_like(Lc)
CM = np.zeros_like(Lc)
Toftw = np.zeros_like(Lc); Tofr21 = np.zeros_like(Lc)
Tofr1 = np.zeros_like(Lc); Tofr2 = np.zeros_like(Lc)

N = int(np.ceil(np.log2(np.abs(ScanLen)))) + 1 # next power of 2 (+1)
nfft = 2**N # Number of FFT points (power of 2)

_xlabel = 'Acquisition number'
if Ts_acq is not None:
    Time_axis = np.arange(N_acqs) * Ts_acq # time vector (one acq every Ts_acq seconds) - s
    _xlabel = 'Time (s)'
    _factor = 1
    if Time_axis[-1] > 120:
        _xlabel = 'Time (min)'
        _factor = 60

# windows are centered at approximated surfaces location
MyWin_PER = US.makeWindow(SortofWin='tukey', WinLen=WinLen_PER,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_PER - int(WinLen_PER/2))
MyWin_PETR = US.makeWindow(SortofWin='tukey', WinLen=WinLen_PETR,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_PETR - int(WinLen_PETR/2))
MyWin_WP = US.makeWindow(SortofWin='tukey', WinLen=WinLen_WP,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_WP - int(WinLen_WP/2))
MyWin_TT = US.makeWindow(SortofWin='tukey', WinLen=WinLen_TT,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_TT - int(WinLen_TT/2))

WP = WP_Ascan * MyWin_WP # window Water Path


# ---------------------------------
# Write results header to text file
# ---------------------------------
if Ts_acq is None:
    if Charac_container or no_container:
        header = 'Cc,Lc'
    else:
        header = 'Lc,LM,CM'
else:
    if Charac_container or no_container:
        header = 't,Lc,Cc'
    else:
        header = 't,Lc,LM,CM'
with open(Results_path, 'w') as f:
    f.write(header+'\n')


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


# ---------
# Sart loop
# ---------
for i in range(N_acqs):
    # -------------------------------------------
    # Acquire signal, temperature and start timer
    # -------------------------------------------
    BUFFERS_DICT, maxSamples, triggerTimeOffsets, triggerTimeOffsetUnits, time_indisposed, triggerInfo = pico.rapid_capture(
            channels, (preTriggerSamples, postTriggerSamples), timebase,
            AvgSamplesNumber, trigger_sigGen, triggertype, gate_time,
            downsampling=(downsampling_ratio_mode, downsampling_ratio))
    ACQmeans = pico.get_data_from_buffersdict(voltage_range_A, voltage_range_B, BUFFERS_DICT)[4]
    TT_Ascan = ACQmeans[TT_channel + dch]
    PE_Ascan = ACQmeans[PE_channel + dch]

    if Ts_acq is not None:
        start_time = time.time() # start timer
    
    if Temperature:
        tmp = arduino.getTemperature(error_msg=f'Warning: wrong temperature data at Acq. #{i+1}/{N_acqs}. Retrying...', 
                                     exception_msg=f'Warning: could not parse temperature data to float at Acq. #{i+1}/{N_acqs}. Retrying...')
        
        if twoSensors:
            means2[i] = tmp[1]
            means1[i] = tmp[0]
        else:
            means1[i] = tmp
        
        Cw = US.speedofsound_in_water(means1[i], method='Abdessamad', method_param=148)
        Cw_vector[i] = Cw
       
    
    # -----------------------------
    # Save temperature and acq data
    # -----------------------------
    if Temperature:
        with open(Temperature_path, 'a') as f:
            if twoSensors:
                row = f'{means1[i]},{means2[i]},{Cw}'
            else:
                row = f'{means1[i]},{Cw}'
            f.write(row+'\n')
    
    if Save_acq_data:
        _mode = 'wb' if i==0 else 'ab' # clear data from previous experiment before writing
        with open(Acqdata_path, _mode) as f:
            TT_Ascan.tofile(f)
            PE_Ascan.tofile(f)
            
            
    # ------------
    # Window scans
    # ------------
    if windowPE:
        PE_R = PE_Ascan * MyWin_PER # extract front surface reflection
        PE_TR = PE_Ascan * MyWin_PETR # extract back surface reflection
    else:
        PE_R = PE_Ascan
        PE_TR = PE_Ascan
    TT = TT_Ascan * MyWin_TT
    
    
    # -------------
    # Control plots
    # -------------
    # Plot every acquisition if Plot_all_acq==True
    # Plot one acquisition in the middle of experiment to see how things are going
    if Plot_all_acq or i==N_acqs//2:
        US.multiplot_tf(np.column_stack((PE_Ascan, TT, WP, PEref_Ascan)).T, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
                    t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
                    f_xlims=([0, 20]), f_ylims=None, f_units='MHz', f_Norm=False,
                    PSD=False, dB=False, label=['PE', 'TT', 'WP', 'PEref'], Independent=True, FigNum='Signals', FgSize=(6.4,4.8))
        US.movefig(location='southeast')
        plt.pause(_plt_pause_time)


    # ----------------
    # TOF computations
    # ----------------
    # Find ToF_TW
    ToF_TW, Aligned_TW, _ = US.CalcToFAscanCosine_XCRFFT(TT, WP, UseCentroid=False, UseHilbEnv=False, Extend=True, Same=False)
    
    if ID:
        if stripIterNo == 2:
            # Iterative Deconvolution: first face
            ToF_RW, StrMat = US.deconvolution(PE_R, PEref_Ascan, stripIterNo=stripIterNo, UseHilbEnv=False)
            ToF_R21 = ToF_RW[1] - ToF_RW[0]
            
            # Plot StrMat
            # -----------
            # fig, axs = plt.subplots(3, num='StrMat', clear=True)
            # USG.movefig(location='southwest')
            # axs[1].set_ylabel('StrMat')
            # axs[2].set_xlabel(_xlabel)
            # axs[0].plot(StrMat[0,:])
            # axs[1].plot(StrMat[1,:])
            # axs[2].plot(StrMat[2,:])
            # plt.tight_layout()
            # plt.pause(_plt_pause_time)
            
            # Iterative Deconvolution: second face
            ToF_TRW, StrMat = US.deconvolution(PE_TR, PEref_Ascan, stripIterNo=stripIterNo, UseHilbEnv=False)
            ToF_TR21 = ToF_TRW[1] - ToF_TRW[0]
        elif stripIterNo == 4:
            # Iterative Deconvolution: first and second face
            ToF_RW, StrMat = US.deconvolution(PE_Ascan, PEref_Ascan, stripIterNo=stripIterNo, UseHilbEnv=False)
            ToF_R21 = ToF_RW[1] - ToF_RW[0]
            ToF_TR21 = ToF_TRW[3] - ToF_TRW[2]
    else:
        MyXcor_PE = US.fastxcorr(PE_Ascan, PEref_Ascan, Extend=True, Same=False)
        Env = US.envelope(MyXcor_PE)
        Real_peaks = US.find_Subsampled_peaks(Env, prominence=0.07*np.max(Env), width=20)
        for i, r in enumerate(Real_peaks):
            if r < len(Env)//2:
                Real_peaks[i] = -(len(Env)-r)
        ToF_R21 = Real_peaks[1] - Real_peaks[0]
        ToF_TR21 = Real_peaks[3] - Real_peaks[2]
        ToF_RW = Real_peaks[:1]
        ToF_TRW = Real_peaks[2:]
    

    ToF_TR1R2 = ToF_TRW[0] - ToF_RW[1] # t_TR1 - t_R2 (this does not work if the second echo's amplitude is larger than the first)
    
    # This should always work
    # -----------------------
    # ToF_PEref = USF.CosineInterpMax(PEref_Ascan, UseHilbEnv=False)
    # absoluteToF_TRW2 = ToF_PEref + ToF_TRW[1]
    # absoluteToF_TRW1 = ToF_PEref + ToF_TRW[0]
    # absoluteToF_RW2 = ToF_PEref + ToF_RW[1]
    # absoluteToF_RW1 = ToF_PEref + ToF_RW[0]
    # if absoluteToF_TRW2 <= absoluteToF_TRW1:
    #     absoluteToF_TR1 = absoluteToF_TRW2
    # else:
    #     absoluteToF_TR1 = absoluteToF_TRW1
        
    # if absoluteToF_RW2 >= absoluteToF_RW1:
    #     absoluteToF_R2 = absoluteToF_RW2
    # else:
    #     absoluteToF_R2 = absoluteToF_RW1
    # ToF_TR1R2 = absoluteToF_TR1 - absoluteToF_R2
    
    Toftw[i] = ToF_TW
    Tofr21[i] = ToF_R21
    Tofr2[i] = ToF_RW[1]
    Tofr1[i] = ToF_RW[0]
    
    
    # -----------------------------------
    # Velocity and thickness computations
    # -----------------------------------
    if Charac_container or no_container:
        if no_container:
            Cc[i] = Cw*(2*np.abs(ToF_TW)/ToF_R21 + 1) # container speed - m/s
            Lc[i] = Cw/2*(2*np.abs(ToF_TW) + ToF_R21)/Fs # container thickness - m     
        else:
            Cc[i] = Cw*(np.abs(ToF_TW)/ToF_R21 + 1) # container speed - m/s
            Lc[i] = Cw/2*(np.abs(ToF_TW) + ToF_R21)/Fs # container thickness - m
    else:
        Lc[i] = Cc*ToF_R21/2/Fs # container thickness - m
        LM[i] = (ToF_R21 + np.abs(ToF_TW) + ToF_TR1R2/2)*Cw/Fs - 2*Lc[i] # material thickness - m
        CM[i] = 2*LM[i]/ToF_TR1R2*Fs # material speed - m/s
    
    
    # ----------------------------------
    # Save results to text file as we go 
    # ----------------------------------
    with open(Results_path, 'a') as f:
        if Ts_acq is None:
            if Charac_container or no_container:
                row = f'{Cc[i]},{Lc[i]}'
            else:
                row = f'{Lc[i]},{LM[i]},{CM[i]}'
        else:
            if Charac_container or no_container:
                row = f'{Time_axis[i]},{Lc[i]},{Cc[i]}'
            else:
                row = f'{Time_axis[i]},{Lc[i]},{LM[i]},{CM[i]}'
        f.write(row+'\n')


    # -----
    # Plots
    # -----
    _xdata = np.arange(i) if Ts_acq is None else Time_axis[:i]/_factor
    
    # Plot Cw
    if Temperature:       
        fig, ax = plt.subplots(1, num='Cw', clear=True)
        US.movefig(location='north')
        ax.set_ylabel('Cw (m/s)')
        ax.set_xlabel(_xlabel)
        ax.scatter(_xdata, Cw_vector[:i], color='white', marker='o', edgecolors='black')
        plt.tight_layout()
        plt.pause(_plt_pause_time)
        
        # Plot temperature
        if Plot_temperature:
            if twoSensors:
                fig, axs = plt.subplots(2, num='Temperature', clear=True)
                US.movefig(location='south')
                axs[0].set_ylabel('Temperature 1 (\u2103)')
                axs[1].set_ylabel('Temperature 2 (\u2103)')
                axs[1].set_xlabel(_xlabel)
                axs[0].scatter(_xdata, means1[:i], color='white', marker='o', edgecolors='black')
                axs[1].scatter(_xdata, means2[:i], color='white', marker='o', edgecolors='black')
            else:
                fig, ax = plt.subplots(1, num='Temperature', clear=True)
                US.movefig(location='south')
                ax.set_ylabel('Temperature (\u2103)')
                ax.set_xlabel(_xlabel)
                ax.scatter(_xdata, means1[:i], color='white', marker='o', edgecolors='black')
            plt.tight_layout()
            plt.pause(_plt_pause_time)
            
    # Plot results
    if Charac_container or no_container:
        fig, axs = plt.subplots(2, num='Results', clear=True)
        US.movefig(location='northeast')
        axs[0].set_ylabel('Cc (m/s)')
        axs[1].set_ylabel('Lc (um)')
        axs[1].set_xlabel(_xlabel)
        axs[0].scatter(_xdata, Cc[:i], color='white', marker='o', edgecolors='black', zorder=2)
        axs[1].scatter(_xdata, Lc[:i]*1e6, color='white', marker='o', edgecolors='black', zorder=2)
        line_Cc = axs[0].axhline(np.mean(Cc[:i]), color='black', linestyle='--', zorder=1) # Plot Cc mean
        line_Lc = axs[1].axhline(np.mean(Lc[:i]*1e6), color='black', linestyle='--', zorder=1) # Plot Lc mean
    else:
        fig, axs = plt.subplots(3, num='Results', clear=True)
        US.movefig(location='northeast')
        axs[0].set_ylabel('Lc (um)')
        axs[1].set_ylabel('LM (mm)')
        axs[2].set_ylabel('CM (m/s)')
        axs[2].set_xlabel(_xlabel)
        axs[0].scatter(_xdata, Lc[:i]*1e6, color='white', marker='o', edgecolors='black', zorder=2)
        axs[1].scatter(_xdata, LM[:i]*1e3, color='white', marker='o', edgecolors='black', zorder=2)
        axs[2].scatter(_xdata, CM[:i], color='white', marker='o', edgecolors='black', zorder=2)            
        line_Lc = axs[0].axhline(np.mean(Lc[:i]*1e6), color='black', linestyle='--', zorder=1) # Plot Lc mean
    plt.tight_layout()
    plt.pause(_plt_pause_time)
    
    
    # --------------------------------
    # Wait for user input or end timer
    # --------------------------------
    if Ts_acq is None:
        input(f'Acquisition #{i+1}/{N_acqs} done. Press any key to continue.')
    else:
        elapsed_time = time.time() - start_time
        time_to_wait = Ts_acq - elapsed_time # time until next acquisition
        print(f'Acquisition #{i+1}/{N_acqs} done.')
        if time_to_wait < 0:
            print(f'Code is slower than Ts_acq = {Ts_acq} s at Acq #{i+1}. Elapsed time is {elapsed_time} s.')
            time_to_wait = 0
        time.sleep(time_to_wait)
    
    # Remove previous mean lines
    if i != N_acqs-1:
        if Charac_container or no_container:
            line_Cc.remove()
        line_Lc.remove()
# -----------
# End of loop
# -----------
pico.stop() # Stop the scope

plt.tight_layout()
if Temperature:
    arduino.close()


# -----------------
# Outlier detection
# -----------------
m = 0.6745
Lc_no_outliers, Lc_outliers, Lc_outliers_indexes = US.reject_outliers(Lc, m=m)
if Charac_container or no_container:
    Cc_no_outliers, Cc_outliers, Cc_outliers_indexes = US.reject_outliers(Cc, m=m)
    if Ts_acq is None:
        axs[0].scatter(Cc_outliers_indexes, Cc_outliers, color='red', zorder=3)
        axs[1].scatter(Lc_outliers_indexes, Lc_outliers*1e6, color='red', zorder=3)
    else:
        axs[0].scatter(Time_axis[Cc_outliers_indexes]/_factor, Cc_outliers, color='red', zorder=3)
        axs[1].scatter(Time_axis[Lc_outliers_indexes]/_factor, Lc_outliers*1e6, color='red', zorder=3)
else:
    LM_no_outliers, LM_outliers, LM_outliers_indexes = US.reject_outliers(LM, m=m)
    CM_no_outliers, CM_outliers, CM_outliers_indexes = US.reject_outliers(CM, m=m)
    if Ts_acq is None:
        axs[0].scatter(Lc_outliers_indexes, Lc_outliers*1e6, color='red', zorder=3)
        axs[1].scatter(LM_outliers_indexes, LM_outliers*1e6, color='red', zorder=3)
        axs[2].scatter(CM_outliers_indexes, CM_outliers, color='red', zorder=3)
    else:
        axs[0].scatter(Time_axis[Lc_outliers_indexes]/_factor, Lc_outliers*1e6, color='red', zorder=3)
        axs[1].scatter(Time_axis[LM_outliers_indexes]/_factor, LM_outliers*1e6, color='red', zorder=3)
        axs[2].scatter(Time_axis[CM_outliers_indexes]/_factor, CM_outliers, color='red', zorder=3)


# -----------------------------
# Write end time to config file
# -----------------------------
_end_time = time.strftime("%Y/%m/%d - %H:%M:%S")
config_dict['End_date'] = _end_time
US.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Experiment ended at {_end_time}.')
print("===================================================\n")


# -------------------------------------
# Compute means and standard deviations
# -------------------------------------
# Lc
Lc_std = np.std(Lc)
Lc_mean = np.mean(Lc)
print("---------------------------------------------------")
print(f'Lc_mean = {Lc_mean*1e6} um')
print(f'Lc_std = {Lc_std*1e6} um')
print(f'Lc = {np.round(Lc_mean*1e6)} \u2a72 {np.round(3*Lc_std*1e6)} um')

# Cc
if Charac_container or no_container:
    Cc_std = np.std(Cc)
    Cc_mean = np.mean(Cc)
    print("---------------------------------------------------")
    print(f'Cc_mean = {Cc_mean} m/s')
    print(f'Cc_std = {Cc_std} m/s')
    print(f'Cc = {np.round(Cc_mean)} \u2a72 {np.round(3*Cc_std)} m/s')
print("---------------------------------------------------")

print("===================================================\n")

# Lc
if Lc_outliers.size != 0:
    print('Without outliers')
    print("---------------------------------------------------")
    Lc_std_no_outliers = np.std(Lc_no_outliers)
    Lc_mean_no_outliers = np.mean(Lc_no_outliers)
    print(f'Lc_mean = {Lc_mean_no_outliers*1e6} um')
    print(f'Lc_std = {Lc_std_no_outliers*1e6} um')
    print(f'Lc = {np.round(Lc_mean_no_outliers*1e6)} \u2a72 {np.round(3*Lc_std_no_outliers*1e6)} um')

# Cc
if (Charac_container or no_container) and Cc_outliers.size != 0:
    Cc_std_no_outliers = np.std(Cc_no_outliers)
    Cc_mean_no_outliers = np.mean(Cc_no_outliers)
    print("---------------------------------------------------")
    print(f'Cc_mean = {Cc_mean_no_outliers} m/s')
    print(f'Cc_std = {Cc_std_no_outliers} m/s')
    print(f'Cc = {np.round(Cc_mean_no_outliers)} \u2a72 {np.round(3*Cc_std_no_outliers)} m/s')
print("---------------------------------------------------")


#%%
# Close the unit
pico.close()