# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 08:59:35 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import time
import numpy as np
import matplotlib.pylab as plt
import SeDaq as SD
import US_Functions as USF
import US_ACQ as ACQ
import US_GenCode as USGC
import US_Graphics as USG
import US_Loaders as USL
import os
import serial

def time2str(seconds) -> str:
    hours = seconds//3600
    minutes = seconds%3600//60
    seconds = seconds - hours*3600 - minutes*60
    s = f'{hours} h, {minutes} min, {seconds} s'
    return s


#%%
########################################################
# Paths and file names to use
########################################################
Path = r'D:\Data\pruebas_acq'
# Path = r'D:\Data\Arnau\Colacao_Plastic_Bottle_characterization'
# Experiment_folder_name = 'test_3_charac_bottle_colacao'
Experiment_folder_name = 'test' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_WP_file_name = 'WP.bin'
Experiment_acqdata_file_basename = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
WP_path = os.path.join(MyDir, Experiment_WP_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_basename)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
if not os.path.exists(MyDir):
    os.makedirs(MyDir)
    print(f'Created new path: {MyDir}')
print(f'Experiment path set to {MyDir}')


#%% 
########################################################
# Parameters and constants
########################################################
# For Experiment_description do NOT use '\n'.
Experiment_description = "Second test of plastic Colacao bottle." \
                        " Container filled with water." \
                        " Excitation_params: Pulse frequency (Hz)."
Fs = 100.0e6                    # Sampling frequency - Hz
Fs_Gencode_Generator = 200.0e6  # Sampling frequency for the gencodes generator - Hz
RecLen = 32*1024                # Maximum range of ACQ - samples (max=32*1024)
Gain_Ch1 = 60                   # Gain of channel 1 - dB
Gain_Ch2 = 35                   # Gain of channel 2 - dB
Attenuation_Ch1 = 0             # Attenuation of channel 1 - dB
Attenuation_Ch2 = 0            # Attenuation of channel 2 - dB
Excitation_voltage = 60         # Excitation voltage (min=20V) - V -- DOESN'T WORK
Fc = 5*1e6                      # Pulse frequency - Hz
Excitation = 'Pulse'            # Excitation to use ('Pulse, 'Chirp', 'Burst') - string
Excitation_params = Fc          # All excitation params - list or float
Smin1, Smin2 = 4_500, 4_500     # starting point of the scan of each channel - samples
Smax1, Smax2 = 9_000, 9_000   # last point of the scan of each channel - samples
AvgSamplesNumber = 25           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels
Ts_acq = 4                      # Time between acquisitions (if None, script waits for user input). Coding time is about 1.5s (so Ts_acq must be >1.5s) - seconds
N_acqs = 500                   # Total number of acquisitions
Charac_container = True         # If True, then the material inside the container is assumed to be water (Cc=Cw) - bool
Reset_Relay = False             # Reset delay: ON>OFF>ON - bool
Save_acq_data = True            # If True, save all acq. data to {Acqdata_path} - bool
Load_WP_from_bin = False        # If True, load WP data from {WP_path} instead of making an acquisiton - bool
Plot_all_acq = True             # If True, plot every acquisition - bool
Temperature = True              # If True, take temperature measurements at each acq. (temperature data is always saved to file) and plot Cw - bool
Plot_temperature = True         # If True, plots temperature measuements at each acq. (has no effect if Temperature==False) - bool
Cw = 1498                       # speed of sound in water - m/s
Cc = 2300                       # speed of sound in the container - m/s
Loc_echo1 = 1300                # position of echo from front surface, approximation - samples
Loc_echo2 = 3500                # position of echo from back surface, approximation - samples
Loc_WP = 3500                   # position of Water Path, approximation - samples
Loc_TT = 3500                   # position of Through Transmission, approximation - samples
WinLen = Loc_echo1 * 2          # window length, approximation
ID = True                       # use Iterative Deconvolution or find_peaks - bool

board = 'Arduino UNO'           # Board type
baudrate = 9600                 # Baudrate (symbols/s)
port = 'COM3'                   # Port to connect to
N_avg = 1                       # Number of temperature measurements to be averaged - int

if Ts_acq is not None:
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
print("------------------------------------------------------------------------")
if Reset_Relay:
    print('Resetting relay...')
    SeDaq.SetRelay(1)
    time.sleep(1) # wait to be sure
    SeDaq.SetRelay(0)
    time.sleep(1) # wait to be sure
    SeDaq.SetRelay(1)
    time.sleep(1) # wait to be sure
    print("------------------------------------------------------------------------")
SeDaq.SetRecLen(RecLen) # initialize record length
# SeDaq.SetExtVoltage(Excitation_voltage) - DOESN'T WORK
SeDaq.SetGain1(Gain_Ch1)
SeDaq.SetGain2(Gain_Ch2)
print(f'Gain of channel 1 set to {SeDaq.GetGain(1)} dB') # return gain of channel 1
print(f'Gain of channel 2 set to {SeDaq.GetGain(2)} dB') # return gain of channel 2
print("------------------------------------------------------------------------")
GenCode = USGC.MakeGenCode(Excitation=Excitation, ParamVal=Excitation_params)
SeDaq.UpdateGenCode(GenCode)
print('Generator code created and updated.')
print("========================================================================\n")


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
               'WP_temperature' : None,
               'Outside_temperature': None,
               'N_avg' : N_avg,
               'Start_date': '',
               'End_date': '',
               'Experiment_description': Experiment_description}
# Descripción material (marca, modelo, dopante)
USL.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Configuration parameters saved to {Config_path}.')
print("========================================================================\n")


#%% 
########################################################################
# Water Path acquisition
########################################################################
if Load_WP_from_bin:
    with open(WP_path, 'rb') as f:
        WP_Ascan = np.fromfile(f)
    print(f'Water path loaded from {WP_path}.')
else:
    WP_Ascan = ACQ.GetAscan_Ch1(Smin1, Smax1, AvgSamplesNumber=AvgSamplesNumber, Quantiz_Levels=Quantiz_Levels)
    print('Water path acquired.')

if Temperature:
    if not ser.isOpen():
        ser.open()
    mean1, mean2 = ACQ.getTemperature(ser, N_avg, 'Warning: wrong temperature data for Water Path.', 'Warning: could not parse temperature data to float for Water Path.')
    
    config_dict['WP_temperature'] = mean1
    config_dict['Outside_temperature'] = mean2
    USL.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
    print(f'Water Path temperature is {mean1} \u00b0C.')
    print(f'Outside temperature is {mean2} \u00b0C.')
    print("========================================================================\n")
    
plt.figure()
plt.plot(WP_Ascan)
plt.title('WP')
USG.movefig(location='n')
plt.pause(0.05)

if Save_acq_data:
    with open(WP_path, 'wb') as f:
        WP_Ascan.tofile(f)

input("Press any key to start the experiment.")
print("========================================================================\n")


#%% 
########################################################################
# Run Acquisition and Computations
########################################################################
# ----------------------------------------
# Initialize variables and window WP_Ascan
# ----------------------------------------
_plt_pause_time = 0.01
if Charac_container:
    Cc = np.zeros(N_acqs)
if Temperature:
    Cw_vector = np.zeros(N_acqs)
Lc = np.zeros(N_acqs)
LM = np.zeros_like(Lc)
CM = np.zeros_like(Lc)
Toftw = np.zeros_like(Lc); Tofr21 = np.zeros_like(Lc)

Smin = (Smin1, Smin2)                   # starting points - samples
Smax = (Smax1, Smax2)                   # last points - samples
ScanLen1 = Smax1 - Smin1                # Total scan length for channel 1 - samples
ScanLen2 = Smax2 - Smin2                # Total scan length for channel 2 - samples
ScanLen = np.max([ScanLen1, ScanLen2])  # Total scan length for computations (zero padding is used) - samples

# Zero pad WP in case each channel has different scan length
if ScanLen1 < ScanLen:
    WP_Ascan = USF.zeroPadding(WP_Ascan, ScanLen)

N = int(np.ceil(np.log2(np.abs(ScanLen)))) + 1 # next power of 2 (+1)
nfft = 2**N # Number of FFT points (power of 2)
if Ts_acq is not None:
    Time_axis = np.arange(N_acqs) * Ts_acq # time vector (one acq every Ts_acq seconds) - s

# windows are centered at approximated surfaces location
MyWin1 = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
               param1=0.25, param2=1, Span=ScanLen, Delay=0)
MyWin2 = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_echo2 - int(WinLen/2))
MyWin_WP = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_WP - int(WinLen/2))
MyWin_TT = USF.makeWindow(SortofWin='tukey', WinLen=WinLen,
               param1=0.25, param2=1, Span=ScanLen, Delay=Loc_TT - int(WinLen/2))

WP = WP_Ascan * MyWin_WP # window Water Path


# ---------------------------------
# Write results header to text file
# ---------------------------------
if Ts_acq is None:
    if Charac_container:
        header = 'Cc,Lc'
    else:
        header = 'Lc,LM,CM'
else:
    if Charac_container:
        header = 't,Lc,Cc'
    else:
        header = 't,Lc,LM,CM'
with open(Results_path, 'w') as f:
    f.write(header+'\n')


# -------------------------------------
# Write temperature header to text file
# -------------------------------------
if Temperature:
    header = 'Inside,Outside,Cw'
    with open(Temperature_path, 'w') as f:
        f.write(header+'\n')
    means1 = np.zeros(N_acqs)
    means2 = np.zeros(N_acqs)
    
    if not ser.isOpen():
        ser.open()


# -------------------------------
# Write start time to config file
# -------------------------------
_start_time = time.strftime("%Y/%m/%d - %H:%M:%S")
config_dict['Start_date'] = _start_time
USL.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Experiment started at {_start_time}.')
print("========================================================================\n")


# ---------
# Sart loop
# ---------
for i in range(N_acqs):
    # -------------------------------------------
    # Acquire signal, temperature and start timer
    # -------------------------------------------
    TT_Ascan, PE_Ascan = ACQ.GetAscan_Ch1_Ch2(Smin, Smax, AvgSamplesNumber=AvgSamplesNumber, Quantiz_Levels=Quantiz_Levels) #acq Ascan
    
    if Ts_acq is not None:
        start_time = time.time() # start timer
    
    if Temperature:
        means1[i], means2[i] = ACQ.getTemperature(ser, N_avg, 'Warning: wrong temperature data at Acq. #{i+1}/{N_acqs}. Retrying...', 'Warning: could not parse temperature data to float at Acq. #{i+1}/{N_acqs}. Retrying...')
        
        Cw = USF.speedofsound_in_water(means1[i], method='Abdessamad', method_param=148)
        Cw_vector[i] = Cw
    
    # -----------------------------
    # Save temperature and acq data
    # -----------------------------  
    with open(Temperature_path, 'a') as f:
        row = f'{means1[i]},{means2[i]},{Cw}'
        f.write(row+'\n')
    
    _mode = 'wb' if i==0 else 'ab' # clear data from previous experiment before writing
    if Save_acq_data:
        with open(Acqdata_path, _mode) as f:
            TT_Ascan.tofile(f)
            PE_Ascan.tofile(f)


    # -----------------------------------------------------------
    # Zero padding in case each channel has different scan length
    # -----------------------------------------------------------
    if ScanLen1 < ScanLen:
        TT_Ascan = USF.zeroPadding(TT_Ascan, ScanLen)
    elif ScanLen2 < ScanLen:
        PE_Ascan = USF.zeroPadding(PE_Ascan, ScanLen)
    
    
    # ------------
    # Window scans
    # ------------
    PE_R = PE_Ascan * MyWin1 # extract front surface reflection
    PE_TR = PE_Ascan * MyWin2 # extract back surface reflection
    TT = TT_Ascan * MyWin_TT
    
    
    # -------------
    # Control plots
    # -------------
    # Plot every acquisition if Plot_all_acq==True
    # Plot one acquisition in the middle of experiment to see how things are going
    if Plot_all_acq or i==N_acqs//2:
        USG.multiplot_tf(np.column_stack((PE_Ascan, TT, WP)).T, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
                    t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
                    f_xlims=([0, 20]), f_ylims=None, f_units='MHz', f_Norm=False,
                    PSD=False, dB=False, label=['PE', 'TT', 'WP'], Independent=True, FigNum='Signals', FgSize=(6.4,4.8))
        USG.movefig(location='southeast')
        plt.pause(_plt_pause_time)


    # ----------------
    # TOF computations
    # ----------------
    # Find ToF_TW
    ToF_TW, Aligned_TW, _ = USF.CalcToFAscanCosine_XCRFFT(TT, WP, UseCentroid=False, UseHilbEnv=False, Extend=False)
    
    if ID:
        # Iterative Deconvolution: first face
        ToF_RW, StrMat = USF.deconvolution(PE_R, WP, stripIterNo=2, UseHilbEnv=False)
        ToF_R21 = ToF_RW[1] - ToF_RW[0]
        
        # Iterative Deconvolution: second face
        ToF_TRW, StrMat = USF.deconvolution(PE_TR, WP, stripIterNo=2, UseHilbEnv=False)
        ToF_TR21 = ToF_TRW[1] - ToF_TRW[0]

    else:
        MyXcor_PE = USF.fastxcorr(PE_Ascan, WP, Extend=True)
        Env = USF.envelope(MyXcor_PE)
        Real_peaks = USF.find_Subsampled_peaks(Env, prominence=0.07*np.max(Env), width=20)
        ToF_R21 = Real_peaks[1] - Real_peaks[0]
        ToF_TR21 = Real_peaks[3] - Real_peaks[2]
        ToF_RW = Real_peaks[:1]
        ToF_TRW = Real_peaks[2:]
    
    ToF_TR1 = np.min(ToF_TRW)
    ToF_R2 = np.max(ToF_RW)
    ToF_TR1R2 = ToF_TR1 - ToF_R2
    
    Toftw[i] = ToF_TW
    Tofr21[i] = ToF_R21
    # -----------------------------------
    # Velocity and thickness computations
    # -----------------------------------        
    if Charac_container:
        Cc[i] = Cw*(np.abs(ToF_TW/ToF_R21) + 1) # container speed - m/s
        Lc[i] = Cw/2*(np.abs(ToF_TW) + np.abs(ToF_R21))/Fs # container thickness - m
    else:
        Lc[i] = Cc*np.abs(ToF_R21)/2/Fs # container thickness - m
        LM[i] = (np.abs(ToF_R21) + ToF_TW + ToF_TR1R2/2)*Cw/Fs - 2*Lc[i] # material thickness - m
        CM[i] = 2*LM[i]/ToF_TR1R2*Fs # material speed - m/s
    
    
    # -----------------------------------
    # Save results to CSV file as we go 
    # -----------------------------------
    with open(Results_path, 'a') as f:
        if Ts_acq is None:
            if Charac_container:
                row = f'{Cc[i]},{Lc[i]}'
            else:
                row = f'{Lc[i]},{LM[i]},{CM[i]}'
        else:
            if Charac_container:
                row = f'{Time_axis[i]},{Lc[i]},{Cc[i]}'
            else:
                row = f'{Time_axis[i]},{Lc[i]},{LM[i]},{CM[i]}'
        f.write(row+'\n')


    # ----------------
    # Plot temperature
    # ----------------
    if Plot_temperature:
        fig, axs = plt.subplots(2, num='Temperature', clear=True)
        USG.movefig(location='south')
        axs[0].set_ylabel('Temperature 1 (\u2103)')
        axs[1].set_ylabel('Temperature 2 (\u2103)')
        
        if Ts_acq is None:
            axs[0].scatter(np.arange(i), means1[:i], color='white', marker='o', edgecolors='black')
            axs[1].scatter(np.arange(i), means2[:i], color='white', marker='o', edgecolors='black')
        else:
            axs[1].set_xlabel('Time (s)')
            axs[0].scatter(Time_axis[:i], means1[:i], color='white', marker='o', edgecolors='black')
            axs[1].scatter(Time_axis[:i], means2[:i], color='white', marker='o', edgecolors='black')
        plt.tight_layout()
        plt.pause(_plt_pause_time)


    # -------
    # Plot Cw
    # -------
    if Temperature:       
        fig, ax = plt.subplots(1, num='Cw', clear=True)
        USG.movefig(location='north')
        ax.set_ylabel('Cw (m/s)')
        if Ts_acq is None:
            ax.scatter(np.arange(i), Cw_vector[:i], color='white', marker='o', edgecolors='black')
        else:
            ax.set_xlabel('Time (s)')
            ax.scatter(Time_axis[:i], Cw_vector[:i], color='white', marker='o', edgecolors='black')
        plt.tight_layout()
        plt.pause(_plt_pause_time)
        

    # ------------
    # Plot points
    # ------------
    if Charac_container:
        fig, axs = plt.subplots(2, num='Results', clear=True)
        USG.movefig(location='northeast')
        axs[0].set_ylabel('Cc (m/s)')
        axs[1].set_ylabel('Lc (um)')
        if Ts_acq is None:
            axs[0].scatter(np.arange(i), Cc[:i], color='white', marker='o', edgecolors='black', zorder=2)
            axs[1].scatter(np.arange(i), Lc[:i]*1e6, color='white', marker='o', edgecolors='black', zorder=2)
        else:
            axs[1].set_xlabel('Time (s)')
            axs[0].scatter(Time_axis[:i], Cc[:i], color='white', marker='o', edgecolors='black', zorder=2)
            axs[1].scatter(Time_axis[:i], Lc[:i]*1e6, color='white', marker='o', edgecolors='black', zorder=2)
        line_Cc = axs[0].axhline(np.mean(Cc[:i]), color='black', linestyle='--', zorder=1) # Plot Cc mean
        line_Lc = axs[1].axhline(np.mean(Lc[:i]*1e6), color='black', linestyle='--', zorder=1) # Plot Lc mean
    else:
        fig, axs = plt.subplots(3, num='Results', clear=True)
        USG.movefig(location='northeast')
        axs[0].set_ylabel('Lc (um)')
        axs[1].set_ylabel('LM (mm)')
        axs[2].set_ylabel('CM (m/s)')
        if Ts_acq is None:
            axs[0].scatter(np.arange(i), Lc[:i]*1e6, color='white', marker='o', edgecolors='black', zorder=2)
            axs[1].scatter(np.arange(i), LM[:i]*1e3, color='white', marker='o', edgecolors='black', zorder=2)
            axs[2].scatter(np.arange(i), CM[:i], color='white', marker='o', edgecolors='black', zorder=2)            
        else:
            axs[2].set_xlabel('Time (s)')
            axs[0].scatter(Time_axis[:i], Lc[:i]*1e6, color='white', marker='o', edgecolors='black', zorder=2)
            axs[1].scatter(Time_axis[:i], LM[:i]*1e3, color='white', marker='o', edgecolors='black', zorder=2)
            axs[2].scatter(Time_axis[:i], CM[:i], color='white', marker='o', edgecolors='black', zorder=2)
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
        print(f'Acquisition #{i+1}/{N_acqs} done. {ToF_TW=}  {ToF_R21=}')
        if time_to_wait < 0:
            print(f'Code is slower than Ts_acq = {Ts_acq} s at Acq #{i+1}. Elapsed time is {elapsed_time} s.')
            time_to_wait = 0
        time.sleep(time_to_wait)
    
    # Remove previous mean lines
    if i != N_acqs-1:
        if Ts_acq is None:
            line_Cc.remove()
        line_Lc.remove()
# -----------
# End of loop
# -----------
plt.tight_layout()
if Temperature:
    try:
        ser.close()
        print(f'Serial communication with {board} at port {port} closed successfully.')
    except Exception as e:
        print(e)


# -----------------
# Outlier detection
# -----------------
m = 0.6745
Lc_no_outliers, Lc_outliers, Lc_outliers_indexes = USF.reject_outliers(Lc, m=m)
if Charac_container:
    Cc_no_outliers, Cc_outliers, Cc_outliers_indexes = USF.reject_outliers(Cc, m=m)
    if Ts_acq is None:
        axs[0].scatter(Cc_outliers_indexes, Cc_outliers, color='red', zorder=3)
        axs[1].scatter(Lc_outliers_indexes, Lc_outliers*1e6, color='red', zorder=3)
    else:
        axs[0].scatter(Time_axis[Cc_outliers_indexes], Cc_outliers, color='red', zorder=3)
        axs[1].scatter(Time_axis[Lc_outliers_indexes], Lc_outliers*1e6, color='red', zorder=3)
else:
    LM_no_outliers, LM_outliers, LM_outliers_indexes = USF.reject_outliers(LM, m=m)
    CM_no_outliers, CM_outliers, CM_outliers_indexes = USF.reject_outliers(CM, m=m)
    if Ts_acq is None:
        axs[0].scatter(Lc_outliers_indexes, Lc_outliers*1e6, color='red', zorder=3)
        axs[1].scatter(LM_outliers_indexes, LM_outliers*1e6, color='red', zorder=3)
        axs[2].scatter(CM_outliers_indexes, CM_outliers, color='red', zorder=3)
    else:
        axs[0].scatter(Time_axis[Lc_outliers_indexes], Lc_outliers*1e6, color='red', zorder=3)
        axs[1].scatter(Time_axis[LM_outliers_indexes], LM_outliers*1e6, color='red', zorder=3)
        axs[2].scatter(Time_axis[CM_outliers_indexes], CM_outliers, color='red', zorder=3)


# -----------------------------
# Write end time to config file
# -----------------------------
_end_time = time.strftime("%Y/%m/%d - %H:%M:%S")
config_dict['End_date'] = _end_time
USL.saveDict2txt(Path=Config_path, d=config_dict, mode='w', delimiter=',')
print(f'Experiment ended at {_end_time}.')
print("============================================================================\n")


# -------------------------------------
# Compute means and standard deviations
# -------------------------------------
# Lc
Lc_std = np.std(Lc)
Lc_mean = np.mean(Lc)
print('--------------------------------------')
print(f'Lc_mean = {Lc_mean*1e6} um')
print(f'Lc_std = {Lc_std*1e6} um')
print(f'Lc = {np.round(Lc_mean*1e6)} \u2a72 {np.round(3*Lc_std*1e6)} um')

# Cc
if Charac_container:
    Cc_std = np.std(Cc)
    Cc_mean = np.mean(Cc)
    print('--------------------------------------')
    print(f'Cc_mean = {Cc_mean} m/s')
    print(f'Cc_std = {Cc_std} m/s')
    print(f'Cc = {np.round(Cc_mean)} \u2a72 {np.round(3*Cc_std)} m/s')
print('--------------------------------------')

print('\n============================================================================\n')

# Lc
if Lc_outliers.size!=0:
    print('Without outliers')
    print('--------------------------------------')
    Lc_std_no_outliers = np.std(Lc_no_outliers)
    Lc_mean_no_outliers = np.mean(Lc_no_outliers)
    print(f'Lc_mean = {Lc_mean_no_outliers*1e6} um')
    print(f'Lc_std = {Lc_std_no_outliers*1e6} um')
    print(f'Lc = {np.round(Lc_mean_no_outliers*1e6)} \u2a72 {np.round(3*Lc_std_no_outliers*1e6)} um')

# Cc
if Charac_container and Cc_outliers.size!=0:
    Cc_std_no_outliers = np.std(Cc_no_outliers)
    Cc_mean_no_outliers = np.mean(Cc_no_outliers)
    print('--------------------------------------')
    print(f'Cc_mean = {Cc_mean_no_outliers} m/s')
    print(f'Cc_std = {Cc_std_no_outliers} m/s')
    print(f'Cc = {np.round(Cc_mean_no_outliers)} \u2a72 {np.round(3*Cc_std_no_outliers)} m/s')
print('--------------------------------------')
