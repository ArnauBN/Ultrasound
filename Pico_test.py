# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:05:05 2022

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>
"""
import numpy as np
import matplotlib.pyplot as plt

import src.ultrasound as US
from src.devices import pico5000a

#%% Parameters
num_bits = 15               # Number of bits to use (8, 12, 14, 15 or 16) - int
Fs = 125e6                  # Desired sampling frequency (Hz) - float


# ------------------
# Arbitrary Waveform
# ------------------
waveform_f0 = 5e6           # Center Frequency of waveform (Hz) - float
waveformSize = 2**10        # Waveform length (power of 2, max=2**15) - int

pulse = US.GC_MakePulse(Param='frequency', ParamVal=waveform_f0, SignalPolarity=2, Fs=Fs)
pulse = pulse[1:-1]*32767
waveform = US.zeroPadding(pulse, waveformSize)
waveform = waveform.astype(np.int16)
waveform_t = np.arange(0,waveformSize)/Fs

# win = scsig.get_window(('gaussian', 15), waveformSize)
# waveform = (np.sin(2*np.pi*waveform_f0*waveform_t)*32767*win).astype(np.int16)

# Computation of the FFT of the waveform
N = int(np.ceil(np.log2(np.abs(waveformSize)))) + 1 # next power of 2 (+1)
nfft = 2**N # Number of FFT points (power of 2)
freq = np.linspace(0, Fs/2, nfft//2)
FFTwaveform = np.fft.fft(waveform-np.mean(waveform), nfft)/nfft
FFTwaveform = FFTwaveform[:nfft//2]


# ---------------
# Channel A setup
# ---------------
coupling_A = 'DC'           # Coupling of channel A ('AC' or 'DC') - str
voltage_range_A = '10mV'      # Voltage range of channel A ('10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V', '50V' or 'MAX') - str
offset_A = 0                # Analog offset of channel A (in volts) - float
enabled_A = 1               # Enable (1) or disable (0) channel A - int


# ---------------
# Channel B setup
# ---------------
coupling_B = 'DC'           # Coupling of channel B ('AC' or 'DC') - str
voltage_range_B = '5V'      # Voltage range of channel B ('10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V', '50V' or 'MAX') - str
offset_B = 0                # Analog offset of channel B (in volts) - float
enabled_B = 1               # Enable (1) or disable (0) channel B - int


# ---------------
# Capture options
# ---------------
channels = 'BOTH'           # 'A', 'B' or 'BOTH' - str
nSegments = 20              # Number of traces to capture and average to reduce noise - int
downsampling_ratio_mode = 0 # Downsampling ratio mode - int
downsampling_ratio = 0      # Downsampling ratio - int


# ---------------
# Trigger options
# ---------------
triggerChannel = 'A'        # 'A', 'B' or 'EXTERNAL' - str
triggerThreshold = 1000      # Trigger threshold in mV - float
enabled_trigger = 1         # Enable (1) or disable (0) trigger - int
direction = 2               # Check API (2=rising) - int
delay = 0                   # time between trigger and first sample (samples) - int
auto_Trigger = 1000         # starts a capture if no trigger event occurs within the specified ms - float
preTriggerSamples = 1000    # Number of samples to capture before the trigger - int
postTriggerSamples = 15_000   # Number of samples to capture after the trigger - int


# ------------------------
# Signal Generator options
# ------------------------
generate_builtin_signal = False      # If True, generate builtin signal - bool
generate_arbitrary_signal = True   # If True, generate arbitrary signal (has priority over builtin) - bool
gate_time = 1000                    # Gate time in milliseconds (only used for gated triggers) - float
BUILTIN_SIGNAL_GENERATOR_DICT = {
    'offsetVoltage'         : 0,            # Offset Voltage (microvolts, uV) - int
    'pkToPk'                : 2_000_000,    # Peak-to-peak voltage (uV) - int
    'wavetype'              : 0,            # Builtin type of waveform - int
    'startFrequency'        : 5e6,          # Frequency (Hz) - float
    'stopFrequency'         : 5e6,          # Stop Frequency of the sweep (Hz) - float
    'increment'             : 0,            # Freq. increment of the sweep (Hz) - float
    'dwellTime'             : 0,            # Time for which the sweep stays at each frequency (s) - float
    'sweepType'             : 0,            # Type of sweep - int
    'shots'                 : 20,            # Number of cycles per trigger. If 0, do sweeps - int
    'sweeps'                : 0,            # Number of sweeps per trigger. If 0, do shots - int
    'triggertype'           : 0,            # Type of trigger - int
    'triggerSource'         : 4,            # Source of trigger - int
    'extInThreshold'        : 0             # Trigger level for EXTERNAL trigger - int
}
ARBITRARY_SIGNAL_GENERATOR_DICT = {
    'offsetVoltage'         : 0,            # Offset Voltage (microvolts, uV) - int
    'pkToPk'                : 4_000_000,    # Peak-to-peak voltage (uV) - int
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
    'triggerSource'         : 1,            # Source of trigger - int
    'extInThreshold'        : 0             # Trigger level for EXTERNAL trigger - int
}


#%% Plot arbitrary waveform
plt.figure('Waveform fft')
# plt.plot(freq*1e-6, 20*np.log10(np.abs(FFTwaveform)))
plt.plot(freq*1e-6, np.abs(FFTwaveform))
plt.xlabel('Frequency (MHz)')
plt.ylabel('FFT Magnitude')
plt.xlim([0,15])

plt.figure('Waveform')
plt.plot(waveform_t*1e9, waveform)
plt.ylabel('Sample count')
plt.xlabel('Sample')
plt.title('Waveform to generate')
plt.tight_layout()


#%% Initial check
# Find out device model (5000a)
pico5000a.check_drivers()


#%% Start
# Create chandle and status ready for use
pico = pico5000a.Pico(num_bits)

#%% Setup
# Set up channel A
pico.setup_channel('A', coupling_A, voltage_range_A, offset_A, enabled_A)

# Set up channel B
pico.setup_channel('B', coupling_B, voltage_range_B, offset_B, enabled_B)

# Set up simple trigger
voltage_range = voltage_range_B if triggerChannel=='B' else voltage_range_A
pico.set_simpleTrigger(enabled_trigger, triggerChannel, voltage_range, triggerThreshold, direction, delay, auto_Trigger)

#%% Get timebase
# Get timebase
timebase, timeIntervalns, maxSamples = pico.get_timebase(Fs, preTriggerSamples + postTriggerSamples, segmentIndex=0)
Real_Fs = 1e9/(2**timebase) # Hz

if generate_arbitrary_signal and ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency'] == ARBITRARY_SIGNAL_GENERATOR_DICT['stopFrequency']:
    pulse_freq = Real_Fs/waveformSize
    if pulse_freq != ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency']:
        ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency'] = pulse_freq
        ARBITRARY_SIGNAL_GENERATOR_DICT['stopFrequency'] = pulse_freq
        print('Frequency of the arbitrary waveform changed to {pulse_freq} Hz.')

#%% Generate signal (SINE)
if generate_arbitrary_signal:
    pico.generate_arbitrary_signal(**ARBITRARY_SIGNAL_GENERATOR_DICT)
    triggertype = ARBITRARY_SIGNAL_GENERATOR_DICT['triggertype']
    triggerSource = ARBITRARY_SIGNAL_GENERATOR_DICT['triggerSource']
elif generate_builtin_signal:
    pico.generate_builtin_signal(**BUILTIN_SIGNAL_GENERATOR_DICT)
    triggertype = BUILTIN_SIGNAL_GENERATOR_DICT['triggertype']
    triggerSource = BUILTIN_SIGNAL_GENERATOR_DICT['triggerSource']

trigger_sigGen = True if triggerSource==4 else False

#%% Capture data
# Run block capture
BUFFERS_DICT, cmaxSamples, triggerTimeOffset, triggerTimeOffsetUnits, time_indisposed = pico.capture(
    channels, (preTriggerSamples, postTriggerSamples), timebase,
    trigger_sigGen, triggertype, gate_time,
    downsampling=(downsampling_ratio_mode, downsampling_ratio), segment_index=0)

# Create time data
t = np.linspace(0, (cmaxSamples - 1) * timeIntervalns, cmaxSamples)

fig, axs = plt.subplots(2)

if channels.upper() in ['A', 'BOTH']:
    # Get buffer
    bufferAMax = BUFFERS_DICT["bufferA0"][0]
    
    # convert ADC counts data to mV
    adc2mVChAMax = pico.adc2millivolts(bufferAMax, voltage_range_A)

    # Plot data
    axs[0].plot(t*1e-3, adc2mVChAMax)
    
if channels.upper() in ['B', 'BOTH']:
    # Get buffer
    bufferBMax = BUFFERS_DICT["bufferB0"][0]
    
    # convert ADC counts data to mV
    adc2mVChBMax = pico.adc2millivolts(bufferBMax, voltage_range_B)

    # Plot data
    axs[1].plot(t*1e-3, adc2mVChBMax)


# Plot config
# axs[0].set_ylim([-plib.str2V(voltage_range_A)*1e3, plib.str2V(voltage_range_A)*1e3])
# axs[1].set_ylim([-plib.str2V(voltage_range_B)*1e3, plib.str2V(voltage_range_B)*1e3])
axs[1].set_xlabel('Time (us)')
axs[0].set_ylabel('Voltage (mV)')
axs[1].set_ylabel('Voltage (mV)')
axs[0].set_title('A')
axs[1].set_title('B')
plt.tight_layout()
plt.show()

# Stop the scope
pico.stop()

#%% fft
samples = preTriggerSamples + postTriggerSamples
N = int(np.ceil(np.log2(np.abs(samples)))) + 1 # next power of 2 (+1)
nfft = 2**N # Number of FFT points (power of 2)

real_Fs = pico.timebase2fs(timebase)
freq = np.linspace(0, real_Fs/2, nfft//2)
FFTmeanB = np.fft.fft(adc2mVChBMax-np.mean(adc2mVChBMax), nfft)/nfft
FFTmeanB = FFTmeanB[:nfft//2]

plt.figure('fft')
plt.plot(freq*1e-6, np.abs(FFTmeanB))
plt.xlabel('Frequency (MHz)')
plt.ylabel('FFT Magnitude (mV)')
# plt.xlim([0,10])


#%% Capture rapid data: nSegments
# Run rapid capture of nSegments
BUFFERS_DICT, cmaxSamples, triggerTimeOffsets, triggerTimeOffsetUnits, time_indisposed, triggerInfo = pico.rapid_capture(
    channels, (preTriggerSamples, postTriggerSamples), timebase,
    nSegments, trigger_sigGen, triggertype, gate_time,
    downsampling=(downsampling_ratio_mode, downsampling_ratio))

pico.print_triggerInfo(triggerInfo)

# Create time data
t = np.linspace(0, (cmaxSamples - 1) * timeIntervalns, cmaxSamples)

arrayAMax, arrayBMax, arrayAMin, arrayBMin, means = pico.get_data_from_buffersdict(voltage_range_A, voltage_range_B, BUFFERS_DICT)

# Stop the scope
pico.stop()


#%% Plot data
fig, axs = plt.subplots(2)
if channels.upper() in ['A', 'BOTH']:
    # Plot data
    # plt.plot(t*1e-3, arrayAMax.T, c='grey')
    # plt.plot(t*1e-3, arrayAMax.T)
    axs[0].plot(t*1e-3, means[0], lw=2)
    pass
if channels.upper() in ['B', 'BOTH']:
    # Plot data
    # plt.plot(t*1e-3, arrayBMax.T, c='grey')
    axs[1].plot(t*1e-3, means[1], lw=2)
    pass

# Plot config
# axs[0].set_ylim([-plib.str2V(voltage_range_A)*1e3, plib.str2V(voltage_range_A)*1e3])
axs[1].set_ylim([-pico5000a.str2V(voltage_range_B)*1e3, pico5000a.str2V(voltage_range_B)*1e3])
axs[1].set_xlabel('Time (us)')
axs[0].set_ylabel('Voltage (mV)')
axs[1].set_ylabel('Voltage (mV)')
axs[0].set_title('A')
axs[1].set_title('B')
plt.tight_layout()
plt.show()


#%% fft
samples = preTriggerSamples + postTriggerSamples
N = int(np.ceil(np.log2(np.abs(samples)))) + 1 # next power of 2 (+1)
nfft = 2**N # Number of FFT points (power of 2)

real_Fs = pico.timebase2fs(timebase)
freq = np.linspace(0, real_Fs/2, nfft//2)
FFTmeanB = np.fft.fft(means[1]-np.mean(means[1]), nfft)/nfft
FFTmeanB = FFTmeanB[:nfft//2]

plt.figure('fft')
plt.plot(freq*1e-6, np.abs(FFTmeanB))
plt.xlabel('Frequency (MHz)')
plt.ylabel('FFT Magnitude (mV)')
# plt.xlim([0,10])


#%% Close
# Close the unit
pico.close()