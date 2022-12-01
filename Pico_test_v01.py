# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:05:05 2022

@author: Alberto
"""
import numpy as np
import matplotlib.pyplot as plt
import ctypes

import Pico5000alib as plib


#%% Arbitrary waveform to generate
# sine wave
_f0 = 5e6
_T0 = 1/_f0
waveformSize = 2**11
_t = np.linspace(0,_T0,waveformSize)
waveform = (np.sin(2*np.pi*_f0*_t)*32767).astype(np.int16) # temporary

# square pulse
# duty_cycle = 0.5
# waveform = np.concatenate([np.ones(int(waveformSize*duty_cycle)), np.zeros(int(waveformSize*(1-duty_cycle)))])*32767
# waveform[waveform==0] = -32768
# waveform = waveform.astype(np.int16)

#%% Parameters
num_bits = 12               # Number of bits to use (8, 12, 14, 15 or 16) - int
Fs = 125e6                  # Sampling frequency (Hz) - float

# Channel A setup
coupling_A = 'DC'           # Coupling of channel A ('AC' or 'DC') - str
voltage_range_A = '2V'      # Voltage range of channel A ('10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V', '50V' or 'MAX') - str
offset_A = 0                # Analog offset of channel A (in volts) - float
enabled_A = 0               # Enable (1) or disable (0) channel A - int

# Channel B setup
coupling_B = 'DC'           # Coupling of channel B ('AC' or 'DC') - str
voltage_range_B = '2V'      # Voltage range of channel B ('10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V', '50V' or 'MAX') - str
offset_B = 0                # Analog offset of channel B (in volts) - float
enabled_B = 1               # Enable (1) or disable (0) channel B - int

# Capture options
channels = 'B'           # 'A', 'B' or 'BOTH' - str
downsampling_ratio_mode = 0 # Downsampling ratio mode - int
downsampling_ratio = 0      # Downsampling ratio - int
time_indisposed = None      # For now, do not modify
lpReady = None              # For now, do not modify
pParameter = None           # For now, do not modify

# Trigger options
triggerChannel = 'B'        # 'A', 'B' or 'EXTERNAL' - str
triggerThreshold = 500      # Trigger threshold in mV - float
enabled_trigger = 1         # Enable (1) or disable (0) trigger - int
direction = 2               # Check API (2=rising) - int
delay = 0                   # time between trigger and first sample (s) - float
auto_Trigger = 1000         # starts a capture if no trigger event occurs within the specified ms - float
preTriggerSamples = 5500    # Number of samples to capture before the trigger - int
postTriggerSamples = 5500   # Number of samples to capture after the trigger - int

# Signal Generator options
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
    'triggerSource'         : 0,            # Source of trigger - int
    'extInThreshold'        : 0             # Trigger level for EXTERNAL trigger - int
}
ARBITRARY_SIGNAL_GENERATOR_DICT = {
    'offsetVoltage'         : 0,            # Offset Voltage (microvolts, uV) - int
    'pkToPk'                : 2_000_000,    # Peak-to-peak voltage (uV) - int
    'startFrequency'        : 1e5,          # Initial frequency of waveform - float
    'stopFrequency'         : 1e5,          # Final frequency before restarting or reversing sweep - float
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


#%% Initial check
# Find out device model (5000a)
plib.check_drivers()


#%% Start
# Create chandle and status ready for use
chandle = ctypes.c_int16()
status = {}

# Start pico
plib.start_pico5000a(chandle, status, num_bits)

#%% Setup
# Set up channel A
plib.setup_channel(chandle, status, 'A', coupling_A, voltage_range_A, offset_A, enabled_A)

# Set up channel B
plib.setup_channel(chandle, status, 'B', coupling_B, voltage_range_B, offset_B, enabled_B)

# Set up simple trigger
voltage_range = voltage_range_B if triggerChannel=='B' else voltage_range_A
plib.set_simpleTrigger(chandle, status, enabled_trigger, triggerChannel, voltage_range, triggerThreshold, direction, delay, auto_Trigger)

#%% Get timebase
# Get timebase
timebase, timeIntervalns, maxSamples = plib.get_timebase(chandle, status, Fs, preTriggerSamples + postTriggerSamples, segmentIndex=0)


#%% Generate signal (SINE)
if generate_arbitrary_signal:
    plib.generate_arbitrary_signal(chandle, status, **ARBITRARY_SIGNAL_GENERATOR_DICT)
    triggertype = ARBITRARY_SIGNAL_GENERATOR_DICT['triggertype']
    triggerSource = ARBITRARY_SIGNAL_GENERATOR_DICT['triggerSource']
elif generate_builtin_signal:
    plib.generate_builtin_signal(chandle, status, **BUILTIN_SIGNAL_GENERATOR_DICT)
    triggertype = BUILTIN_SIGNAL_GENERATOR_DICT['triggertype']
    triggerSource = BUILTIN_SIGNAL_GENERATOR_DICT['triggerSource']

trigger_sigGen = True if triggerSource==4 else False

#%% Capture data
# Run block capture
BUFFERS_DICT, cmaxSamples, triggerTimeOffset, triggerTimeOffsetUnits, time_indisposed = plib.capture(
    chandle, status, channels, (preTriggerSamples, postTriggerSamples), timebase,
    trigger_sigGen, triggertype, gate_time,
    downsampling=(downsampling_ratio_mode, downsampling_ratio), segment_index=0)

# Create time data
t = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)

if channels.upper() in ['A', 'BOTH']:
    # Get buffer
    bufferAMax = BUFFERS_DICT["bufferA0"][0]
    
    # convert ADC counts data to mV
    adc2mVChAMax = plib.adc2millivolts(chandle, status, bufferAMax, voltage_range_A)

    # Plot data
    plt.plot(t, adc2mVChAMax)
    
if channels.upper() in ['B', 'BOTH']:
    # Get buffer
    bufferBMax = BUFFERS_DICT["bufferB0"][0]
    
    # convert ADC counts data to mV
    adc2mVChBMax = plib.adc2millivolts(chandle, status, bufferBMax, voltage_range_B)

    # Plot data
    plt.plot(t, adc2mVChBMax)


# Plot data
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (mV)')
plt.show()

# Stop the scope
plib.stop_pico5000a(chandle, status)


#%% Capture rapid data: 10 segments
# Run rapid capture of 10 segments
BUFFERS_DICT, cmaxSamples, triggerTimeOffsets, triggerTimeOffsetUnits, time_indisposed, triggerInfo = plib.rapid_capture(
    chandle, status, channels, (preTriggerSamples, postTriggerSamples), timebase,
    10, trigger_sigGen, triggertype, gate_time,
    downsampling=(downsampling_ratio_mode, downsampling_ratio))

plib.print_triggerInfo(triggerInfo)

# Create time data
t = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value)

arrayAMax, arrayBMax, arrayAMin, arrayBMin = plib.get_data_from_buffersdict(chandle, status, voltage_range_A, voltage_range_B, BUFFERS_DICT)

means = np.zeros([4, len(t)])
for i, a in enumerate([arrayAMax, arrayBMax, arrayAMin, arrayBMin]):
    if len(a) != 0:
        means[i] = np.mean(a, axis=0)
    else:
        means[i] = np.full(len(t), None)

if channels.upper() in ['A', 'BOTH']:
    # Get buffer
    # bufferAMax0 = BUFFERS_DICT["bufferA0"][0]
    # bufferAMax1 = BUFFERS_DICT["bufferA1"][0]
    # bufferAMax2 = BUFFERS_DICT["bufferA2"][0]
    # bufferAMax3 = BUFFERS_DICT["bufferA3"][0]
    # bufferAMax4 = BUFFERS_DICT["bufferA4"][0]
    # bufferAMax5 = BUFFERS_DICT["bufferA5"][0]
    # bufferAMax6 = BUFFERS_DICT["bufferA6"][0]
    # bufferAMax7 = BUFFERS_DICT["bufferA7"][0]
    # bufferAMax8 = BUFFERS_DICT["bufferA8"][0]
    # bufferAMax9 = BUFFERS_DICT["bufferA9"][0]
    
    # convert ADC counts data to mV
    # adc2mVChAMax = plib.adc2millivolts(chandle, status, bufferAMax, voltage_range_A)

    # Plot data
    plt.plot(t, arrayAMax.T, c='grey')
    plt.plot(t, means[0], lw=2, c='k')
    
if channels.upper() in ['B', 'BOTH']:
    # Get buffer
    # bufferBMax0 = BUFFERS_DICT["bufferB0"][0]
    # bufferBMax1 = BUFFERS_DICT["bufferB1"][0]
    # bufferBMax2 = BUFFERS_DICT["bufferB2"][0]
    # bufferBMax3 = BUFFERS_DICT["bufferB3"][0]
    # bufferBMax4 = BUFFERS_DICT["bufferB4"][0]
    # bufferBMax5 = BUFFERS_DICT["bufferB5"][0]
    # bufferBMax6 = BUFFERS_DICT["bufferB6"][0]
    # bufferBMax7 = BUFFERS_DICT["bufferB7"][0]
    # bufferBMax8 = BUFFERS_DICT["bufferB8"][0]
    # bufferBMax9 = BUFFERS_DICT["bufferB9"][0]
    
    # # convert ADC counts data to mV
    # adc2mVChBMax0 = plib.adc2millivolts(chandle, status, bufferBMax0, voltage_range_B)
    # adc2mVChBMax1 = plib.adc2millivolts(chandle, status, bufferBMax1, voltage_range_B)
    # adc2mVChBMax2 = plib.adc2millivolts(chandle, status, bufferBMax2, voltage_range_B)
    # adc2mVChBMax3 = plib.adc2millivolts(chandle, status, bufferBMax3, voltage_range_B)
    # adc2mVChBMax4 = plib.adc2millivolts(chandle, status, bufferBMax4, voltage_range_B)
    # adc2mVChBMax5 = plib.adc2millivolts(chandle, status, bufferBMax5, voltage_range_B)
    # adc2mVChBMax6 = plib.adc2millivolts(chandle, status, bufferBMax6, voltage_range_B)
    # adc2mVChBMax7 = plib.adc2millivolts(chandle, status, bufferBMax7, voltage_range_B)
    # adc2mVChBMax8 = plib.adc2millivolts(chandle, status, bufferBMax8, voltage_range_B)
    # adc2mVChBMax9 = plib.adc2millivolts(chandle, status, bufferBMax9, voltage_range_B)

    # B = np.array([adc2mVChBMax0,
    #               adc2mVChBMax1,
    #               adc2mVChBMax2,
    #               adc2mVChBMax3,
    #               adc2mVChBMax4,
    #               adc2mVChBMax5,
    #               adc2mVChBMax6,
    #               adc2mVChBMax7,
    #               adc2mVChBMax8,
    #               adc2mVChBMax9])
    # avg_B = np.mean(B, axis=0)

    # Plot data
    plt.plot(t, arrayBMax.T, c='grey')
    plt.plot(t, means[1], lw=2, c='k')


# Plot data
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (mV)')
plt.show()

# Stop the scope
plib.stop_pico5000a(chandle, status)


#%% Close
# Close the unit
plib.close_pico5000a(chandle, status)