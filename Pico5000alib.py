# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 18:08:54 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

Dependencies
============
- PicoSDK: <https://www.picotech.com/downloads/_lightbox/pico-software-development-kit-64bit>
- Python wrappers for PicoSDK: <https://github.com/picotech/picosdk-python-wrappers>
- numpy
- pandas
- ctypes
- time

CODES
=====
Wave type
---------
0 - SINE

1 - SQUARE

2 - TRIANGLE

3 - RAMP_UP

4 - RAMP_DOWN

5 - SINC

6 - GAUSSIAN

7 - HALF_SINE

8 - DC_VOLTAGE

9 - PWM

10 - WHITE_NOISE

11 - PRBS

12 - ARBITRARY


Sweep type
----------
0 - UP

1 - DOWN

2 - UPDOWN

3 - DOWNUP


Trigger type
------------
0 - RISING

1 - FALLING

2 - GATE_HIGH

3 - GATE_LOW


Trigger source
--------------
0 - NONE

1 - SCOPE

2 - AUX

3 - EXT

4 - SOFTWARE


Trigger Time Offset Units
-------------------------
0 - FS

1 - PS

2 - NS

3 - US

4 - MS

5 - S


Threshold direction
-------------------
0 - ABOVE

1 - BELOW

2 - RISING

3 - FALLING

4 - RISING_OR_FALLING

5 - ABOVE_LOWER

6 - BELOW_LOWER

7 - RISING_LOWER

8 - FALLING_LOWER

9 - POSITIVE_RUNT

10 - NEGATIVE_RUNT


Downsampling ratio mode
-----------------------
0 - NONE

1 - AGGREGATE

2 - DECIMATE

3 - AVERAGE

4 - DISTRIBUTION

"""
from picosdk.discover import find_all_units
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import assert_pico_ok, mV2adc, adc2mV
import numpy as np
import pandas as pd
import ctypes
import time


#%% ========== PARSERS ==========
def _parseResolution(num_bits):
    '''
    Returns the resolution code corresponding to the specified number of bits.

    Parameters
    ----------
    num_bits : int or str
        number of bits. Available values are 8, 12, 14, 15 and 16.

    Raises
    ------
    ValueError
        Wrong number of bits. Accepted values are: 8, 12, 14, 15 and 16.

    Returns
    -------
    resolution : ctypes.c_int32
        The resolution code.

    Arnau, 29/11/2022
    '''
    key = 'PS5000A_DR_' + str(num_bits) + 'BIT'
    if key not in ps.PS5000A_DEVICE_RESOLUTION:
        raise ValueError('Wrong number of bits. Accepted values are: 8, 12, 14, 15 and 16.')
    resolution = ps.PS5000A_DEVICE_RESOLUTION[key]
    return resolution

def _parseChannel(channel):
    '''
    Returns the channel code corresponding to the specified string.

    Parameters
    ----------
    channel : str
        Port. Accepted values are 'A', 'B' and 'EXTERNAL'.

    Raises
    ------
    ValueError
        Wrong channel. Accepted values are: 'A', 'B' and 'EXTERNAL'.

    Returns
    -------
    ch : ctypes.c_int32
        The channel code.

    Arnau, 29/11/2022
    '''
    key = 'PS5000A_CHANNEL_' + channel.upper()
    if key not in ps.PS5000A_CHANNEL:
        raise ValueError("Wrong channel. Accepted values are: 'A' and 'B'.")
    ch = ps.PS5000A_CHANNEL[key]
    return ch

def _parseCoupling(coupling):
    '''
    Returns the coupling code corresponding to the specified string.

    Parameters
    ----------
    coupling : str
        Coupling type. Accepted values are 'DC' and 'AC'.

    Raises
    ------
    ValueError
        Wrong coupling. Accepted values are: 'DC' and 'AC'.

    Returns
    -------
    coupling_type : ctypes.c_int32
        The coupling code.

    Arnau, 29/11/2022
    '''
    key = 'PS5000A_' + coupling.upper()
    if key not in ps.PS5000A_COUPLING:
        raise ValueError("Wrong coupling. Accepted values are: 'DC' and 'AC'.")
    coupling_type = ps.PS5000A_COUPLING[key]
    return coupling_type

def _parseVoltageRange(voltage_range):
    '''
    Returns the voltage range code corresponding to the specified string.

    Parameters
    ----------
    voltage_range : str
        Voltage range. Accepted values are '10mV', '20mV', '50mV', '100mV',
        '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.

    Raises
    ------
    ValueError
        Wrong voltage range. Accepted values are: '10mV', '20mV', '50mV',
        '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.

    Returns
    -------
    chRange : ctypes.c_int32
        The voltage range code.

    Arnau, 29/11/2022
    '''
    key = 'PS5000A_MAX_RANGES' if voltage_range.upper() == 'MAX' else 'PS5000A_' + voltage_range.upper()
    if key not in ps.PS5000A_RANGE:
        raise ValueError("Wrong voltage range. Accepted values are:" \
                         "'10mV', '20mV', '50mV', '100mV', '200mV', '500mV', " \
                         "'1V', '2V', '5V', '10V', '20V', '50V' and 'MAX'.")
    chRange = ps.PS5000A_RANGE["PS5000A_20V"]
    return chRange

#%% ========== DEVICE ==========
def check_drivers():
    '''
    Finds all Pico units and prints their info.

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    scopes = find_all_units()
    for scope in scopes:
        print(scope.info)
        scope.close()

def start_pico5000a(chandle, status, num_bits):
    '''
    Open 5000a PicoScope and set the number of bits.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    num_bits : int or str
        number of bits. Available values are 8, 12, 14, 15 and 16.

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    # Set resolution
    resolution = _parseResolution(num_bits)
    
    # Returns handle to chandle for use in future API functions
    status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(chandle), None, resolution)
    
    try:
        assert_pico_ok(status["openunit"])
    except: # PicoNotOkError:
    
        powerStatus = status["openunit"]
    
        if powerStatus == 286:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        elif powerStatus == 282:
            status["changePowerSource"] = ps.ps5000aChangePowerSource(chandle, powerStatus)
        else:
            raise
    
        assert_pico_ok(status["changePowerSource"])

def stop_pico5000a(chandle, status):
    '''
    Stops the scope device from sampling data.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    # Stop the scope
    status["stop"] = ps.ps5000aStop(chandle)
    assert_pico_ok(status["stop"])
    print(f'Stopped with status:\n{status}')

def close_pico5000a(chandle, status):
    '''
    Closes the unit, releasing the handle.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    # Close the unit
    status["close"] = ps.ps5000aCloseUnit(chandle)
    assert_pico_ok(status["close"])
    print(f'Closed with status:\n{status}')

#%% ========== FUNCTIONS ==========
def set_resolution(chandle, status, num_bits):
    '''
    Sets the resolution given the number of bits to use.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    num_bits : int or str
        number of bits. Available values are 8, 12, 14, 15 and 16.

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    resolution = _parseResolution(num_bits)
    status["setDeviceResolution"] = ps.ps5000aSetDeviceResolution(chandle, resolution)
    assert_pico_ok(status["setDeviceResolution"])

def setup_channel(chandle, status, channel, coupling, voltage_range, offset, enabled):
    '''
    Setup specified channel.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    channel : str
        Channel to setup. Accepted values are 'A' and 'B'.
    coupling : str
        Coupling type. Accepted values are 'DC' and 'AC'.
    voltage_range : str
        Voltage range. Accepted values are '10mV', '20mV', '50mV', '100mV',
        '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.
    offset : float
        Analog offset in volts.
    enabled : int
        Enable channel (1) or disable channel (0).

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    # Set channel to use
    ch = _parseChannel(channel)
    
    # Coupling
    coupling_type = _parseCoupling(coupling)
   
    # Voltage range
    chRange = _parseVoltageRange(voltage_range)
    
    # Setup channel
    status["setChA"] = ps.ps5000aSetChannel(chandle, ch, enabled, coupling_type, chRange, offset)
    assert_pico_ok(status["setChA"])

def fs2timebase(Fs):
    '''
    Convert sampling frequency to timebase.

    Parameters
    ----------
    Fs : float
        Sampling frequency in Hz.

    Returns
    -------
    n : int
        Timebase so that 2**n is the allowed sampling period in nanoseconds.

    Arnau, 29/11/2022
    '''
    Ts = 1e9/Fs # ns
    n = int(np.log2(Ts))
    if 2**n < Ts:
        n += 1
    return n

def timebase2fs(n):
    '''
    Convert timebase to sampling frequency.

    Parameters
    ----------
    n : int
        Timebase so that 2**n is the sampling period in nanoseconds.

    Returns
    -------
    Fs : float
        Sampling frequency in Hz.

    Arnau, 29/11/2022
    '''
    Ts = 2**n # ns
    Fs = 1e9/Ts # Hz
    return Fs

def millivolts2adc(chandle, status, v, voltage_range):
    '''
    Converts millivotls to adc count.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    v : float
        Millivolts to convert to adc count.
    voltage_range : str
        Voltage range. Accepted values are '10mV', '20mV', '50mV', '100mV',
        '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.

    Returns
    -------
    v_adc : int
        The corresponding adc count.

    Arnau, 29/11/2022
    '''
    parsed_chRange = _parseVoltageRange(voltage_range)
    maxADC = get_maxADC(chandle, status)
    v_adc = int(mV2adc(v, parsed_chRange, maxADC))
    return v_adc

def adc2millivolts(chandle, status, bufferADC, voltage_range):
    '''
    Converts a raw adc count buffer to millivolts.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    bufferADC : array of ctypes.c_int16
        The buffered adc values to convert to millivolts.
    voltage_range : str
        Voltage range. Accepted values are '10mV', '20mV', '50mV', '100mV',
        '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.

    Returns
    -------
    buffermV : ndarray
        The buffer in millivolts.

    Arnau, 30/11/2022
    '''
    parsed_chRange = _parseVoltageRange(voltage_range)
    maxADC = get_maxADC(chandle, status)
    buffermV = adc2mV(bufferADC, parsed_chRange, maxADC)
    return np.array(buffermV)

def print_triggerInfo(triggerInfo):
    '''
    Print trigger information given the PS5000A_TRIGGER_INFO object.

    Parameters
    ----------
    triggerInfo : ps.PS5000A_TRIGGER_INFO
        Trigger information to be printed.

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    lst = []
    for i in triggerInfo:
        lst.append([i.segmentIndex, i.status, i.triggerIndex, i.triggerTime, i.timeUnits, i.timeStampCounter])
    df = pd.DataFrame(lst, columns=['segmentIndex', 'PICO_STATUS', 'triggerIndex', 'triggerTime', 'timeUnits', 'timeStampCounter'])
    print(df.to_string(index=False))
    
#%% ========== CAPTURE DATA ==========
def capture(chandle, status, channels, samples, timebase, trigger_sigGen, sigGen_triggertype, sigGen_gate_time, downsampling=(0,0), segment_index=0):
    '''
    Capture Block.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    channels : str or tuple
        Channels to setup. Accepted values are 'A', 'B' and 'both'.
    samples : int or tuple of ints
        If tuple, the number of samples before and after the trigger. If not a 
        tuple, the total number of samples to be captured (50% before and 50%
        after the trigger)
    timebase : int
        Timebase so that 2**timebase is the sampling period in nanoseconds.
    trigger_sigGen : bool
        If True, do a software trigger on the signal generator.
    sigGen_triggertype : int
        Code of the type of trigger to use. Available types:
            0 - RISING
            1 - FALLING
            2 - GATE_HIGH
            3 - GATE_LOW
    sigGen_gate_time : float, optional
        The gate time in milliseconds for trigger types 2 or 3 (gated triggers).
        Unused for trigger types 0 or 1 (edge triggers). The default is 1000.
    downsampling : tuple
        Tuple of 2 ints: (ratio_mode, ratio). The default is (0,0).
    segment_index : int, optional
        Specifies memory segment to use. The default is 0.
        
    Returns
    -------
    BUFFERS_DICT : dict
        dictionary containing all buffered data. Sintax of keys and values are
        as follows :
            
                BUFFERS_DICT[f'bufferX{segment_index}'] = [bufferMax, bufferMin]
            
            where X is A or B. The bufferMin is only used for downsampling.
    cmaxSamples : int
        The actual number of samples.
    triggerTimeOffset : int
        Time offset of waveform with respect to trigger.
    triggerTimeOffsetUnits : int
        Units of the time offset.
            0 - FS
            1 - PS
            2 - NS
            3 - US
            4 - MS
            5 - S
    time_indisposed : int
        Time, in milliseconds, that the scope spent collecting samples.
    
    Arnau, 29/11/2022
    '''
    if isinstance(samples, tuple):
        maxsamples = samples[0] + samples[1]
        preTriggerSamples, postTriggerSamples = samples
    else:
        maxsamples = samples
        preTriggerSamples = samples//2
        postTriggerSamples = samples - preTriggerSamples
    
    time_indisposed = ctypes.c_int32()
    status["runBlock"] = ps.ps5000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, 
                                            timebase, ctypes.byref(time_indisposed), segment_index, None, None)
    assert_pico_ok(status["runBlock"])
    
    # Create buffers ready for assigning pointers for data collection
    BUFFERS_DICT = {} # BUFFERS_DICT[bufferX0] = (bufferMaxX0, bufferMinX0), bufferMinX0 is used for downsampling
    if channels.upper()=='A' or channels.upper()=='BOTH':
        key = f"bufferA{segment_index}"
        BUFFERS_DICT[key] = [(ctypes.c_int16 * maxsamples)(), (ctypes.c_int16 * maxsamples)()]

        # Set data buffer location for data collection from channel A
        chA = _parseChannel('A')
        status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(chandle, chA, ctypes.byref(BUFFERS_DICT[key][0]), ctypes.byref(BUFFERS_DICT[key][1]), maxsamples, segment_index, downsampling[0])
        assert_pico_ok(status["setDataBuffersA"])
    
    if channels.upper()=='B' or channels.upper()=='BOTH':
        key = f"bufferB{segment_index}"
        BUFFERS_DICT[key] = [(ctypes.c_int16 * maxsamples)(), (ctypes.c_int16 * maxsamples)()]
        
        # Set data buffer location for data collection from channel B
        chB = _parseChannel('B')
        status["setDataBuffersB"] = ps.ps5000aSetDataBuffers(chandle, chB, ctypes.byref(BUFFERS_DICT[key][0]), ctypes.byref(BUFFERS_DICT[key][1]), maxsamples, segment_index, downsampling[0])
        assert_pico_ok(status["setDataBuffersB"])

    overflow = ctypes.c_int16() # create overflow loaction
    cmaxSamples = ctypes.c_int32(maxsamples) # create converted type maxSamples

    if trigger_sigGen:
        trigger_generator(chandle, status, sigGen_triggertype, sigGen_gate_time)

    # Check for data collection to finish using ps5000aIsReady
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))
    
    # Retrieve data from scope to buffers assigned above
    status["getValues"] = ps.ps5000aGetValues(chandle, 0, ctypes.byref(cmaxSamples), downsampling[1], downsampling[0], segment_index, ctypes.byref(overflow))
    assert_pico_ok(status["getValues"])

    # Get time offsets for each waveform
    triggerTimeOffset = ctypes.c_int64()
    triggerTimeOffsetUnits = ctypes.c_char()
    status["GetTriggerTimeOffset64"] = ps.ps5000aGetTriggerTimeOffset64(chandle, ctypes.byref(triggerTimeOffset), ctypes.byref(triggerTimeOffsetUnits), 0)
    assert_pico_ok(status["GetTriggerTimeOffset64"])   
    
    return BUFFERS_DICT, cmaxSamples.value, triggerTimeOffset.value, int.from_bytes(triggerTimeOffsetUnits.value, 'big'), time_indisposed.value

def rapid_capture(chandle, status, channels, samples, timebase, nSegments, trigger_sigGen, sigGen_triggertype, sigGen_gate_time, downsampling=(0,0)):
    '''
    Rapid capture of {nSegments} segments. The first segment index is always 0.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    channels : str or tuple
        Channels to setup. Accepted values are 'A', 'B' and 'both'.
    samples : int or tuple of ints
        If tuple, the number of samples before and after the trigger. If not a 
        tuple, the total number of samples to be captured (50% before and 50%
        after the trigger)
    timebase : int
        Timebase so that 2**timebase is the sampling period in nanoseconds.
    nSegments : int
        Number of segments to capture.
    trigger_sigGen : bool
        If True, do a software trigger on the signal generator for every 
        segment.
    sigGen_triggertype : int
        Code of the type of trigger to use. Available types:
            0 - RISING
            1 - FALLING
            2 - GATE_HIGH
            3 - GATE_LOW
    sigGen_gate_time : float, optional
        The gate time in milliseconds for trigger types 2 or 3 (gated triggers).
        Unused for trigger types 0 or 1 (edge triggers). The default is 1000.
    downsampling : tuple
        Tuple of 2 ints: (ratio_mode, ratio). The default is (0,0).

    Returns
    -------
    BUFFERS_DICT : dict
        dictionary containing all buffered data. Sintax of keys and values are
        as follows :
            
                BUFFERS_DICT[f'bufferX{i}'] = [bufferMax, bufferMin]
            
            where X is A or B and i refers to each segment index. The bufferMin
            is only used for downsampling.
    cmaxSamples : int
        The actual number of samples.
    triggerTimeOffsets : ndarray of int
        Time offset of every segment captured.
    triggerTimeOffsetUnits : int
        Units of the TriggerTimeOffsets vector.
            0 - FS
            1 - PS
            2 - NS
            3 - US
            4 - MS
            5 - S
    time_indisposed : int
        Time, in milliseconds, that the scope spent collecting samples.
    triggerInfo : array of ps.PS5000A_TRIGGER_INFO
        Trigger information of every segment.
    
    Arnau, 29/11/2022
    '''
    if isinstance(samples, tuple):
        maxsamples = samples[0] + samples[1]
        preTriggerSamples, postTriggerSamples = samples
    else:
        maxsamples = samples
        preTriggerSamples = samples//2
        postTriggerSamples = samples - preTriggerSamples
    
    overflow = ctypes.c_int16() # Creates a overlow location for data
    cmaxSamples = ctypes.c_int32(maxsamples) # Creates converted types maxsamples
    
    status["MemorySegments"] = ps.ps5000aMemorySegments(chandle, nSegments, ctypes.byref(cmaxSamples))
    assert_pico_ok(status["MemorySegments"])
    
    # sets number of captures
    status["SetNoOfCaptures"] = ps.ps5000aSetNoOfCaptures(chandle, nSegments)
    assert_pico_ok(status["SetNoOfCaptures"])

    # Starts the block capture
    time_indisposed = ctypes.c_int32()
    status["runblock"] = ps.ps5000aRunBlock(chandle, preTriggerSamples, postTriggerSamples, 
                                            timebase, ctypes.byref(time_indisposed), 0, None, None)
    assert_pico_ok(status["runblock"])
    
    BUFFERS_DICT = {} # BUFFERS_DICT[bufferXi] = (bufferMaxXi, bufferMinXi), bufferMinXi is used for downsampling
    if channels.upper()=='A' or channels.upper()=='BOTH':
        chA = _parseChannel('A')
        for i in range(nSegments):
            # Create buffers ready for assigning pointers for data collection
            BUFFERS_DICT[f"bufferA{i}"] = [(ctypes.c_int16 * maxsamples)(), (ctypes.c_int16 * maxsamples)()]
            
            # Setting the data buffer location for data collection from channel chA
            status["SetDataBuffers"] = ps.ps5000aSetDataBuffers(chandle, chA, 
                                    ctypes.byref(BUFFERS_DICT[f"bufferA{i}"][0]), 
                                    ctypes.byref(BUFFERS_DICT[f"bufferA{i}"][1]), 
                                    maxsamples, i, downsampling[0])
            assert_pico_ok(status["SetDataBuffers"])
    
    if channels.upper()=='B' or channels.upper()=='BOTH':
        chB = _parseChannel('B')
        for i in range(nSegments):
            # Create buffers ready for assigning pointers for data collection
            BUFFERS_DICT[f"bufferB{i}"] = [(ctypes.c_int16 * maxsamples)(), (ctypes.c_int16 * maxsamples)()]
            
            # Setting the data buffer location for data collection from channel chA
            status["SetDataBuffers"] = ps.ps5000aSetDataBuffers(chandle, chB, 
                                    ctypes.byref(BUFFERS_DICT[f"bufferB{i}"][0]), 
                                    ctypes.byref(BUFFERS_DICT[f"bufferB{i}"][1]), 
                                    maxsamples, i, downsampling[0])
            assert_pico_ok(status["SetDataBuffers"])
                
    overflow = (ctypes.c_int16 * nSegments)() # Creates an overlow location for data
    cmaxSamples = ctypes.c_int32(maxsamples) # Creates converted types maxsamples
    
    if trigger_sigGen:
        for _ in range(nSegments):
            trigger_generator(chandle, status, sigGen_triggertype, sigGen_gate_time)
    
    # Check for data collection to finish using ps5000aIsReady
    ready = ctypes.c_int16(0)
    check = ctypes.c_int16(0)
    while ready.value == check.value:
        status["isReady"] = ps.ps5000aIsReady(chandle, ctypes.byref(ready))
    
    status["GetValuesBulk"] = ps.ps5000aGetValuesBulk(chandle, ctypes.byref(cmaxSamples), 0, nSegments-1, downsampling[1], downsampling[0], ctypes.byref(overflow))
    assert_pico_ok(status["GetValuesBulk"])

    # Get time offsets for each waveform
    triggerTimeOffsets = (ctypes.c_int64*nSegments)()
    triggerTimeOffsetUnits = ctypes.c_char()
    status["GetValuesTriggerTimeOffsetBulk64"] = ps.ps5000aGetValuesTriggerTimeOffsetBulk64(chandle, ctypes.byref(triggerTimeOffsets), ctypes.byref(triggerTimeOffsetUnits), 0, nSegments-1)
    assert_pico_ok(status["GetValuesTriggerTimeOffsetBulk64"])

    # Create array of ps.PS5000A_TRIGGER_INFO for each memory segment
    triggerInfo = (ps.PS5000A_TRIGGER_INFO*nSegments) ()
    status["GetTriggerInfoBulk"] = ps.ps5000aGetTriggerInfoBulk(chandle, ctypes.byref(triggerInfo), 0, nSegments-1)
    assert_pico_ok(status["GetTriggerInfoBulk"])

    return BUFFERS_DICT, cmaxSamples.value, np.array(list(map(int,triggerTimeOffsets))), int.from_bytes(triggerTimeOffsetUnits.value, 'big'), time_indisposed.value, triggerInfo

#%% ========== GETS ==========
def get_maxADC(chandle, status):
    '''
    Find maximum ADC count value

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.

    Returns
    -------
    maxADC : ctypes.c_int16()
        The maximum ADC count value.

    Arnau, 29/11/2022
    '''
    maxADC = ctypes.c_int16()
    status["maximumValue"] = ps.ps5000aMaximumValue(chandle, ctypes.byref(maxADC))
    assert_pico_ok(status["maximumValue"])
    return maxADC

def get_minADC(chandle, status):
    '''
    Find minimum ADC count value

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.

    Returns
    -------
    minADC : ctypes.c_int16()
        The minimum ADC count value.

    Arnau, 29/11/2022
    '''
    minADC = ctypes.c_int16()
    status["minimumValue"] = ps.ps5000aMinimumValue(chandle, ctypes.byref(minADC))
    assert_pico_ok(status["minimumValue"])
    return minADC

def get_timebase(chandle, status, Fs, samples, segmentIndex=0):
    '''
    Get timebase.
    
    Warning :
        It may not be possible to access all Timebases as all channels are 
        enabled by default when opening the scope. To access these Timebases,
        set any unused analogue channels to off.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    Fs : float
        Desired sampling frequency in Hz.
    samples : int
        Number of samples to capture.
    segmentIndex : int, optional
        Index of segment in buffer: from 0 to 31. The default is 0.

    Returns
    -------
    timebase : int
        The timebase so that 2**timebase == sampling_period in nanoseconds
    timeIntervalns : float
        The returned time interval in nanoseconds.
    returnedMaxSamples : int
        The returned number of samples.

    Arnau, 29/11/2022
    '''
    timebase = fs2timebase(Fs)
    print(f'Sampling frequency is set to {timebase2fs(timebase)} MHz')

    timeIntervalns = ctypes.c_float()
    returnedMaxSamples = ctypes.c_int32()
    status["getTimebase2"] = ps.ps5000aGetTimebase2(chandle, timebase, samples, ctypes.byref(timeIntervalns), ctypes.byref(returnedMaxSamples), segmentIndex)
    assert_pico_ok(status["getTimebase2"])
    return timebase, timeIntervalns.value, returnedMaxSamples.value
    
def get_MinMax(chandle, status):
    '''
    Return the minimum and maximum values and size allowed for the arbitrary
    waveform generator.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.

    Returns
    -------
    MinVal : int
        Lowest sample value allowed.
    MaxVal : int
        Highest sample value allowed.
    MinSize : int
        Minimum size allowed.
    MaxSize : int
        Maximum size allowed.

    Arnau, 29/11/2022
    '''
    MinVal = ctypes.c_int32()
    MaxVal = ctypes.c_int32()
    MinSize = ctypes.c_int32()
    MaxSize = ctypes.c_int32()
    status["sigGenArbitraryMinMaxValues"] = ps.ps5000aSigGenArbitraryMinMaxValues(chandle, ctypes.byref(MinVal), ctypes.byref(MaxVal), ctypes.byref(MinSize), ctypes.byref(MaxSize))
    assert_pico_ok(status["sigGenArbitraryMinMaxValues"])
    return -MinVal.value, MaxVal.value, MinSize.value, MaxSize.value

def _get_deltaPhase(chandle, status, frequency, indexMode, bufferLength):
    '''
    Transform the desired frequency to phase samples.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    frequency : float
        The desired frequency in Hz.
    indexMode : int
        Single mode (0) or Dual mode (1).
            If 0 (single): the whole waveform must be specified, which allows
            asymmetric waveforms.
            If 1 (dual): only have of the waveform must be specified. The 
            buffer is then read a second time (backwards) to generate the other
            symmetric half of the waveform. The resulting waveform will be
            twice as long.
    bufferLength : int
        Length of the waveform. Maximum value is 2**15.

    Returns
    -------
    phase : ctypes.c_int32
        The phase samples that generate the specified frequency.

    Arnau, 30/11/22
    '''
    _frequency = 0.1 if frequency==0 else frequency
    phase = ctypes.c_uint32()
    status["sigGenFrequencyToPhase"] = ps.ps5000aSigGenFrequencyToPhase(chandle, _frequency, indexMode, bufferLength, ctypes.byref(phase))
    assert_pico_ok(status["sigGenFrequencyToPhase"])
    return phase
    
def get_data_from_buffersdict(chandle, status, voltage_range_A, voltage_range_B, buffers_dict: dict):
    '''
    Converts the data in the dictionary to mV. The function returns 4 numpy 
    arrays: arrayAMax, arrayBMax, arrayAMin and arrayBMin.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    voltage_range_A : str
        Voltage range of channel A. Accepted values are: '10mV', '20mV', '50mV',
        '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.
    voltage_range_B : str
        Voltage range of channel B. Accepted values are: '10mV', '20mV', '50mV',
        '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.
    buffers_dict : dict
        Dictionary containing data from channel A and/or B. Each value in the
        dictionary should contain 2 buffers: raw data and downsampled data.
        Raw data is return in arrayAMax and arrayBMax, while downsampled data 
        is returned in arrayAMin and arrayBMin. If one of the channels has no
        data, the corresponding two arrays have length zero.

    Returns
    -------
    arrayAMax : ndarray
        Array of raw data from channel A in millivolts. Traces are stored 
        row-wise.
    arrayBMax : ndarray
        Array of raw data from channel B in millivolts. Traces are stored 
        row-wise.
    arrayAMin : ndarray
        Array of downsampled data from channel A in millivolts. Traces are 
        stored row-wise.
    arrayBMin : ndarray
        Array of downsampled data from channel B in millivolts. Traces are 
        stored row-wise.

    Arnau, 01/12/2022
    '''
    lstAMax = []
    lstBMax = []
    lstAMin = []
    lstBMin = []
    for k, v in buffers_dict.items():       
        if 'A' in k:
            MaxA = adc2millivolts(chandle, status, v[0], voltage_range_A)
            MinA = adc2millivolts(chandle, status, v[1], voltage_range_A)
            lstAMax.append(MaxA)
            lstAMin.append(MinA)
            
        elif 'B' in k:
            MaxB = adc2millivolts(chandle, status, v[0], voltage_range_B)
            MinB = adc2millivolts(chandle, status, v[1], voltage_range_B)
            lstBMax.append(MaxB)
            lstBMin.append(MinB)
    
    arrayAMax = np.array(lstAMax)
    arrayBMax = np.array(lstBMax)
    arrayAMin = np.array(lstAMin)
    arrayBMin = np.array(lstBMin)   
    
    return arrayAMax, arrayBMax, arrayAMin, arrayBMin
    
#%% ========== TRIGGERS ==========
def set_simpleTrigger(chandle, status, enabled, channel, chRange, threshold, direction, delay, auto_Trigger):
    '''
    Set a simple Trigger.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    enabled : int
        Enable trigger (1) or disable trigger (0).
    channel : str
        Channel to setup. Accepted values are 'A', 'B' and 'EXTERNAL'.
    chRange : str
        Voltage range. Accepted values are Wrong voltage range. Accepted values
        are '10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', 
        '5V', '10V', '20V', '50V' and 'MAX'.
    threshold : float
        Trigger threshold in mV.
    direction : int
        Code of the direction of the trigger. Check PicoScope's API.
    delay : float
        Time between trigger and first sample in seconds.
    auto_Trigger : float
        Starts a capture if no trigger event occurs within the specified milliseconds.

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    ch = _parseChannel(channel)
    threshold_adc = millivolts2adc(chandle, status, threshold, chRange)
    status["trigger"] = ps.ps5000aSetSimpleTrigger(chandle, enabled, ch, threshold_adc, direction, delay, auto_Trigger)
    assert_pico_ok(status["trigger"])


#TODO
def set_AdvancedTrigger(chandle, status):
    '''
    NOT YET IMPLEMENTED.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.

    Raises
    ------
    NotImplementedError
        Not yet implemented. Do not use.
        
    Returns
    -------
    None.

    Arnau, 30/11/2022
    '''
    raise NotImplementedError('Not yet implemented. Do not use.')

    

#%% ========== SIGNAL GENERATOR ==========
def generate_arbitrary_signal(
        chandle, status, offsetVoltage, pkToPk, startFrequency,
        stopFrequency, increment, dwellCount, arbitraryWaveform,
        arbitraryWaveformSize, sweepType, indexMode, shots, sweeps, 
        triggertype, triggerSource, extInThreshold):
    '''
    Generate arbitrary signal.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    offsetVoltage : int
        Offset Voltage in microvolts.
    pkToPk : int
        Peak-to-peak voltage of the waveform in microvolts.
    startFrequency : float
        Initial frequency of the waveform in Hz.
    stopFrequency : float
        Final frequency of the waveform in Hz.
    increment : float
        Frequency step in Hz added every time the dwellCount period expires.
    dwellCount : int
        Number of 50 nanoseconds steps between succesive frequency increments.
        Determines the rate at which the generator sweeps the output frequency.
    arbitraryWaveform : array_like of int16
        The desired waveform to be generated. Elements of the array must be of 
        16-bit integer, e.g. np.int16.
    arbitraryWaveformSize : int
        The length of the waveform. Maximum value is 2**15.
    sweepType : int
        Code of the type of sweep. Available types:
            0 - UP
            1 - DOWN
            2 - UPDOWN
            3 - DOWNUP
    indexMode : int
        Single mode (0) or Dual mode (1).
            If 0 (single): the whole waveform must be specified, which allows
            asymmetric waveforms.
            If 1 (dual): only have of the waveform must be specified. The 
            buffer is then read a second time (backwards) to generate the other
            symmetric half of the waveform. The resulting waveform will be
            twice as long.
    shots : int
        Number of cycles of the waveform to be produced after a trigger event.
        If 0, sweep the frequency as specified by sweeps.
        If not 0, sweep must be 0.
    sweeps : int
        Number of times to sweep the frequency after a trigger event.
        If 0, produce a number of cycles specified by shots.
        If not 0, shots must be 0.
    triggertype : int
        Code of the type of trigger to use. Available types:
            0 - RISING
            1 - FALLING
            2 - GATE_HIGH
            3 - GATE_LOW
    triggerSource : int
        Code of the trigger source. Available types:
            0 - NONE
            1 - SCOPE
            2 - AUX
            3 - EXT
            4 - SOFTWARE
    extInThreshold : int
        Trigger level in case of EXTERNAL trigger.

    Returns
    -------
    None.

    Arnau, 30/11/2022
    '''
    cstartDeltaPhase = _get_deltaPhase(chandle, status, startFrequency, indexMode, arbitraryWaveformSize)
    cstopDeltaPhase = _get_deltaPhase(chandle, status, stopFrequency, indexMode, arbitraryWaveformSize)
    cdeltaPhaseIncrement = _get_deltaPhase(chandle, status, increment, indexMode, arbitraryWaveformSize)

    carbitraryWaveform = (ctypes.c_int16 * arbitraryWaveformSize)(*arbitraryWaveform) # convert numpy array to ctypes
    csweepType = ctypes.c_int32(sweepType)
    cindexMode = ctypes.c_int32(indexMode)
    ctriggertype = ctypes.c_int32(triggertype)
    ctriggerSource = ctypes.c_int32(triggerSource)
    
    status["setSigGenArbitrary"] = ps.ps5000aSetSigGenArbitrary(
        chandle, 
        offsetVoltage, 
        pkToPk, 
        cstartDeltaPhase,
        cstopDeltaPhase,
        cdeltaPhaseIncrement,
        dwellCount,
        ctypes.byref(carbitraryWaveform),
        arbitraryWaveformSize,
        csweepType, 
        ctypes.c_int32(0), # not used in Pico 5000a
        cindexMode,
        shots, 
        sweeps, 
        ctriggertype, 
        ctriggerSource, 
        extInThreshold)
    assert_pico_ok(status["setSigGenArbitrary"])


def generate_builtin_signal(
        chandle, status, offsetVoltage, pkToPk, wavetype, startFrequency,
        stopFrequency, increment, dwellTime, sweepType, operation, shots,
        sweeps, triggertype, triggerSource, extInThreshold):
    '''
    Generate builtin signal.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    offsetVoltage : int
        Offset Voltage in microvolts.
    pkToPk : int
        Peak-to-peak voltage of the waveform in microvolts.
    wavetype : int
        Code of the built-in wave type. Available types:
            0 - SINE
            1 - SQUARE
            2 - TRIANGLE
            3 - RAMP_UP
            4 - RAMP_DOWN
            5 - SINC
            6 - GAUSSIAN
            7 - HALF_SINE
            8 - DC_VOLTAGE
            9 - WHITE_NOISE
    startFrequency : float
        Start frequency of the signal in Hz.
    stopFrequency : float
        Stop frequency of the signal in Hz.
    increment : float
        Increment of frequency in Hz.
    dwellTime : float
        The time for which the sweep stays at each frequency in seconds.
    sweepType : int
        Code of the type of sweep. Available types:
            0 - UP
            1 - DOWN
            2 - UPDOWN
            3 - DOWNUP
    operation : int
        Code of the extra operations for the waveform type. Not used in 5000a.
    shots : int
        Number of cycles of the waveform to be produced after a trigger event.
        If 0, sweep the frequency as specified by sweeps.
        If not 0, sweep must be 0.
    sweeps : int
        Number of times to sweep the frequency after a trigger event.
        If 0, produce a number of cycles specified by shots.
        If not 0, shots must be 0.
    triggertype : int
        Code of the type of trigger to use. Available types:
            0 - RISING
            1 - FALLING
            2 - GATE_HIGH
            3 - GATE_LOW
    triggerSource : int
        Code of the trigger source. Available types:
            0 - NONE
            1 - SCOPE
            2 - AUX
            3 - EXT
            4 - SOFTWARE
    extInThreshold : int
        Trigger level in case of EXTERNAL trigger.

    Returns
    -------
    None.

    Arnau, 29/11/2022
    '''
    cwavetype = ctypes.c_int32(wavetype)
    csweepType = ctypes.c_int32(sweepType)
    coperation = ctypes.c_int32(operation)
    ctriggertype = ctypes.c_int32(triggertype)
    ctriggerSource = ctypes.c_int32(triggerSource)
    
    status["setSigGenBuiltInV2"] = ps.ps5000aSetSigGenBuiltInV2(
        chandle, 
        offsetVoltage, 
        pkToPk, 
        cwavetype, 
        startFrequency, 
        stopFrequency, 
        increment, 
        dwellTime, 
        csweepType, 
        coperation, 
        shots, 
        sweeps, 
        ctriggertype, 
        ctriggerSource, 
        extInThreshold)
    assert_pico_ok(status["setSigGenBuiltInV2"])

def trigger_generator(chandle, status, triggertype, gate_time=1000):
    '''
    Trigger the signal generator.

    Parameters
    ----------
    chandle : ctypes.c_int16
        handle.
    status : dict
        status dictionary.
    triggertype : int
        Code of the type of trigger to use. Available types:
            0 - RISING
            1 - FALLING
            2 - GATE_HIGH
            3 - GATE_LOW
    gate_time : float, optional
        The gate time in milliseconds for trigger types 2 or 3 (gated triggers).
        Unused for trigger types 0 or 1 (edge triggers). The default is 1000.

    Returns
    -------
    None.

    ''' 
    if triggertype in [0, 1]:
        status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(chandle, 0)
        assert_pico_ok(status["sigGenSoftwareControl"])
        
    elif triggertype==3:
        status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(chandle, 0)
        assert_pico_ok(status["sigGenSoftwareControl"])
        time.sleep(gate_time*1e-3)
        status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(chandle, 1)
        assert_pico_ok(status["sigGenSoftwareControl"])
        
    elif triggertype==2:
        status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(chandle, 1)
        assert_pico_ok(status["sigGenSoftwareControl"])
        time.sleep(gate_time*1e-3)
        status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(chandle, 0)
        assert_pico_ok(status["sigGenSoftwareControl"])