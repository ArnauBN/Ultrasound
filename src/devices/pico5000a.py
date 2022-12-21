# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:51:21 2022
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

#%% ========== CLASS ==========
class Pico:
    def __init__(self, num_bits=None):
        self.chandle = ctypes.c_int16()
        self.status = {}
        
        # optional start
        if num_bits is not None:
            self.start(num_bits)

    def start(self, num_bits=8):
        '''
        Open 5000a PicoScope and set the number of bits.
    
        Parameters
        ----------
        num_bits : int or str, optional
            number of bits. Available values are 8, 12, 14, 15 and 16. Default
            is 8.
    
        Returns
        -------
        None.
    
        Arnau, 20/12/2022
        '''
        # Set resolution
        resolution = parseResolution(num_bits)
        
        # Returns handle to chandle for use in future API functions
        self.status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(self.chandle), None, resolution)
        
        try:
            assert_pico_ok(self.status["openunit"])
        except: # PicoNotOkError:
        
            powerStatus = self.status["openunit"]
        
            if powerStatus == 286:
                self.status["changePowerSource"] = ps.ps5000aChangePowerSource(self.chandle, powerStatus)
            elif powerStatus == 282:
                self.status["changePowerSource"] = ps.ps5000aChangePowerSource(self.chandle, powerStatus)
            else:
                raise
        
            assert_pico_ok(self.status["changePowerSource"])
    
    def stop(self):
        '''
        Stops the scope device from sampling data.
    
        Returns
        -------
        None.
    
        Arnau, 20/12/2022
        '''
        # Stop the scope
        self.status["stop"] = ps.ps5000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])
        print(f'Stopped with status:\n{self.status}')
    
    def close(self):       
        '''
        Closes the unit, releasing the handle.
    
        Returns
        -------
        None.
    
        Arnau, 20/12/2022
        '''
        # Close the unit
        self.status["close"] = ps.ps5000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])
        print(f'Closed with status:\n{self.status}')


    # ---------
    # FUNCTIONS
    # ---------
    def set_resolution(self, num_bits):
        '''
        Sets the resolution given the number of bits to use.
    
        Parameters
        ----------
        num_bits : int or str
            number of bits. Available values are 8, 12, 14, 15 and 16.
    
        Returns
        -------
        None.
    
        Arnau, 20/12/2022
        '''
        resolution = parseResolution(num_bits)
        self.status["setDeviceResolution"] = ps.ps5000aSetDeviceResolution(self.chandle, resolution)
        assert_pico_ok(self.status["setDeviceResolution"])
    
    def setup_channel(self, channel, coupling, voltage_range, offset, enabled):
        '''
        Setup specified channel.
    
        Parameters
        ----------
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
    
        Arnau, 20/12/2022
        '''
        # Set channel to use
        ch = parseChannel(channel)
        
        # Coupling
        coupling_type = parseCoupling(coupling)
       
        # Voltage range
        chRange = parseVoltageRange(voltage_range)
        
        # Setup channel
        self.status["setChA"] = ps.ps5000aSetChannel(self.chandle, ch, enabled, coupling_type, chRange, offset)
        assert_pico_ok(self.status["setChA"])

    def millivolts2adc(self, v, voltage_range):
        '''
        Converts millivotls to adc count.
    
        Parameters
        ----------
        v : float
            Millivolts to convert to adc count.
        voltage_range : str
            Voltage range. Accepted values are '10mV', '20mV', '50mV', '100mV',
            '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.
    
        Returns
        -------
        v_adc : int
            The corresponding adc count.
    
        Arnau, 20/12/2022
        '''
        parsed_chRange = parseVoltageRange(voltage_range)
        maxADC = self.get_maxADC()
        v_adc = int(mV2adc(v, parsed_chRange, maxADC))
        return v_adc
    
    def adc2millivolts(self, bufferADC, voltage_range):
        '''
        Converts a raw adc count buffer to millivolts.
    
        Parameters
        ----------
        bufferADC : array of ctypes.c_int16
            The buffered adc values to convert to millivolts.
        voltage_range : str
            Voltage range. Accepted values are '10mV', '20mV', '50mV', '100mV',
            '200mV', '500mV', '1V', '2V', '5V', '10V', '20V' and '50V'.
    
        Returns
        -------
        buffermV : ndarray
            The buffer in millivolts.
    
        Arnau, 20/12/2022
        '''
        parsed_chRange = parseVoltageRange(voltage_range)
        maxADC = self.get_maxADC()
        buffermV = adc2mV(bufferADC, parsed_chRange, maxADC)
        return np.array(buffermV)

    # ------------
    # CAPTURE DATA
    # ------------
    def capture(self, channels, samples, timebase, trigger_sigGen, sigGen_triggertype, sigGen_gate_time, downsampling=(0,0), segment_index=0):
        '''
        Capture Block.
    
        Parameters
        ----------
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
        
        Arnau, 20/12/2022
        '''
        if isinstance(samples, tuple):
            maxsamples = samples[0] + samples[1]
            preTriggerSamples, postTriggerSamples = samples
        else:
            maxsamples = samples
            preTriggerSamples = samples//2
            postTriggerSamples = samples - preTriggerSamples
        
        time_indisposed = ctypes.c_int32()
        self.status["runBlock"] = ps.ps5000aRunBlock(self.chandle, preTriggerSamples, postTriggerSamples, 
                                                timebase, ctypes.byref(time_indisposed), segment_index, None, None)
        assert_pico_ok(self.status["runBlock"])
        
        # Create buffers ready for assigning pointers for data collection
        BUFFERS_DICT = {} # BUFFERS_DICT[bufferX0] = (bufferMaxX0, bufferMinX0), bufferMinX0 is used for downsampling
        if channels.upper()=='A' or channels.upper()=='BOTH':
            key = f"bufferA{segment_index}"
            a = (ctypes.c_int16 * maxsamples)()
            b = (ctypes.c_int16 * maxsamples)()
            BUFFERS_DICT[key] = [a, b]
    
            # Set data buffer location for data collection from channel A
            chA = parseChannel('A')
            self.status["setDataBuffersA"] = ps.ps5000aSetDataBuffers(self.chandle, chA, ctypes.byref(BUFFERS_DICT[key][0]), ctypes.byref(BUFFERS_DICT[key][1]), maxsamples, segment_index, downsampling[0])
            assert_pico_ok(self.status["setDataBuffersA"])
        
        if channels.upper()=='B' or channels.upper()=='BOTH':
            key = f"bufferB{segment_index}"
            c = (ctypes.c_int16 * maxsamples)()
            d = (ctypes.c_int16 * maxsamples)()
            BUFFERS_DICT[key] = [c, d]
            
            # Set data buffer location for data collection from channel B
            chB = parseChannel('B')
            self.status["setDataBuffersB"] = ps.ps5000aSetDataBuffers(self.chandle, chB, ctypes.byref(BUFFERS_DICT[key][0]), ctypes.byref(BUFFERS_DICT[key][1]), maxsamples, segment_index, downsampling[0])
            assert_pico_ok(self.status["setDataBuffersB"])
    
        overflow = ctypes.c_int16() # create overflow loaction
        cmaxSamples = ctypes.c_int32(maxsamples) # create converted type maxSamples
    
        if trigger_sigGen:
            self.trigger_generator(sigGen_triggertype, sigGen_gate_time)
    
        # Check for data collection to finish using ps5000aIsReady
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.status["isReady"] = ps.ps5000aIsReady(self.chandle, ctypes.byref(ready))
        
        # Retrieve data from scope to buffers assigned above
        self.status["getValues"] = ps.ps5000aGetValues(self.chandle, 0, ctypes.byref(cmaxSamples), downsampling[1], downsampling[0], segment_index, ctypes.byref(overflow))
        assert_pico_ok(self.status["getValues"])
    
        # Get time offsets for each waveform
        triggerTimeOffset = ctypes.c_int64()
        triggerTimeOffsetUnits = ctypes.c_char()
        self.status["GetTriggerTimeOffset64"] = ps.ps5000aGetTriggerTimeOffset64(self.chandle, ctypes.byref(triggerTimeOffset), ctypes.byref(triggerTimeOffsetUnits), 0)
        assert_pico_ok(self.status["GetTriggerTimeOffset64"])   
        
        return BUFFERS_DICT, cmaxSamples.value, triggerTimeOffset.value, int.from_bytes(triggerTimeOffsetUnits.value, 'big'), time_indisposed.value
    
    def rapid_capture(self, channels, samples, timebase, nSegments, trigger_sigGen, sigGen_triggertype, sigGen_gate_time, downsampling=(0,0)):
        '''
        Rapid capture of {nSegments} segments. The first segment index is always 0.
    
        Parameters
        ----------
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
        maxSamples : int
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
        
        Arnau, 20/12/2022
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
        
        self.status["MemorySegments"] = ps.ps5000aMemorySegments(self.chandle, nSegments, ctypes.byref(cmaxSamples))
        assert_pico_ok(self.status["MemorySegments"])
        
        # sets number of captures
        self.status["SetNoOfCaptures"] = ps.ps5000aSetNoOfCaptures(self.chandle, nSegments)
        assert_pico_ok(self.status["SetNoOfCaptures"])
    
        # Starts the block capture
        time_indisposed = ctypes.c_int32()
        self.status["runblock"] = ps.ps5000aRunBlock(self.chandle, preTriggerSamples, postTriggerSamples, 
                                                timebase, ctypes.byref(time_indisposed), 0, None, None)
        assert_pico_ok(self.status["runblock"])
        
        BUFFERS_DICT = {} # BUFFERS_DICT[bufferXi] = (bufferMaxXi, bufferMinXi), bufferMinXi is used for downsampling
        if channels.upper()=='A' or channels.upper()=='BOTH':
            chA = parseChannel('A')
            for i in range(nSegments):
                # Create buffers ready for assigning pointers for data collection
                a = (ctypes.c_int16 * maxsamples)()
                b = (ctypes.c_int16 * maxsamples)()
                BUFFERS_DICT[f"bufferA{i}"] = [a, b]
    
                # Setting the data buffer location for data collection from channel chA
                self.status["SetDataBuffers"] = ps.ps5000aSetDataBuffers(self.chandle, chA, 
                                                                         ctypes.byref(BUFFERS_DICT[f"bufferA{i}"][0]),
                                                                         ctypes.byref(BUFFERS_DICT[f"bufferA{i}"][1]),
                                                                         maxsamples, i, downsampling[0])
                assert_pico_ok(self.status["SetDataBuffers"])
        
        if channels.upper()=='B' or channels.upper()=='BOTH':
            chB = parseChannel('B')
            for i in range(nSegments):
                # Create buffers ready for assigning pointers for data collection
                c = (ctypes.c_int16 * maxsamples)()
                d = (ctypes.c_int16 * maxsamples)()
                BUFFERS_DICT[f"bufferB{i}"] = [c, d]
                
                # Setting the data buffer location for data collection from channel chA
                self.status["SetDataBuffers"] = ps.ps5000aSetDataBuffers(self.chandle, chB, 
                                                                         ctypes.byref(BUFFERS_DICT[f"bufferB{i}"][0]),
                                                                         ctypes.byref(BUFFERS_DICT[f"bufferB{i}"][1]),
                                                                         maxsamples, i, downsampling[0])
                assert_pico_ok(self.status["SetDataBuffers"])
                    
        overflow = (ctypes.c_int16 * nSegments)() # Creates an overlow location for data
        cmaxSamples = ctypes.c_int32(maxsamples) # Creates converted types maxsamples
        
        if trigger_sigGen:
            for _ in range(nSegments):
                self.trigger_generator(sigGen_triggertype, sigGen_gate_time)
        
        # Check for data collection to finish using ps5000aIsReady
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.status["isReady"] = ps.ps5000aIsReady(self.chandle, ctypes.byref(ready))
        
        self.status["GetValuesBulk"] = ps.ps5000aGetValuesBulk(self.chandle, ctypes.byref(cmaxSamples), 0, nSegments-1, downsampling[1], downsampling[0], ctypes.byref(overflow))
        assert_pico_ok(self.status["GetValuesBulk"])
    
        # Get time offsets for each waveform
        triggerTimeOffsets = (ctypes.c_int64*nSegments)()
        triggerTimeOffsetUnits = ctypes.c_char()
        self.status["GetValuesTriggerTimeOffsetBulk64"] = ps.ps5000aGetValuesTriggerTimeOffsetBulk64(self.chandle, ctypes.byref(triggerTimeOffsets), ctypes.byref(triggerTimeOffsetUnits), 0, nSegments-1)
        assert_pico_ok(self.status["GetValuesTriggerTimeOffsetBulk64"])
    
        # Create array of ps.PS5000A_TRIGGER_INFO for each memory segment
        triggerInfo = (ps.PS5000A_TRIGGER_INFO*nSegments)()
        self.status["GetTriggerInfoBulk"] = ps.ps5000aGetTriggerInfoBulk(self.chandle, ctypes.byref(triggerInfo), 0, nSegments-1)
        assert_pico_ok(self.status["GetTriggerInfoBulk"])
    
        return BUFFERS_DICT, cmaxSamples.value, np.array(list(map(int,triggerTimeOffsets))), int.from_bytes(triggerTimeOffsetUnits.value, 'big'), time_indisposed.value, triggerInfo

    # ----
    # GETS
    # ----
    def get_maxADC(self):
        '''
        Find maximum ADC count value
    
        Returns
        -------
        maxADC : ctypes.c_int16()
            The maximum ADC count value.
    
        Arnau, 20/12/2022
        '''
        maxADC = ctypes.c_int16()
        self.status["maximumValue"] = ps.ps5000aMaximumValue(self.chandle, ctypes.byref(maxADC))
        assert_pico_ok(self.status["maximumValue"])
        return maxADC
    
    def get_minADC(self):
        '''
        Find minimum ADC count value
    
        Returns
        -------
        minADC : ctypes.c_int16()
            The minimum ADC count value.
    
        Arnau, 20/12/2022
        '''
        minADC = ctypes.c_int16()
        self.status["minimumValue"] = ps.ps5000aMinimumValue(self.chandle, ctypes.byref(minADC))
        assert_pico_ok(self.status["minimumValue"])
        return minADC

    def get_timebase(self, Fs, samples, segmentIndex=0):
        '''
        Get timebase.
        
        Warning :
            It may not be possible to access all Timebases as all channels are 
            enabled by default when opening the scope. To access these Timebases,
            set any unused analogue channels to off.
    
        Parameters
        ----------
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
    
        Arnau, 20/12/2022
        '''
        timebase = fs2timebase(Fs)
        print(f'Sampling frequency is set to {timebase2fs(timebase)} MHz')
    
        timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int32()
        self.status["getTimebase2"] = ps.ps5000aGetTimebase2(self.chandle, timebase, samples, ctypes.byref(timeIntervalns), ctypes.byref(returnedMaxSamples), segmentIndex)
        assert_pico_ok(self.status["getTimebase2"])
        return timebase, timeIntervalns.value, returnedMaxSamples.value
        
    def get_MinMax(self):
        '''
        Return the minimum and maximum values and size allowed for the arbitrary
        waveform generator.
    
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
    
        Arnau, 20/12/2022
        '''
        MinVal = ctypes.c_int32()
        MaxVal = ctypes.c_int32()
        MinSize = ctypes.c_int32()
        MaxSize = ctypes.c_int32()
        self.status["sigGenArbitraryMinMaxValues"] = ps.ps5000aSigGenArbitraryMinMaxValues(self.chandle, ctypes.byref(MinVal), ctypes.byref(MaxVal), ctypes.byref(MinSize), ctypes.byref(MaxSize))
        assert_pico_ok(self.status["sigGenArbitraryMinMaxValues"])
        return -MinVal.value, MaxVal.value, MinSize.value, MaxSize.value
    
    def get_deltaPhase(self, frequency, indexMode, bufferLength):
        '''
        Transform the desired frequency to phase samples.
    
        Parameters
        ----------
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
    
        Arnau, 20/12/22
        '''
        _frequency = 0.1 if frequency==0 else frequency
        phase = ctypes.c_uint32()
        self.status["sigGenFrequencyToPhase"] = ps.ps5000aSigGenFrequencyToPhase(self.chandle, _frequency, indexMode, bufferLength, ctypes.byref(phase))
        assert_pico_ok(self.status["sigGenFrequencyToPhase"])
        return phase
        
    def get_data_from_buffersdict(self, voltage_range_A, voltage_range_B, buffers_dict: dict):
        '''
        Converts the data in the dictionary to mV. The function returns 5 numpy 
        arrays: arrayAMax, arrayBMax, arrayAMin, arrayBMin and means.
        
        The first 4 arrays contain the raw data (Max) or the downsampled data (Min)
        for every trace. The last array contains the mean trace of the first 4 
        arrays. This means that the first 4 arrays have shapes NxL and the 5th
        has shape 4xL, where N is the number of traces found and L is the length of
        every trace.
    
        Parameters
        ----------
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
            Array of shape NxL of raw data from channel A in millivolts.
        arrayBMax : ndarray
            Array of shape NxL of raw data from channel B in millivolts.
        arrayAMin : ndarray
            Array of shape NxL of downsampled data from channel A in millivolts.
        arrayBMin : ndarray
            Array of shape NxL of downsampled data from channel B in millivolts.
        means : ndarray
            Matrix of shape 4xL containing the average trace of the previous 4
            arrays. If one of the arrays has no data, the corresponding fow of this
            vector is filled with None.
        
        Arnau, 20/12/2022
        '''
        lstAMax = []
        lstBMax = []
        lstAMin = []
        lstBMin = []
        for k, v in buffers_dict.items():       
            if 'A' in k:
                MaxA = self.adc2millivolts(v[0], voltage_range_A)
                MinA = self.adc2millivolts(v[1], voltage_range_A)
                lstAMax.append(MaxA)
                lstAMin.append(MinA)
                
            elif 'B' in k:
                MaxB = self.adc2millivolts(v[0], voltage_range_B)
                MinB = self.adc2millivolts(v[1], voltage_range_B)
                lstBMax.append(MaxB)
                lstBMin.append(MinB)
        
        arrayAMax = np.array(lstAMax)
        arrayBMax = np.array(lstBMax)
        arrayAMin = np.array(lstAMin)
        arrayBMin = np.array(lstBMin)   
        
        try:
            L = len(arrayAMax[0])
        except:
            L = len(arrayBMax[0])
        means = np.zeros([4, L])
        for i, a in enumerate([arrayAMax, arrayBMax, arrayAMin, arrayBMin]):
            if len(a) != 0:
                means[i] = np.mean(a, axis=0)
            else:
                means[i] = np.full(L, None)
        
        return arrayAMax, arrayBMax, arrayAMin, arrayBMin, means

    # --------
    # TRIGGERS
    # --------
    def set_simpleTrigger(self, enabled, channel, chRange, threshold, direction, delay, auto_Trigger):
        '''
        Set a simple Trigger.
    
        Parameters
        ----------
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
    
        Arnau, 20/12/2022
        '''
        ch = parseChannel(channel)
        threshold_adc = self.millivolts2adc(threshold, chRange)
        self.status["trigger"] = ps.ps5000aSetSimpleTrigger(self.chandle, enabled, ch, threshold_adc, direction, delay, auto_Trigger)
        assert_pico_ok(self.status["trigger"])
    
    #TODO
    def set_AdvancedTrigger(self):
        '''
        NOT YET IMPLEMENTED.
    
        Raises
        ------
        NotImplementedError
            Not yet implemented. Do not use.
            
        Returns
        -------
        None.
    
        Arnau, 20/12/2022
        '''
        raise NotImplementedError('Not yet implemented. Do not use.')

    # ----------------
    # SIGNAL GENERATOR
    # ----------------
    def generate_arbitrary_signal(self, offsetVoltage, pkToPk, startFrequency,
            stopFrequency, increment, dwellCount, arbitraryWaveform,
            arbitraryWaveformSize, sweepType, indexMode, shots, sweeps, 
            triggertype, triggerSource, extInThreshold):
        '''
        Generate arbitrary signal.
    
        Parameters
        ----------
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
    
        Arnau, 20/12/2022
        '''
        cstartDeltaPhase = self.get_deltaPhase(startFrequency, indexMode, arbitraryWaveformSize)
        cstopDeltaPhase = self.get_deltaPhase(stopFrequency, indexMode, arbitraryWaveformSize)
        cdeltaPhaseIncrement = self.get_deltaPhase(increment, indexMode, arbitraryWaveformSize)
    
        carbitraryWaveform = (ctypes.c_int16 * arbitraryWaveformSize)(*arbitraryWaveform) # convert numpy array to ctypes
        csweepType = ctypes.c_int32(sweepType)
        cindexMode = ctypes.c_int32(indexMode)
        ctriggertype = ctypes.c_int32(triggertype)
        ctriggerSource = ctypes.c_int32(triggerSource)
        coperation = ctypes.c_int32(0) # not used in Pico 5000a
        
        self.status["setSigGenArbitrary"] = ps.ps5000aSetSigGenArbitrary(
            self.chandle, 
            offsetVoltage, 
            pkToPk, 
            cstartDeltaPhase,
            cstopDeltaPhase,
            cdeltaPhaseIncrement,
            dwellCount,
            ctypes.byref(carbitraryWaveform),
            arbitraryWaveformSize,
            csweepType, 
            coperation,
            cindexMode,
            shots, 
            sweeps, 
            ctriggertype, 
            ctriggerSource, 
            extInThreshold)
        assert_pico_ok(self.status["setSigGenArbitrary"])

    def generate_builtin_signal(self, offsetVoltage, pkToPk, wavetype,
            startFrequency, stopFrequency, increment, dwellTime, sweepType,
            shots, sweeps, triggertype, triggerSource, extInThreshold):
        '''
        Generate builtin signal.
    
        Parameters
        ----------
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
    
        Arnau, 20/12/2022
        '''
        cwavetype = ctypes.c_int32(wavetype)
        csweepType = ctypes.c_int32(sweepType)
        coperation = ctypes.c_int32(0) # not used in Pico 5000a
        ctriggertype = ctypes.c_int32(triggertype)
        ctriggerSource = ctypes.c_int32(triggerSource)
        
        self.status["setSigGenBuiltInV2"] = ps.ps5000aSetSigGenBuiltInV2(
            self.chandle, 
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
        assert_pico_ok(self.status["setSigGenBuiltInV2"])

    def trigger_generator(self, triggertype, gate_time=1000):
        '''
        Trigger the signal generator.
    
        Parameters
        ----------
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
        
        Arnau, 20/12/2022
        ''' 
        if triggertype in [0, 1]:
            self.status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(self.chandle, 0)
            assert_pico_ok(self.status["sigGenSoftwareControl"])
            
        elif triggertype==3:
            self.status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(self.chandle, 0)
            assert_pico_ok(self.status["sigGenSoftwareControl"])
            time.sleep(gate_time*1e-3)
            self.status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(self.chandle, 1)
            assert_pico_ok(self.status["sigGenSoftwareControl"])
            
        elif triggertype==2:
            self.status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(self.chandle, 1)
            assert_pico_ok(self.status["sigGenSoftwareControl"])
            time.sleep(gate_time*1e-3)
            self.status["sigGenSoftwareControl"] = ps.ps5000aSigGenSoftwareControl(self.chandle, 0)
            assert_pico_ok(self.status["sigGenSoftwareControl"])
    
    def stopSigGen(self):
        '''
        Stop the signal generator. This is done by generating a DC voltage of zero
        volts.
    
        Returns
        -------
        None.
    
        '''
        self.generate_builtin_signal(0, 0, 8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)
    

#%% ========== PARSERS ==========
def parseResolution(num_bits):
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

def parseChannel(channel):
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

def parseCoupling(coupling):
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

def parseVoltageRange(voltage_range):
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
    chRange = ps.PS5000A_RANGE[key]
    return chRange


#%% ========== DISCOVERY ==========
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


#%% ========== FUNCTIONS ==========
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

def str2V(s: str):
    '''
    Parse voltage as string to volts as float.

    Parameters
    ----------
    s : str
        Voltage to convert to volts. 
        Examples: '10mV', '3V', '346.123uV', '1.5kV'.

    Raises
    ------
    ValueError
        Wrong voltage.

    Returns
    -------
    V : float
        The voltage (in volts) as a float.

    '''
    if s[-1].upper() != 'V':
        raise ValueError("Wrong voltage. String must end with uV, mV, V or kV")
    if ' ' in s:
        raise ValueError("Wrong voltage. Must not contain spaces.")
    if '-' in s:
        raise ValueError("Wrong voltage. Must not contain dashes.")
    
    if s[-2]=='u':
        factor = 1e-6
        v = s[:-2]
    elif s[-2]=='m':
        factor = 1e-3
        v = s[:-2]
    elif s[-2]=='k':
        factor = 1e3
        v = s[:-2]
    else:
        factor = 1
        v = s[:-1]
    return float(v)*factor


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





#%% ========== TESTING THE CLASS ==========
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Parameters
    num_bits = 15               # Number of bits to use (8, 12, 14, 15 or 16) - int
    Fs = 125e6                  # Desired sampling frequency (Hz) - float


    # ------------------
    # Arbitrary Waveform
    # ------------------
    waveform_f0 = 2e6           # Center Frequency of waveform (Hz) - float
    waveformSize = 2**11        # Waveform length (power of 2, max=2**15) - int

    _samples = int(np.round(1/waveform_f0*Fs/2))
    pulse = np.ones(_samples)*32767
    waveform = np.append(pulse, np.zeros(waveformSize - _samples))
    waveform = waveform.astype(np.int16)


    # ---------------
    # Channel A setup
    # ---------------
    coupling_A = 'DC'           # Coupling of channel A ('AC' or 'DC') - str
    voltage_range_A = '5V'      # Voltage range of channel A ('10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V', '50V' or 'MAX') - str
    offset_A = 0                # Analog offset of channel A (in volts) - float
    enabled_A = 1               # Enable (1) or disable (0) channel A - int


    # ---------------
    # Channel B setup
    # ---------------
    coupling_B = 'DC'           # Coupling of channel B ('AC' or 'DC') - str
    voltage_range_B = '5V'      # Voltage range of channel B ('10mV', '20mV', '50mV', '100mV', '200mV', '500mV', '1V', '2V', '5V', '10V', '20V', '50V' or 'MAX') - str
    offset_B = 0                # Analog offset of channel B (in volts) - float
    enabled_B = 0               # Enable (1) or disable (0) channel B - int


    # ---------------
    # Capture options
    # ---------------
    channels = 'A'              # 'A', 'B' or 'BOTH' - str
    nSegments = 20              # Number of traces to capture and average to reduce noise - int
    downsampling_ratio_mode = 0 # Downsampling ratio mode - int
    downsampling_ratio = 0      # Downsampling ratio - int


    # ---------------
    # Trigger options
    # ---------------
    triggerChannel = 'A'        # 'A', 'B' or 'EXTERNAL' - str
    triggerThreshold = 500      # Trigger threshold in mV - float
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
        'triggerSource'         : 1,            # Source of trigger - int
        'extInThreshold'        : 0             # Trigger level for EXTERNAL trigger - int
    }
    # Find out device model (5000a)
    check_drivers()
    time.sleep(3)

    # Start pico
    pico = Pico(num_bits)
    time.sleep(3)

    # Setup
    # Set up channel A
    pico.setup_channel('A', coupling_A, voltage_range_A, offset_A, enabled_A)

    # Set up channel B
    pico.setup_channel('B', coupling_B, voltage_range_B, offset_B, enabled_B)

    # Set up simple trigger
    voltage_range = voltage_range_B if triggerChannel=='B' else voltage_range_A
    pico.set_simpleTrigger(enabled_trigger, triggerChannel, voltage_range, triggerThreshold, direction, delay, auto_Trigger)
    time.sleep(3)

    # Get timebase
    timebase, timeIntervalns, maxSamples = pico.get_timebase(Fs, preTriggerSamples + postTriggerSamples, segmentIndex=0)
    Real_Fs = 1e9/(2**timebase) # Hz

    if generate_arbitrary_signal and ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency'] == ARBITRARY_SIGNAL_GENERATOR_DICT['stopFrequency']:
        pulse_freq = Real_Fs/waveformSize
        if pulse_freq != ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency']:
            ARBITRARY_SIGNAL_GENERATOR_DICT['startFrequency'] = pulse_freq
            ARBITRARY_SIGNAL_GENERATOR_DICT['stopFrequency'] = pulse_freq
            print('Frequency of the arbitrary waveform changed to {pulse_freq} Hz.')
    time.sleep(3)
    
    # Generate signal
    if generate_arbitrary_signal:
        pico.generate_arbitrary_signal(**ARBITRARY_SIGNAL_GENERATOR_DICT)
        triggertype = ARBITRARY_SIGNAL_GENERATOR_DICT['triggertype']
        triggerSource = ARBITRARY_SIGNAL_GENERATOR_DICT['triggerSource']
    elif generate_builtin_signal:
        pico.generate_builtin_signal(**BUILTIN_SIGNAL_GENERATOR_DICT)
        triggertype = BUILTIN_SIGNAL_GENERATOR_DICT['triggertype']
        triggerSource = BUILTIN_SIGNAL_GENERATOR_DICT['triggerSource']

    trigger_sigGen = True if triggerSource==4 else False
    time.sleep(3)


    # Capture rapid data: nSegments   
    BUFFERS_DICT, cmaxSamples, triggerTimeOffsets, triggerTimeOffsetUnits, time_indisposed, triggerInfo = pico.rapid_capture(
        channels, (preTriggerSamples, postTriggerSamples), timebase,
        nSegments, trigger_sigGen, triggertype, gate_time,
        downsampling=(downsampling_ratio_mode, downsampling_ratio))
    means = pico.get_data_from_buffersdict(voltage_range_A, voltage_range_B, BUFFERS_DICT)[4]
    
    start_time = time.time()
    
    # Create time data
    t = np.linspace(0, (cmaxSamples - 1) * timeIntervalns, cmaxSamples)
        
    # Plot
    fig, axs = plt.subplots(1, num='Signal', clear=True)
    axs[0].plot(t*1e-3, means[0], lw=2)
    axs[0].set_ylim([-pico.str2V(voltage_range_A)*1e3, pico.str2V(voltage_range_A)*1e3])
    axs[1].set_xlabel('Time (us)')
    axs[0].set_ylabel('Voltage (mV)')
    axs[0].set_title('Channel A')
        
    elapsed_time = time.time() - start_time
    print(f'Elapsed time is {elapsed_time} s.')

    # Stop the scope
    pico.stop()

    # Close the unit
    pico.close()