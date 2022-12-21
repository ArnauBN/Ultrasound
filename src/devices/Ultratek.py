# -*- coding: utf-8 -*-
"""
Ultratek Class for USB-UT350
see http://www.usultratek.com/products/usb_ut350.htm

Created on Mon Jul 23 13:30:31 2012
@author: Miguel Molero
"""

import time, os
import os
import ctypes
from pathlib import Path
import numpy as np

# dir_path = os.getcwd()
# dir_path = "C:\Windows" #location of USBUT.dll, USBUT.lib

class Ultratek:
    def __init__(self):
        """
        int USBUTParms(int mode, int parm, DWORD wParam, int lParam);

        """

        # self.usbUT = ctypes.windll.LoadLibrary(dir_path + "\USBUT.dll")
        self.usbUT = ctypes.windll.LoadLibrary(Path(__file__).parents[2] / "./DLL/USBUT.dll")
        self.number = ctypes.create_string_buffer('\000' * 20)
        self.data = ctypes.create_string_buffer('\000' * 8191)


    def init(self):
        """
        Description:    Initializes the device.
        wParam:         Not used
        lParam:         Not used
        Return Value:   1
        """

        INITIALIZE = 5000

        if ( self.usbUT.USBUTParms(INITIALIZE, 0, 0, 0) == -500 ):
            return False
        else:
            return True


    def initWithParameters(self, parmsUTcard):

        handler = self.init()
        if handler:
            self.setSamplingRate(parmsUTcard.SamplingRate)
            self.setHardwareHPF(option='12kHz')
            self.setTransducerMode(option=1)
            self.setBufferLength(parmsUTcard.NumberOfSamples)
            self.setTriggerDelay(parmsUTcard.TriggerDelay)

            self.setPulseVoltage(parmsUTcard.PulseVoltage)
            self.setADTriggerSource(parmsUTcard.TriggerSource)
            self.setPulseWidth(parmsUTcard.PulseWidth)
            self.setToOneBurstFrequency(parmsUTcard.BurstFrequency, parmsUTcard.Polarity)
            self.setGain(parmsUTcard.Gain)
            self.setDCOffset(511)
            self.setLowPassFilter(48000000)


    def isDataReady(self):
        """
        Description:    Checks the status of the data acquisition process.

        wParam:         Not Used
        lParam:         Not Used
        Return Value:   DEVICE_NOT_READY (-500) if the USB-UT is not present
                    <0 if there is no data in the memory
                    >0 if there is data in the memory

        """
        ISDATAREADY = 1000
        return self.usbUT.USBUTParms(ISDATAREADY, 0, 0, 0)


    def softwareTrigger(self):
        """
        Description:    Generates a software trigger which causes the device to emit
                        a pulse and then digitize the response.  Note that in order
                        to generate a software trigger the trigger source must be set
                        to 'Software Trigger'.  Refer to SETADTRIGGERSOURCE in order
                        to set the trigger source.
        wParam:         Not Used
        lParam:         Not Used
        Return Value:   1 if successful
        """
        SOFTWARETRIGGER = 1001
        self.usbUT.USBUTParms(SOFTWARETRIGGER, 0, 0, 0);


    def setSamplingRate(self, Fs):
        """
        Description:    Sets the sampling rate.
        wParam:         Index from 0 to 3.
                    0: 50 MHz (default)
                    1: 25 MHz
                    2: 12.5 MHz
                    3: 6.25 MHz

        lParam:         Not Used
        Return Value:   1

        """
        if Fs == 50:
            option = 0
        elif Fs == 25:
            option = 1
        elif Fs == 12.5:
            option = 2
        elif Fs == 6.25:
            option = 3
        else:
            option = 0

        SETSAMPLINGRATE = 1002
        self.usbUT.USBUTParms(SETSAMPLINGRATE, 0, option, 0)


    def setLowPassFilter(self, frequency):
        """
        Description:    Sets the frequency of the low pass filter.
        wParam:         Frequency in Hz
        lParam:         Not Used
        Return Value:   1
        0=No Filter, 1=48 MHz, 2=28 MHz, 3=18 Mhz, 4=8.8 MHz, 5=7.5 MHz,
        6=6.7 MHz, 7=5.9 MHz
        """

        SETLOWPASSFILTER = 1004
        self.usbUT.USBUTParms(SETLOWPASSFILTER, 0, frequency, 0)


    def setHighPassFilter(self, frequency):
        """
        Description:    Sets the frequency of the high pass filter.
        wParam:         Frequency in Hz
        lParam:         Not Used
        Return Value:   1
        """

        SETHIGHPASSFILTER = 1005
        self.usbUT.USBUTParms(SETHIGHPASSFILTER, 0, frequency, 0)


    def setTransducerMode(self, option=1):
        """ASDK_SETTRANSDUCERMODE (mode = 2037), Set Transducer Mode
        This mode sets the transducer method, pulse/echo or through transmission (single or dual elements).
        parm: not used
        wParam:  0=pulse/echo, 1=through transmission

        """
        self.usbUT.USBUTParms(1008, 0, option, 0)


    def setBufferLength(self, N_Samples):
        """
        Description:    Sets the length of the buffer to be taken during the next
                        data acquisition cycle.
        wParam:         Number of samples ranging from 1 to 8190.  The default value
                        is 2000.
        lParam:         Not Used
        Return Value:   1
        """
        SETBUFFERLENGTH = 1009

        self.usbUT.USBUTParms(SETBUFFERLENGTH, 0, N_Samples, 0)


    def setTriggerDelay(self, N_Samples):
        """

        Description:    Sets the number of samples to skip after the pulse is
                        emitted before taking data.
        wParam:         Number of samples ranging from 1 to 16370.  The default
                        value is 2.
        lParam:         Not Used
        Return Value:   1

        """
        SETTRIGGERDELAY = 1010
        self.usbUT.USBUTParms(SETTRIGGERDELAY, 0, N_Samples, 0);


    def setPulseVoltage(self, Volt):
        """
        Description:    Sets the pulse voltage.
        wParam:         Index from 0 to 255.
                        The formula for the voltage is:
                        V = -(4 + 4000 / (0.392157 * Index + 13.5))
        lParam:         Not Used
        Return Value:   1
        """

        SETPULSEVOLTAGE = 1011

        if Volt < 40:
            Volt = 40
        if Volt > 300:
            Volt = 300

        Index = int((1.0 / 0.392157) * (  (4000 / (  Volt - 4  ) ) - 13.5   ))

        if Index > 255:
            Index = 255
        if Index < 0:
            Index = 0

        self.usbUT.USBUTParms(SETPULSEVOLTAGE, 0, Index, 0)


    def setPulseWidth(self, width):
        """
        Description:    Sets the pulse width for the normal pulser or the number of
                        half cycles for the tone burst pulser.
        wParam:         Index from 0 to 255.  Default is 0 which is the minimum
                        pulse width.
        lParam:         Not Used
        Return Value:   1
        """

        SETPULSEWIDTH = 1012
        Index = width if width <= 32 else 32
        self.usbUT.USBUTParms(SETPULSEWIDTH, 0, Index, 0)


    def setGain(self, Gain):
        """
        Description:    Sets the receiver gain.
        wParam:         Index from 0 to 1023.  The receiver gain can be set from
                        -20dB to 80dB with approximately a 0.1dB resolution.
                        Gain in dB = 0.0978 * (Index - 204)
        lParam:         Not Used
        Return Value:   1
        """

        SETGAIN = 1013

        if Gain >= 80:
            Gain = 80

        if Gain <= -20:
            Gain = -19.5

        Index = int((Gain / 0.0978) + 204)
        self.usbUT.USBUTParms(SETGAIN, 0, Index, 0)


    def setDCOffset(self, offset):
        """
        Description:    Sets the receiver DC offset.
        wParam:         Index from 0 to 1023.  The default is 511 which is 2.5V
        lParam:         Not Used
        Return Value:   1
        """
        SETDCOFFSET = 1014
        self.usbUT.USBUTParms(SETDCOFFSET, 0, offset, 0)


    def setADTriggerSource(self, whichTrigger):
        """
        Description:    Sets the trigger source of the analog to digital converter.

        wParam:     Index from 0 to 2.
                    0: +External Trigger
                    1:  Software Trigger
                    2: -External Trigger

        lParam:         Not Used
        Return Value:   1
        """
        SETADTRIGGERSOURCE = 1016

        if whichTrigger == 'software':
            option = 1
        elif whichTrigger == 'externalPos':
            option = 0
        elif whichTrigger == 'externalNeg':
            option = 2

        self.usbUT.USBUTParms(SETADTRIGGERSOURCE, 0, option, 0);


    def getEncoderCounter(self, whichEncoder=0):
        """
        Description:    Gets the encoder counter value.
        wParam:         Index from 0 to 3.
                    0: 1st encoder (X-axis)
                    1: 2nd encoder (Y-axis)
                    2: 3rd encoder (Z-axis)
                    3: 4th encoder (W-axis)
        lParam:         Not Used
        Return Value:   Counter value from -8388608 to 8388607
        """
        GETENCODERCOUNTER = 1017
        return self.usbUT.USBUTParms(GETENCODERCOUNTER, 0, whichEncoder, 0);


    def setEncoderCounter(self, whichEncoder=0, value=0):
        """
        Description:    Sets the encoder counter value.
        wParam:         Index from 0 to 3.
                    0: 1st encoder (X-axis)
                    1: 2nd encoder (Y-axis)
                    2: 3rd encoder (Z-axis)
                    3: 4th encoder (W-axis)
        lParam:         Counter value from -8388608 to 8388607
        Return Value:   1
        """
        SETENCODERCOUNTER = 1018
        self.usbUT.USBUTParms(SETENCODERCOUNTER, 0, whichEncoder, value);


    def getData(self, N_Samples):
        """
        Description:    Gets the digitized data from memory after data acquisition.
        wParam:         Pointer to the buffer (unsigned char *) to receive the data.
        lParam:         Number of bytes to transfer
        Return Value:   Number of bytes received from the device
        """
        GETDATA = 1019
        self.usbUT.USBUTParms(GETDATA, 0, self.data, N_Samples)

        #data=[]
        #for i in range(0,N_Samples):
        #    data.append( ord( self.data[i]  ) )
        #return np.array(data)

        #Faster!!!!!!
        return np.array(map(ord, self.data[0:N_Samples]))


    def getModelNumber(self):
        """
        Description:    Gets the model number of the device.
        wParam:         Pointer to the buffer (char *) to receive the data.  Buffer
                        must be at least 20 characters.
        lParam:         Not used
        Return Value:   1
        """
        GETMODELNUMBER = 1023
        self.usbUT.USBUTParms(GETMODELNUMBER, 0, self.number, 0)
        print("Model Number: ", repr(self.number.value))


    def getSerialNumber(self):
        """
        Description:    Gets the serial number of the device.
        wParam:         Pointer to the buffer (char *) to receive the data.  Buffer
                        must be at least 20 characters.
        lParam:         Not used
        Return Value:   1
        """
        GETSERIALNUMBER = 1024
        self.usbUT.USBUTParms(GETSERIALNUMBER, 0, self.number, 0);
        print("Serial Number: ", repr(self.number.value))


    def getOptionByte1(self):
        """
        Description:    Gets option byte 1 which contains options present on the
                        device.
        wParam:         Not used
        lParam:         Not used
        Return Value:   Option byte 1:      (0: Option not installed)
                    Bit 7: DAC              (1: Option is installed)
                    Bit 6: 1st Encoder
                    Bit 5: 2nd Encoder
                    Bit 4: Not used
                    Bit 3: Sync Out
                    Bit 2: Not Used
                    Bit 1: Not Used
                    Bit 0: Not Used
        """
        GETOPTIONBYTE1 = 1034
        return self.usbUT.USBUTParms(GETOPTIONBYTE1, 0, 0, 0);


    def setPRF(self, frequency=100):
        """
        Description:    Sets the PRF.
        wParam:         Frequency (0=Off)
        lParam:         Not used
        Return Value:   1
        """
        SETPRF = 1038
        self.usbUT.USBUTParms(SETPRF, 0, frequency, 0)


    def setIOPort(self, whichPort=0, value=255):
        """
        Description:    Sets the I/O port value.
        wParam:         I/O port to set:
                        0: Port 0
                        1: Port 1
        lParam:     Value for output port.  It will be ignored if the port is
                    configured as an input port.
        Return Value:   1
        *******************************************************************************/
        """
        SETIOPORT = 1039
        self.usbUT.USBUTParms(SETIOPORT, 0, whichPort, value)


    def configureIOPort(self, IO=0, configuration=255):
        """
        Description:    Configures the I/O port.
        wParam:         I/O port to configure:
                    0: Port 0
                    1: Port 1
        lParam:         I/O Port configuration byte
                    Bit             7   6   5   4   3   2   1   0
                    Configuration   x   x   x   x   x   x   x   x
                    x: 0 for Input
                    x: 1 for Output
        Return Value:   1
        """

        CONFIGUREIOPORT = 1040
        self.usbUT.USBUTParms(CONFIGUREIOPORT, 0, IO, configuration)


    def setToOneBurstFrequency(self, frequency, polarity='negative'):
        """

           Description: Sets the frequency of the tone burst pulser.
           wParam:          Frequency in kHz from 20 to 10,000
           lParam:          Polarity: 0 = Positive, 1 = Negative
           Return Value:    1
        """
        SETTONEBURSTFREQUENCY = 1046
        if polarity == 'negative':
            self.usbUT.USBUTParms(SETTONEBURSTFREQUENCY, 0, frequency, 1)
        elif polarity == 'positive':
            self.usbUT.USBUTParms(SETTONEBURSTFREQUENCY, 0, frequency, 0)
        else:
            self.usbUT.USBUTParms(SETTONEBURSTFREQUENCY, 0, frequency, 1)


    def setHardwareHPF(self, option):
        """
        ASDK_SETHARDWAREHPF (mode = 2036), Set Hardware Filter
        This mode sets the hardware high pass filter for new version of the hardware.
        parm: not used
        wParam: hardware high pass filter for USB-UT350 only 0=12kHz, 1=600 kHz only

        """
        if option == '12kHz':
            self.usbUT.USBUTParms(1047, 0, 0, 0)
        elif option == '600kHz':
            self.usbUT.USBUTParms(1047, 0, 1, 0)
        else:
            self.usbUT.USBUTParms(1047, 0, 0, 0)


    def setAveraging(self, number):
        """
        Description:    Configures averaging.
        wParam:         Not used
        lParam:         Number of waveforms to average for each waveform returned.
        Return Value:   1
        """
        SETAVERAGING = 1075
        self.average = number
        self.usbUT.USBUTParms(SETAVERAGING, 0, 0, number)


    def setFilter(self, choose='lowPassFilter', frequency=1000000):
        """
        Description:    Sets the filter specified by wParam.
        wParam:         Index from 0 to 1:
                    0: Low Pass Filter
                    1: High Pass Filter
        lParam:         Filter frequency in Hz.
        Return Value:   1
        """
        SETFILTER = 1076
        if choose == 'lowPassFilter':
            self.usbUT.USBUTParms(SETFILTER, 0, 0, frequency)
        else:
            self.usbUT.USBUTParms(SETFILTER, 0, 1, frequency)


    def setEncoderMode(self, mode=1):
        """
        Description:    Sets the encoder pins to function either as encoder pins or
                        digital I/O.  When these pins are set to function as digital
                        I/O port 1, the encoders will be disabled.
        wParam:         Not used
        lParam:         0: Encoder pins function as encoder pins.
                        1: Encoder pins function as digital I/O port 1.
        Return Value:   1
        """
        SETENCODERMODE = 5050
        self.usbUT.USBUTParms(SETENCODERMODE, 0, 0, mode)


    def setSpeaker(self):

        SETSPEAKER = 5051
        self.usbUT.USBUTParms(SETSPEAKER, 0, 0, 1)
        time.sleep(0.1)
        self.usbUT.USBUTParms(SETSPEAKER, 0, 0, 0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    card = Ultratek()
    handler = card.init()
    if handler:
        card.setSamplingRate(6.25)
        card.setBufferLength(1800)
        card.setGain(40)
        card.setTriggerDelay(10)
        card.setADTriggerSource('software')

    print("Single acquisition mode")

    card.softwareTrigger()

    while card.isDataReady(): # Wait for the data to be available
        start = time.time()
        data = card.getData(1800) # Discard initial garbage packet
        print((time.time() - start) * 1e3)
        data = card.getData(1800)

    plt.figure()
    plt.plot(data)
    plt.show()







