# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:27:21 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import serial
import numpy as np
import time


#%%
class Arduino:
    '''Class for temperature readings via serial comm. with the Arduino board'''
    def __init__(self, board='Arduino UNO', baudrate=9600, port='COM3', twoSensors=False, N_avg=1, timeout=None):
        self.board = board                 # Board type
        self.N_avg = N_avg                 # Number of temperature measurements to be averaged - int
        self.twoSensors = twoSensors
        
        self.intDigits = 2
        self.floatDigits = 3
        self.sepLength = 0
        
        self.ser = serial.Serial(port, baudrate, timeout=timeout)  # open comms

    @property
    def port(self):
        '''Serial communication port'''
        return self.ser.port

    @port.setter
    def port(self, p):
        self.ser.port = p

    @property
    def baudrate(self):
        '''Serial communication baudrate'''
        return self.ser.baudrate

    @baudrate.setter
    def baudrate(self, b):
        self.ser.baudrate = b
    
    def open(self):
        '''
        Open serial connection.

        Returns
        -------
        None.

        '''
        if not self.ser.isOpen():
            self.ser.open()

    def close(self):
        '''
        Close serial connection.

        Returns
        -------
        None.

        '''
        try:
            if self.ser.isOpen():
                self.ser.close()
                print(f'Serial communication with {self.board} at port {self.port} closed successfully.')
            else:
                print(f'Serial communication with {self.board} at port {self.port} was already closed.')
        except Exception as e:
            print(e)

    def _parseTemperatureData(self, line):
        '''
        Converts a line of serial data to floats.

        Parameters
        ----------
        line : bytes
            See self.getTemperature() for more info.

        Returns
        -------
        temperature1 : float
            Temperature of sensor 1.
        temperature2 : float
            Temperature of sensor 2.

        Arnau, 01/02/2023
        '''
        if self.twoSensors:
            threshold = self.intDigits + self.floatDigits + 1 # +1 due to the decimal point itself
            temp1 = float(line[:threshold])
            temp2 = float(line[threshold+self.sepLength:-1])
            return temp1, temp2
        return float(line)
    
    def getTemperature(self, error_msg: str=None, exception_msg: str=None):
        '''
        Get a temperature reading via serial communication. The expected format
        of the readings is one float if self.twoSensors==False or two floats if 
        self.twoSensors==True. The expected number of integer digits and float
        digits are self.intDigits and self.floatDigits, respectively. In the
        case where self.twoSensors==True, the separation between both floats 
        can have any length (including zero).
        
        Examples of accepted formats are:
            b'12.34567.890'
            
            b'12.345'
            
            b'12.345 67.890'
            
            b'12.345-----67.890'
        
        If N_avg > 1, then N_avg readings are taken and averaged. An error
        message is printed when the float numbers are parsed correctly but do
        not follow this format. An exception message is printed if the numbers
        could not be parsed to float. If the messages are None, then nothing is
        printed. In all cases, the readings are retaken until a good reading is
        found.
        
        Please note that the serial communication should already be open and it
        is NOT closed by this function.
    
        Parameters
        ----------
        error_msg : str, optional
            Error message to print when a float number is correctly parsed but does
            not follow the proper format (xx.xxx). The default is None.
        exception_msg : str, optional
            Error message to print when an exception is raised when attempting to
            parse the bytes of data to float numbers. The default is None.
    
        Returns
        -------
        mean1 : float
            Measurement of sensor 1.
        mean2 : float
            Measurement of sensor 2.
    
        Arnau, 30/01/2023
        '''
        lines = [None]*self.N_avg # init. list of lines
        temp1 = np.zeros(self.N_avg);
        if self.twoSensors:
            temp2 = np.zeros(self.N_avg) # init. float temperature vectors
        GoodMeasurement = False
        while not GoodMeasurement:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        
            start_time = time.time()
            while self.ser.read() != b'\n':
                if time.time() - start_time > 5:
                    print('Timeout')
                    break
        
            for i in range(self.N_avg):
                lines[i] = self.ser.readline()
            
            GoodMeasurement = True
            
            for i in range(self.N_avg):
                try:
                    if lines[i] != b'':
                        if self.twoSensors:
                            temp1[i], temp2[i] = self._parseTemperatureData(lines[i])
                        else:
                            temp1[i] = self._parseTemperatureData(lines[i])
                    else:
                        if error_msg is not None:
                            print(error_msg)
                        GoodMeasurement = False
        
                except Exception:
                    if exception_msg is not None:
                        print(exception_msg)
                    GoodMeasurement = False
                    
        mean1 = np.mean(temp1)
        if self.twoSensors:
            mean2 = np.mean(temp2)
            return mean1, mean2
        return mean1


