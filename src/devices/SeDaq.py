from ctypes import c_uint8, c_uint16, c_double, byref, cdll
import numpy as np
from pathlib import Path

def Gencode_from_file(f_n):
    with open(f_n, 'r') as f:
        lines = []
        for line in f:
            line = line.strip('\r\n')
            lines.append(line)
    return lines

def ClosestPowerOf2(gencode_len):
    # Cambiado ligeramente del original, que esta comentado
    # N = 0
    # while 2**N < gencode_len:
    #     N = N + 1
    
    # return N
    return int(np.ceil(np.log2(np.abs(gencode_len))))

class SeDaqDLL:
    '''Class for handling the Lithuanian acquisition system'''
    def __init__(self, dllPath=None):
        if dllPath is None:
            cmd = cdll.LoadLibrary((Path(__file__).parents[2] / "./SeDaqDLL.dll").__str__())
        else:
            cmd = cdll.LoadLibrary(dllPath)

        self.SeDaqDLL_SetExcVoltage = cmd.SeDaqDLL_SetExcVoltage
        self.SeDaqDLL_SetSoftTrig = cmd.SeDaqDLL_SetSoftTrig
        self.SeDaqDLL_SetExcWave = cmd.SeDaqDLL_SetExcWave
        self.SeDaqDLL_SetGain = cmd.SeDaqDLL_SetGain
        self.SeDaqDLL_GetAScan = cmd.SeDaqDLL_GetAScan
        self.SeDaqDLL_SetRecLen = cmd.SeDaqDLL_SetRecLen
        self.SeDaqDLL_SetBankDelay= cmd.SeDaqDLL_SetBankDelay
        self.SeDaqDLL_Init = cmd.SeDaqDLL_Init
        self.SeDaqDLL_EnableExc = cmd.SeDaqDLL_EnableExc
        self.SeDaqDLL_UartSend = cmd.SeDaqDLL_UartSend
        self.SeDaqDLL_UartGet = cmd.SeDaqDLL_UartGet
        self.SeDaqDLL_SetRelay = cmd.SeDaqDLL_SetRelay
        self.SeDaqDLL_UartRead = cmd.SeDaqDLL_UartRead
        self.SeDaqDLL_GetGain = cmd.SeDaqDLL_GetGain
                
        ADC1 = c_uint16 * (1024*32)
        ADC2 = c_uint16 * (1024*32)
        self.DataADC1 = ADC1()
        self.DataADC2 = ADC2()
        
        self.GenCodes = []
        self.RecLen = 1024*32
    
        self.AvgSamplesNumber = 10
        self.Quantiz_Levels = 1024
    
    # ============ GETS ============    
    def GetAScan(self):
        '''
        Makes a trigger and acquires data on both channels.

        Returns
        -------
        None.

        Docstring: Arnau, 20/02/2023
        '''
        self.SeDaqDLL_SetSoftTrig(1)
        self.SeDaqDLL_GetAScan(byref(self.DataADC1),1)
        self.SeDaqDLL_GetAScan(byref(self.DataADC2),2)
    
    def GetAScan1(self):
        '''
        Makes a trigger and acquires data on channel 1.

        Returns
        -------
        None.

        Docstring: Arnau, 20/02/2023
        '''
        self.SeDaqDLL_SetSoftTrig(1)
        self.SeDaqDLL_GetAScan(byref(self.DataADC1),1)
        
    def GetAScan2(self):
        '''
        Makes a trigger and acquires data on channel 2.

        Returns
        -------
        None.

        Docstring: Arnau, 20/02/2023
        '''
        self.SeDaqDLL_SetSoftTrig(1)
        self.SeDaqDLL_GetAScan(byref(self.DataADC2),2)

    def GetCycleAScan1(self, CycleNo):
        CycleAScan1 = []
        for i in range(CycleNo):
            self.GetAScan1()
            CycleAScan1.append(self.DataADC1[:self.RecLen])
        return np.array(CycleAScan1)
    
    def GetGain(self, ch):
        '''
        Returns the current gain of the specified channel.

        Parameters
        ----------
        ch : int
            Channel (1 or 2).

        Returns
        -------
        int
            Gain in dB of the channel.

        Aranu, 20/02/2023
        '''
        return self.SeDaqDLL_GetGain(ch)

    def GetGain1(self):
        '''
        Returns the current gain of channel 1.

        Returns
        -------
        int
            Gain in dB of channel 1.

        Aranu, 20/02/2023
        '''
        return self.SeDaqDLL_GetGain(1)
    
    def GetGain2(self):
        '''
        Returns the current gain of channel 2.

        Returns
        -------
        int
            Gain in dB of channel 2.

        Aranu, 20/02/2023
        '''
        return self.SeDaqDLL_GetGain(2)



    def _GetAscan(self, ch, Smin, Smax):
        '''
        Returns the acquired Ascan from Smin to Smax samples of the specified 
        channel as a numpy array. The Ascan is scaled by self.Quantiz_Levels
        and self.AvgSamplesNumber.

        Parameters
        ----------
        ch : int
            Channel (1 or 2).
        Smin : int
            First sample.
        Smax : int
            Last sample.

        Returns
        -------
        Ascan : ndarray
            The acquiered Ascan.

        Docstring: Aranu, 20/02/2023
        '''
        Ascan = np.zeros(Smax - Smin)
        Flag = self.AvgSamplesNumber

        while Flag > 0:
            self.GetAScan() # get Ascan
            if ch==1:
                Aux = np.array(list(map(float, self.DataADC1[Smin:Smax]))) # get Ascan
            else:
                Aux = np.array(list(map(float, self.DataADC2[Smin:Smax]))) # get Ascan
            Aux = (Aux - self.Quantiz_Levels/2) / self.Quantiz_Levels # Normalize 
            Aux = Aux - np.mean(Aux) # remove mean
            
            if not(np.all(Aux==0.0)):	            
                Ascan = Ascan + Aux		
                Flag -= 1
                		
        Ascan = Ascan / self.AvgSamplesNumber #calculate averaged Ascan
        Ascan = Ascan - np.mean(Ascan) #substract mean value
        return Ascan

    def GetAscan_Ch2(self, Smin, Smax):
        '''
        Returns the acquired Ascan from Smin to Smax samples of channel 2 as a 
        numpy array. The Ascan is scaled by self.Quantiz_Levels and
        self.AvgSamplesNumber.

        Parameters
        ----------
        Smin : int
            First sample.
        Smax : int
            Last sample.

        Returns
        -------
        Ascan : ndarray
            The acquiered Ascan.

        Aranu, 20/02/2023
        '''
        return self._GetAscan(2, Smin, Smax)

    def GetAscan_Ch1(self, Smin, Smax):
        '''
        Returns the acquired Ascan from Smin to Smax samples of channel 1 as a 
        numpy array. The Ascan is scaled by self.Quantiz_Levels and
        self.AvgSamplesNumber.

        Parameters
        ----------
        Smin : int
            First sample.
        Smax : int
            Last sample.

        Returns
        -------
        Ascan : ndarray
            The acquiered Ascan.

        Aranu, 20/02/2023
        '''
        return self._GetAscan(1, Smin, Smax)

    def GetAscan_Ch1_Ch2(self, Smin, Smax):
        '''
        Returns the acquired Ascan of both channels. If Smin and Smax are
        tuples, channel 1 correspons to the first element of the tuple and
        channel 2 corresponds to the second element of the tuple. If Smin or
        Smax are not tuples, both channels have the same Smin and Smax.

        Parameters
        ----------
        Smin : tuple or int
            First sample.
        Smax : tuple or int
            Last sample.

        Returns
        -------
        Ascan_Ch1 : ndarray
            The acquiered Ascan of channel 1.
        Ascan_Ch2 : ndarray
            The acquiered Ascan of channel 2.

        Aranu, 20/02/2023
        '''
        if isinstance(Smin, tuple):
            Smin1, Smin2 = Smin
        else:
            Smin1 = Smin
            Smin2 = Smin
        if isinstance(Smax, tuple):
            Smax1, Smax2 = Smax
        else:
            Smax1 = Smax
            Smax2 = Smax
        
        Ascan_Ch2 = np.zeros(Smax2-Smin2)
        Ascan_Ch1 = np.zeros(Smax1-Smin1)
        Flag = self.AvgSamplesNumber
        while Flag > 0:
            self.GetAScan() # get Ascan        
            
            Aux_Ch1 = np.array(list(map(float, self.DataADC1[Smin1:Smax1]))) # get Ascan
            Aux_Ch2 = np.array(list(map(float, self.DataADC2[Smin2:Smax2]))) # get Ascan
            
            Aux_Ch1 = (Aux_Ch1 - self.Quantiz_Levels/2) / self.Quantiz_Levels # Normalize
            Aux_Ch2 = (Aux_Ch2 - self.Quantiz_Levels/2) / self.Quantiz_Levels # Normalize
            
            Aux_Ch1 = Aux_Ch1 - np.mean(Aux_Ch1) # remove mean
            Aux_Ch2 = Aux_Ch2 - np.mean(Aux_Ch2) # remove mean
            
            if not(np.all(Aux_Ch2==0.0)) and not(np.all(Aux_Ch1==0.0)):	            
                Ascan_Ch2 = Ascan_Ch2 + Aux_Ch2
                Ascan_Ch1 = Ascan_Ch1 + Aux_Ch1
                Flag -= 1
        
        Ascan_Ch2 = Ascan_Ch2 / self.AvgSamplesNumber # calculate averaged Ascan
        Ascan_Ch2 = Ascan_Ch2 - np.mean(Ascan_Ch2) # substract mean value
        Ascan_Ch1 = Ascan_Ch1 / self.AvgSamplesNumber # calculate averaged Ascan
        Ascan_Ch1 = Ascan_Ch1 - np.mean(Ascan_Ch1) # substract mean value
        return Ascan_Ch1, Ascan_Ch2


    # ============ SETS ============ 
    def SetRecLen(self, RecLen):
        '''
        Sets the total recording length for both channels in samples.

        Parameters
        ----------
        RecLen : int
            Total recording length.

        Returns
        -------
        None.

        Docstring: Aranu, 20/02/2023
        '''
        self.SeDaqDLL_SetRecLen(RecLen, 1)
        self.SeDaqDLL_SetRecLen(RecLen, 2)
        self.RecLen = RecLen
           
    def SetGenCode(self, GenCodeNo):
        '''
        Set the Generator Code to the specified number.

        Parameters
        ----------
        GenCodeNo : int
            Number (starting with 1) of the GenCode.

        Returns
        -------
        None.

        Docstring: Aranu, 20/02/2023
        '''
        GenArrayTo = self.GenCodes[GenCodeNo-1] # 
        BytesTot = len(GenArrayTo)
        self.SeDaqDLL_SetExcWave(byref(GenArrayTo),BytesTot,0)
                
    def SetBankDelay(self, BankDelay):
        self.SetBankDelay(c_double(BankDelay),1)
        self.SetBankDelay(c_double(BankDelay),2)

    def SetGain1(self, gain):
        '''
        Set the gain of channel 1.

        Parameters
        ----------
        gain : int
            Gain in dB.

        Returns
        -------
        None.

        Docstring: Aranu, 20/02/2023
        '''
        self.SeDaqDLL_SetGain(c_double(gain),1)

    def SetGain2(self, gain):
        '''
        Set the gain of channel 2.

        Parameters
        ----------
        gain : int
            Gain in dB.

        Returns
        -------
        None.

        Docstring: Aranu, 20/02/2023
        '''
        self.SeDaqDLL_SetGain(c_double(gain),2)

    def SetExtVoltage(self, voltage):
        '''DOES NOT WORK.
        Set the excitation voltage.

        Parameters
        ----------
        voltage : float
            Voltage in volts.

        Returns
        -------
        None.

        Docstring: Aranu, 20/02/2023
        '''
        self.SeDaqDLL_SetExcVoltage(voltage)

    def SetRelay(self, mode):
        '''DOES NOT WORK.
        Turn the relay on (True) or off (False).

        Parameters
        ----------
        mode : bool
            If True, turn the relay on.

        Returns
        -------
        None.

        Docstring: Aranu, 20/02/2023
        '''
        self.SeDaqDLL_SetRelay(mode)


    # ============ GENCODE ============ 
    def UpdateGenCode(self, gencode):
        '''
        Update the current GenCode to the specified one.

        Parameters
        ----------
        gencode : list or ndarray
            GenCode to set.

        Returns
        -------
        None.

        Docstring: Aranu, 20/02/2023
        '''
        gencode_len = len(gencode)
        N = ClosestPowerOf2(gencode_len)
        BytesTot = 2**N
        GenArrayTo = (c_uint8 * BytesTot)()
        for i in range(BytesTot):
            if i<gencode_len:
                GenArrayTo[i] = c_uint8(int(gencode[i]))
            else:
                GenArrayTo[i] = c_uint8(0)
        BytesTot = len(GenArrayTo)
        self.SeDaqDLL_SetExcWave(byref(GenArrayTo),BytesTot,0)
    
    def AddGenCode(self, file_name):
        gencode = Gencode_from_file(file_name)

        gencode_len = len(gencode)
        N = ClosestPowerOf2(gencode_len)
        BytesTot = 2**N

        GenArrayTo = (c_uint8 * BytesTot)()
        for i in range(BytesTot):
            if i<gencode_len:
                GenArrayTo[i] = c_uint8(int(gencode[i]))
            else:
                GenArrayTo[i] = c_uint8(0)
        self.GenCodes.append(GenArrayTo)