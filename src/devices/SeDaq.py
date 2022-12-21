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
    def __init__(self, dllPath=None):
        # cmd = cdll.LoadLibrary(r"SeDaqDLL.dll")
        if dllPath is None:
            cmd = cdll.LoadLibrary(Path(__file__).parents[2] / "./DLL/SeDaqDLL.dll")
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
    
    
    # ============ GETS ============    
    def GetAScan(self):
        self.SeDaqDLL_SetSoftTrig(1)
        self.SeDaqDLL_GetAScan(byref(self.DataADC1),1)
        self.SeDaqDLL_GetAScan(byref(self.DataADC2),2)
    
    def GetAScan1(self):
        self.SeDaqDLL_SetSoftTrig(1)
        self.SeDaqDLL_GetAScan(byref(self.DataADC1),1)
        
    def GetAScan2(self):
        self.SeDaqDLL_SetSoftTrig(1)
        self.SeDaqDLL_GetAScan(byref(self.DataADC2),2)

    def GetCycleAScan1(self, CycleNo):
        CycleAScan1 = []
        for i in range(CycleNo):
            self.GetAScan1()
            CycleAScan1.append(self.DataADC1[:self.RecLen])
        return np.array(CycleAScan1)
    
    def GetGain(self, ch):
        return self.SeDaqDLL_GetGain(ch)

    def GetGain1(self, ch):
        return self.SeDaqDLL_GetGain(1)
    
    def GetGain2(self, ch):
        return self.SeDaqDLL_GetGain(2)


    # ============ SETS ============ 
    def SetRecLen(self, RecLen):
        self.SeDaqDLL_SetRecLen(RecLen, 1)
        self.SeDaqDLL_SetRecLen(RecLen, 2)
        self.RecLen = RecLen
           
    def SetGenCode(self, GenCodeNo):
        GenArrayTo = self.GenCodes[GenCodeNo-1] # 
        BytesTot = len(GenArrayTo)
        self.SeDaqDLL_SetExcWave(byref(GenArrayTo),BytesTot,0)
                
    def SetBankDelay(self, BankDelay):
        self.SetBankDelay(c_double(BankDelay),1)
        self.SetBankDelay(c_double(BankDelay),2)

    def SetGain1(self, gain):
        self.SeDaqDLL_SetGain(c_double(gain),1)

    def SetGain2(self, gain):
        self.SeDaqDLL_SetGain(c_double(gain),2)

    def SetExtVoltage(self, voltage):
        self.SeDaqDLL_SetExcVoltage(voltage)

    def SetRelay(self, mode):
        self.SeDaqDLL_SetRelay(mode)


    # ============ GENCODE ============ 
    def UpdateGenCode(self, gencode):
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