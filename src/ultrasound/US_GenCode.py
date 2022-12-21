# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:51:54 2019

@author: Alberto
"""

import numpy as np
from scipy import signal

#, ArrLenMax = 1024*16, gencodeNo = 1, FileName='', DeadTime_Samples=0, CancelDuration=10, AddZerosInFront_Samples=10):
#ArrLenMax = 1024*16 Maximum of array
#        gencodeNo = number of the gencode to add to the name when saving
#        FileName = full name including path where to save the generated gencode
#    Optional not used so far:
#        DeadTime_Samples = 0
#        CancelDuration=10
#        AddZerosInFront=10

# Pulse GenCode
def GC_MakePulse(Param = 'frequency', ParamVal = 5e6, SignalPolarity = 2, Fs = 200e6):
    """
    Make Pulse gencode
    Inputs
        Param = ['Frequency' or 'Duration' or 'Samples']
        ParamVal = value, units must be [Hz or sec or Samples]
        SignalPolarity = [2 or 1 or -1], polarity of the gencode
        Fs = Sampling frequency in Hz   
    Outputs
        Signal
    """ 
    if Param.upper() == 'FREQUENCY':
        samples = int(np.round(1./ParamVal*Fs/2))        
    elif Param.upper() == 'DURATION':
        samples = int(np.round(ParamVal*Fs))
    elif Param.upper() == 'SAMPLES':
        samples = int(ParamVal)
    Signal = np.zeros(samples+2)
    Signal[1:samples+1]=1
    # signal polarity
    if SignalPolarity == -1 or SignalPolarity == 1:
        Signal = Signal*SignalPolarity        
    return Signal

# Chirp GenCode
def GC_MakeChirp(Fstart=2e6,Fend=8e6,Duration=3e-6, method='linear', Phase=270, SignalPolarity = 2, Fs = 200e6):
    """
    Make Chirp gencode
    Inputs
        Fstart = start (lower) frequency 
        Fend = end (higher) frequency 
        Duration = duration of the pulse in seconds
        Phase = phase of the pulse in degrees
        method = sort of chirp {‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’}
        SignalPolarity = [2 or 1 or -1], polarity of the gencode
        Fs = Sampling frequency in Hz       
    Outputs
        Signal
    """    
    t = np.arange(0,Duration-1/Fs,1/Fs)
    SignalChirp=signal.chirp(t, Fstart, Duration, Fend, method, Phase); 
    nr=0
    Signal = np.zeros(len(SignalChirp))
    for Value in SignalChirp:
        if Value>0:
            Signal[nr]=1
        else:
            Signal[nr]=0
        nr+=1
    # signal polarity
    if SignalPolarity == -1 or SignalPolarity == 1:
        Signal = Signal*SignalPolarity
    else:
        Signal=(Signal*2)-1
        Signal[0]=0
        Signal[-1]=0
        
    return Signal

# Chirp GenCode
def GC_MakeBurst(Fo=5e6, NoCycles=5, SignalPolarity=2, Fs=200e6):
    """
    Make Burst gencode
    Inputs
        Fo = Central frequency Hz
        NoCycles = Number of cycles (integer)        
        SignalPolarity = [2 or 1 or -1], polarity of the gencode
        Fs = Sampling frequency in Hz       
    Outputs
        Signal
    """       
    PeriodSamples = int(np.round(Fs/Fo))
    Cycle = np.zeros(PeriodSamples)
    Cycle[0:int(np.round(PeriodSamples/2))-1] = 1
    Signal = np.zeros(1)
    for i in range(0,NoCycles):
        Signal = np.concatenate((Signal,Cycle))
    # signal polarity
    if SignalPolarity == -1 or SignalPolarity == 1:
        Signal = Signal*SignalPolarity
    else:
        Signal=(Signal*2)-1
        Signal[0]=0
        Signal[-1]=0
    
    return Signal

def Signa2GenCode(Signal, SignalPolarity = 2, DeadTime_Samples=0, CancelDuration=0, AddZerosInFront_Samples=0):
    """
    convert signal to excitation gencode
    Inputs
        Signal = signal to convert
        DeadTime_Samples = 
        CancelDuration = 
        AddZerosInFront_Samples =
    Outputs
        GenCode = Generated GenCode
    """
    GenCode1 = np.zeros(len(Signal))
    GenCode2 = np.zeros(len(Signal))
    nr=0
    for Value in Signal:        
        if Value>0:
            GenCode1[nr]=1
        elif Value<0:
            GenCode2[nr]=1
        nr+=1
    GenCode=GenCode1*2+GenCode2
    
    #Add dead time:
    if DeadTime_Samples>0:    
        for nr in range(1,len(GenCode)-1):
            if GenCode1[nr]>GenCode1[nr+1]:
                GenCode1[(nr-DeadTime_Samples+1):nr]=0
                GenCode[(nr-DeadTime_Samples+1):nr]=0
            if GenCode2[nr]>GenCode2[nr+1]:
                GenCode2[(nr-DeadTime_Samples+1):nr]=0
                GenCode[(nr-DeadTime_Samples+1):nr]=0
            
    #Add canceling:
    if CancelDuration>0:        
        GenCode=np.concatenate((GenCode[:-1], np.ones(CancelDuration)*3))
        GenCode1=np.concatenate((GenCode1[:-1], np.ones(CancelDuration)))
        GenCode2=np.concatenate((GenCode2[:-1], np.ones(CancelDuration)))


    #Add zeros in front if delay needed, comment out if not used
    if AddZerosInFront_Samples>0:
        GenCode=np.concatenate((np.zeros(AddZerosInFront_Samples), GenCode))
        GenCode1=np.concatenate((np.zeros(AddZerosInFront_Samples), GenCode1))
        GenCode2=np.concatenate((np.zeros(AddZerosInFront_Samples), GenCode2))
    
    return GenCode


def MakeGenCode(Excitation='Pulse', Param='frequency',ParamVal = 5e6, SignalPolarity = 2, Fs = 200e6,
                DeadTime_Samples=0, CancelDuration=0, AddZerosInFront_Samples=0):
    """
    Make excitation gencodes. 
    Inputs
        Excitation = {'Pulse','Chirp','Burst'}
        Param = {'Frequency','Duration'}: use only for pulse
        ParamVal = 
            {Fo or Duraion} for Pulse
            {Fstart,Fend,Duration,Method,Phase} for Chirp
            {Fo,NoCycles} for Burst
        SignalPolarity = {2, or 1 or -1}, usually 2 (bipolar)
        Fs = Sampling frequency (200e6 for usual KTU pulser/receiver)
        DeadTime_Samples = Not sure, leave 0
        CancelDuration = Not sure, leave 0
        AddZerosInFront_Samples = Not sure, leave 0
    Outputs
        GenCode = Produced GenCode
    """
    if Excitation.upper()=='PULSE':
        Signal = GC_MakePulse(Param = Param, ParamVal = ParamVal, SignalPolarity = SignalPolarity, Fs = Fs)
    elif Excitation.upper()=='CHIRP':
        Signal = GC_MakeChirp(Fstart=ParamVal[0], Fend=ParamVal[1], Duration=ParamVal[2], method=ParamVal[3], Phase=ParamVal[4], SignalPolarity = SignalPolarity, Fs = Fs)
    elif Excitation.upper()=='BURST':
        Signal = GC_MakeBurst(Fo=ParamVal[0], NoCycles=ParamVal[1], SignalPolarity = SignalPolarity, Fs = Fs)
    GenCode = Signa2GenCode(Signal, SignalPolarity = SignalPolarity, DeadTime_Samples=DeadTime_Samples, CancelDuration=CancelDuration, AddZerosInFront_Samples=AddZerosInFront_Samples)
    return GenCode

