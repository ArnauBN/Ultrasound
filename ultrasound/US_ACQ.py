'''
Tool Box to control and acquiare signals using KTU hardware
13/02/2019
Alberto Rodriguez
'''

import numpy as np
from SeDaq import *
import matplotlib.pylab as plt
import serial
#import winsound

''' Get Ascans '''
#Get Ascans ->
def GetAscan_Ch2(Smin, Smax, AvgSamplesNumber = 10, Quantiz_Levels = 1024):
    '''
    Get Ascan from channel 2 only and extract data between Smin and Smax. Data
    is normalized according to quantization levels and averaged according to
    the number of samples acquired at each point.

    Parameters
    ----------
    Smin : tuple or int
        First sample.
    Smax : tuple or int
        Last Sample.
    AvgSamplesNumber : int, optional
        Number of Ascan to average in each acq. The default is 10.
    Quantiz_Levels : int, optional
        Number of levels of acq (2^B). The default is 1024.

    Returns
    -------
    Ascan_Ch2 : ndarray
        Acquired Ascan of channel 2.

    Revised: Arnau, 12/12/2022
    '''
    SeDaq = SeDaqDLL()
    Ascan_Ch2 = np.zeros(Smax-Smin)
    Flag = AvgSamplesNumber

    while Flag > 0:
        SeDaq.GetAScan() #get Ascan        
        Aux_Ch2 = np.array(list(map(float,SeDaq.DataADC2[Smin:Smax]))) #get Ascan WP
        Aux_Ch2 = (Aux_Ch2 - Quantiz_Levels/2)/Quantiz_Levels # Normalize 
        Aux_Ch2 = Aux_Ch2 - np.mean(Aux_Ch2) # remove mean
        
        if not(np.all(Aux_Ch2==0.0)):	            
            Ascan_Ch2 = Ascan_Ch2 + Aux_Ch2		
            Flag -= 1
            		
    Ascan_Ch2 = Ascan_Ch2 / AvgSamplesNumber #calculate averaged Ascan
    Ascan_Ch2 = Ascan_Ch2 - np.mean(Ascan_Ch2) #substract mean value
    return Ascan_Ch2

def GetAscan_Ch1(Smin, Smax, AvgSamplesNumber = 10, Quantiz_Levels = 1024):
    '''
    Get Ascan from channel 1 only and extract data between Smin and Smax. Data
    is normalized according to quantization levels and averaged according to
    the number of samples acquired at each point.

    Parameters
    ----------
    Smin : tuple or int
        First sample.
    Smax : tuple or int
        Last Sample.
    AvgSamplesNumber : int, optional
        Number of Ascan to average in each acq. The default is 10.
    Quantiz_Levels : int, optional
        Number of levels of acq (2^B). The default is 1024.

    Returns
    -------
    Ascan_Ch2 : ndarray
        Acquired Ascan of channel 1.

    Revised: Arnau, 12/12/2022
    '''
    SeDaq = SeDaqDLL()
    Ascan_Ch1 = np.zeros(Smax-Smin)
    Flag = AvgSamplesNumber
    while Flag > 0:
        SeDaq.GetAScan() #get Ascan        
        Aux_Ch1 = np.array(list(map(float,SeDaq.DataADC1[Smin:Smax]))) #get Ascan WP
        Aux_Ch1 = (Aux_Ch1 - Quantiz_Levels/2)/Quantiz_Levels # Normalize 
        Aux_Ch1 = Aux_Ch1 - np.mean(Aux_Ch1) # remove mean
        
        if not(np.all(Aux_Ch1==0.0)):	            
            Ascan_Ch1 = Ascan_Ch1 + Aux_Ch1		
            Flag -= 1
            		
    Ascan_Ch1 = Ascan_Ch1 / AvgSamplesNumber #calculate averaged Ascan
    Ascan_Ch1 = Ascan_Ch1 - np.mean(Ascan_Ch1) #substract mean value
    return Ascan_Ch1

def GetAscan_Ch1_Ch2(Smin, Smax, AvgSamplesNumber = 10, Quantiz_Levels = 1024):
    '''
    Get Ascans. It checks that Ascans are not zeros, because sometimes happens...

    Parameters
    ----------
    Smin : tuple or int
        First sample of both channels.
    Smax : tuple or int
        Last Sample of both channels.
    AvgSamplesNumber : int, optional
        Number of Ascan to average in each acq. The default is 10.
    Quantiz_Levels : int, optional
        Number of levels of acq (2^B). The default is 1024.

    Returns
    -------
    Ascan_Ch1 : ndarray
        Acquired Ascan of channel 1.
    Ascan_Ch2 : ndarray
        Acquired Ascan of channel 2.
    
    Revised: Arnau, 12/12/2022
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
    
    SeDaq = SeDaqDLL()
    Ascan_Ch2 = np.zeros(Smax2-Smin2)
    Ascan_Ch1 = np.zeros(Smax1-Smin1)
    Flag = AvgSamplesNumber
    while Flag > 0:
        SeDaq.GetAScan() #get Ascan        
        Aux_Ch2 = np.array(list(map(float,SeDaq.DataADC2[Smin2:Smax2]))) #get Ascan WP
        Aux_Ch2 = (Aux_Ch2 - Quantiz_Levels/2)/Quantiz_Levels # Normalize 
        Aux_Ch2 = Aux_Ch2 - np.mean(Aux_Ch2) # remove mean
        Aux_Ch1 = np.array(list(map(float,SeDaq.DataADC1[Smin1:Smax1]))) #get Ascan WP
        Aux_Ch1 = (Aux_Ch1 - Quantiz_Levels/2)/Quantiz_Levels # Normalize 
        Aux_Ch1 = Aux_Ch1 - np.mean(Aux_Ch1) # remove mean
        
        if not(np.all(Aux_Ch2==0.0)) and not(np.all(Aux_Ch1==0.0)):	            
            Ascan_Ch2 = Ascan_Ch2 + Aux_Ch2
            Ascan_Ch1 = Ascan_Ch1 + Aux_Ch1
            Flag -= 1
            		
    Ascan_Ch2 = Ascan_Ch2 / AvgSamplesNumber #calculate averaged Ascan
    Ascan_Ch2 = Ascan_Ch2 - np.mean(Ascan_Ch2) #substract mean value
    Ascan_Ch1 = Ascan_Ch1 / AvgSamplesNumber #calculate averaged Ascan
    Ascan_Ch1 = Ascan_Ch1 - np.mean(Ascan_Ch1) #substract mean value
    return Ascan_Ch1, Ascan_Ch2
	
def Plot_Ascan_tf(Ascan, Units_t = 1e6, Units_F = 1e6, Fs=100e6, Fmin = 0,Fmax = 50e6, FigNum=1, FigTitle='Original Ascan'):
    '''Plots Ascan in time and frequency, between Fmin and Fmax
    Units_t is constant to normalize time axis
    Units_F is constant to normalize frequency axis
    Fs sampling frequency
    Fmin lower limit, considering Units_F
    Fmax upper limit, considering Units_F
    '''
    Time_Axis = np.arange(0, Ascan.size)/Fs*Units_t
    fig, axs = plt.subplots(2, 1, num=FigNum, clear=True)
    axs[0].plot(Time_Axis, Ascan)
#    axs[0].set_title(FigTitle)
    axs[0].set_xlabel('time (us)')
    axs[0].set_ylabel('Ascan')
    # calculate fft
    MyFFT = np.fft.fft(Ascan)
    Freq_Axis = np.arange(0, MyFFT.size)*Fs/MyFFT.size/Units_F
    # freq subplot
    axs[1].plot(Freq_Axis, np.abs(MyFFT))
    axs[1].set_xlabel('frequency (MHz)')
#    axs[1].set_title('subplot 2')
    axs[1].set_ylabel('FFT')
    axs[1].set_xlim(Fmin,Fmax)    
    fig.suptitle(FigTitle)
    fig.tight_layout()
    plt.show()

def GenCodeList_Info(FileName):
    '''Load information from GenCode list, according to its ionner format
    Input
       FileName: Name of the file, including full path
    Outputs
       Name: Name of the gencode
       Sort: Sort of excitation, text ('chirp','pulse','burst')
       ProgGenCLKfreqMHz: ProgGenCLKfreqMHz, float
       F1: Lower (chirp) or central (burst, pulse) frequency in MHz, float
       F2: higher (chirp) or central (burst, pulse) frequency in MHz, float
       Cycles: number of cycles of the burst, float
       Duration: Duration of the signal, us, float
       Polarity: Polarity of the pulse (-1, 1, 2), integer    
    '''
    try:
        GenCodeList = np.loadtxt(FileName, dtype={'names': ('Name', 'Sort', 'Fclk', 'F1','F2','NCyc','Dur','Pol'),
                     'formats': ('S10', 'S5','f','f','f','f','f','int')}, delimiter=';')
        titulo=[]
        for GNo in range(GenCodeList.size):
            if GenCodeList[GNo][1]=='chirp':
                patata=str(GenCodeList[GNo][0]) + ': ' + str(GenCodeList[GNo][1]) + ' BW(MHz)=[' + str(GenCodeList[GNo][3]) + ',' +\
                str(GenCodeList[GNo][4]) + '] ' + u'\u0394\u03C4=' + str(GenCodeList[GNo][6]) + u'\u03BCs'
            
            elif GenCodeList[GNo][1]=='burst':
                patata = str(GenCodeList[GNo][0]) + ': ' + str(GenCodeList[GNo][1]) + ' Fc=' + str(GenCodeList[GNo][3]) + \
                'MHz NoCycles=' + str(GenCodeList[GNo][5]) + u' \u0394\u03C4=' + str(GenCodeList[GNo][6]) + u'\u03BCs'
            else:
                patata = str(GenCodeList[GNo][0]) + ': ' + str(GenCodeList[GNo][1]) + ' Fc=' + str(GenCodeList[GNo][3])  +\
                'MHz ' + u'\u0394\u03C4=' + str(GenCodeList[GNo][6]) + u'\u03BCs'
            titulo.append(patata)
        return GenCodeList, titulo
    except Exception as e:
        print(e)
        return ''


def _parseTemperatureData(line, twoSensors=True, intDigits=2, floatDigits=3, sepLength=0):
    if twoSensors:
        threshold = intDigits + floatDigits + 1 # +1 due to the decimal point itself
        temp1 = float(''.join(list(map(chr, line[:threshold]))))
        temp2 = float(''.join(list(map(chr, line[threshold+sepLength:-1]))))
        return temp1, temp2
    return float(line)

def getTemperature(ser: serial.Serial, N_avg: int=1, twoSensors: bool=True, intDigits: int=2, floatDigits: int=3, sepLength: int=0, error_msg: str=None, exception_msg: str=None):
    '''
    Get a temperature reading via serial communication (ser). The expected 
    format of the readings are b'12.34567.890' which is then parsed to floats
    as 12.345 and 67.890 (these numbers are just an example). If N_avg > 1,
    then N_avg readings are taken and averaged. An error message is printed 
    when the float numbers are parsed correctly but do not follow this format. 
    An exception message is printed if the numbers could not be parsed to float.
    If the message are None, then nothing is printed. In all cases, the
    readings are retaken until a good reading is found.
    
    Please note that the serial communication should already be opened and it
    is NOT closed by this function.

    Parameters
    ----------
    ser : serial.Serial object
        The serial communication object.
    N_avg : int, optional
        Number of reading to take and average. The default is 1.
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

    Arnau, 08/11/2022
    '''
    lines = [None]*N_avg # init. list of lines
    temp1 = np.zeros(N_avg);
    if twoSensors:
        temp2 = np.zeros(N_avg) # init. float temperature vectors
    GoodMeasurement = False
    while not GoodMeasurement:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    
        while ser.read() != b'\n':
            pass
    
        for i in range(N_avg):
            lines[i] = ser.readline()
        
        GoodMeasurement = True
        
        for i in range(N_avg):
            try:
                if lines[i] != b'':
                    if twoSensors:
                        temp1[i], temp2[i] = _parseTemperatureData(lines[i],
                                                                   twoSensors=twoSensors,
                                                                   intDigits=intDigits,
                                                                   floatDigits=floatDigits,
                                                                   sepLength=sepLength)
                    else:
                        temp1[i] = _parseTemperatureData(lines[i], twoSensors=False)
                else:
                    if error_msg is not None:
                        print(error_msg)
                    GoodMeasurement = False
    
            except Exception:
                if exception_msg is not None:
                    print(exception_msg)
                GoodMeasurement = False
                
        mean1 = np.mean(temp1)
        if twoSensors:
            mean2 = np.mean(temp2)
            return mean1, mean2
    return mean1