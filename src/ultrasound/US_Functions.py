# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:47:56 2020.

@author: alrom
"""
import numpy as np
from scipy import signal
import winsound
from scipy.signal import find_peaks
import re
import os


def fastxcorr(x, y, Extend=True, Same=False):
    '''
    Calculate xcor using fft.

    Parameters
    ----------
    x : 1D ndarray
        First signal. Array of floats.
    y : 1D ndarray
        Second signal. Array of floats.
    Extend : bool, optional
        If True, extend to correct dimensionality of result. The default is
        True.
    Same : bool, optional
        If True, return result with dimensions equal longest. The default is
        False.

    Returns
    -------
    xcor : ndarray
        Cross correlation between x and y.

    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    if Extend:
        nfft = len(x)+len(y)-1
    else:
        nfft = np.maximum(len(x), len(y))
    z = np.real(np.fft.ifft(np.fft.fft(x, nfft) * np.conj(np.fft.fft(y, nfft))))
    if Same:
        return z[:np.maximum(len(x), len(y))]
    else:
        return z


def centroid(x, UseHilbEnv=False):
    '''
    Calculate centorid of a signal or of its envelope. It is used mainly for
    delays.

    Parameters
    ----------
    x : 1D ndarray
        Input signal. Array of floats.
    UseHilbEnv : bool, optional
        If True, use envelope to find the centroid. The default is False.

    Returns
    -------
    c : float
        The centroid of x.

    Revised: Arnau, 12/12/2022
    '''
    x = envelope(x)
    n = np.arange(len(x))
    return np.sum(n*(x**2))/np.sum(x**2)


def nextpow2(i):
    '''
    Calculate next power of 2, i. e., minimum N so that 2*N>i.

    Parameters
    ----------
    i : int
        Number to calculate its next power of 2.

    Returns
    -------
    n : int
        Closest power of 2 so that 2**n>i.

    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    n = int(np.log2(i))
    if 2**n < i:
        n += 1
    return n


def ShiftSubsampleByfft(Signal, Delay):
    '''
    Delay signal in subsample precision using FFT. Arbitrary subsampling delay
    can be applied.

    Parameters
    ----------
    Signal : ndarray
        Array to be shifted.
    Delay : float
        Delay in subsample precision.

    Returns
    -------
    s : ndarray
        Shifted signal

    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    N = np.size(Signal)  # signal length
    HalfN = np.floor(N / 2)  # length of the semi-frequency axis in frequency domain
    FAxis1 = np.arange(HalfN + 1) / N  # Positive semi-frequency axis
    FAxis2 = (np.arange(HalfN + 2, N + 1, 1) - (N + 1)) / N  # Negative semi-frequency axis
    FAxis = np.concatenate((FAxis1, FAxis2))  # Full reordered frequency axis

    return np.real(np.fft.ifft(np.fft.fft(Signal) * np.exp(-1j*2*np.pi*FAxis*Delay)))


def CosineInterpMax_generic(MySignal, UseHilbEnv=False):
    '''
    Calculate the location of the maximum in subsample basis using cosine
    interpolation. The correction for correlation is NOT performed.

    Parameters
    ----------
    MySignal : ndarray
        Input signal.
    UseHilbEnv : bool, optional
        If True, uses envelope instead of raw signal. The default is False.

    Returns
    -------
    DeltaToF : float
        Location of the maximum in subsample precision.

    Arnau, 10/01/2023
    '''
    if UseHilbEnv:
        MySignal = np.absolute(signal.hilbert(MySignal))
    MaxLoc = np.argmax(np.abs(MySignal))  # find index of maximum
    N = MySignal.size  # signal length
    A = MaxLoc - 1  # left proxima
    B = MaxLoc + 1  # Right proxima
    if MaxLoc == 0:  # Check if maxima is in the first or the last sample
        A = N - 1
    elif MaxLoc == N - 1:
        B = 0
    
    # calculate interpolation maxima according to cosine interpolation
    Alpha = np.arccos((MySignal[A] + MySignal[B]) / (2 * MySignal[MaxLoc]))
    Beta = np.arctan((MySignal[A] - MySignal[B]) / (2 * MySignal[MaxLoc] * np.sin(Alpha)))
    Px = Beta / Alpha

    # Calculate ToF in samples
    DeltaToF = MaxLoc - Px
    return DeltaToF

def CosineInterpMax(MySignal, xcor=True, UseHilbEnv=False):
    '''
    Calculate the location of the maximum in subsample basis using cosine
    interpolation.

    Parameters
    ----------
    MySignal : ndarray
        Input signal.
    xcor : bool, optional
        If True, assumes the signal is a full correlation, therefore the sign
        of the location of the maximum is adjusted accordingly. The default is
        True.
    UseHilbEnv : bool, optional
        If True, uses envelope instead of raw signal. The default is False.

    Returns
    -------
    DeltaToF : float
        Location of the maximum in subsample precision.

    Alberto, 10/11/2020
    Revised: Arnau, 06/02/2023
    '''
    # plt.plot(MySignal)
    if UseHilbEnv:
        MySignal = np.absolute(signal.hilbert(MySignal))
        # plt.plot(MySignal)
    MaxLoc = np.argmax(np.abs(MySignal))  # find index of maximum
    N = MySignal.size  # signal length
    A = MaxLoc - 1  # left proxima
    B = MaxLoc + 1  # Right proxima
    if MaxLoc == 0:  # Check if maxima is in the first or the last sample
        A = N - 1
    elif MaxLoc == N - 1:
        B = 0
    
    # calculate interpolation maxima according to cosine interpolation
    Alpha = np.arccos((MySignal[A] + MySignal[B]) / (2 * MySignal[MaxLoc]))
    Beta = np.arctan((MySignal[A] - MySignal[B]) / (2 * MySignal[MaxLoc] * np.sin(Alpha)))
    Px = Beta / Alpha

    # Calculate ToF in samples
    DeltaToF = MaxLoc - Px

    # Check whether delay is to the right or to the left and correct ToF
    if xcor and MaxLoc > N/2:
        DeltaToF = -(N - DeltaToF)
    # Returned value is DeltaToF, the location of the maxima in subsample basis
    return DeltaToF


def CalcToFAscanCosine_XCRFFT(Data, Ref, UseCentroid=False, UseHilbEnv=False, Extend=True, Same=False):
    '''
    Used to align one Ascan to a Reference by ToF subsample estimate using
    cosine interpolation. Also returns ToFmap and Xcorr. Xcoor is calculated
    using FFT.
    It uses cosine interpolation or centroid (UseCentroid=True) to approximate
    peak location. If UseHilbEnv=True, uses envelope for the delay instead
    of raw signal.

    Parameters
    ----------
    Data : ndarray
        Ascan.
    Ref : ndarray
        Reference to align
    UseCentroid : bool, optional
        If True, use centroid instead of maximum. The default is False.
    UseHilbEnv : boo, optional
        If True, use envelope instead of raw signal. The default is False.
    Extend : bool, optional
        Used for the cross correlation. If True, extend xcor to correct 
        dimensionality of result. The default is True.
    Same : bool, optional
        Used for the cross correlation. If True, the xcor has dimensions equal
        longest. The default is False.
    
    Returns
    -------
    DeltaToF : float
        Time of flight between pulses.
    AlignedData : ndarray
        Aligned array to Ref.
    MyXcor : ndarray
        Cross correlation.
    
    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    try:
        # Calculates xcorr in frequency domain
        MyXcor = fastxcorr(Data, Ref, Extend=Extend, Same=Same)
        # determine time of flight
        if UseCentroid:
            DeltaToF = centroid(MyXcor, UseHilbEnv=UseHilbEnv)
        else:
            DeltaToF = CosineInterpMax(MyXcor, UseHilbEnv=UseHilbEnv)
        # Delay to align
        AlignedData = ShiftSubsampleByfft(Data, DeltaToF)
        return DeltaToF, AlignedData, MyXcor

    except Exception as ex:
        print(ex)


def align2zero(Data, UseCentroid=False, UseHilbEnv=False):
    '''
    Align signal to zero in order to flatten its phase. It delays the signal
    so that its maximum or centroid (UseCentroid=True) is located at the
    origin. It uses the signal or its envelope (UseHilbEnv=True).

    Parameters
    ----------
    Data : ndarray
        Ascan
    UseCentroid : bool, optional
        If True, use centroid instead of maximum. The default is False.
    UseHilbEnv : bool, optional
        If True, use envelope instead of raw signal. The default is False.

    Returns
    -------
    AlignedData : ndarray
        Aligned array to Ref
    ZeroToF : float
        Time of flight to zero.

    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    # determine time of flight
    if UseCentroid:
        ZeroToF = centroid(Data, UseHilbEnv=UseHilbEnv)
    else:
        ZeroToF = CosineInterpMax(Data, UseHilbEnv=UseHilbEnv)
        
    # Delay to align
    AlignedData = ShiftSubsampleByfft(Data, ZeroToF)
    return ZeroToF, AlignedData



def lfilt(InData, SoF, CutOffFreq, Fs, FOrder=4):
    '''
    Filters data by linear filtering. The filter is a butterwoth, with
    cut of frequency Fc=CutOffFreq in Hz according to given Fs in Hz. Note that
    it is the teorethical nominal desired cutoff frequency. Filter can be
    highpass or lowpass.

    Parameters
    ----------
    InData : ndarray
        Input signal.
    SoF : str, {'low','high'}
        Choose between low or high pass filter.
    CutOffFreq : float
        Cut off frequency in Hz.
    Fs : float
        Sampling frequency in Hz.
    FOrder : int, optional
        Order of the filter. The default is 4.

    Returns
    -------
    OutData : ndarray
        Filtered data.

    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    try:
        Fc = CutOffFreq/Fs*2
        b, a = signal.butter(FOrder, Fc, SoF)
        zi = signal.lfilter_zi(b, a)
        return signal.lfilter(b, a, InData, zi=zi*InData[0])
    except Exception:
        print('lfilt, Check parameters, something is wonrg.')


def filtfilt(InData, SoF, CutOffFreq, Fs, FOrder=4):
    '''
    Filters a signal using filtfilt algorithm  so that no delay is produced.
    Note that the frequency response of the resulting filter is the square
    of the original one. The filter is a butterwoth, with
    cutoff frequency Fc=CutOffFreq in Hz according to given Fs in Hz. Note that
    it is the teorethical nominal desired cut of frequency. Filter can be
    highpass or lowpass.

    Parameters
    ----------
    InData : ndrray float
        Inputa dato to filter
    SoF : str, {'low','hogh'}
        Choose between low or high pass filter.
    CutOffFreq : float
        Cut off frequency in Hz.
    Fs : float
        Sampling frequency in Hz.
    FOrder : int, optional
        Order of the filter to be applied. The default is 4.

    Returns
    -------
    OutData : ndarray
        Filtered data.

    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    try:
        Fc = CutOffFreq/Fs*2
        b, a = signal.butter(FOrder, Fc, SoF)
        return signal.filtfilt(b, a, InData)
    except Exception:
        print('filtfilt, Check parameters, something is wonrg.')


def deconvolution(Data, Ref, stripIterNo=2, UseHilbEnv=False, Extend=True, Same=False):
    '''
    Applies iterative deconvolution to a signal using a reference signal.
    It delays the reference to align it to the maximum (or centroid) of the
    signal (or its envelope) and then subtracts it (scaled to have the same
    amplitude), so that the remainder has the main echo supressed. The
    process can be repeated in a successive algorithm, stripping the succesive
    echoes. Note that due to errors (mainly if there is overlap between echoes)
    the error raises with each iteration. It returns the ToF vector with the
    location of the maxima (or centroids) of the successive echoes (or its 
    envelopes), and also the matrix of the resulting striped signals at each
    iteration, including in the first row the original signal.
    It consumes a lot of memory when working with big Bscans or Cscans.

    Parameters
    ----------
    Data : ndarray
        Ascan
    Ref : ndarray
        Reference Ascan.
    stripIterNo : int, optional
        Number of iterations of the deconvolution. The default is 2.
    UseHilbEnv : bool, optional
        If True, use hilbert envelope maximum. The default is False.
    Extend : bool, optional
        Used for the cross correlation. If True, extend xcor to correct 
        dimensionality of result. The default is True.
    Same : bool, optional
        Used for the cross correlation. If True, the xcor has dimensions equal
        longest. The default is False.
    
    Returns
    -------
    ToF : ndarray
        Succesive ToF in subsample basis.
    StrMat : 2D-ndarray
        Striping matrix with shape (stripIterNo, len(Data)).


    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    RAmp = np.sum(np.power(Ref, 2)) / len(Ref)  # mean power of reference
    StrMat = np.zeros((stripIterNo+1, len(Data)))  # reallocates stripped matrix
    ToF = np.zeros(stripIterNo)  # preallocates ToF
    StrMat[0, :] = Data
    for LayerNo in np.arange(stripIterNo):
        ToF[LayerNo] = CalcToFAscanCosine_XCRFFT(StrMat[LayerNo, :], Ref, UseHilbEnv=UseHilbEnv, Extend=Extend, Same=Same)[0]
        RefShifted = ShiftSubsampleByfft(Ref, ToF[LayerNo])  # Shift the ref to the Ascan position
        Amp = np.sum(StrMat[LayerNo, :] * RefShifted) / len(RefShifted) / RAmp  # Amplitude scaling factor
        StrMat[LayerNo+1, :] = StrMat[LayerNo, :] - RefShifted * Amp  # strip Ascan
    return ToF, StrMat

def zeroPadding(Data, NewLen):
    '''
    Extend arrays (Ascans) by zero padding.
    
    Used for zeropadding signals for 1D/2D/3D arrays. Signal must be in the 
    last dimension [-1]. If the signal is longer than the desired final 
    length (NewLen), it does nothing.

    Parameters
    ----------
    Data : ndarray
        Input matrix, 1D/2D/3D.
    NewLen : int
        New length of the Ascans.

    Returns
    -------
    Data : ndarray
        Extended data.

    Alberto, 10/11/2020.
    Revised: Arnau, 12/12/2022
    '''
    OldLen = Data.shape[-1]
    if NewLen > OldLen:
        if Data.ndim == 1:
            Data = np.append(Data, np.zeros(NewLen - OldLen))
        elif Data.ndim == 2:
            Data = np.concatenate((Data, np.zeros((Data.shape[0], NewLen - OldLen))), axis=1)
        else:
            Data = np.concatenate((Data, np.zeros((Data.shape[0], Data.shape[1], NewLen - OldLen))), axis=2)
        return Data


def findIndexInAxis(Data, Value):
    '''
    Find index of value in sorted 1D array.
    Find the index of the closest value in the input vector, which is a 
    sorted array, usually an axis, so that Data[index] is the closes to Value.

    Parameters
    ----------
    Data : ndarray
        Input vector.
    Value : float
        Value we are looking for.
    
    Returns
    -------
    idx : int
        Index or None.

    Alberto, 10/11/2020.
    Docstring: Arnau, 12/12/2022
    '''
    if Value in Data:  # if value is in array
        return np.where(Data == Value)[0][0]
    elif Value < Data[0]:  # if value is lower than any element of the array
        return 0
    elif Value > Data[-1]:  # if value is higher than any element of the array
        return len(Data)-1
    else:
        a = np.where(Data < Value)[0][-1]
        A = np.where(Data > Value)[0][0]
        if np.abs(np.abs(Value)-np.abs(Data[a])) < np.abs(np.abs(Value)-np.abs(Data[A])):
            return a
        else:
            return A

def envelope(Data, axis=-1):
    '''
    Calculate the envelope of a signal as the absolute value of its hilbert
    transform.

    Parameters
    ----------
    Data : ndarray
        Input signal.
    axis : int
        Axis along which to calculate the envelope. The default is -1.

    Returns
    -------
    e : ndarray
        The envelope of the input signal.

    Alberto, 10/11/2020
    Revised: Arnau, 12/12/2022
    '''
    if Data.ndim > 1:
        return np.abs(signal.hilbert(Data, axis=axis))
    else:
        return np.abs(signal.hilbert(Data))

def moving_average(Data, WinLen):
    '''
    Apply undelayed moving average with rectangular window.

    It calculates undelayed moving average, that is, first zeropads signal and
    window, and then delays windows to zero to avoid delay in the output. Note
    that the zeropadding is made to the length of the resulting non-overlapped
    result, therefore only first samples are correct, that is, being DataLen
    the length of the signal and WinLen the length of the window, the zero
    padding is that so that the final length is DataLen+WinLen-1, so after
    processing only the first DataLen sampes are correct. That is why odd
    number of samples in window is mandatory.
    MA is calculated using correlation, as windows are symmetric.

    Parameters
    ----------
    Data : ndarray
        Input data (1D/2D/3D).
    WinLen : int
        Lenth of window. Must be odd.

    Returns
    -------
    MAData : ndarray
        Averaged signal.

    Alberto, 09/11/2020
    Docstring: Arnau, 12/12/2022
    '''
    if WinLen % 2 == 0:
        WinLen += 1
    DataLen = Data.shape[-1]
    NewLen = DataLen + WinLen - 1  # Desired len for convolution
    Data = zeroPadding(Data, NewLen)  # zeropadding
    MyWin = np.ones(WinLen) / WinLen  # build rectangular window
    MyWin = zeroPadding(MyWin, NewLen)  # zeropadding
    MyWin = np.roll(MyWin, -int((WinLen - 1) / 2))  # delays win to center at 0
    # calculate convolution (correlation) as MA and return valid points
    if Data.ndim == 1:
        return fastxcorr(Data, MyWin)[0:DataLen]
    elif Data.ndim == 2:
        return fastxcorr(Data, MyWin)[:, 0:DataLen]
    elif Data.ndim == 3:
        return fastxcorr(Data, MyWin)[:, :, 0:DataLen]

def normalizeAscans(Data):
    '''
    Normalize each Ascan separately.

    Parameters
    ----------
    Data : ndarray
        Matrix to normalize.

    Returns
    -------
    NormData : ndarray
        Normalized matrix.

    Docstring: Arnau, 12/12/2022
    '''
    copied = Data.copy()
    if Data.ndim == 1:
        copied = Data / np.amax(np.abs(Data))
    elif Data.ndim == 2:
        for i in np.arange(Data.shape[0]):
            copied[i, :] = Data[i, :] / np.amax(np.abs(Data[i, :]))
    else:
        for i in np.arange(Data.shape[0]):
            for j in np.arange(Data.shape[1]):
                copied[i, j, :] = Data[i, j, :] / np.amax(np.abs(Data[i, j, :]))
    return copied


def makeWindow(SortofWin='boxcar', WinLen=512,
               param1=1, param2=1, Span=0, Delay=0):
    '''
    Make window.
    
    Parameters
    ----------
    SortofWin : str, optional
        can be any of the following (entered as plaintext):
            boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman,
            blackmanharris, nuttall, barthannkaiser (needs beta), gaussian (needs standard deviation),
            general_gaussian (needs power, width), slepian (needs width), dpss (needs normalized half-bandwidth),
            chebwin (needs attenuation), exponential (needs decay scale), tukey (needs taper fraction)
        The default is 'boxcar'.
    WinLen : int, optional
        Length of the desired window. The default is 512.
    param1 : float, optional
        beta (kaiser), std (Gaussian), power (general gaussian),
        width (slepian), norm h-b (dpss), attenuation (chebwin),
        decay scale (exponential), tapper fraction (tukey).
        The default is 1.
    param2 : float, optional
        Width (general gaussian). The default is 1.
    Span : int, optional
        Final length of the required window, in case expansion needed. The 
        default is 0.
    Delay : int, optional
        Required delay to the right, in samples. The default is 0.
    
    Return
    ------
    Window : ndarray
        1D array of lenth WinLen or Span.
        
    Docstring: Arnau, 12/12/2022
    Correction: Arnau, 24/05/2023
        ShiftSubsampleByfft(MyWin, -Delay) changed to
        ShiftSubsampleByfft(MyWin, Delay)
    '''
    
    lstWinWithParameter = ['barthannkaiser','gaussian','slepian','dpss',
                           'chebwin','exponential','tukey']
    if any(SortofWin.lower() in x for x in lstWinWithParameter):
        if SortofWin.lower() == 'general_gaussian':
            MyWin = signal.get_window(('general_gaussian', param1, param2), WinLen)
        else:
            MyWin = signal.get_window((str(SortofWin.lower()), param1), WinLen)
    else:
        MyWin = signal.get_window(str(SortofWin.lower()), WinLen)
    if not(Span == 0):  # if Span, add zeros to the end
        MyWin = np.append(MyWin, np.zeros(Span - WinLen))
    if not(Delay == 0):  #if Delay, circshift to right Delay samples
        if isinstance(Delay, int):
            MyWin = np.roll(MyWin, int(Delay))  # if interger use numpy roll
        else:
            MyWin = ShiftSubsampleByfft(MyWin, Delay)  # if float use subsample
    return MyWin

def CheckIntegrityScan(Scan, FixIt=True):
    '''
    Check integrity of Bascan or Csacn (non zero arrays).

    Parameters
    ----------
    Scan : ndarray
        Ascan, Bscan or Cscan.
    FixIt : bool, optional
        If True, fix data.

    Returns
    -------
    Scan : ndarray
        Fixed Ascan.
    FoundError : bool
        If True, an error was found.

    Docstring: Arnau, 12/12/2022
    '''
    def checkarr(arr):
        if np.isnan(np.dot(arr, arr)) or np.all(arr == 0):
            return False
        else:
            return True
            
    FoundError = False
    if FixIt:
        OutScan = np.zeros_like(Scan)
    Dimensions = len(Scan.shape)
    if Dimensions == 2:
        if not checkarr(Scan[0,:]):
            FoundError = True
            i = 1
            while i <= Scan.shape[0]:
                if checkarr(Scan[i,:]):
                    Scan[0,:] = Scan[i,:]
                    i = Scan.shape[0]+1
                else:
                    i+=1
        for i in np.arange(1,Scan.shape[0]):
            if not checkarr(Scan[i,:]):
                FoundError = True
                Scan[i,:] = Scan[i-1,:]
    return Scan, FoundError
                
                    
def makeBeep(frequency = 2500, duration = 1000):
    '''
    Make a beep sound with the given frequency and duration.

    Parameters
    ----------
    frequency : float, optional
        Frequency in Hz. The default is 2500.
    duration : float, optional
        Duration in milliseconds. The default is 1000.

    Returns
    -------
    None.

    Docstring: Arnau, 12/12/2022
    '''
    #frequency = 2500  # Set Frequency To 2500 Hertz
    #duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)            
        

def extendedHilbert(Data, NoSamples=None, Zeros=True):
    '''
    Calculate hilbert transform extending signal before calculating
    the analitical signal, to prevent boundary problems.    
    Extension is bilateral (starting and ending of signal).

    Parameters
    ----------
    Data : ndarray
        Input data.
    NoSamples : int or None, optional
        Number of samples to be added. If None, calculate 10%. Default is None.
    Zeros : bool, optional
        If True, add zeros, otherwise reflect corresponding samples. The 
        default is True.

    Returns
    -------
    HT : ndarray
        The Hilbert transform.

    Docstring: Arnau, 12/12/2022
    '''
    if NoSamples==None:
        NoSamples = int(len(Data/10))    
    
    Env_temp = signal.hilbert( np.concatenate( ( np.zeros((NoSamples)), Data, np.zeros((NoSamples)) ) ) )    
    return Env_temp[NoSamples:-NoSamples]


def Normalize(Data):
    '''
    Normalize data.

    Parameters
    ----------
    Data : ndarray
        Input data.

    Returns
    -------
    NormData : ndarray
        Normalized data.

    '''
    return Data/np.amax(np.abs(Data))


def generateLettersList(N, backslash=True):
    '''
    Generates list of N strings consisting of ordered capital letters. If N is 
    greater than the alphabet, it continues with AA, AB, AC,... If backslash is
    True, elements of the list start with '\', i.e., \AA, \AB, \AC,... Note 
    that Python will represent this as a double backslash ('\\').

    Parameters
    ----------
    N : int
        Desired number of items in list.
    backslash : bool, optional
        If True, elements of resulting list are preceeded by '\\'. The default 
        is True.

    Returns
    -------
    lettersList : list
        List of letters.

    Arnau, 01/10/2022
    '''
    if backslash:
        lettersList = [r"\\"]*N
    else:
        lettersList = ['']*N
    start_ascii, stop_ascii = 65, 90 # A, Z
    Nletters = stop_ascii - start_ascii + 1
    Laps = N // Nletters # integer divison
    if Laps==0:
        for i in range(N):
            lettersList[i] += chr(start_ascii+i)
    else:
        auxList = ['']*Laps
        for Lap in range(Laps):
            auxList[Lap] = chr(start_ascii+Lap)
            for i in range(Nletters):
                lettersList[Nletters*Lap+i] += chr(start_ascii+i)
            if Nletters*(Lap+2) < N:
                lettersList[Nletters*(Lap+1):Nletters*(Lap+2)] = [s + auxList[Lap] for s in lettersList[Nletters*(Lap+1):Nletters*(Lap+2)]]
            else:
                lettersList[Nletters*(Lap+1):] = [s + auxList[Lap] for s in lettersList[Nletters*(Lap+1):]]
        lettersList[Nletters*Laps:] = [s + chr(start_ascii+i) for i,s in enumerate(lettersList[Nletters*Laps:])]
    
    return lettersList


def find_Subsampled_peaks(Data, **kwargs):
    '''
    Uses the scipy.signal.find_peaks() function to find the peaks and then uses
    cosine interpolation to find the subsampled peaks. See 
    scipy.signal.find_peaks() documentation for more details.

    Parameters
    ----------
    Data : ndarray
        A signal with peaks.

    Returns
    -------
    Real_peaks : ndarray
        Subsampled peaks that satisfy all given conditions.

    Arnau, 24/05/2023
    '''
    if np.isnan(np.sum(Data)):
        return np.nan
    peaks, _ = find_peaks(Data, **kwargs)

    # now we use cosine interpolation to be more precise... although not really important
    Real_peaks = []
    for p in peaks:
        A = p - 1
        B = p + 2
        MaxLoc = p
        Alpha = np.arccos((Data[A] + Data[B]) / (2 * Data[MaxLoc]))
        Beta = np.arctan((Data[A] - Data[B]) / (2 * Data[MaxLoc] * np.sin(Alpha)))
        Px = Beta / Alpha
        
        # recalculate peak location in samples adding the interpolated value
        Real_peaks.append(p - Px)
    return np.array(Real_peaks)


def Zscore(data):
    '''
    Z-score of len(data) observations computed as Z=(x-mean)/std.
    
    Parameters
    ----------
    data : ndarray
        Input dataset.

    Returns
    -------
    Z : ndarray
        The Z-score of every observation.

    Arnau
    '''
    aux = np.array(data)
    Z = (aux - np.mean(aux))/np.std(aux)
    return Z

def reject_outliers(data, m=0.6745):
    '''
    Outlier detection using the Z-score, computed as Z = (x-mean)/std.

    Parameters
    ----------
    data : ndarray
        Input dataset.
    m : float, optional
        Outlier threshold in Z scale. The recommended value is 0.6745. The 
        default is 0.6745.

    Returns
    -------
    new_data : ndarray
        Original data without outliers.
    outliers : ndarray
        The removed outliers.
    outliers_indexes : ndarray
        The indices of the outliers in the original data array.

    Arnau
    '''
    aux = np.array(data)
    Z = Zscore(data)
    new_data = aux[np.abs(Z)<m]
    outliers = aux[np.abs(Z)>=m]
    outliers_indexes = np.where(np.abs(Z)>=m)[0]
    return new_data, outliers, outliers_indexes

def maskOutliers(data, m: float=0.6745, UseMedian: bool=False):
    '''
    Return a numpy masked array where the masked values are the outliers of the
    data (row-wise in the case of a 2D matrix) given a certain threshold `m`.
    If UseMedian is True, outliers are difned from the median instead of the
    mean.

    Parameters
    ----------
    data : ArrayLike
        Input 2D matrix. The Outliers are found row-wise.
    m : float, optional
        Outlier threshold. The default is 0.6745.
    UseMedian : bool, optional
        If True, use median instead of mean. The default is False.

    Returns
    -------
    data_masked : numpy.ma.core.MaskedArray
        Array with masked outliers.

    Arnau, 09/11/2023
    '''
    if data.ndim == 1:
        if UseMedian:
            med = (data - np.median(data))/np.std(data)
            outliers_indexes = np.where(np.abs(med) >= m)[0]
        else:
            outliers_indexes = reject_outliers(data, m=m)[2]
        _temp = np.zeros(len(data))
        _temp[outliers_indexes] = outliers_indexes
        mask = np.array(_temp, dtype=bool)
        data_masked = np.ma.masked_array(data, mask=mask)
        return data_masked
    else:
        masks = np.zeros([1, data.shape[1]])
        for x in data:
            if UseMedian:
                med = (x - np.median(x))/np.std(x)
                outliers_indexes = np.where(np.abs(med) >= m)[0]
            else:
                outliers_indexes = reject_outliers(x, m=m)[2]
            _temp = np.zeros(len(x))
            _temp[outliers_indexes] = outliers_indexes
            mask = np.array(_temp, dtype=bool)
            if 0 in outliers_indexes:
                mask[0] = True
            masks = np.vstack([masks, mask])
        masks = masks[1:,:]
        data_masked = np.ma.masked_array(data, mask=masks)
        return data_masked


def winwidth2n(win, width):
    '''
    Return the number of samples that a window should have in order to obtain 
    the desired main lobe spectral width (defined to the first null).

    Parameters
    ----------
    win : str or tuple
        Window type.
    width : float
        Spectral width in descrete frequency.

    Raises
    ------
    NotImplementedError
        Window not available.

    Returns
    -------
    n : int
        The number of samples that the window should have.

    Arnau, 02/12/22
    '''
    if isinstance(win, tuple):
        if win[0]=='kaiser':
            return np.sqrt(1 + win[1]**2)/width
        else:
            raise NotImplementedError('Window not available.')
    else:
        AVAILABLE_WINS_DICT = {'boxcar' : 2/width,
                                'rect' : 2/width,
                                'rectangular' : 2/width,
                                'triang' : 4/width, 
                                'blackman' : 6/width,
                                'hamming' : 4/width,
                                'hann' : 4/width,
                                'bartlett' : 4/width,
                                'flattop' : 10/width,
                                'parzen' : 8/width,
                                'blackmanharris' : 8/width,
                                'exponential' : 4.4375/width}
        if win not in AVAILABLE_WINS_DICT:
            raise NotImplementedError('Window not available.')
        return int(AVAILABLE_WINS_DICT[win])

def n2winwidth(win, n):
    '''
    Return the main lobe spectral width (defined to the first null) of the
    specified window for a given sample length.

    Parameters
    ----------
    win : str or tuple
        Window type.
    n : int
        The number of samples of the window.

    Raises
    ------
    NotImplementedError
        Window not available.

    Returns
    -------
    width : float
        The width of the window in descrete frequency.

    Arnau, 02/12/22
    '''
    if isinstance(win, tuple):
        if win[0]=='kaiser':
            return np.sqrt(1 + win[1]**2)/n
        else:
            raise NotImplementedError('Window not available.')
    else:
        AVAILABLE_WINS_DICT = {'boxcar' : 2/n,
                                'rect' : 2/n,
                                'rectangular' : 2/n,
                                'triang' : 4/n, 
                                'blackman' : 6/n,
                                'hamming' : 4/n,
                                'hann' : 4/n,
                                'bartlett' : 4/n,
                                'flattop' : 10/n,
                                'parzen' : 8/n,
                                'blackmanharris' : 8/n,
                                'exponential' : 4.4375/n}
        if win not in AVAILABLE_WINS_DICT:
            raise NotImplementedError('Window not available.')
        return AVAILABLE_WINS_DICT[win]


def time2str(seconds) -> str:
    '''
    Convert the number of seconds to a string with format: X h, X min, X s.

    Parameters
    ----------
    seconds : int or float
        The number of seconds to convert to hours, minutes and seconds.

    Returns
    -------
    s : str
        Formatted string: {hours} h, {minutes} min, {seconds} s.

    Arnau, 21/12/2022
    '''
    hours = seconds//3600
    minutes = seconds%3600//60
    seconds = round(seconds - hours*3600 - minutes*60)
    s = f'{hours} h, {minutes} min, {seconds} s'
    return s

def str2time(strtime) -> int:
    '''
    Convert the given string (time) to seconds (int). The input time must
    specify hours, minutes and seconds.

    Parameters
    ----------
    strtime : str
        The time to converto to seconds.

    Returns
    -------
    seconds : int
        The corresponding number of seconds.

    Arnau, 31/05/2023
    '''
    factors = [3600, 60, 1]
    list_hms = [int(s) for s in re.findall(r'\d+', strtime)]
    seconds = factors[0]*list_hms[0] + factors[1]*list_hms[1] + factors[2]*list_hms[2]
    return seconds

def getNbins(x, mode: str='auto', **kwargs):
    '''
    Returns the appropiate number of bins to compute the histogram of x with
    the given mode. See numpy.histogram_bin_edges for more information.

    Parameters
    ----------
    x : ndarray
        Input data.
    mode : str
        The computation algorithm to find the optimal number of bins. See 
        numpy.histogram_bin_edges docstring. The default is 'auto'.

    Returns
    -------
    Nbins : int
        The optimal number of bins.

    Arnau 23/12/2022
    '''
    edges = np.histogram_bin_edges(x, bins=mode, **kwargs)
    Nbins = int(edges.size-1)
    return Nbins

def hist(x, bins=None, density: bool=False, range=None, mode: str='auto'):
    '''
    Compute the histogram of x. Returns the histogram values, bin edges and 
    width of bins. If x is a 2D matrix, the histogram is computed for every
    row and the returned values are stores in lists.

    Parameters
    ----------
    x : ndarray
        Input data.
    bins : int, optional
        If None, the optimal number of bins is computed with the algorithm 
        given by mode. See numpy.histogram() docstring. The default is None.
    density : bool, optional
        See numpy.histogram() docstring. The default is False.
    range : tuple or None, optional
        See numpy.histogram() docstring. The default is None.
    mode : str, optional
        See numpy.histogram_bin_edges. The default is 'auto'.

    Returns
    -------
    h : ndarray or list of ndarrays
        The values of the histogram.
    b : ndarray or list of ndarrays
        Bin edges (length(hist)+1).
    width : float or list of floats
        Width of the bins.

    Arnau 09/01/2023
    '''
    if bins is None:
        bins = getNbins(x, mode=mode, range=range)

    if x.ndim==1:
        h, b = np.histogram(x, bins=bins, density=density, range=range)
        width = b[1] - b[0]
    elif x.ndim==2:
        h = []
        b = []
        width = []
        for row in x:
            hi, bi = np.histogram(row, bins=bins, density=density, range=range)
            widthi = bi[1] - bi[0]
            h.append(hi)
            b.append(bi)
            width.append(widthi)
    return h, b, width

def slice_array(x, slicesize, overlap=0.5):
    '''
    Slice the given array with a moving window. Slices can be overlaped.

    Parameters
    ----------
    x : ndarray
        Input data.
    slicesize : int
        Window size in samples.
    overlap : float, optional
        Float between 0 and 1. Specifies the overlap amount. The default is 0.5.

    Returns
    -------
    sliced : ndarray
        Matrix containing a slice in each row (N x slicesize).
    idxs : ndarray
        Matrix containing the original indices of the slices in each row
        (N x slicesize). Can be used for plotting, e.g.:
            plt.scatter(idxs, sliced, color='k', marker='.')
    
    Arnau 23/12/2022
    '''
    n = slicesize
    m = int(n*overlap)
    
    idxs = np.arange(len(x))
    idxs = np.lib.stride_tricks.sliding_window_view(idxs, n)[::n-m, :]
    sliced = np.lib.stride_tricks.sliding_window_view(x, n)[::n-m, :]
    return sliced, idxs


def max_in_slice(x, slice_idxs, axis=None):
    '''
    Find maximum of an array in the given indices. Indices do not need to be
    consecutive.

    Parameters
    ----------
    x : ndarray
        Input array.
    slice_idxs : ndarray or list
        Indices of x where to find the maximum.
    axis : int or None, optional
        Axis parameter for numpy.max. The default is None.

    Returns
    -------
    MaxLoc : int or ndarray
        The index of the maximum (or indices if x.ndims>1).
    MaxVal : float or ndarray
        The value of the maximum (or values if x.ndims>1).

    Arnau 12/01/2023
    '''
    m = np.argmax(x[slice_idxs], axis=axis)
    MaxLoc = slice_idxs[m]
    MaxVal = x[MaxLoc]
    return MaxLoc, MaxVal

def min_in_slice(x, slice_idxs, axis=None):
    '''
    Find minimum of an array in the given indices. Indices do not need to be
    consecutive.

    Parameters
    ----------
    x : ndarray
        Input array.
    slice_idxs : ndarray or list
        Indices of x where to find the minimum.
    axis : int or None, optional
        Axis parameter for numpy.min. The default is None.

    Returns
    -------
    MinLoc : int or ndarray
        The index of the minimum (or indices if x.ndims>1).
    MinVal : float or ndarray
        The value of the minimum (or values if x.ndims>1).

    Arnau 12/01/2023
    '''
    m = np.argmin(x[slice_idxs], axis=axis)
    MinLoc = slice_idxs[m]
    MinVal = x[MinLoc]
    return MinLoc, MinVal

def max_in_range(x, lims, indep=None, axis=None):
    '''
    Find maximum of an array in the given range of indices (if indep is None)
    or in the range of indices corresponding to the given values of indep.
    
    indep must have the same shape as x.
    
    In either case, the lower limit is included and the upper limit is
    excluded.
    
    Example:
        >>> x = np.arange(5,20)
        >>> f = np.linspace(0, 10, len(x))
        >>> lims = (3.5, 5.3)
        >>> max_in_range(x=x, lims=lims, indep=f, axis=None)
        (7, 12)

    Parameters
    ----------
    x : ndarray
        Input array.
    lims : tuple of ints or tuple of floats
        If indep is None, these are the first and last indeces of the range. 
        Else, these are the values in indep that define the range.
    indep : ndarray or None, optional
        Array with the same shape as x (or None, unused). The range is defined
        with the values in this array. The default is None.
    axis : int or None, optional
        Axis parameter for numpy.max. The default is None.

    Returns
    -------
    MaxLoc : int or ndarray
        The index of the maximum (or indices if x.ndims>1).
    MaxVal : float or ndarray
        The value of the maximum (or values if x.ndims>1).

    Arnau, 12/01/2023
    '''
    if indep is None:
        MaxLoc = np.argmax(x[lims[0]:lims[-1]], axis=axis) + lims[0]
    else:
        condition = np.logical_and(indep >= lims[0], indep < lims[-1])
        MaxLoc = np.argmax(x[condition], axis=axis) + np.where(condition==True)[0][0]
    MaxVal = x[MaxLoc]
    return MaxLoc, MaxVal

def min_in_range(x, lims, indep=None, axis=None):
    '''
    Find minimum of an array in the given range of indices (if indep is None)
    or in the range of indices corresponding to the given values of indep.
    
    indep must have the same shape as x.
    
    In either case, the lower limit is included and the upper limit is
    excluded.
    
    Example:
        >>> x = np.arange(5,20)
        >>> f = np.linspace(0, 10, len(x))
        >>> lims = (3.5, 5.3)
        >>> min_in_range(x=x, lims=lims, indep=f, axis=None)
        (5, 10)

    Parameters
    ----------
    x : ndarray
        Input array.
    lims : tuple of ints or tuple of floats
        If indep is None, these are the first and last indeces of the range. 
        Else, these are the values in indep that define the range.
    indep : ndarray or None, optional
        Array with the same shape as x (or None, unused). The range is defined
        with the values in this array. The default is None.
    axis : int or None, optional
        Axis parameter for numpy.min. The default is None.

    Returns
    -------
    MinLoc : int or ndarray
        The index of the minimum (or indices if x.ndims>1).
    MinVal : float or ndarray
        The value of the minimum (or values if x.ndims>1).

    Arnau, 12/01/2023
    '''
    if indep is None:
        MinLoc = np.argmin(x[lims[0]:lims[-1]], axis=axis) + lims[0]
    else:
        condition = np.logical_and(indep >= lims[0], indep < lims[-1])
        MinLoc = np.argmin(x[condition], axis=axis) + np.where(condition==True)[0][0]
    MinVal = x[MinLoc]
    return MinLoc, MinVal



def find_nearest(x, value):
    '''
    Finds the nearest value in x.

    Parameters
    ----------
    x : ndarray
        Input array.
    value : float
        Value to find in x. The nearest value in x is returned.

    Returns
    -------
    idx : int
        The index of value_found.
    value_found : float
        The closest value in x to the input value.

    Arnau, 10/02/2023
    '''
    array = np.asarray(x)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def round2signif(x):
    '''
    Round x to the first significant digit.

    Parameters
    ----------
    x : int
        Scalar to round.

    Returns
    -------
    out : int
        The rounded value.

    Arnau, 02/05/2023
    '''
    return np.round(x, -int(np.floor(np.log10(np.abs(x)))))


def polgen(deg: int):
    '''
    Returns a function that generates a polynomial of the specified degree. 
    This new function has two parameters (in this order): 
        - The coefficients of the polynomial starting with the zero'th order.
        - The independent variable, x.

    Parameters
    ----------
    deg : int
        Degree of the desired polynomial.

    Returns
    -------
    pol : function
        Function that generates polynomial values given the coefficients and 
        the independent variable. Example:
            >>> pol = polgen(2)
            >>> pol([1,2,3], 5)
            57
            >>> 1 + 2*4 + 3*(4**2)
            57
            >>> pol([1,2,3], [3,4])
            array([34, 57])
            >>> 1 + 2*np.array([3,4]) + 3*(np.array([3,4])**2)
            array([34, 57])

    Arnau, 18/07/2023
    '''
    def pol(coeffs, x):
        aux = np.array(x)
        result = 0
        for i in range(deg + 1):
            result += coeffs[i]*(aux**i)
        return result
    return pol


def CoefficientOfDetermination(y, f, nparams: int=2):
    '''
    Compute the adjusted coefficient of determination. The number of estimated 
    parameters is `nparams` (e.g. `2` for a linear regression).

    Parameters
    ----------
    y : float or ArrayLike
        Measured values.
    f : float or ArrayLike
        Estimated values.
    nparams : int, optional
        The number of estimated parameters. The default is 2.

    Returns
    -------
    r2 : float or ArrayLike
        The adjusted coefficient of determination (R squared).

    Arnau, 18/07/2023
    '''
    SSres = np.sum((y - f)**2)
    SStot = np.sum((y - np.mean(y))**2)
    
    n = len(y)
    p = nparams - 1
    dftot = n - 1
    dfres = n - p
    
    adjusted_SSres = SSres/dfres
    adjusted_SStot = SStot/dftot
    
    return 1 - adjusted_SSres / adjusted_SStot









def find_numbers_in_string(input_string: str, IgnoreSign: bool=False, IgnoreDot: bool=False) -> list:
    '''
    Finds all numbers in a string and returns them in a list.

    Parameters
    ----------
    input_string : str
        Input string.
    IgnoreSign : bool, optional
        If True, do not use '-' as the sign of the numbers (e.g. 'ab12g-34')
        would result in [12, 34] instead of [12, -34]. The default is False.
    IgnoreDot : bool, optional
        If True, do not use '.' as the decimal point (e.g. 'ab12.3g-45.6')
        would result in [12, 3, -45, 6] instead of [12.3, -45.6]. The default
        is False.

    Returns
    -------
    numbers : list
        List of all numbers found. Each number is converted to int or float
        accordingly.

    Arnau, 14/08/2023
    '''
    number_pattern = r'-?\d+\.\d+|-?\d+' # default
    if IgnoreSign and IgnoreDot:
        number_pattern = r'\d+'
    if IgnoreSign and (not IgnoreDot):
        number_pattern = r'\d+\.\d+|\d+'
    if (not IgnoreSign) and IgnoreDot:
        number_pattern = r'-?\d+'
    matches = re.findall(number_pattern, input_string)
    return [float(match) if '.' in match else int(match) for match in matches]

def sort_strings_by_number(str_list: list, numidx: int=0, reverse: bool=False, **kwargs) -> list:
    '''
    Sort a list of string using the {numidx}th number found in them. If numidx
    is 0, the first number found in each string is used as the sorting key. If
    numidx is -1, the last one is used instead. Any index is allowed.

    Parameters
    ----------
    str_list : list[str]
        List of string to sort.
    numidx : int, optional
        The index of the number in the string to use as key (e.g., 0 is the
        first number found in each string while -1 is the last). The default
        is 0.
    reverse : bool, optional
        If True, the sorted list is reversed (descending order). The default is
        False.
    **kwargs : keyword args
        Keyword arguments for `find_numbers_in_string`.

    Returns
    -------
    sorted_str_list : list[str]
        Sorted list of strings.

    Arnau, 14/08/2023
    '''    
    def keyfunc(s):
        return find_numbers_in_string(s, **kwargs)[numidx]
    return sorted(str_list, key=keyfunc, reverse=reverse)

def get_dir_names(Path: str=None) -> list:
    '''
    Returns a list of all directory names inside a specified path. 
    Subdirectories are not searched. If Path is None, the current working
    directory is used.

    Parameters
    ----------
    Path : str, optional
        Path in which to search directories. The default is None (current
        working directory).

    Returns
    -------
    dir_names = list[str]
        List of directory names found.

    Arnau, 14/08/2023
    '''
    p = os.getcwd() if Path is None else Path
    return [name for name in os.listdir(p) if os.path.isdir(os.path.join(p, name))]


def apply2listElements(inlst: list, func: callable, *args) -> list:
    '''
    Apply function to each element of a list recursively.

    Parameters
    ----------
    obj : list
        Input list.
    func : callable
        Function to apply to each element.
    *args : args
        Extra arguments for func.

    Returns
    -------
    outlst : list
        Resulting list.

    Arnau, 23/08/2023
    '''
    if isinstance(inlst, list):
        return [apply2listElements(v, func, *args) for v in inlst]
    else:
        return func(inlst, *args)

def multiplyMaskedArraybyScalar(inarr, scalar):
    '''
    Multiply each element in a numpy masked array by a scalar, INCLUDING THE
    MASKED VALUES.

    Parameters
    ----------
    arr : numpy.MaskedArray
        Input numpy masked array.
    scalar : float or int
        Scalar to multiply the array by.

    Returns
    -------
    outarr : numpy.MaskedArray
        Resulting numpy masked array.

    Arnau, 23/08/2023
    '''
    return np.ma.masked_array(inarr.data*scalar, mask=inarr.mask)


def popBadSpecimens(BatchName: str, Specimens: list) -> list:
    '''
    Removes bad specimen names for the given list. A good specimen name is the
    one that is only made of the batch name and a number. 
    
    Usage example:
        >>> BatchName = 'A'
        >>> Specimens = ['A1', 'A2', 'D3', '4A', 'A-5']
        >>> popBadSpecimens(BatchName, Specimens)
        ['A1', 'A2', '4A', 'A-5']

    Parameters
    ----------
    BatchName : str
        Batch name.
    Specimens : list[str]
        List of specimen names.

    Returns
    -------
    GoodSpecimens : list[str]
        List of good specimen names.

    Arnau, 29/08/2023
    '''
    GoodSpecimens = []
    for s in Specimens:
        if isGoodSpecimen(BatchName, s):
            GoodSpecimens.append(s)
    return GoodSpecimens

def isNumber(n) -> bool:
    '''
    Check if the given variable can be converted to a float.

    Parameters
    ----------
    n : any
        Input variable.

    Returns
    -------
    bool
        True if the input variable can be converted to float.

    Arnau, 25/08/2023
    '''
    try:
        float(n)
    except:
        return False
    return True

def isGoodSpecimen(BatchName: str, SpecimenName: str) -> bool:
    '''
    Returns True if the given specimen name is a valid name for the specified
    batch. A good specimen name is the one that is only made of the batch name
    and a number. 

    Usage example:
        >>> BatchName = 'A'
        >>> SpecimenName = 'A1'
        >>> isGoodSpecimen(BatchName, SpecimenName)
        True
        >>> SpecimenName = 'An1'
        >>> isGoodSpecimen(BatchName, SpecimenName)
        False
    
    Parameters
    ----------
    BatchName : str
        Batch name.
    SpecimenName : str
        Specimen name.

    Returns
    -------
    GoodSpecimen : bool
        True if the specimen name is valid.

    Arnau, 29/08/2023
    '''
    return isNumber(SpecimenName.replace(BatchName, ''))


def fillMaskedArrayWithPrevVal(x):
    '''
    Fill numpy.MaskedArray masked values with their corresponding previous
    unmasked value.

    Parameters
    ----------
    x : numpy.MaskedArray
        Input masked array.

    Returns
    -------
    data : ndarray
        Array with the same number of elements as the input masked array where.
    
    Arnau, 09/11/2023
    '''
    mask = x.mask.copy()
    data = x.data.copy()
    outlier_indices = np.where(mask)[0]
    for i in outlier_indices:
        if i==0:
            prev = 1
            while prev in outlier_indices:
                prev += 1
            data[i] = data[i + prev]
        else:
            data[i] = data[i-1]
    return data