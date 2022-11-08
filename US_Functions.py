# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:47:56 2020.

@author: alrom
"""
import matplotlib.pylab as plt
import numpy as np
from scipy import signal
import sys
import winsound
from scipy.signal import find_peaks

#TODO: ckeck the length of xcorr, maybe not correct to ensure valid samples
def fastxcorr(x, y, Extend=False, Same=True):
    """
    Calculate xcor using fft. Asumes vectors are columnwise.

    Parameters
    ----------
    x : comlumnwise np.array of floats. Is a 1D columnwise vector
    y : comlumnwise np.array of floats. Is a 1D columnwise vector
    Extend : boolean, if True, extend to correct dimensionality of result
    Same : boolean, if True, return result with dimensions equal longest

    Returns
    -------
    cross correlation between x and y

    Calculates cross correlation between signals in the frequency domain.
    Length of ffts is equal to the length of the longest signal.
    Alberto, 10/11/2020
    """
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
    """
    Calculate centorid of a vector.

    Inputs
        x : np.array 1D, input vetor.
        UseHilbEnv : bollean, if True use envelope. Initial False.
    Outputs
        centroid.

    Calculates the centroid of a signal or of its envelope (UseHilbert=True).
    It is used mainly for delays.
    """
    x = envelope(x)
    n = np.arange(len(x))
    return np.sum(n*(x**2))/np.sum(x**2)


def nextpow2(i):
    """
    Calculate next power of 2.

    Parameters
    ----------
    i : int
        Numer to calculate next power of 2

    Returns
    -------
    n : int
        closet power of 2 so that 2**n>i.

    Calculate next power of 2, i. e., minimum N so that 2*N>i
    Alberto, 10/11/2020
    """
    n = int(np.log2(i))
    if 2**n < i:
        n += 1
    return n


def ShiftSubsampleByfft(Signal, Delay):
    """
    Delay signal in subsample precision using FFT.

    Parameters
    ----------
    Signal : np.array
        Array to be shifted
    Delay : +float
        delay in subsample precision

    Returns
    -------
    Shifted signal

    Delays a signal in frequency domain, so subsampling arbitrary delay can be
    applied.
    Alberto, 10/11/2020
    """
    N = np.size(Signal)  # signal length
    HalfN = np.floor(N / 2)  # length of the semi-frequency axis in frequency domain
    FAxis1 = np.arange(HalfN + 1) / N  # Positive semi-frequency axis
    FAxis2 = (np.arange(HalfN + 2, N + 1, 1) - (N + 1)) / N  # Negative semi-frequency axis
    FAxis = np.concatenate((FAxis1, FAxis2))  # Full reordered frequency axis

    return np.real( np.fft.ifft(np.fft.fft(Signal) * np.exp(1j*2*np.pi*FAxis*Delay)))


def CosineInterpMax(MySignal, UseHilbEnv=False):
    """
    Calculate the location of the maximum in subsample basis.
    Uses cosine interpolation.

    Parameters
    ----------
    MySignal : np.array
        input signal
    UseHilbEnv : boolean, optional
        If True, uses envelope instead of raw signal. The default is False.

    Returns
    -------
    DeltaToF : +float
        Location of the maxim in subsample precision.

    Calculates the location of the maximum of a signal in subsample basis
    using cosine interpolation.
    Alberto, 10/11/2020
    """
    if UseHilbEnv:
        MySignal = np.absolute(signal.hilbert(MySignal))
    MaxLoc = np.argmax(np.abs(MySignal))  # find index of maximum
    N = MySignal.size  # signal length
    A = MaxLoc - 1  # left proxima
    B = MaxLoc + 1  # Right proxima
    if MaxLoc == 0:  # Check if maxima is in the first of the last sample
        A = MySignal.size - 1
    elif MaxLoc == MySignal.size-1:
        B = 0
        # calculate interpolation maxima according to cosine interpolation
    Alpha = np.arccos((MySignal[A] + MySignal[B]) / (2 * MySignal[MaxLoc]))
    Beta = np.arctan((MySignal[A] - MySignal[B]) / (2 * MySignal[MaxLoc] * np.sin(Alpha)))
    Px = Beta / Alpha
    # Calculate ToF in samples
    DeltaToF = MaxLoc - Px
    # Check wherter if delay is to the right or to the left and correct ToF
    if MaxLoc > N/2:
        DeltaToF = -(N - DeltaToF)
    # Returned value is DeltaToF, the location of the maxima in subsample basis
    return DeltaToF


def CalcToFAscanCosine_XCRFFT(Data, Ref, UseCentroid=False, UseHilbEnv=False, Extend=False):
    """
    Calculate cross correlation in frequency domain.

    Parameters
    ----------
      Data = Ascan
      Ref = Reference to align
      UseCentroid : Boolean, if True, use centroid instead of maximum
      UseHilbEnv = Boolean, True to envelope instead of raw signal

    Returns
    -------
      DeltaToF = Time of flight between pulses.
      AlignedData = Aligned array to Ref.
      MyXcor : cross correlation.

    Used to align one Ascan to a Reference by ToF subsample estimate using
    cosine interpolation. Also returns ToFmap and Xcorr. Xcoor is calculated
    using FFT.
    It uses cosine interpolation or centroid (UseCentroid=True) to approximate
    peak location. IF UseHilbEnv=True, uses envelope for the delay instead
    of raw signal.
    Alberto, 10/11/2020
    """
    try:
        # Calculates xcorr in frequency domain
        MyXcor = fastxcorr(Data, Ref, Extend=Extend)
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
    """
    Align signal 2 zero. it is intended to flatten the phase.

    Parameters
    ----------
      Data = Ascan
      UseCentroid : Boolena, if True, use centroid instead of maximum
      UseHilbEnv = boolean, True if using hilbert transform

    Returns
    -------
      AlignedData = Aligned array to Ref
      ZeroToF = Time of flight to zero.

    Align signal to zero in order to flatten its phase. It delays the signal
    so that its maximum or centroid (UseCentroid=True) is located at the
    origin. It uses the signal or its envelope (UseHilbEnv=True).
    Alberto, 10/11/2020
    """
    # try:
    # determine time of flight
    if UseCentroid:
        ZeroToF = centroid(Data, UseHilbEnv=UseHilbEnv)
    else:
        ZeroToF = CosineInterpMax(Data, UseHilbEnv=UseHilbEnv)
    # Delay to align
    AlignedData = ShiftSubsampleByfft(Data, ZeroToF)
    return ZeroToF, AlignedData

    # except Exception as ex:
    #     print(ex)


def lfilt(InData, SoF, CutOffFreq, Fs, FOrder=4):
    """
    Filter Signal using butterworth filter.

    Parameters
    ----------
    InData : np.array pof float, Input signal
    SoF : str, {'low','high'}, Choose between low or high pass filter
    CutOffFreq : +float, Cut off frequency in Hz
    Fs : +float, Sampling frequency in Hz
    FOrder : +int, order of the filter, Order of the filter, default is 4.

    Returns
    -------
    filtered data

    Filters data by linear filtering. The filter is a butterwoth, with
    cut of frequency Fc=CutOffFreq in Hz according to given Fs in Hz. Note that
    it is the teorethical nominal desired cut of frequency. Filter can be
    highpass or lowpass.
    Alberto, 10/11/2020
    """
    try:
        Fc = CutOffFreq/Fs*2
        b, a = signal.butter(FOrder, Fc, SoF)
        zi = signal.lfilter_zi(b, a)
        return signal.lfilter(b, a, InData, zi=zi*InData[0])
    except Exception:
        print('lfilt, Check parameters, something is wonrg.')


def filtfilt(InData, SoF, CutOffFreq, Fs, FOrder=4):
    """
    Filter Signal using filtfilt butterworth filter.

    Parameters
    ----------
    InData : np.array float
        Inputa dato to filter
    SoF : str, {'low','hogh'}
        Choose between low or high pass filter
    CutOffFreq : +float
        Cut off frequency in Hz
    Fs : +float
        Sampling frequency in Hz
    FOrder : +int, order of the filter
        Order of the filter to be applied. The default is 4.

    Returns
    -------
    filtered data

    Filters a signal using filtfilt algorithm  so that no delay is produced.
    Note that the frequency response of the resulting filter is the square
    of the original one. The filter is a butterwoth, with
    cut of frequency Fc=CutOffFreq in Hz according to given Fs in Hz. Note that
    it is the teorethical nominal desired cut of frequency. Filter can be
    highpass or lowpass.
    Alberto, 10/11/2020
    """
    try:
        Fc = CutOffFreq/Fs*2
        b, a = signal.butter(FOrder, Fc, SoF)
        return signal.filtfilt(b, a, InData)
    except Exception:
        print('filtfilt, Check parameters, something is wonrg.')


def deconvolution(Data, Ref, stripIterNo=2, UseHilbEnv=False):
    """
    Iterative deconvolution.

    Parameters
    ----------
    Data : np.array of float, Ascan
    Ref : np.array of float, Ascan
    stripIterNo : +int, number of iterations of the deconvolution, default=2
    UseHilbEnv : Boolean, if True use hilber envelope maximum. default False

    Returns
    -------
    ToF np.array of succesive ToF, float, in subsample basis
    StrMat striping matrix, no.array, (stripIterNo, len(Data)).

    Applies iterative deconvolution to a signal using a reference signal.
    It delays de reference to align it to the maximum (or centroid) of the
    signal (or its envelope) and then subtract it (scaled to have the same
    amplitude), so that the remainder has the main echo supressed. The
    process can be repeated in a successive algorithm, stripping the succesive
    echoes. Note that due to errors (mainly if there is overlap between echoes)
    the error raises with each iteration. It returns the ToF vector with the
    location of the maxima (or centroids) of the successive echoes (or its 
    envelopes), and also the matrix of the resulting striped signals at each
    iteration, including in the first row the original signal.
    It consumes a lot of memory when working with big Bscans or Cscans.
    Alberto, 10/11/2020
    """
    RAmp = np.sum(np.power(Ref, 2)) / len(Ref)  # mean power of reference
    StrMat = np.zeros((stripIterNo+1, len(Data)))  # reallocates stripped matrix
    ToF = np.zeros(stripIterNo)  # preallocates ToF
    StrMat[0, :] = Data
    for LayerNo in np.arange(stripIterNo):
        ToF[LayerNo] = CalcToFAscanCosine_XCRFFT(StrMat[LayerNo, :], Ref, UseHilbEnv=UseHilbEnv)[0]
        RefShifted = ShiftSubsampleByfft(Ref, -ToF[LayerNo])  # Shift the ref to the Ascan position
        Amp = np.sum(StrMat[LayerNo, :] * RefShifted) / len(RefShifted) / RAmp  # Amplitude scaling factor
        StrMat[LayerNo+1, :] = StrMat[LayerNo, :] - RefShifted * Amp  # strip Ascan
    return ToF, StrMat

# RAmp=np.sum(np.power(Ref,2))/ScanLen # mean power of refference 
# for i in range(NumAscans):   
#     DeltaToF = USF.CalcToFAscanCosine_XCRFFT(MyScan.Data[i,:],Ref,UseHilbEnv=True)[0] #time of flight of first echo        
#     RefShifted = USF.ShiftSubsampleByfft(Ref,-DeltaToF) #  Shift the ref to the Ascan position
#     Amp = np.sum(MyScan.Data[i,:] * RefShifted) / len(RefShifted) / RAmp # Amplitude Ponderation factor
#     MyScan.Data[i,:] = MyScan.Data[i,:] - RefShifted*Amp # strip pulse


def zeroPadding(Data, NewLen):
    """
    Extend arrays (Ascans) by zero padding.

    Parameters
    ----------
    Data : np.array, input matrix, 1D/2D/3D
    NewLen : New length of the Ascans

    Returns
    -------
    Data : Extended data

    Used for zeropadding signals for 1D/2D/3D arrays. Signal must be in the 
    last dimension [-1]. If the signal is longer than the desired final 
    length (NewLen), it does nothing.
    Alberto, 10/11/2020.
    """
    OldLen = Data.shape[-1]
    if NewLen > OldLen:
        if Data.ndim == 1:
            Data = np.append(Data, np.zeros(NewLen - OldLen))
        elif Data.ndim == 2:
            Data = np.concatenate((Data, np.zeros((Data.shape[0], NewLen - OldLen))),axis=1)
        else:
            Data = np.concatenate((Data, np.zeros((Data.shape[0], Data.shape[1], NewLen - OldLen))),axis=2)
        return Data


def findIndexInAxis(Data, Value):
    """
    Find index of value in sorted 1D array.

    Parameters
    ----------
    Data : Array
    Value : Value we are looking for
    Returns
    -------
    Index or None.

    Find the index of the closest value in the input vector, which is a 
    sorted array, usually an axis, so that Data[index] is the closes to Value.
    Alberto, 10/11/2020.
    """
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
    """
    Calculate envelopes as absolute value of hilbert transform.

    Parameters
    ----------
    Data : +float array
    axis : int, axis along whcih to calculate, default -1

    Returns
    -------
    envelope.

    Calculate the envelope of a signal as the absolute value of its hilbert
    transform.
    Alberto, 10/11/2020
    """
    if Data.ndim > 1:
        return np.abs(signal.hilbert(Data, axis=axis))
    else:
        return np.abs(signal.hilbert(Data))

def moving_average(Data, WinLen):
    """
    Apply undelayed moving average with rectangular window.

    Parameters
    ----------
    Data : +float, input data (1D/2D/3D)
    WinLen : +int odd, lenth of window

    Returns
    -------
    Averaged signal

    It calculates undelayed moving average, that is, first zeropads signal and
    window, and then delays windows to zero to avoid delay in the output. Note
    that the zeropadding is made to the length of the resulting non-overlapped
    result, therefore only first samples are correct, that is, being DataLen
    the length of the signal and WinLen the length of the window, the zero
    padding is that so that finale length is DataLen+WinLen-1, so after
    processing only the first DataLen sampes are correct. That is why odd
    number of samples in window is mandatory.
    MA is calculated using correlation, as windows are symmetric.
    Alberto, 09/11/2020
    """
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
    """
    Normalize each Ascan separately

    Parameters
    ----------
    Data : Data matrix to normalize

    Returns
    -------
    Normalized matrix

    """
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
    """
    Make window.
    
    Parameters
    ----------
    SortofWin : str, can be any of the following, entered as plaintext
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman,
        blackmanharris, nuttall, barthannkaiser (needs beta), gaussian (needs standard deviation),
        general_gaussian (needs power, width), slepian (needs width), dpss (needs normalized half-bandwidth),
        chebwin (needs attenuation), exponential (needs decay scale), tukey (needs taper fraction)
    WinLen : +int, Length of the desired window
    param1 = +float, beta (kaiser), std (Gaussian), power (general gaussian),
        width (slepian), norm h-b (dpss), attenuation (chebwin),
        decay scale (exponential), tapper fraction (tukey)
    param2 : +float, width (general gaussian)
    Span : final length of the required window, in case expansion needed
    Delay : Required delay to the right, in samples
    
    Return
    ------
    Window : 1D array of lenth WinLen or span
    """
    
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
            MyWin = ShiftSubsampleByfft(MyWin, -Delay)  # if float use subsample
    return MyWin

def CheckIntegrityScan(Scan,FixIt = True):
    '''
    Check integrity of Bascan or Csacn (non zero arrays)

    Parameters
    ----------
    Ascan : Bascan or Cscan
    FixIt : Boolean, if True, fix data

    Returns
    -------
    Scan : fixed Ascan
    FoundError : Boolean, if it found error Ascans

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
    #frequency = 2500  # Set Frequency To 2500 Hertz
    #duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)            
        

def extendedHilbert(Data, NoSamples=None, Zeros=True):
    '''
    Calculate hilbert transform extendindg signal before calculating
    the anañitical signal, to prevent boundary problems.    
    Extension is bilateral (starting and ending of signal)

    Parameters
    ----------
    Data : array of float, innput data 
    NoSamples : number samples to be added, if None, calculate 10%
    Zeros : bollean, if True, add zeros, otherwise reflect corresponding samples

    Returns
    -------
    HT : Hilbert transform

    '''
    if NoSamples==None:
        NoSamples = int(len(Data/10))    
    
    Env_temp = signal.hilbert( np.concatenate( ( np.zeros((NoSamples)), Data, np.zeros((NoSamples)) ) ) )    
    return Env_temp[NoSamples:-NoSamples]


def Normalize(Data):
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
    Data : sequence
        A signal with peaks.

    Returns
    -------
    Real_peaks : ndarray
        Subsampled peaks that satisfy all given conditions.

    '''
    if np.isnan(np.sum(Data)):
        return np.ones(4)*np.nan
    peaks, _ = find_peaks(Data, **kwargs)

    # now we use cosine interpolation to be more precise... although not really important
    Real_peaks = np.zeros((4))
    for i in np.arange(4):
        A = peaks[i]-1
        B = peaks[i]+2
        MaxLoc = peaks[i]
        Alpha = np.arccos((Data[A] + Data[B]) / (2 * Data[MaxLoc]))
        Beta = np.arctan((Data[A] - Data[B]) / (2 * Data[MaxLoc] * np.sin(Alpha)))
        Px = Beta / Alpha
        
        # recalculate peak location in samples adding the interpolated value
        Real_peaks[i] = peaks[i] - Px
    return Real_peaks


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

    '''
    aux = np.array(data)
    Z = Zscore(data)
    new_data = aux[np.abs(Z)<m]
    outliers = aux[np.abs(Z)>=m]
    outliers_indexes = np.where(np.abs(Z)>=m)[0]
    return new_data, outliers, outliers_indexes

def speedofsound_in_water(T, method: str, method_param=None):
    '''
    Compute speed of sound in pure water as a function of temperature with the
    given method. Some methods require parameters given in method_param.
    
    Available methods:
        'Bilaniuk'
        'Marczak'
        'Lubbers'
        'Abdessamad'
    
    Available method parameters:
        'Bilaniuk':
            112
            36
            148
        'Lubbers':
            '15-35'
            '10-40'
        'Abdessamad':
            148
            36

    Parameters
    ----------
    T : float or ndarray
        Temperature of water in Celsius.
    method : str
        Method to use.
    method_param : int or str
        parameter for the method. Default is None.

    Returns
    -------
    c : float or ndarray
        the speed of sound in pure (distilled) water in m/s.

    Arnau, 08/11/2022
    '''
    if method.lower()=='bilaniuk':
        # Bilaniuk and Wong (0-100 ºC) - year 1993-1996
        if method_param==112:
            c = 1.40238742e3 + 5.03821344*T - 5.80539349e-2*(T**2) + \
                3.32000870e-4*(T**3) - 1.44537900e-6*(T**4) + 2.99402365e-9*(T**5)
        elif method_param==36:
            c = 1.40238677e3 + 5.03798765*T - 5.80980033e-2*(T**2) + \
                3.34296650e-4*(T**3) - 1.47936902e-6*(T**4) + 3.14893508e-9*(T**5)
        elif method_param==148:
            c = 1.40238744e3 + 5.03836171*T - 5.81172916e-2*(T**2) + \
                3.34638117e-4*(T**3) - 1.48259672e-6*(T**4) + 3.16585020e-9*(T**5)
    elif method.lower()=='marczak':
        # Marczak (0-95 ºC) - year 1997
        c = 1.402385e3 + 5.038813*T - 5.799136e-2*(T**2) + 3.287156e-4*(T**3) - \
            1.398845e-6*(T**4) + 2.787860e-9*(T**5)
    elif method.lower()=='lubbers':
        if method_param=='15-35':
            # Lubbers and Graaff (15-35 ºC) - year 1998
            c = 1404.3 + 4.7*T - 0.04*(T**2)
        elif method_param=='10-40':
            # Lubbers and Graaff (10-40 ºC) - year 1998
            c = 1405.03 + 4.62*T - 3.83e-2*(T**2)
    elif method.lower()=='abdessamad':
        # Abdessamad, Malaoui & Iqdour, Radouane & Ankrim, Mohammed & Zeroual, 
        # Abdelouhab & Benhayoun, Mohamed & Quotb, K.. (2005). 
        # New model for speed of sound with temperature in pure water. 
        # AMSE Review (Association for the Advancement of Modelling and Simulation
        # Techniques in Enterprises). 74. 12.10-12.13.
        if method_param==148:
            # (0.001-95.126 ºC)
            c = 1.569678141e3*np.exp(-((T-5.907868678e1)/(-3.443078912e2))**2) - \
                2.574064370e4*np.exp(-((T+3.705052160e2)/(-1.601257116e2))**2)
        elif method_param==36:
            # (0.056-74.022 ºC)
            c = 1.567302324e3*np.exp(-((T-6.101414576e1)/(-3.388027429e2))**2) - \
                1.468922269e4*np.exp(-((T+3.255477156e2)/(-1.478114724e2))**2)
    return c