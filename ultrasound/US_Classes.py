# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:14:47 2020.

@author: alrom
"""
from scipy import signal
import numpy as np
import US_Functions as USF
import sys

class Material:
    """

    Clase para transductores.

        Atributes:
            Name : str, specimen name
            Material : str, material
            Description : str, description, default is None.
            Origin : str, Origin of the specimen
            Cs : +float, trans. speed of sound m/s, default is None
            Cl : +float, long. speed of sound m/s, default is None
            Thickness : +float, Thickness, default is None.

        Methods:
            self.info() : prints information.
    """

    def __init__(self, Name, Material=None, Description=None,
                 Origin='unknown', Cs=None, Cl=None, Thickness=None):
        self.Name = Name
        self.Material = Material
        self.Origin = Origin
        self.Description = Description
        self.Cs = Cs
        self.Cl = Cl
        self.Thickness = Thickness

    def info(self):
        """
        Print info of the specimen.

        Returns
        -------
        None.

        """
        print('Specimen info: ')
        print('--> Name: {}'.format(self.Name))
        print('--> Material: {}'.format(self.Material))
        print('--> Description: {}'.format(self.Description))
        print('--> Origin: {}'.format(self.Origin))
        print('--> Speed of Sound (Cs) {} m/s: '.format(self.Cs))
        print('--> Speed of Sound (Cl) {} m/s: '.format(self.Cl))
        print('--> Thickness: {} mm: '.format(self.Thickness))


class Transducer:
    """

    Clase para transductores.

        Atributes:
            Fc : float, Central frequency in Hz
            BW : +float, bandwidth. If lower than 2, considered as ratio,
                otherwise as BW in Hz
            FcNom : Nominal Fc in Hz, if any, +float or None
            BWNom : Nominal BW in Hz, if any, +float or None
            name : str, name of the transducer
            origin : str, origin of transducer
            DelayLine = None or str, delay line description if any
            DelayLineLen = +float, length of delayline if any, None
            description : str, description and comments, if any
            Diameter : +float, Diameter in mm
            Focus : +float or None, focal distance if any.
                If None, consideerd as unfocused.
            Focused : Boolean, True if focused
            BWR : +float, Relative bandwidth, Fc/BW
            Flow : +float, lower freq. of the band, Fc-BW/2
            Fhigh : +float, higher freq. of the band, Fc+BW/2

        Methods:
            self.info() : prints information.
    """

    def __init__(self, Fc, BW, FcNom=None, BWNom=None, Diameter=0,
                 Focus=None, Name='unknown', Origin='unknown', DelayLine=None,
                 DelayLineLen=None, Description=None):
        self.Fc = Fc
        self.BW = BW
        self.FcNom = FcNom
        self.BWNom = BWNom
        self.Diameter = Diameter
        self.Focus = Focus
        self.Name = Name
        self.DelayLine = DelayLine
        self.DelayLineLen = DelayLineLen
        self.Origin = Origin
        self.Description = Description

        if Focus is None:
            self.Focused = False
        else:
            self.Focused = True

        self.BWR = Fc/BW

        self.Flow = Fc-self.BW/2
        self.Fhigh = Fc+self.BW/2

    def info(self):
        """
        Print info of the transducer.

        Returns
        -------
        None.

        """
        print('Transducer info: ')
        print('--> Name: {}'.format(self.Name))
        print('--> Fc: {} MHz'.format(self.Fc/1e6))
        print('--> BW: {} MHz'.format(self.BW/1e6))
        if self.Focus is not None:
            print('--> Focus: {} mm'.format(self.Focus))
        else:
            print('--> Unfocused.')
        if self.DelayLine is not None:
            print('--> Delay line of {} of {} mm'.format(self.DelayLine, self.DelayLineLength))
        print('--> origin: {}'.format(self.Origin))
        if self.Description is not None:
            print('--> description: {}'.format(self.Description))

class ACQ:
    """

    Clase para ACQ systems.

        Atributes:
            Fs : +float, Sampling frequency in sps
            Fdac : +float, digital to analog frequency sampling in Hz, default 200e6
            bps : +int, bits per sample
            chansno = +int, number of channels
            name : str, name of the equipment
            origin : str, origin of equipment
            Description : str, description and comments, if any.

    """

    def __init__(self, Fs, Fdac=200e6, Bps=10, Chansno=2, Name='unknown',
                 Origin='unknown', Description=None):
        self.Fs = Fs
        self.Fdac = Fdac
        self.Bps = Bps
        self.Chansno = Chansno
        self.Name = Name
        self.Origin = Origin
        self.Description = Description
        self.Qlevels = 2**Bps

    def info(self):
        """
        Print info of ACQ system.

        Returns
        -------
        None.

        """
        print('ACQ equipment info: ')
        print('--> Name: {}'.format(self.Name))
        print('--> Fs: {} MHz'.format(self.Fs/1e6))
        print('--> bps: {} bits'.format(self.Bps))
        print('--> origin: {}'.format(self.Origin))
        if self.Description is not None:
            print('--> description: {}'.format(self.Description))


class Excitation:
    """

    Class for excitations.

        Atributes:
            GenCode : array, signal used for excitation ACQ
            Fs : +float, sampling frequency used to generate the signal
            Signal : str, sort of signal {pulse, burst, chirp, APWP}
            Amp : str, sort of exc. {'analog', 'rectangular'}
            Fc : +float, Central frequency in Hz
            BW : +float, bandwidth in Hz
            Duration : +float, duration in s
            Description = sort of excitation used.

    """

    def __init__(self, GenCode=None, Fs=None, Signal='pulse', Amp='None',
                 Fc=None, BW=None, Duration=None, Description=None):
        self.GenCode = GenCode
        self.Fs = Fs
        self.Signal = Signal
        self.Amp = Amp
        self.Fc = Fc
        self.BW = BW
        self.Duration = Duration
        self.Description = Description

    def info(self):
        """
        Print info of excitation.

        Returns
        -------
        None.

        """
        print('Excitation info: ')
        print('--> Signal: {}'.format(self.Signal))
        print('--> Amp: {}'.format(self.Amp))
        print('--> Fc: {} MHz'.format(self.Fc/1e6))
        print('--> BW: {} MHz'.format(self.BW/1e6))
        print('--> Duration: {} ms'.format(self.Duration*1e3))
        if self.Description is not None:
            print('--> description: {}'.format(self.Description))


class USScan:
    """

    Class for ultrasonic scans.

        Atributes:
            Data : np.array with samples, 1D/2D/3D
            Ref : Reference Ascan, default None -> take.one from data
            RefMat : Material Object for the reference signal
            NoXsteps : +int, number of X steps
            NoYsteps : +int, number of Y steps
            AscanLen : length of Ascan in samples
            XstepSize : size of X step in mm, default 0
            YstepSize : size of Y step in mm, default 0
            Fc : +float, Central frequency in Hz
            Fs : +float, sampling frequency in Hz
            BW : +float, signal bandwidth
            Cs : +float, trans. Speed of sound in media, tentative
            Cl : +float, long. Speed of sound in media, tentative
            Trans : Transducer object
            Acq : acq object.
            Excitation: Excitation object
            Material: Material object.
            CheckIntgerity : Boolean, True to check there is not wrong Ascans
            DataMemory : a replica of Data when copy_Memory() has been called

        Methods:
            self.info() : to print info of data
            self.align(UseCentroid=False, UseHilbEnv=False)
                To align data matrix to reference. Returns ToF matrix and
                replace Data with aligned Data
            self.align2zero(UseCentroid=False, UseHilbEnv=False, Who=None)
                To align input matrix to 0. Returns ToF matrix and
                replace input with aligned input. Who selects who is aligned,
                Data (None) or Ref ('Ref')
            self.delayData(ToF, UseCentroid=False, UseHilbEnv=False, Who=None)
                To delay input matrix to specific ToF. Replace input with
                delayed input. Who selects who is aligned,
                Data (None) or Ref ('Ref').
            self.copy_Memory()
                Copy self.Data in self.DataMemory
            self.relese_Memory()
                Delete self.DataMemory, to release memory
            self.reset_Data()
                Gets back memorized data -> self.Data = self.DataMemory
            self.pulseCompression(MoveBack=True, UseCentroid=False, UseHilbEnv=False)
                Compute pulse compression using Ref signal, undelayed (MoveBack)
                It uses circular cross correlation with FFT.
            self.filtfilt(SoF, CutOffFreq, FOrder=4, Who = None)
                Replace self.Data with a filtered version using filtfilt.
                Filter is a butterworth filter, lowpass or highpass.
                CutOffFreq must be in Hz and it is the theoretical. Who selects
                who is aligned, Data (None) or Ref ('Ref').
            self.lfilt(SoF, CutOffFreq, FOrder=4, Who = None)
                Replace self.Data with a filtered version using linear filtering.
                Filter is a butterworth filter, lowpass or highpass.
                CutOffFreq must be in Hz and it is the theoretical. Who selects
                who is aligned, Data (None) or Ref ('Ref').
            self.stripFrontface(UseHilbEnv=False)
                strips fromt face echo using deconvolution and stores ToF of
                frontface in self.strToF. Replace Self.Data with striped Data.
    """

    def __init__(self, Data, Ref=None, RefMat=None, XstepSize=0, YstepSize=0,
                 Fc=0, Fs=0, BW=0, Cs=0, Cl=0, Trans=None, Acq=None,
                 Excitation=None, Material=None, CheckIntgerity=False):
        # try:
        self.Dimensions = len(Data.shape)
        if CheckIntgerity:
            self.Data, FoundFalse = USF.CheckIntegrityScan(Data)
        else:
            self.Data = Data
        self.Ref = Ref
        self.RefMat = RefMat
        if self.Dimensions == 1:  # case Ascan
            self.SoS = 'Ascan'
            self.AscanLen = Data.shape[0]
            self.NoXsteps = None
            self.NoYsteps = None
            if self.Ref is None:
                self.Ref = Data
        elif self.Dimensions == 2:  # case Bscan
            self.SoS = 'Bscan'
            self.NoYsteps = None
            [self.NoXsteps, self.AscanLen] = Data.shape
            
            if self.Ref is None:
                self.Ref = Data[int(self.NoXsteps/2), :]
        else:  # case Cscan
            self.SoS = 'Cscan'
            [self.NoXsteps, self.NoYsteps, self.AscanLen] = Data.shape
            if self.Ref is None:
                self.Ref = Data[int(self.NoXsteps/2), int(self.NoYsteps/2), :]
        self.XstepSize = XstepSize
        self.YstepSize = YstepSize
        self.Trans = Trans
        self.Acq = Acq
        self.Excitation = Excitation
        self.Material = Material
        if Trans is None:
            self.Fc = Fc
        else:
            self.Fc = Excitation.Fc
        if Acq is not None:
            self.Fs = Acq.Fs
        else:
            self.Fs = Fs
        if Excitation is None:
            self.BW = BW
        else:
            self.BW = Excitation.BW
        if Material is not None:
            self.Cs = Material.Cs
            self.Cs = Material.Cl
        else:
            self.Cs = Cs
            self.Cs = Cl
        # except:  # catch *all* exceptions
        #     e = sys.exc_info()[0]
        #     print("US scan object couln't be created.")
        #     print("Error: {}".format(e))

    def info(self):
        """
        Print info of Bscan.

        Returns
        -------
        None.

        """
        print('{} info:'.format(self.SoS))
        print('--> {} Length: {} samples'.format(self.SoS, self.AscanLen))
        if self.SoS != 'Ascan':
            print('--> No of Ascans in X axes: {} samples'.format(self.NoXsteps))
            if self.XstepSize == 0:
                print('--> Step size: Unknown')
            else:
                print('--> Step size X: {} mm'.format(self.XstepSize))
                print('--> X length of Scan: {} mm'.format(self.XstepSize * self.NoXsteps))
            if self.SoS != 'Bscan':
                if self.YstepSize == 0:
                    print('--> Step size: Unknown')
                else:
                    print('--> Step size Y: {} mm'.format(self.YstepSize))
                    print('--> Y length of Scan: {} mm'.format(self.YstepSize * self.NoYsteps))
        print('--> Fs: {} MHz'.format(self.Fs/1e6))
        print('--> Fc: {} MHz'.format(self.Fc/1e6))
        print('--> BW: {} MHz'.format(self.BW/1e6))
        if self.Material.Material is None:
            print('--> Material unknown.')
        else:
            print('--> Analized material: {} (Cs={} m/s).'.format(self.Material.Material, self.Material.Cs))
        if self.Ref is not None:
            if self.RefMat is None:
                print('--> External reference of unknown origin.')
            else:
                print('--> External reference of {}.'.format(self.RefMat.Material))
        else:
            print('--> Internal reference.')
                

    def align(self, UseCentroid=False, UseHilbEnv=False):
        """
        Align Data to Reference, using subsample shift.

        Parameters
        ----------
        UseHilbEnv: Boolean, optional
            if True, uses hilbert envelope to find maximum

        Returns
        -------
        self.Data aligned
        self.ToF, np.array with time of fligt.
        """
        if self.Dimensions == 1:
            self.ToF, self.Data = USF.CalcToFAscanCosine_XCRFFT(self.Data, self.Ref, UseCentroid=UseCentroid, UseHilbEnv=UseHilbEnv)
        elif self.Dimensions == 2:
            self.ToF = np.zeros(self.NoXsteps)
            for i in np.arange(self.NoXsteps):
                self.ToF[i], self.Data[i, :] , borra= USF.CalcToFAscanCosine_XCRFFT(self.Data[i, :], self.Ref, UseCentroid=UseCentroid, UseHilbEnv=UseHilbEnv)
        else:
            self.ToF = np.zeros((self.NoXsteps, self.NoYsteps))
            for i in np.arange(self.NoXsteps):
                for j in np.arange(self.NoYsteps):
                    self.ToF[i, j], self.Data[i, j, :] = USF.CalcToFAscanCosine_XCRFFT(self.Data[i, j, :], self.Ref, UseCentroid=UseCentroid, UseHilbEnv=UseHilbEnv)

    def align2zero(self, UseCentroid=False, UseHilbEnv=False, Who=None):
        """
        Align Data to zero, using subsample shift.

        Parameters
        ----------        
        UseCentroid : Boolean, if True, use centroid instead of maximum
        UseHilbEnv: Boolean, optional
            if True, uses hilbert envelope to find maximum
        Who : str, indicates who is delayed. None -> Data, 'Ref'->Ref

        Returns
        -------
        self.Data aligned to zero
        self.ZeroToF, np.array with time of fligt to zero.
        """
        if Who == 'Ref':
            self.ZeroToF_Ref, self.Ref = USF.align2zero(self.Ref, UseCentroid=UseCentroid, UseHilbEnv=UseHilbEnv)
        else:
            if self.Dimensions == 1:
                self.ZeroToF, self.Data = USF.align2zero(self.Data, UseCentroid=UseCentroid, UseHilbEnv=UseHilbEnv)
            elif self.Dimensions == 2:
                self.ZeroToF = np.zeros(self.NoXsteps)
                for i in np.arange(self.NoXsteps):
                    self.ZeroToF[i], self.Data[i, :] = USF.align2zero(self.Data[i, :], UseCentroid=UseCentroid, UseHilbEnv=UseHilbEnv)
            else:
                self.ToF = np.zeros((self.NoXsteps, self.NoYsteps))
                for i in np.arange(self.NoXsteps):
                    for j in np.arange(self.NoYsteps):
                        self.ZeroToF[i, j], self.Data[i, j, :] = USF.align2zero(self.Data[i, j, :], UseCentroid=UseCentroid, UseHilbEnv=UseHilbEnv)

    def delayData(self, ToF, UseCentroid=False, UseHilbEnv=False, Who=None):
        """
        Align Data to specific ToF, using subsample shift.

        Parameters
        ----------
        ToF : +float, samples of delay, in subsample
        UseCentroid : Boolean, if True, use centroid instead of maximum
        UseHilbEnv: Boolean, optional
            if True, uses hilbert envelope to find maximum
        Who : str, indicates who is delayed. None -> Data, 'Ref'->Ref

        Returns
        -------
        self.Data delayed

        """
        if Who == 'Ref':
            self.Ref = USF.ShiftSubsampleByfft(self.Ref, ToF)
        else:
            if self.Dimensions == 1:
                self.Data = USF.ShiftSubsampleByfft(self.Data, ToF)
            elif self.Dimensions == 2:                
                for i in np.arange(self.NoXsteps):
                    self.Data[i, :] = USF.ShiftSubsampleByfft(self.Data[i, :], ToF[i])
            else:                
                for i in np.arange(self.NoXsteps):
                    for j in np.arange(self.NoYsteps):
                        self.Data[i, :] = USF.ShiftSubsampleByfft(self.Data[i, j, :], ToF[i, j])


    def copy_Memory(self):
        """
        Reset self.Data to the one stored in self.MemoryData.

        Returns
        -------
        self.DataMemory.

        """
        self.DataMemory = np.copy(self.Data)

    def relese_Memory(self):
        """
        delete self.MemoryData.

        Returns
        -------

        """
        del self.DataMemory

    def reset_Data(self):
        """
        Reset self.Data to the one stored in self.DataMemory.

        Returns
        -------
        self.DataMemory.

        """
        
        self.Data = np.copy(self.DataMemory)

    def pulseCompression(self, MoveBack=True, UseCentroid=False, UseHilbEnv=False):
        """
        Make pulse compression by cross correlation in frequency.

        Parameters
        ----------
        MoveBack : Boolean, to avoid delay, default is True
        UseCentroid : Boolean, to use centroid for alignment, default is False
        UseHilbEnv : Boolean, to use envelope for maxima, default is False

        Returns
        -------
        self.Compressed.

        Calculates pulse compression using the reference as adapted filter.
        Compression is made using FFT circular cross correlation. Before
        processing, Referecne is delayed to origin (if MoveBack=True) so that
        there is no delay in the resulting signal. If selected, result is
        normalized by the squared magnitude of the frequency response of the
        reference.
        Alberto, 10/11/2020
        """
        if MoveBack:
            if UseCentroid:
                DeltaToF = USF.centroid(self.Ref, UseHilbEnv=UseHilbEnv)
            else:
                DeltaToF = USF.CosineInterpMax(self.Ref, UseHilbEnv=UseHilbEnv)
            Ref = USF.ShiftSubsampleByfft(self.Ref, DeltaToF)
        else:
            Ref = self.Ref
        self.Compressed = np.zeros_like(self.Data)
        if self.Dimensions == 1:
            self.Compressed = USF.fastxcorr(self.Data, Ref)
        elif self.Dimensions == 2:
            for i in np.arange(self.NoXsteps):
                self.Compressed[i, :] = USF.fastxcorr(self.Data[i, :], Ref)
        else:
            for i in np.arange(self.NoXsteps):
                for j in np.arange(self.NoYsteps):
                    self.Compressed[i, j, :] = USF.fastxcorr(self.Data[i, j, :], Ref)

    def filtfilt(self, SoF, CutOffFreq, FOrder=4, Who = None):
        """
        Filter Data sing filtfilt algorithm. Butterwoth filter.

        Parameters
        ----------
        SoF : str, {'low','high'}, select low or high pass filter.
        CutOffFreq : +float, cut off frequency, in Hz
        FOrder : +int, filter order. The default is 4.
        Who : str, indicates who is filtered. None -> Data, 'Ref'->Ref

        Returns
        -------
        None.

        """
        if Who == 'Ref':
            self.Ref = USF.filtfilt(self.Ref, SoF, CutOffFreq, self.Fs, FOrder=FOrder)
        else:
            if self.Dimensions == 1:
                self.Data = USF.filtfilt(self.Data, SoF, CutOffFreq, self.Fs, FOrder=FOrder)
            elif self.Dimensions == 2:
                for i in np.arange(self.NoXsteps):
                    self.Data[i, :] = USF.filtfilt(self.Data[i, :], SoF, CutOffFreq, self.Fs, FOrder=FOrder)
            else:
                for i in np.arange(self.NoXsteps):
                    for j in np.arange(self.NoYsteps):
                        self.Data[i, j, :] = USF.filtfilt(self.Data[i, j, :], SoF, CutOffFreq, self.Fs, FOrder=FOrder)

    def lfilt(self, SoF, CutOffFreq, FOrder=4, Who = None):
        """
        Filter Data and Ref using lfilt algorithm. Butterwoth filter.

        Parameters
        ----------
        SoF : str, {'low','high'}, select low or high pass filter.
        CutOffFreq : +float, cut off frequency, in Hz
        FOrder : +int, filter order. The default is 4.
        Who : str, indicates who is filtered. None -> Data, 'Ref'->Ref

        Returns
        -------
        None.

        """
        if Who == 'Ref':
            self.Ref = USF.lfilt(self.Ref, SoF, CutOffFreq, self.Fs, FOrder=FOrder)
        else:
            if self.Dimensions == 1:
                self.Data = USF.lfilt(self.Data, SoF, CutOffFreq, self.Fs, FOrder=FOrder)
            elif self.Dimensions == 2:
                for i in np.arange(self.NoXsteps):
                    self.Data[i, :] = USF.lfilt(self.Data[i, :], SoF, CutOffFreq, self.Fs, FOrder=FOrder)
            else:
                for i in np.arange(self.NoXsteps):
                    for j in np.arange(self.NoYsteps):
                        self.Data[i, j, :] = USF.lfilt(self.Data[i, j, :], SoF, CutOffFreq, self.Fs, FOrder=FOrder)

    def stripFrontface(self, UseHilbEnv=False):
        """
        Strip front-face surface from data.

        Parameters
        ----------
        UseHilbEnv : Boolean, if True uses envelope, default False
        
        Returns
        -------
        None.
        self.strToF : ToF frontface - Ref

        """
        if self.Dimensions == 1:
                self.strToF, aux = USF.deconvolution(self.Data, self.Ref, stripIterNo=1, UseHilbEnv=UseHilbEnv)
                self.Data = aux[1,:]
        elif self.Dimensions == 2:
            self.strToF = np.zeros(self.NoXsteps)
            for i in np.arange(self.NoXsteps):
                self.strToF[i], aux = USF.deconvolution(self.Data[i, :], self.Ref, stripIterNo=1, UseHilbEnv=UseHilbEnv)
                self.Data[i, :] = aux[1,:]
        else:
            self.strToF = np.zeros((self.NoXsteps,self.NoYsteps))            
            for i in np.arange(self.NoXsteps):
                for j in np.arange(self.NoYsteps):
                    self.strToF[i, j], aux= USF.deconvolution(self.Data[i, j, :], self.Ref, stripIterNo=1, UseHilbEnv=UseHilbEnv)
                    self.Data[i, j, :] = aux[1,:]


#TODO Get time of flight with particular Ascan
