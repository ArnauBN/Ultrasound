# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:54:06 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

SP = Signal Processing

"""
from scipy import signal as scsig
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pylab as plt

from . import US_Functions as USF
from . import US_Loaders as USL
from . import US_Graphics as USG
from . import US_SoS as USS


#%%
class RealtimeSP:
    def __init__(self, Path, compute=False, material='resin', Cw_material='water', Verbose=False):
        '''Class for processing experiment data. 
        If compute is True, the constructor computes everything. 
        If not, it is loaded from results.txt.'''
        self.Path = Path
        self.name = Path.split('\\')[-1]
        self.Cw_material = Cw_material
        self.material = material
        
        self.paths() # set paths
        if Verbose: self.checkFiles()
        self.load() # load data
        self.setParamsFromFiles()
        
        self.LPFtemperature()
        if compute:
            self.computeTOF() # Compute Time-of-Flights
            self.computeResults() # Compute results
        
        
        # =======
        # METHODS
        # =======
        # paths()
        # loadConfigDict()
        # loadResultsDict()
        # loadTemperatureDict()
        # loadWP()
        # loadTTandPE()
        # loadPEref()
        # loadPEref2()
        
        # checkFiles()
        # load()
        # setParamsFromFiles()
        # windowAscans(Loc_WP: int=3300, Loc_TT: int=3200, Loc_PER: int=1000, Loc_PETR: int=7000,
        #              WinLen_WP: int=1000, WinLen_TT: int=1000, WinLen_PER: int=1300, WinLen_PETR: int=1300)
        
        
        # =========
        # VARIABLES
        # =========
        # self.config_dict
        # self.results_dict
        # self.temperature_dict
        # self.WPraw
        # self.TTraw
        # self.PEraw
        # self.PEref
        # self.PEref2
        # ---------
        # self.Fs
        # self.Ts
        # self.N_acqs
        # self.temperature_inside
        # self.temperature_outside
        # self.Cw
        # self.Cfromtemp
        # self.Lc
        # self.L
        # self.C
        # self.Cc
        # ---------
        # self.Win_PER
        # self.Win_PETR
        # self.Win_WP
        # self.Win_TT
        # self.WP
        # self.PE_R
        # self.PE_TR
        # self.TT
        # self.windowed
        # ---------
        
        
        
        
    
    def paths(self):
        '''
        Set all file paths of the experiment. Expected file names are:
             'config.txt'
             'PEref.bin'
             'PEref2.bin'
             'WP.bin'
             'acqdata.bin'
             'temperature.txt'
             'results.txt'

        Returns
        -------
        None.

        Arnau, 12/05/2023
        '''
        self.Experiment_config_file_name = 'config.txt' # Without Backslashes
        self.Experiment_PEref_file_name = 'PEref.bin'
        self.Experiment_PEref2_file_name = 'PEref2.bin'
        self.Experiment_WP_file_name = 'WP.bin'
        self.Experiment_acqdata_file_name = 'acqdata.bin'
        self.Experiment_Temperature_file_name = 'temperature.txt'
        self.Experiment_Results_file_name = 'results.txt'
        
        self.Config_path = os.path.join(self.Path, self.Experiment_config_file_name)
        self.PEref_path = os.path.join(self.Path, self.Experiment_PEref_file_name)
        self.PEref2_path = os.path.join(self.Path, self.Experiment_PEref2_file_name)
        self.WP_path = os.path.join(self.Path, self.Experiment_WP_file_name)
        self.Acqdata_path = os.path.join(self.Path, self.Experiment_acqdata_file_name)
        self.Temperature_path = os.path.join(self.Path, self.Experiment_Temperature_file_name)
        self.Results_path = os.path.join(self.Path, self.Experiment_Results_file_name)
    
    def loadConfigDict(self):
        '''
        Load Config dictionary.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        self.config_dict = USL.load_config(self.Config_path) if os.path.isfile(self.Config_path) else None
    
    def loadResultsDict(self):
        '''
        Load Results dictionary.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        self.results_dict = USL.load_columnvectors_fromtxt(self.Results_path) if os.path.isfile(self.Results_path) else None
    
    def loadTemperatureDict(self):
        '''
        Load Temperature dictionary.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        self.temperature_dict = USL.load_columnvectors_fromtxt(self.Temperature_path) if os.path.isfile(self.Temperature_path) else None
    
    def loadWP(self):
        '''
        Load Water-Path.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        if os.path.isfile(self.WP_path):
            with open(self.WP_path, 'rb') as f:
                self.WPraw = np.fromfile(f)
        else:
            self.WPraw = None
    
    def loadTTandPE(self):
        '''
        Load Acq data (TE and PE).

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        if os.path.isfile(self.Acqdata_path) and self.config_dict is not None:
            self.TTraw, self.PEraw = USL.load_bin_acqs(self.Acqdata_path, self.config_dict['N_acqs'])
        else:
            self.TTraw, self.PEraw = None, None
    
    def loadPEref(self):
        '''
        Load Pulse-Echo reference.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        if os.path.isfile(self.PEref_path):
            with open(self.PEref_path, 'rb') as f:
                self.PEref = np.fromfile(f)
        else:
            self.PEref = None
    
    def loadPEref2(self):
        '''
        Load Pulse-Echo reference for second face.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        if os.path.isfile(self.PEref2_path):
            with open(self.PEref2_path, 'rb') as f:
                self.PEref2 = np.fromfile(f)
        else:
            if self.PEref is not None:
                self.PEref2 = self.PEref.copy()
            else:
                self.PEref2 = None
    
    def checkFiles(self):
        '''
        Check which files do not exist.

        Returns
        -------
        None.
        
        Arnau, 01/12/2023
        '''
        if not os.path.isfile(self.Config_path):
            print(f'Config File Not Found: {self.Config_path}')
        if not os.path.isfile(self.Results_path):
            print(f'Results File Not Found: {self.Results_path}')
        if not os.path.isfile(self.Temperature_path):
            print(f'Temperature File Not Found: {self.Temperature_path}')
        if not os.path.isfile(self.WP_path):
            print(f'Water-Path File Not Found: {self.WP_path}')
        if not os.path.isfile(self.Acqdata_path):
            print(f'Acq. data File Not Found: {self.Acqdata_path}')
        if not os.path.isfile(self.PEref_path):
            print(f'Pulse-Echo Reference File Not Found: {self.PEref_path}')
        if not os.path.isfile(self.PEref2_path):
            print(f'Second Face Pulse-Echo Reference File Not Found: {self.PEref2_path}')
    
    def load(self):
        '''
        Load all data.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        self.loadConfigDict()
        self.loadResultsDict()
        self.loadTemperatureDict()
        self.loadWP()
        self.loadTTandPE()
        self.loadPEref()
        self.loadPEref2()
    
    def setParamsFromFiles(self):
        '''
        Set several parameters from config, results and temperature files.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        self.Fs = self.config_dict['Fs']
        self.Ts = self.config_dict['Ts_acq']
        self.N_acqs = self.config_dict['N_acqs']
        self.temperature_inside = self.temperature_dict['Inside'] if 'Inside' in self.temperature_dict else None
        self.temperature_outside = self.temperature_dict['Outside'] if 'Outside' in self.temperature_dict else None
        self.Cw = USS.temp2sos(self.temperature_outside, material=self.Cw_material) if 'Outside' in self.temperature_dict else self.temperature_dict['Cw']
        self.Cfromtemp = USS.temp2sos(self.temperature_inside, material=self.material)
        self.Lc = self.results_dict['Lc']
        if 'LM' in self.results_dict: self.L = self.results_dict['LM']
        if 'L' in self.results_dict: self.L = self.results_dict['L']
        if 'CM' in self.results_dict: self.C = self.results_dict['CM']
        if 'C' in self.results_dict: self.C = self.results_dict['C']
        if 'Cc' in self.results_dict: self.Cc = self.results_dict['Cc']
    
    def windowAscans(self, Loc_WP: int=3300, Loc_TT: int=3200, Loc_PER: int=1000, Loc_PETR: int=7000,
                     WinLen_WP: int=1000, WinLen_TT: int=1000, WinLen_PER: int=1300, WinLen_PETR: int=1300):
        '''
        Window the data.

        Parameters
        ----------
        Loc_WP : int, optional
            Location of the Water-Path window. The default is 3500.
        Loc_TT : int, optional
            Location of the Through-Transmission window. The default is 3300.
        Loc_PER : int, optional
            Location of the frontface window. The default is 1450.
        Loc_PETR : int, optional
            Location of the backface window. The default is 7600.
        WinLen_WP : int, optional
            Length of the Water-Path window. The default is 800.
        WinLen_TT : int, optional
            Length of the Through-Transmission window. The default is 800.
        WinLen_PER : int, optional
            Length of the frontface window. The default is 800.
        WinLen_PETR : int, optional
            Length of the backface window. The default is 800.

        Returns
        -------
        None.

        Arnau, 12/05/2023
        '''
        Smin1 = self.config_dict['Smin1']
        Smin2 = self.config_dict['Smin2']
        Smax1 = self.config_dict['Smax1']
        Smax2 = self.config_dict['Smax2']
        ScanLen = np.max([Smax1 - Smin1,  Smax2 - Smin2])  # Total scan length for computations (zero padding is used) - samples

        self.Win_PER = USF.makeWindow(SortofWin='tukey', WinLen=WinLen_PER,
                                     param1=0.25, param2=1, Span=ScanLen, Delay=Loc_PER - int(WinLen_PER/2))
        self.Win_PETR = USF.makeWindow(SortofWin='tukey', WinLen=WinLen_PETR,
                                      param1=0.25, param2=1, Span=ScanLen, Delay=Loc_PETR - int(WinLen_PETR/2))
        self.Win_WP = USF.makeWindow(SortofWin='tukey', WinLen=WinLen_WP,
                                    param1=0.25, param2=1, Span=ScanLen, Delay=Loc_WP - int(WinLen_WP/2))
        self.Win_TT = USF.makeWindow(SortofWin='tukey', WinLen=WinLen_TT,
                                    param1=0.25, param2=1, Span=ScanLen, Delay=Loc_TT - int(WinLen_TT/2))

        self.WP = self.WPraw * self.Win_WP # window Water Path
        self.PE_R = (self.PEraw.T * self.Win_PER).T # extract front surface reflection
        self.PE_TR = (self.PEraw.T * self.Win_PETR).T # extract back surface reflection
        self.TT = (self.TTraw.T * self.Win_TT).T # window Through Transmission
        
        self.windowed = True
    
    def computeTOF_old(self, windowXcor=False, correction=True, filter_tofs_pe=True, UseHilbEnv=False, windowSecondFace=True):
        '''
        Compute Time-Of-Flights.

        Parameters
        ----------
        windowXcor : bool, optional
            If True, window correlation between TT and WP to obtain the ToF of
            the first TT. Default is False.
        correction : bool, optional
            If True, apply the correction. The default is True.
        filter_tofs_pe : bool, optional
            If True, apply a filter to the correction to reduce noise. The
            filter is a Low-Pass IIR filter of order 2 and cutoff frequency 2
            mHz. The default is False.
        UseHilbEnv : bool, optional
            If True, uses envelope instead of raw signal. The default is False.
        windowSecondFace : bool, optional
            If True, apply windowing to the second face's echoes to ensure that
            the first one is found first. Default is True.
        
        Returns
        -------
        None.
        
        Arnau, 09/11/2023
        '''
        def TOF(x, y, UseHilbEnv):
            return USF.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=UseHilbEnv, UseCentroid=False)[0]
        def ID(x, y, UseHilbEnv):
            return USF.deconvolution(x, y, stripIterNo=2, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]
        def ID4(x, y, UseHilbEnv):
            return USF.deconvolution(x, y, stripIterNo=4, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]
        def windowedXcorTOF(x, y, UseHilbEnv):
            MyXcor = USF.fastxcorr(x, y, Extend=True, Same=False)
            peaks = USF.find_Subsampled_peaks(np.abs(USF.envelope(MyXcor)), distance=50, prominence=0.01)
            peak = peaks[peaks > len(MyXcor)/2][0]
            Win = USF.makeWindow(SortofWin='tukey', WinLen=50,
                                param1=0.25, param2=1, Span=len(MyXcor), Delay=peak - int(50/2))
            MyXcor *= Win
            return USF.CosineInterpMax(MyXcor, UseHilbEnv=UseHilbEnv)
        def FindMax(x):
            return USF.CosineInterpMax(x, xcor=False, UseHilbEnv=True)
        
        if correction:
            self.tofs_pe = np.apply_along_axis(TOF, 0, self.PEraw, self.PEraw[:,0], UseHilbEnv)/2
            if 1/self.Ts < 2*self.lpf_fc:
                print(f'tofs_pe does not have frequency components beyond {self.lpf_fc*1e3} mHz, therefore it is not filtered.')
                self.tofs_pe_lpf = self.tofs_pe
            else:
                b_IIR, a_IIR = scsig.iirfilter(self.lpf_order, 2*self.lpf_fc*self.Ts, btype='lowpass')
                self.tofs_pe_lpf = scsig.filtfilt(b_IIR, a_IIR, self.tofs_pe)
            tof_correction = self.tofs_pe_lpf if filter_tofs_pe else self.tofs_pe           
        
        self.tw = FindMax(self.WP) + self.config_dict['Smin1']
        if self.windowed:
            self.tT = np.apply_along_axis(FindMax, 0, self.TT) + self.config_dict['Smin1']
        else:
            self.tT = np.apply_along_axis(FindMax, 0, self.TTraw) + self.config_dict['Smin1']
        
        if self.windowed:
            if windowXcor:
                self.ToF_TW = np.apply_along_axis(windowedXcorTOF, 0, self.TT, self.WP, UseHilbEnv)
            else:
                self.ToF_TW = np.apply_along_axis(TOF, 0, self.TT, self.WP, UseHilbEnv)
            if correction:
                self.ToF_TW = self.ToF_TW - tof_correction # correction
            print('ToF_TW done.')
    
            # Iterative Deconvolution: first face
            self.ToF_RW = np.apply_along_axis(ID, 0, self.PE_R, self.PEref, UseHilbEnv)
            self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]
            print('ToF_RW done.')
    
            # Second face
            if windowSecondFace:
                _PEtemp = self.PE_TR.copy()
                mref = np.argmax(self.PEref)
                _toftemp = np.apply_along_axis(TOF, 0, _PEtemp, self.PEref, True)
                for i, t in enumerate(_toftemp):
                    _PEtemp[int(t + mref - 70) : int(t + mref + 70), i] = 0
                _toftemp2 = np.apply_along_axis(TOF, 0, _PEtemp, self.PEref, True)
                self._PEtemp  = _PEtemp
                self.ToF_TRW = np.zeros_like(self.ToF_RW)
                for i in range(len(_toftemp)):
                    if _toftemp[i] > _toftemp2[i]:
                        self.ToF_TRW[0,i] = _toftemp2[i]
                        self.ToF_TRW[1,i] = _toftemp[i]
                    else:
                        self.ToF_TRW[0,i] = _toftemp[i]
                        self.ToF_TRW[1,i] = _toftemp2[i]
            else:
                # Iterative Deconvolution: second face
                # self.ToF_TRW = np.apply_along_axis(ID, 0, self.PE_TR, self.PEref2, UseHilbEnv)
                if os.path.isfile(self.PEref2_path):
                    self._temp = np.apply_along_axis(ID, 0, self.PE_TR, self.PEref2, True)
                    self.ToF_TRW = self._temp.copy()
                    self.ToF_TRW[1] = self._temp[1]
                    self.ToF_TRW[0] = self._temp[0]
                    self.peref2_tof = TOF(self.PEref2, self.PEref, True)
                    self.ToF_TRW = self.ToF_TRW + self.peref2_tof
                else:
                    self.ToF_TRW = np.apply_along_axis(ID, 0, self.PE_TR, self.PEref, True)
            self.ToF_TR21 = self.ToF_TRW[1] - self.ToF_TRW[0]
            self.ToF_TR1R2 = self.ToF_TRW[0] - self.ToF_RW[1]
            print('ToF_TRW done.')
        else:
            if windowXcor:
                self.ToF_TW = np.apply_along_axis(windowedXcorTOF, 0, self.TTraw, self.WPraw, UseHilbEnv)
            else:
                self.ToF_TW = np.apply_along_axis(TOF, 0, self.TTraw, self.WPraw, UseHilbEnv)
            if correction:
                self.ToF_TW = self.ToF_TW - tof_correction # correction
            print('ToF_TW done.')
    
            self.ToF_RW = np.apply_along_axis(ID4, 0, self.PEraw, self.PEref, UseHilbEnv)
            self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]
            self.ToF_TR21 = self.ToF_RW[3] - self.ToF_RW[2]
            self.ToF_TR1R2 = self.ToF_RW[2] - self.ToF_RW[1]
            print('ToF_RW and ToF_TRW done.')           

    def computeTOF(self, windowXcor=False, filter_tofs_pe=True, UseHilbEnv=False, windowSecondFace=True):
        '''
        Compute Time-Of-Flights.

        Parameters
        ----------
        windowXcor : bool, optional
            If True, window correlation between TT and WP to obtain the ToF of
            the first TT. Default is False.
        filter_tofs_pe : bool, optional
            If True, apply a filter to the correction to reduce noise. The
            filter is a Low-Pass IIR filter of order 2 and cutoff frequency 2
            mHz. The default is False.
        UseHilbEnv : bool, optional
            If True, uses envelope instead of raw signal. The default is False.
        windowSecondFace : bool, optional
            If True, apply windowing to the second face's echoes to ensure that
            the first one is found first. Default is True.
        
        Returns
        -------
        None.
        
        Arnau, 08/01/2024
        '''
        def TOF(x, y, UseHilbEnv):
            return USF.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=UseHilbEnv, UseCentroid=False)[0]
        def ID(x, y, UseHilbEnv):
            return USF.deconvolution(x, y, stripIterNo=2, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]
        def ID4(x, y, UseHilbEnv):
            return USF.deconvolution(x, y, stripIterNo=4, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]
        def windowedXcorTOF(x, y, UseHilbEnv):
            MyXcor = USF.fastxcorr(x, y, Extend=True, Same=False)
            peaks = USF.find_Subsampled_peaks(np.abs(USF.envelope(MyXcor)), distance=50, prominence=0.01)
            peak = peaks[peaks > len(MyXcor)/2][0]
            Win = USF.makeWindow(SortofWin='tukey', WinLen=50,
                                param1=0.25, param2=1, Span=len(MyXcor), Delay=peak - int(50/2))
            MyXcor *= Win
            return USF.CosineInterpMax(MyXcor, UseHilbEnv=UseHilbEnv)
        def FindMax(x):
            return USF.CosineInterpMax(x, xcor=False, UseHilbEnv=True)
        
        wp    = self.WP.copy()    if self.windowed else self.WPraw.copy()
        tt    = self.TT.copy()    if self.windowed else self.TTraw.copy()
        pe_r  = self.PE_R.copy()  if self.windowed else self.PEraw.copy()
        pe_tr = self.PE_TR.copy() if self.windowed else self.PEraw.copy()
        
        self.cw0 = USS.temp2sos(self.config_dict['WP_temperature'], material='water')
        self.tw0 = FindMax(wp) + self.config_dict['Smin1']
        self.tw = self.cw0 / self.Cw_lpf * self.tw0
        
        # Correct Water-Paths
        self.WPs = np.zeros([len(wp),len(self.tw)])
        for i,t in enumerate(self.tw):
            self.WPs[:,i] = USF.ShiftSubsampleByfft(wp, t - self.tw0)
        
        # Compute Through-Transmission ToFs with corrected Water-Paths
        self.ToF_TW = np.zeros(len(self.tw))
        for i in range(len(self.tw)):
            self.ToF_TW[i] = windowedXcorTOF(tt[:,i], self.WPs[:,i], UseHilbEnv=UseHilbEnv) if windowXcor else TOF(tt[:,i], self.WPs[:,i], UseHilbEnv=UseHilbEnv)
        print('ToF_TW done.')
        
        if self.windowed:
            # Iterative Deconvolution: first face
            self.ToF_RW = np.apply_along_axis(ID, 0, pe_r, self.PEref, UseHilbEnv)
            self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]
            print('ToF_RW done.')
    
            # Second face
            if windowSecondFace:
                mref = np.argmax(self.PEref)
                _toftemp = np.apply_along_axis(TOF, 0, pe_tr, self.PEref, True)
                for i, t in enumerate(_toftemp):
                    pe_tr[int(t + mref - 70) : int(t + mref + 70), i] = 0
                _toftemp2 = np.apply_along_axis(TOF, 0, pe_tr, self.PEref, True)
                self._pe_tr  = pe_tr
                self.ToF_TRW = np.zeros_like(self.ToF_RW)
                for i in range(len(_toftemp)):
                    if _toftemp[i] > _toftemp2[i]:
                        self.ToF_TRW[0,i] = _toftemp2[i]
                        self.ToF_TRW[1,i] = _toftemp[i]
                    else:
                        self.ToF_TRW[0,i] = _toftemp[i]
                        self.ToF_TRW[1,i] = _toftemp2[i]
            else:
                # Iterative Deconvolution: second face
                if os.path.isfile(self.PEref2_path):
                    self._temp = np.apply_along_axis(ID, 0, pe_tr, self.PEref2, True)
                    self.ToF_TRW = self._temp.copy()
                    self.ToF_TRW[1] = self._temp[1]
                    self.ToF_TRW[0] = self._temp[0]
                    self.peref2_tof = TOF(self.PEref2, self.PEref, True)
                    self.ToF_TRW = self.ToF_TRW + self.peref2_tof
                else:
                    self.ToF_TRW = np.apply_along_axis(ID, 0, pe_tr, self.PEref, True)
            self.ToF_TR21 = self.ToF_TRW[1] - self.ToF_TRW[0]
            self.ToF_TR1R2 = self.ToF_TRW[0] - self.ToF_RW[1]
            print('ToF_TRW done.')
        else:
            self.ToF_RW = np.apply_along_axis(ID4, 0, self.PEraw, self.PEref, UseHilbEnv)
            self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]
            
            self.ToF_TR21 = self.ToF_RW[3] - self.ToF_RW[2]
            self.ToF_TR1R2 = self.ToF_RW[2] - self.ToF_RW[1]
            print('ToF_RW and ToF_TRW done.')


    def computeResults(self, Cc=2726, charac_container=False):
        '''
        Compute results.

        Parameters
        ----------
        Cc : float, optional
            Speed of sound in the container in m/s. The default is 5490.
        charac_container : bool, optional
            If True, compute Cc and Lc assuming water inside. The default is
            False.
        
        Returns
        -------
        None.

        Arnau, 09/01/2024
        '''
        self.Cc = Cc
        cw = self.Cw_lpf

        if charac_container:
            self.Cc = cw*(np.abs(self.ToF_TW)/self.ToF_R21 + 1) # container speed - m/s
            self.Lc = cw/2*(np.abs(self.ToF_TW) + self.ToF_R21)/self.Fs # container thickness - m
        else:
            self.Lc = self.Cc*self.ToF_R21/2/self.Fs # container thickness - m
            self.L = (self.ToF_R21 + np.abs(self.ToF_TW) + self.ToF_TR1R2/2)*cw/self.Fs - 2*self.Lc # material thickness - m
            self.C = 2*self.L/self.ToF_TR1R2*self.Fs # material speed - m/s
            
    def computeResults_old(self, Cc=2726, charac_container=False, cw=None):
        '''
        Compute results.

        Parameters
        ----------
        Cc : float, optional
            Speed of sound in the container in m/s. The default is 5490.
        charac_container : bool, optional
            If True, compute Cc and Lc assuming water inside. The default is
            False.
        cw : float, optional
            Speed of sound in water in m/s. If it is None, then 
            cw = self.Cw[0]. The default is None.
        
        Returns
        -------
        None.

        Arnau, 27/06/2023
        '''
        self.Cc = Cc
        if cw == None: cw = self.Cw[0]

        if charac_container:
            self.Cc = cw*(np.abs(self.ToF_TW)/self.ToF_R21 + 1) # container speed - m/s
            self.Lc = cw/2*(np.abs(self.ToF_TW) + self.ToF_R21)/self.Fs # container thickness - m
        else:
            self.Lc = self.Cc*self.ToF_R21/2/self.Fs # container thickness - m
            self.L = (self.ToF_R21 + np.abs(self.ToF_TW) + self.ToF_TR1R2/2)*cw/self.Fs - 2*self.Lc # material thickness - m
            self.C = 2*self.L/self.ToF_TR1R2*self.Fs # material speed - m/s
    
    def computeResultsFinal_old(self, Cc=2726, lpf_temperature=True):
        self.Cc = Cc
        cw0 = USS.temp2sos(self.config_dict['WP_temperature'], material='water')
        cw = self.Cw_lpf if lpf_temperature else self.Cw
        
        self.Lc = self.Cc*self.ToF_R21/2/self.Fs # container thickness - m
        self.L = (cw0*self.tw + cw*(self.ToF_R21 - self.tT + self.ToF_TR1R2/2))/self.Fs - 2*self.Lc # material thickness - m
        self.C = 2*self.L/self.ToF_TR1R2*self.Fs # material speed - m/s
        
    def LPFtemperature(self):
        '''
        Filter the temperature readings with a Low-Pass IIR filter of order 2 
        and cutoff frequency 2 mHz.
    
        Returns
        -------
        None.
    
        Arnau, 27/07/2023
        '''
        self.lpf_order = 2
        self.lpf_fc = 2e-3 # Hz
        if 1/self.Ts < 2*self.lpf_fc:
            print(f'Signal does not have frequency components beyond {self.lpf_fc*1e3} mHz, therefore it is not filtered.')
        else:
            b_IIR, a_IIR = scsig.iirfilter(self.lpf_order, 2*self.lpf_fc*self.Ts, btype='lowpass')
            self.temperature_inside_lpf = scsig.filtfilt(b_IIR, a_IIR, self.temperature_inside)
            self.temperature_outside_lpf = scsig.filtfilt(b_IIR, a_IIR, self.temperature_outside)
            self.Cw_lpf = USS.temp2sos(self.temperature_outside_lpf, material=self.Cw_material)
            self.Cfromtemp_lpf = USS.temp2sos(self.temperature_inside_lpf, material=self.material)
    
    def saveResults(self):
        '''
        Save results in {self.Results_path} path column-wise with a header. 
        Saved data is t, Lc, L and C with header t,Lc,LM,CM.
        Separator is a comma.

        Returns
        -------
        None.

        Arnau, 30/05/2023
        '''
        with open(self.Results_path, 'w') as f:
            row = 't,Lc,LM,CM'
            f.write(row+'\n')

        Time_axis = np.arange(0, self.N_acqs)*self.Ts
        with open(self.Results_path, 'a') as f:
            for i in range(self.N_acqs):
                row = f'{Time_axis[i]},{self.Lc[i]},{self.L[i]},{self.C[i]}'
                f.write(row+'\n')

    def saveCw(self):
        '''
        Save temperature (raw and lpf) and Cw (raw and lpf) in
        {self.Temperature_path} path column-wise with a header. Saved data is
        temperature, temperature_lpf, Cw and Cw_lpf with header
        Inside,Inside_lpf,Cw,Cw_lpf. Separator is a comma.

        Returns
        -------
        None.

        Arnau, 17/11/2023
        '''
        if 'Outside' in self.temperature_dict:
            with open(self.Temperature_path, 'w') as f:
                row = 'Inside,Inside_lpf,Outside,Outside_lpf,Cw,Cw_lpf'
                f.write(row+'\n')
    
            with open(self.Temperature_path, 'a') as f:
                for i in range(self.N_acqs):
                    row = f'{self.temperature_inside[i]},{self.temperature_inside_lpf[i]},{self.temperature_outside[i]},{self.temperature_outside_lpf[i]},{self.Cw[i]},{self.Cw_lpf[i]}'
                    f.write(row+'\n')
        else:
            with open(self.Temperature_path, 'w') as f:
                row = 'Inside,Inside_lpf,Cw,Cw_lpf'
                f.write(row+'\n')
    
            with open(self.Temperature_path, 'a') as f:
                for i in range(self.N_acqs):
                    row = f'{self.temperature_inside[i]},{self.temperature_inside_lpf[i]},{self.Cw[i]},{self.Cw_lpf[i]}'
                    f.write(row+'\n')


# =============================================================================
#%%
# =============================================================================

class DogboneSP:
    def __init__(self, Path, compute=True, imgName=None, Verbose=False, **kwargs):
        '''Class for processing experiment data. If compute is True, the constructor computes everything.'''
        self.Path = Path
        self.name = Path.split('\\')[-1]
        self.imgName = imgName
        
        self.paths() # set paths
        if Verbose: self.checkFiles()
        self.load() # load data
        self.setParamsFromFiles()
        self.angleAndScanpos() # get rotation angle and scanner position vector
        self.LPFtemperature() # Low-Pass filter temperature
        
        if compute:
            self.computeAll(**kwargs)
            
        
        
        # =======
        # METHODS
        # =======
        # paths()
        # loadConfigDict()
        # loadResultsDict()
        # loadTemperatureDict()
        # loadScanpattern()
        # loadWP()
        # loadTTandPE()
        # loadPEref()
        # loadPEref2()
        # loadImg()
        # loadArchDensity()
        # checkFiles()
        # load()
        # setParamsFromFiles()
        # windowAscans(Loc_WP: int=3300, Loc_TT: int=3200, Loc_PER: int=1000, Loc_PETR: int=7000,
        #              WinLen_WP: int=1000, WinLen_TT: int=1000, WinLen_PER: int=1300, WinLen_PETR: int=1300)
        
        
        # =========
        # VARIABLES
        # =========
        # self.config_dict
        # self.results_dict
        # self.temperature_dict
        # self.WPraw
        # self.TTraw
        # self.PEraw
        # self.PEref
        # self.PEref2
        # self.scanpattern
        # self.img
        # self.archdensity
        # ---------
        # self.Fs
        # self.Ts
        # self.N_acqs
        # self.temperature
        # self.Cw
        # self.L
        # self.CL
        # self.Cs
        # self.density
        # self.shear_modulus
        # self.young_modulus
        # self.bulk_modulus
        # self.poisson_ratio
        # ---------
        # self.Ridx
        # self.theta
        # self.scanpos_step
        # self.scanpos
        # ---------
        
        
        
        # ---------
        # self.Win_PER
        # self.Win_PETR
        # self.Win_WP
        # self.Win_TT
        # self.WP
        # self.PE_R
        # self.PE_TR
        # self.TT
        # self.windowed
        # ---------
        
        
        
        
        
        
        
    
    def paths(self):
        '''
        Set all file paths of the experiment. Expected file names are:
             'config.txt'
             'PEref.bin'
             'WP.bin'
             'acqdata.bin'
             'temperature.txt'
             'scanpath.txt'
            f'img{self.name}.jpg' or f'{self.imgName}'

        Returns
        -------
        None.

        Arnau, 04/01/2024
        '''
        self.Experiment_config_file_name = 'config.txt' # Without Backslashes
        self.Experiment_PEref_file_name = 'PEref.bin'
        self.Experiment_PEref2_file_name = 'PEref2.bin'
        self.Experiment_WP_file_name = 'WP.bin'
        self.Experiment_acqdata_file_name = 'acqdata.bin'
        self.Experiment_Temperature_file_name = 'temperature.txt'
        self.Experiment_scanpath_file_name = 'scanpath.txt'
        self.Experiment_img_file_name = f'img{self.name}.jpg' if self.imgName is None else self.imgName
        self.Experiment_results_file_name = 'results.txt'
        self.Experiment_archdensity_file_name = 'archdensity.txt'
        
        self.Config_path = os.path.join(self.Path, self.Experiment_config_file_name)
        self.PEref_path = os.path.join(self.Path, self.Experiment_PEref_file_name)
        self.PEref2_path = os.path.join(self.Path, self.Experiment_PEref2_file_name)
        self.WP_path = os.path.join(self.Path, self.Experiment_WP_file_name)
        self.Acqdata_path = os.path.join(self.Path, self.Experiment_acqdata_file_name)
        self.Temperature_path = os.path.join(self.Path, self.Experiment_Temperature_file_name)
        self.Scanpath_path = os.path.join(self.Path, self.Experiment_scanpath_file_name)
        self.Img_path = os.path.join(self.Path, self.Experiment_img_file_name)
        self.Results_path = os.path.join(self.Path, self.Experiment_results_file_name)
        self.Archdensity_path = os.path.join(self.Path, self.Experiment_archdensity_file_name)
    
    
    
    def loadConfigDict(self):
        '''
        Load Config dictionary.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        self.config_dict = USL.load_config(self.Config_path) if os.path.isfile(self.Config_path) else None
    
    def loadResultsDict(self):
        '''
        Load Results dictionary.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        self.results_dict = USL.load_columnvectors_fromtxt(self.Results_path) if os.path.isfile(self.Results_path) else None
    
    def loadTemperatureDict(self):
        '''
        Load Temperature dictionary.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        self.temperature_dict = USL.load_columnvectors_fromtxt(self.Temperature_path) if os.path.isfile(self.Temperature_path) else None
    
    def loadScanpattern(self):
        '''
        Load scanpattern.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        self.scanpattern = USL.load_columnvectors_fromtxt(self.Scanpath_path, delimiter=',', header=False, dtype=str) if os.path.isfile(self.Scanpath_path) else None
    
    def loadWP(self):
        '''
        Load Water-Path.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        if os.path.isfile(self.WP_path):
            with open(self.WP_path, 'rb') as f:
                self.WPraw = np.fromfile(f)
        else:
            self.WPraw = None
        self.WP = self.WPraw.copy()
    
    def loadTTandPE(self):
        '''
        Load Acq data (TE and PE).

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        if os.path.isfile(self.Acqdata_path) and self.config_dict is not None:
            self.TTraw, self.PEraw = USL.load_bin_acqs(self.Acqdata_path, self.config_dict['N_acqs'])
        else:
            self.TTraw, self.PEraw = None, None
        self.TT, self.PE = self.TTraw.copy(), self.PEraw.copy()
    
    def loadPEref(self):
        '''
        Load Pulse-Echo reference.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        if os.path.isfile(self.PEref_path):
            with open(self.PEref_path, 'rb') as f:
                self.PEref = np.fromfile(f)
        else:
            self.PEref = None
    
    def loadPEref2(self):
        '''
        Load Pulse-Echo reference for second face.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        if os.path.isfile(self.PEref2_path):
            with open(self.PEref2_path, 'rb') as f:
                self.PEref2 = np.fromfile(f)
        else:
            if self.PEref is not None:
                self.PEref2 = self.PEref.copy()
            else:
                self.PEref2 = None
    
    def loadImg(self):
        '''
        Load Image of specimen.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        self.img = mpimg.imread(self.Img_path) if os.path.isfile(self.Img_path) else None

    def loadArchDensity(self):
        '''
        Load Archimedean density.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        self.archdensity = float(USL.load_columnvectors_fromtxt(self.Archdensity_path, header=False, dtype=float)) if os.path.isfile(self.Archdensity_path) else None
    
    def checkFiles(self):
        '''
        Check which files do not exist.

        Returns
        -------
        None.
        
        Arnau, 01/12/2023
        '''
        if not os.path.isfile(self.Config_path):
            print(f'Config File Not Found: {self.Config_path}')
        if not os.path.isfile(self.Results_path):
            print(f'Results File Not Found: {self.Results_path}')
        if not os.path.isfile(self.Temperature_path):
            print(f'Temperature File Not Found: {self.Temperature_path}')
        if not os.path.isfile(self.WP_path):
            print(f'Water-Path File Not Found: {self.WP_path}')
        if not os.path.isfile(self.Acqdata_path):
            print(f'Acq. data File Not Found: {self.Acqdata_path}')
        if not os.path.isfile(self.PEref_path):
            print(f'Pulse-Echo Reference File Not Found: {self.PEref_path}')
        if not os.path.isfile(self.PEref2_path):
            print(f'Second Face Pulse-Echo Reference File Not Found: {self.PEref2_path}')
        if not os.path.isfile(self.Scanpath_path):
            print(f'Scanpattern File Not Found: {self.Scanpath_path}')
        if not os.path.isfile(self.Img_path):
            print(f'Image File Not Found: {self.Img_path}')
        if not os.path.isfile(self.Archdensity_path):
            print(f'Archimedean Density File Not Found: {self.Archdensity_path}')
    
    def load(self):
        '''
        Load all data.

        Returns
        -------
        None.

        Arnau, 01/12/2023
        '''
        self.loadConfigDict()
        self.loadResultsDict()
        self.loadTemperatureDict()
        self.loadScanpattern()
        self.loadWP()
        self.loadTTandPE()
        self.loadPEref()
        self.loadPEref2()
        self.loadImg()
        self.loadArchDensity()
    
    def setParamsFromFiles(self):
        '''
        Set several parameters from config, results and temperature files.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        self.Fs = self.config_dict['Fs']
        self.Ts = self.config_dict['Ts_acq']
        self.N_acqs = self.config_dict['N_acqs']
        self.temperature = self.temperature_dict['Inside'] if 'Inside' in self.temperature_dict else None
        self.Cw = self.temperature_dict['Cw'] if 'Cw' in self.temperature_dict else USS.temp2sos(self.temperature, material='water')
        
        self.L             = self.results_dict['L']             if self.results_dict is not None and 'L'             in self.results_dict else None
        self.CL            = self.results_dict['CL']            if self.results_dict is not None and 'CL'            in self.results_dict else None
        self.Cs            = self.results_dict['Cs']            if self.results_dict is not None and 'Cs'            in self.results_dict else None
        self.density       = self.results_dict['density']       if self.results_dict is not None and 'density'       in self.results_dict else None
        self.shear_modulus = self.results_dict['shear_modulus'] if self.results_dict is not None and 'shear_modulus' in self.results_dict else None
        self.young_modulus = self.results_dict['young_modulus'] if self.results_dict is not None and 'young_modulus' in self.results_dict else None
        self.bulk_modulus  = self.results_dict['bulk_modulus']  if self.results_dict is not None and 'bulk_modulus'  in self.results_dict else None
        self.poisson_ratio = self.results_dict['poisson_ratio'] if self.results_dict is not None and 'poisson_ratio' in self.results_dict else None
    
    def angleAndScanpos(self):
        '''
        Obtain rotation angle nad generate position vector.

        Returns
        -------
        None.

        Arnau, 11/12/2023
        '''
        self.Ridx = [np.where(self.scanpattern == s)[0][0] for s in self.scanpattern if 'R' in s][0] + 1
        self.theta = float(self.scanpattern[self.Ridx-1][1:]) * np.pi / 180
        self.scanpos_step = float(self.scanpattern[0][1:])
        self.scanpos = np.arange(self.Ridx)*self.scanpos_step # mm

    def LPFtemperature(self):
        '''
        Filter the temperature readings with a Low-Pass IIR filter of order 2 
        and cutoff frequency 2 mHz.

        Returns
        -------
        None.

        Arnau, 16/03/2023
        '''
        self.lpf_order = 2
        self.lpf_fc = 2e-3 # Hz
        if 1/self.Ts < 2*self.lpf_fc:
            print(f'Signal does not have frequency components beyond {self.lpf_fc*1e3} nHz, therefore it is not filtered.')
        else:
            b_IIR, a_IIR = scsig.iirfilter(self.lpf_order, 2*self.lpf_fc*self.Ts, btype='lowpass')
            self.temperature_lpf = scsig.filtfilt(b_IIR, a_IIR, self.temperature)
            self.Cw_lpf = USS.temp2sos(self.temperature_lpf, material='water')
    
    # def computeTOF(self, UseHilbEnv: bool=False, UseCentroid: bool=False, WindowTTshear: bool=False, Loc_TT: int=None):
    #     '''
    #     Compute all Time-Of-Flights.

    #     Parameters
    #     ----------
    #     UseHilbEnv : bool, optional
    #         Use maximum of the correlation's envelope. The default is False.
    #     UseCentroid : bool, optional
    #         Use centroid of the correlation. The default is False.
    #     WindowTTshear : bool, optional
    #         Window the shear TT pulse. The default is False.
    #     Loc_TT : int, optional
    #         Location of the TT window. If None, it is automatically estimated
    #         using the average of the first 20 mm of the specimen. The default
    #         is None.
            
    #     Returns
    #     -------
    #     None.

    #     Arnau, 31/08/2023
    #     '''
    #     def TOF(x, y):
    #         return USF.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=UseHilbEnv, UseCentroid=UseCentroid)[0]

    #     def ID(x, y):           
    #         return USF.deconvolution(x, y, stripIterNo=2, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]

    #     if WindowTTshear:
    #         if Loc_TT is None:
    #             self.ToF_TW = np.apply_along_axis(TOF, 0, self.TT, self.WP)
    #             self.winTT(Loc_TT)
    #             self.ToF_TW = np.apply_along_axis(TOF, 0, self.windowedTT, self.WP)
    #         else:
    #             self.winTT(Loc_TT)
    #             self.ToF_TW = np.apply_along_axis(TOF, 0, self.windowedTT, self.WP)
    #     else:
    #         self.ToF_TW = np.apply_along_axis(TOF, 0, self.TT, self.WP) # tt - tw
        
    #     self.ToF_RW = np.apply_along_axis(ID, 0, self.PE, self.PEref)
    #     self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]

    # def computeTOFFinal(self, UseHilbEnv: bool=False, UseCentroid: bool=False, WindowTTshear: bool=False, Loc_TT: int=None):
    #     def TOF(x, y):
    #         return USF.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=UseHilbEnv, UseCentroid=UseCentroid)[0]
    #     def ID(x, y):           
    #         return USF.deconvolution(x, y, stripIterNo=2, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]
    #     def FindMax(x):
    #         return USF.CosineInterpMax(x, xcor=False, UseHilbEnv=False)

    #     self.tw = FindMax(self.WP) + self.config_dict['Smin1']
    #     if WindowTTshear:
    #         if Loc_TT is None:
    #             self.ToF_TW = np.apply_along_axis(TOF, 0, self.TT, self.WP)
    #             self.winTT(Loc_TT)
    #             self.tT = np.apply_along_axis(FindMax, 0, self.windowedTT) + self.config_dict['Smin1']
    #         else:
    #             self.winTT(Loc_TT)
    #             self.tT = np.apply_along_axis(FindMax, 0, self.windowedTT) + self.config_dict['Smin1']
    #     else:
    #         self.tT = np.apply_along_axis(FindMax, 0, self.TT) + self.config_dict['Smin1']
        
    #     self.ToF_RW = np.apply_along_axis(ID, 0, self.PE, self.PEref)
    #     self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]


    def computeTOF(self, UseHilbEnv: bool=False, UseCentroid: bool=False, WindowTTshear: bool=False, Loc_TT: int=None):
        def TOF(x, y):
            return USF.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=UseHilbEnv, UseCentroid=UseCentroid)[0]
        def ID(x, y):           
            return USF.deconvolution(x, y, stripIterNo=2, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]
        def FindMax(x):
            return USF.CosineInterpMax(x, xcor=False, UseHilbEnv=True)
        
        self.cw0 = USS.temp2sos(self.config_dict['WP_temperature'], material='water')
        self.tw0 = FindMax(self.WP) + self.config_dict['Smin1']
        self.tw = self.cw0 / self.Cw_lpf * self.tw0
        
        self.WPs = np.zeros([len(self.WP),len(self.tw)])
        for i,t in enumerate(self.tw):
            self.WPs[:,i] = USF.ShiftSubsampleByfft(self.WP, t - self.tw0)
        
        if WindowTTshear:
            if Loc_TT is None:
                self.ToF_TW = np.apply_along_axis(TOF, 0, self.TT, self.WP)
            self.winTT(Loc_TT)
        
        self.ToF_TW = np.zeros(len(self.tw))
        for i in range(len(self.tw)):
            tt = self.windowedTT[:,i] if WindowTTshear else self.TT[:,i]
            self.ToF_TW[i] = USF.CalcToFAscanCosine_XCRFFT(tt, self.WPs[:,i], UseHilbEnv=UseHilbEnv, UseCentroid=UseCentroid)[0] # ts - tw
                        
        self.ToF_RW = np.apply_along_axis(ID, 0, self.PE, self.PEref)
        self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]
        
    # def computeResults(self, Cw_mode='mean'):
    #     '''
    #     Compute Thickness (L), Longitudinal velocity (CL), Shear velocity (Cs).

    #     Parameters
    #     ----------
    #     Cw_mode : str, optional
    #         This parameter defines what speed of sound in water to use. The
    #         available options are:
    #             'temperature':
    #                 The Cw obtained from the temperature measurements.
    #             'mean':
    #                 The mean of the Cw obtained from the temperature
    #                 measurements.
    #             'lpf':
    #                 The low-pass filtered Cw obtained from the temperature
    #                 measurements.
    #             any other value:
    #                 The Cw measured directly with the ToF (config_dict['Cw']).
    #         The default is 'mean'.

    #     Returns
    #     -------
    #     None.

    #     Arnau, 16/03/2023
    #     '''
    #     cw = self.getCwfromCwMode(Cw_mode)
    #     cw_aux = np.asarray([cw]).flatten()[::-1]

    #     self.L = cw/2*(-2*self.ToF_TW + self.ToF_R21)/self.Fs # thickness - m
    #     self.CL = cw*(-2*self.ToF_TW/self.ToF_R21 + 1) # longitudinal velocity - m/s
    #     self.Cs = cw_aux / np.sqrt(np.sin(self.theta)**2 + (cw_aux * self.ToF_TW[::-1] / (self.L * self.Fs) + np.cos(self.theta))**2) # shear velocity - m/s

    #     self.CL = self.CL[:self.Ridx]
    #     self.L = self.L[:self.Ridx]
    #     self.Cs = self.Cs[:self.Ridx]

    # def computeResultsFinal(self, lpf_temperature=True):        
    #     cw0 = USS.temp2sos(self.config_dict['WP_temperature'], material='water')
    #     cw = self.Cw_lpf if lpf_temperature else self.Cw
    #     cw_aux = np.asarray([cw]).flatten()[::-1]
        
    #     self.L = ((cw0*self.tw - cw*self.tT) + cw*self.ToF_R21/2)/self.Fs # thickness - m
    #     self.CL = 2*(cw0*self.tw - cw*self.tT)/self.ToF_R21 + cw # longitudinal velocity - m/s
    #     self.Cs = cw_aux / np.sqrt(np.sin(self.theta)**2 + ((cw_aux*self.tT[::-1] - cw0*self.tw) / (self.L * self.Fs) + np.cos(self.theta))**2) # shear velocity - m/s

    #     self.CL = self.CL[:self.Ridx]
    #     self.L = self.L[:self.Ridx]
    #     self.Cs = self.Cs[:self.Ridx]

    def computeResults(self):
        '''
        Compute Thickness (L), Longitudinal velocity (CL), Shear velocity (Cs).

        Returns
        -------
        None.

        Arnau, 05/12/2023
        '''
        cw = self.Cw_lpf
        cw_aux = np.asarray([cw]).flatten()[::-1]
                
        self.L = cw/2*(-2*self.ToF_TW + self.ToF_R21)/self.Fs # thickness - m
        self.CL = cw*(-2*self.ToF_TW/self.ToF_R21 + 1) # longitudinal velocity - m/s
        self.Cs = cw_aux / np.sqrt(np.sin(self.theta)**2 + (cw_aux * self.ToF_TW[::-1] / (self.L * self.Fs) + np.cos(self.theta))**2) # shear velocity - m/s

        self.CL = self.CL[:self.Ridx]
        self.L = self.L[:self.Ridx]
        self.Cs = self.Cs[:self.Ridx]

    # def getCwfromCwMode(self, Cw_mode):
    #     self.Cw_mode = Cw_mode
    #     if type(Cw_mode) is not str:
    #         cw = self.config_dict['Cw']
    #     else:
    #         if Cw_mode.lower() not in ['mean', 'lpf', 'temperature']:
    #             cw = self.config_dict['Cw']
    #         elif Cw_mode.lower()=='temperature':
    #             cw = self.Cw
    #         elif Cw_mode.lower()=='mean':
    #             cw = np.mean(self.Cw)
    #         elif Cw_mode.lower()=='lpf':
    #             cw = self.Cw_lpf
    #     return cw

    def computeDensity(self, UseAvgAcrossGains: bool=False):
        '''
        Compute density using the already recorded reference amplitude. The 
        recorded references are for 15, 16, 17, 18, 19 and 20 dB of gain at 
        50us.

        Parameters
        ----------
        UseAvgAcrossGains : bool, optional
            If False, the reference used is the one that has the same gain as
            the experiment (note that 19 and 20 dB are already in saturation),
            no gain correction is applied. If True, the references for 15, 16,
            17 and 18 dB are gain-corrected and averaged. Then the experiment
            amplitude is corrected with its own gain. The default is False.

        Returns
        -------
        None.

        Arnau, 16/03/2023
        '''
        self.Arefs = np.array([0.32066701, 0.36195303, 0.40814156, 0.45066097, 0.45507314, 0.45697591])
        self.ArefsGains = np.array([15, 16, 17, 18, 19, 20])

        G = self.config_dict['Gain_Ch2']
        if UseAvgAcrossGains:
            self.AR1 = np.max(np.abs(self.PE)*(10**(-G/20)), axis=0)[:self.Ridx]
            self.Aref = np.mean(self.Arefs[:4]*(10**(-self.ArefsGains[:4]/20)))
        else:
            idx = np.where(self.ArefsGains==G)[0][0]
            self.Aref = self.Arefs[idx]
            self.AR1 = np.max(np.abs(self.PE), axis=0)[:self.Ridx]
            
        Zw = 1.48e6 # acoustic impedance of water (N.s/m)
        d = Zw / self.CL * (self.Aref + self.AR1) / (self.Aref - self.AR1) # density (kg/m^3)
        self.density = d * 1e-3 # density (g/cm^3)

    def computeModuli(self, UseArchDensity=True):
        '''
        Computes the Shear modulus, Young's modulus, Bulk modulus and Poisson's
        ratio. To do this, the density, longitudinal velocity and shear
        velocity are used.

        Returns
        -------
        None.

        Arnau, 05/08/2023
        '''
        Cs2 = self.Cs**2
        CL2 = self.CL**2
        d = self.archdensity if UseArchDensity else self.density
        self.shear_modulus = d * Cs2
        self.young_modulus = d * Cs2 * (3*CL2 - 4*Cs2) / (CL2 - Cs2)
        self.bulk_modulus  = d * (CL2 - 4*Cs2/3)
        self.poisson_ratio = d * (CL2 - 2*Cs2) / (2*(CL2 - Cs2))

    def computeAll(self, **kwargs):
        self.computeTOF(**kwargs) # Compute Time-of-Flights
        self.computeResults() # Compute results
        self.computeDensity() # Compute density
        self.computeModuli() # Compute mechanical properties

    def saveResults(self, binary=False):
        '''
        Save results in {self.Results_path} file. Variables are sved in columns
        with the first row as header. The saved variables are:
            self.scanpos
            self.L
            self.CL
            self.Cs
            self.density

        Returns
        -------
        None.

        Arnau, 15/05/2023
        '''
        _wmode = 'wb' if binary else 'w'
        _amode = 'ab' if binary else 'a'
        with open(self.Results_path, _wmode) as f:
            row = 'scanpos,L,CL,Cs,density,shear_modulus,young_modulus,bulk_modulus,poisson_ratio'
            f.write(row+'\n')
        with open(self.Results_path, _amode) as f:
            for i in range(len(self.CL)):
                row = f'{self.scanpos[i]},{self.L[i]},{self.CL[i]},{self.Cs[i]},{self.density[i]},{self.shear_modulus[i]},{self.young_modulus[i]},{self.bulk_modulus[i]},{self.poisson_ratio[i]}'
                f.write(row+'\n')

    def winTT(self, Loc_TT=None):
        '''
        Window TT pulses for shear velocity.

        Parameters
        ----------
        Loc_TT : int, optional
            Location of the TT window. If None, it is automatically estimated
            using the average of the first 20 mm of the specimen.
            
        Returns
        -------
        None.

        Arnau, 31/08/23
        '''
        self.windowedTT = self.TT.copy()
        
        position_limit = 20 # mm
        WinLen_TT = 80 # samples
        ScanLen = self.config_dict['Smax1'] - self.config_dict['Smin1']
        
        if Loc_TT is None:
            idx = USF.find_nearest(self.scanpos, position_limit)[0]
            tofs = np.mean(self.ToF_TW[::-1][:idx])
            wp_tofs = USF.CosineInterpMax(self.WP, xcor=False)
            Loc = wp_tofs + tofs
        else:
            Loc = Loc_TT

        MyWin_TT = USF.makeWindow(SortofWin='tukey', WinLen=WinLen_TT,
                        param1=0.25, param2=1, Span=ScanLen, Delay=Loc - int(WinLen_TT/2))
        for i in range(self.Ridx, self.N_acqs):
            self.windowedTT[:,i] = self.TT[:,i] * MyWin_TT
    
    def plotGUI(self, ptxt: str='northwest'):
        '''
        Plot data (CL, Cs, L, TT, PE and img) in an interactive GUI.

        Parameters
        ----------
        ptxt : str, optional
            Location of the display for the cursor's current coordinates. The
            default is 'northwest'.

        Returns
        -------
        None.

        Arnau, 21/03/2023
        '''
        Smin = self.config_dict['Smin1']
        Smax = self.config_dict['Smax1']
        if 'length' in self.config_dict:
            length = self.config_dict['length']
        elif 'Length' in self.config_dict:
            length = self.config_dict['Length']
        else:
            length = None
        t = np.arange(Smin, Smax) / self.Fs * 1e6 # us
        USG.dogboneGUI(self.scanpos, t, self.CL, self.Cs, self.L*1e3, self.PE, self.TT, self.img, length=length, ptxt=ptxt)

    def plotTemperature(self):
        '''
        Plot the measured temperature with the low-pass filtered version and 
        the corresponding Cw.

        Returns
        -------
        ax1 : AxesSubplot
            Top axis handle.
        ax2 : AxesSubplot
            Bottom axis handle.

        Arnau, 16/03/2023
        '''
        ax1, ax2 = plt.subplots(2)[1]
        ax1.scatter(np.arange(self.N_acqs), self.temperature, marker='.', color='k')
        ax1.plot(self.temperature_lpf, 'r', lw=3)
        ax1.set_ylabel('Temperature (\u2103)')
        
        ax2.scatter(np.arange(self.N_acqs), self.Cw, marker='.', color='k')
        ax2.plot(self.Cw_lpf, 'r', lw=3)
        ax2.set_ylabel('Cw (m/s)')
        
        plt.tight_layout()
        
        return ax1, ax2


#%%
# =============================================================================
# =============================================================================


class BasicSP:
    '''Class with only static methods to compute basic properties of signals.
    Signals must be numpy arrays.'''
    
    @staticmethod
    def mean(x, ddof=0):
        '''
        Compute the mean of x. Produces the same result as np.mean(x) if 
        ddof==0.

        Parameters
        ----------
        x : ndarray
            Input signal.
        ddof : int, optional
            Degrees of freedom. The default is 0.

        Returns
        -------
        result: float
            The mean of x.

        Arnau, 24/07/2023
        '''
        return x.sum() / (len(x) - ddof)

    @staticmethod
    def var(x, ddof=0):
        '''
        Compute the variance of x. Produces the same result as np.var(x, ddof).

        Parameters
        ----------
        x : ndarray
            Input signal.
        ddof : int, optional
            Degrees of freedom. The default is 0.

        Returns
        -------
        result: float
            The variance of x.

        Arnau, 24/07/2023
        '''
        return BasicSP.mean(np.abs(x - x.mean())**2, ddof=ddof)

    @staticmethod
    def std(x, ddof=0):
        '''
        Compute the standard deviation of x. Produces the same result as 
        np.std(x, ddof).

        Parameters
        ----------
        x : ndarray
            Input signal.
        ddof : int, optional
            Degrees of freedom. The default is 0.

        Returns
        -------
        result: float
            The standard deviation of x.

        Arnau, 24/07/2023
        '''
        return np.sqrt(BasicSP.var(x, ddof=ddof))
    
    @staticmethod
    def energy(x):
        '''
        Computes the energy of a signal. Produces the same result as
        np.correlate(x, x, mode='valid')[0]

        Parameters
        ----------
        x : ndarray
            Input signal.

        Returns
        -------
        result: float
            The energy of x.

        Arnau, 24/07/2023
        '''
        return np.sum(np.abs(x)**2)
    
    @staticmethod
    def stdpower(x, ddof=0):
        '''
        Computes the power of a signal by computing its variance. Also known as
        'average power'. If ddof==1, it is almost the same as BasicSP.power(x).

        Parameters
        ----------
        x : ndarray
            Input signal.
        ddof : int, optional
            Degrees of freedom. The default is 0.

        Returns
        -------
        result: float
            The power (variance) of x.

        Arnau, 24/07/2023
        '''
        return BasicSP.var(x, ddof=ddof)

    @staticmethod
    def power(x):
        '''
        Computes the power of a signal. Also known as 'average power'. Produces
        the same result as:
            np.sum(BasicSP.PSD(x)) / len(x)
        and:
            np.sum(BasicSP.PSDfromAutocorr(x)) / len(x)

        Parameters
        ----------
        x : ndarray
            Input signal.

        Returns
        -------
        result: float
            The power of x.

        Arnau, 24/07/2023
        '''
        return BasicSP.energy(x) / len(x)

    @staticmethod
    def rms(x):
        '''
        Computes the root-mean-square amplitude of x (it is the square root of 
        its BasicSP.power(x)).

        Parameters
        ----------
        x : ndarray
            Input signal.

        Returns
        -------
        result: float
            The rms value of x.

        Arnau, 24/07/2023
        '''
        return np.sqrt(BasicSP.power(x))
    
    @staticmethod
    def cv(x, ddof=0):
        '''
        Compute the CV (Coefficient of Variation) of a signal. Also known as 
        RSD (Relative Standard Deviation) or NRMSD (Normalized Root-Mean-Square
        Deviation)

        Parameters
        ----------
        x : ndarray
            Input signal.
        ddof : int, optional
            Degrees of freedom. The default is 0.

        Returns
        -------
        result: float
            The Coefficient of Variation of x.

        Arnau, 24/07/2023
        '''
        return BasicSP.std(x, ddof=ddof) / BasicSP.mean(x, ddof=ddof)
    
    @staticmethod
    def statisticalSNR(x, ddof=0): 
        '''
        Computes the SNR of x from its CV (Coefficient of Variation). Here, 
        this SNR is defined as a power ratio. In some references, one may find 
        the definition to be an amplitude ratio. This SNR is only useful for
        non-negative values.

        Parameters
        ----------
        x : ndarray
            Input signal.
        ddof : int, optional
            Degrees of freedom. The default is 0.

        Returns
        -------
        result: float
            The SNR of x.

        Arnau, 24/07/2023
        '''
        return 1 / BasicSP.cv(x, ddof=ddof)**2

    @staticmethod
    def PSD(x, nfft=None):
        '''
        Computes the PSD (Power Spectral Density) of a signal in W/Hz. If 
        nfft==2*len(x)-1, this produces the same result as 
        BasicSP.PSDfromAutocorr(x).

        Parameters
        ----------
        x : ndarray
            Input signal.
        nfft : int, optional
            Number of FFT points. The default is None.

        Returns
        -------
        result: ndarray
            The PSD of x.

        Arnau, 24/07/2023
        '''
        n = len(x) if nfft is None else nfft
        return np.abs(np.fft.fft(x, n)**2) / n
    
    @staticmethod
    def autocorr(x):
        '''
        Computes the autocorrelation of x. The resulting length is 2*len(x)-1.

        Parameters
        ----------
        x : ndarray
            Input signal.

        Returns
        -------
        result: ndarray
            The autocorrelation of x.

        Arnau, 24/07/2023
        '''
        return np.correlate(x, x, mode='full')

    @staticmethod
    def PSDfromAutocorr(x):
        '''
        Computes the PSD of x from its autocorrelation using the
        Wiener-Khinchin theorem. The resulting length is 2*len(x)-1.

        Parameters
        ----------
        x : ndarray
            Input signal.

        Returns
        -------
        result: ndarray
            The PSD of x.

        Arnau, 24/07/2023
        '''
        return np.abs(np.fft.fft(BasicSP.autocorr(x))) / (2*len(x)-1)