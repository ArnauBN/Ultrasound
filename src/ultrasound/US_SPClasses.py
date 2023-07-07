# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:54:06 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

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
    def __init__(self, Path, compute=False):
        '''Class for processing experiment data. If compute is True, the constructor computes everything. If not, it is loaded from results.txt.'''
        self.Path = Path
        self.name = Path.split('\\')[-1]
        
        self._paths() # set paths
        self._load() # load data
        
        self.LPFtemperature()
        if compute:
            self.computeTOF() # Compute Time-of-Flights
            self.computeResults() # Compute results
    
    def _paths(self):
        '''
        Set all file paths of the experiment. Expected file names are:
             'config.txt'
             'PEref.bin'
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
        self.Experiment_WP_file_name = 'WP.bin'
        self.Experiment_acqdata_file_name = 'acqdata.bin'
        self.Experiment_Temperature_file_name = 'temperature.txt'
        self.Experiment_Results_file_name = 'results.txt'
        
        self.Config_path = os.path.join(self.Path, self.Experiment_config_file_name)
        self.PEref_path = os.path.join(self.Path, self.Experiment_PEref_file_name)
        self.WP_path = os.path.join(self.Path, self.Experiment_WP_file_name)
        self.Acqdata_path = os.path.join(self.Path, self.Experiment_acqdata_file_name)
        self.Temperature_path = os.path.join(self.Path, self.Experiment_Temperature_file_name)
        self.Results_path = os.path.join(self.Path, self.Experiment_Results_file_name)
    
    def _load(self):
        '''
        Load all data.

        Returns
        -------
        None.

        Arnau, 12/05/2023
        '''
        self.config_dict, self.results_dict, self.temperature_dict, self.WPraw, self.TTraw, self.PEraw = USL.load_all(self.Path)
        
        with open(self.PEref_path, 'rb') as f:
            self.PEref = np.fromfile(f)
        
        self.Fs = self.config_dict['Fs']
        self.Ts = self.config_dict['Ts_acq']
        self.N_acqs = self.config_dict['N_acqs']
        self.temperature = self.temperature_dict['Inside']
        self.Cw = self.temperature_dict['Cw']
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
    
    def computeTOF(self, windowXcor=False, correction=True, filter_tofs_pe=True, UseHilbEnv=False):
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
            
        Returns
        -------
        None.
        
        Arnau, 18/05/2023
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
        
        if correction:
            self.tofs_pe = np.apply_along_axis(TOF, 0, self.PEraw, self.PEraw[:,0], UseHilbEnv)/2
            if 1/self.Ts < 2*self.lpf_fc:
                print(f'tofs_pe does not have frequency components beyond {self.lpf_fc*1e3} mHz, therefore it is not filtered.')
                self.tofs_pe_lpf = self.tofs_pe
            else:
                b_IIR, a_IIR = scsig.iirfilter(self.lpf_order, 2*self.lpf_fc*self.Ts, btype='lowpass')
                self.tofs_pe_lpf = scsig.filtfilt(b_IIR, a_IIR, self.tofs_pe)
        tof_correction = self.tofs_pe_lpf if filter_tofs_pe else self.tofs_pe           
        
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
            
            # Iterative Deconvolution: second face
            # self.ToF_TRW = np.apply_along_axis(ID, 0, self.PE_TR, self.PEref, UseHilbEnv)
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

    def computeResults(self, Cc=5490, charac_container=False, cw=None):
        '''
        Compute results.

        Parameters
        ----------
        Cc : float, optional
            Speed of sound in the container in m/s. The default is 4459.
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

    def LPFtemperature(self):
        '''
        Filter the temperature readings with a Low-Pass IIR filter of order 2 
        and cutoff frequency 2 mHz.
    
        Returns
        -------
        None.
    
        Arnau, 16/05/2023
        '''
        self.lpf_order = 2
        self.lpf_fc = 2e-3 # Hz
        if 1/self.Ts < 2*self.lpf_fc:
            print(f'Signal does not have frequency components beyond {self.lpf_fc*1e3} mHz, therefore it is not filtered.')
        else:
            b_IIR, a_IIR = scsig.iirfilter(self.lpf_order, 2*self.lpf_fc*self.Ts, btype='lowpass')
            self.temperature_lpf = scsig.filtfilt(b_IIR, a_IIR, self.temperature)
            self.Cw_lpf = USS.temp2sos(self.temperature_lpf, material='water')
    
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


# =============================================================================
# =============================================================================


class DogboneSP:
    def __init__(self, Path, compute=True, imgName=None):
        '''Class for processing experiment data. If compute is True, the constructor computes everything.'''
        self.Path = Path
        self.name = Path.split('\\')[-1]
        self.imgName = imgName
        
        self._paths() # set paths
        self._load() # load data
        self._angleAndScanpos() # get rotation angle and scanner position vector
        self.LPFtemperature() # Low-Pass filter temperature
        
        if compute:
            self.computeTOF() # Compute Time-of-Flights
            self.computeResults() # Compute results
            self.computeDensity() # Compute density
    
    def _paths(self):
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

        Arnau, 16/03/2023
        '''
        self.Experiment_config_file_name = 'config.txt' # Without Backslashes
        self.Experiment_PEref_file_name = 'PEref.bin'
        self.Experiment_WP_file_name = 'WP.bin'
        self.Experiment_acqdata_file_name = 'acqdata.bin'
        self.Experiment_Temperature_file_name = 'temperature.txt'
        self.Experiment_scanpath_file_name = 'scanpath.txt'
        self.Experiment_img_file_name = f'img{self.name}.jpg' if self.imgName is None else self.imgName
        self.Experiment_results_file_name = 'results.txt'
        
        self.Config_path = os.path.join(self.Path, self.Experiment_config_file_name)
        self.PEref_path = os.path.join(self.Path, self.Experiment_PEref_file_name)
        self.WP_path = os.path.join(self.Path, self.Experiment_WP_file_name)
        self.Acqdata_path = os.path.join(self.Path, self.Experiment_acqdata_file_name)
        self.Temperature_path = os.path.join(self.Path, self.Experiment_Temperature_file_name)
        self.Scanpath_path = os.path.join(self.Path, self.Experiment_scanpath_file_name)
        self.Img_path = os.path.join(self.Path, self.Experiment_img_file_name)
        self.Results_path = os.path.join(self.Path, self.Experiment_results_file_name)
    
    def _load(self):
        '''
        Load all data.

        Returns
        -------
        None.

        Arnau, 16/05/2023
        '''
        # Config
        self.config_dict = USL.load_config(self.Config_path)
        self.Fs = self.config_dict['Fs']
        self.N_acqs = self.config_dict['N_acqs']
        self.Ts = self.config_dict['Ts_acq']
        
        # Data
        self.TT, self.PE = USL.load_bin_acqs(self.Acqdata_path, self.config_dict['N_acqs'])
        
        # Temperature and CW
        self.temperature_dict = USL.load_columnvectors_fromtxt(self.Temperature_path)
        self.temperature = self.temperature_dict['Inside']
        self.Cw = self.temperature_dict['Cw']
        
        # Scan pattern
        self.scanpattern = USL.load_columnvectors_fromtxt(self.Scanpath_path, delimiter=',', header=False, dtype=str)
        
        # WP
        with open(self.WP_path, 'rb') as f:
            self.WP = np.fromfile(f)
        
        # PE ref
        with open(self.PEref_path, 'rb') as f:
            self.PEref = np.fromfile(f)
        
        # Image
        if os.path.exists(self.Img_path):
            self.img = mpimg.imread(self.Img_path)
        else:
            self.img = None

    def _angleAndScanpos(self):
        '''
        Obtain rotation angle nad generate position vector.

        Returns
        -------
        None.

        Arnau, 16/03/2023
        '''
        self.Ridx = [np.where(self.scanpattern == s)[0][0] for s in self.scanpattern if 'R' in s][0] + 1
        self.theta = float(self.scanpattern[self.Ridx-1][1:]) * np.pi / 180
        step = float(self.scanpattern[0][1:])
        self.scanpos = np.arange(self.Ridx)*step # mm

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
            self.Cw_lpf = USF.speedofsound_in_water(self.temperature_lpf)
    
    def computeTOF(self, UseHilbEnv: bool=False, UseCentroid: bool=False, WindowTTshear: bool=False, Loc_TT: int=1175):
        '''
        Compute all Time-Of-Flights.

        Parameters
        ----------
        UseHilbEnv : bool, optional
            Use maximum of the correlation's envelope. The default is False.
        UseCentroid : bool, optional
            Use centroid of the correlation. The default is False.
        WindowTTshear : bool, optional
            Window the shear TT pulse. The default is False.
        Loc_TT : int, optional
            Location of the TT window. The default is 1175.
            
        Returns
        -------
        None.

        Arnau, 23/03/2023
        '''
        def TOF(x, y):
            # m1 = US.CosineInterpMax(x, xcor=False)
            # m2 = US.CosineInterpMax(y, xcor=False)
            # return m1 - m2
            
            # xh = np.absolute(scsig.hilbert(x))
            # yh = np.absolute(scsig.hilbert(y))
            # return US.CalcToFAscanCosine_XCRFFT(xh, yh, UseHilbEnv=False, UseCentroid=False)[0]

            return USF.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=UseHilbEnv, UseCentroid=UseCentroid)[0]

        def ID(x, y):
            # xh = np.absolute(scsig.hilbert(x))
            # yh = np.absolute(scsig.hilbert(y))
            # return US.deconvolution(xh, yh, stripIterNo=2, UseHilbEnv=False, Extend=True, Same=False)[0]
            
            return USF.deconvolution(x, y, stripIterNo=2, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]

        self.winTT(WindowTTshear, Loc_TT)
        self.ToF_TW = np.apply_along_axis(TOF, 0, self.windowedTT, self.WP)
        self.ToF_RW = np.apply_along_axis(ID, 0, self.PE, self.PEref)
        self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]
        
        
        # #%%
        # # cw = np.mean(Cw)
        # # cw = config_dict['Cw']
        # # cw = Cw
        # cw = Cw2
        # cw_aux = np.asarray([cw]).flatten()[::-1]

        # def TOF(x, y, Cwx, Cwy, Fs=100e6):
        #     m1 = US.CosineInterpMax(x, xcor=False)
        #     m2 = US.CosineInterpMax(y, xcor=False)
        #     return (Cwx*m1 - Cwy*m2) / Fs

        # def TOF2(x, y):
        #     return US.CalcToFAscanCosine_XCRFFT(x,y)[0]

        # def ID(x, y):
        #     return US.deconvolution(x, y)[0]

        # # ToF_TW = np.apply_along_axis(TOF, 0, TT, WP, Cw, Cw[0], Fs=Fs)
        # ToF_TW = np.zeros(N_acqs)
        # for i in range(N_acqs):
        #     ToF_TW[i] = TOF(TT[:,i], WP, cw[i], cw[0], Fs=Fs)
        #     # ToF_TW[i] = TOF(TT[:,i], WP, cw, cw, Fs=Fs)
        #     # ToF_TW[i] = TOF2(TT[:,i], WP)

        # # ToF_TW = cw*ToF_TW/Fs
        # # ToF_TW = cw[0]*ToF_TW/Fs

        # ToF_RW = np.apply_along_axis(ID, 0, PE, PEref)
        # ToF_R21 = ToF_RW[1] - ToF_RW[0]
        # ToF_R21 = cw*ToF_R21/Fs


        # L = np.abs(ToF_TW) + ToF_R21/2 # thickness - m   
        # CL = 2*np.abs(ToF_TW)/(ToF_R21/cw) + cw # longitudinal velocity - m/s
        # Cs = cw_aux / np.sqrt(np.sin(theta)**2 + (np.abs(ToF_TW[::-1])/L + np.cos(theta))**2) # shear velocity - m/s

        # CL = CL[:Ridx]
        # L = L[:Ridx]
        # Cs = Cs[:Ridx]
        
    def computeResults(self, Cw_mode='mean'):
        '''
        Compute Thickness (L), Longitudinal velocity (CL), Shear velocity (Cs).

        Parameters
        ----------
        Cw_mode : str, optional
            This parameter defines what speed of sound in water to use. The
            available options are:
                'temperature':
                    The Cw obtained from the temperature measurements.
                'mean':
                    The mean of the Cw obtained from the temperature
                    measurements.
                'lpf':
                    The low-pass filtered Cw obtained from the temperature
                    measurements.
                any other value:
                    The Cw measured directly with the ToF (config_dict['Cw']).
            The default is 'mean'.

        Returns
        -------
        None.

        Arnau, 16/03/2023
        '''
        if type(Cw_mode) is not str:
            cw = self.config_dict['Cw']
        else:
            if Cw_mode.lower() not in ['mean', 'lpf', 'temperature']:
                cw = self.config_dict['Cw']
            elif Cw_mode.lower()=='temperature':
                cw = self.Cw
            elif Cw_mode.lower()=='mean':
                cw = np.mean(self.Cw)
            elif Cw_mode.lower()=='lpf':
                cw = self.Cw_lpf
        cw_aux = np.asarray([cw]).flatten()[::-1]

        self.L = cw/2*(2*np.abs(self.ToF_TW) + self.ToF_R21)/self.Fs # thickness - m
        self.CL = cw*(2*np.abs(self.ToF_TW)/self.ToF_R21 + 1) # longitudinal velocity - m/s
        self.Cs = cw_aux / np.sqrt(np.sin(self.theta)**2 + (cw_aux * np.abs(self.ToF_TW[::-1]) / (self.L * self.Fs) + np.cos(self.theta))**2) # shear velocity - m/s

        self.CL = self.CL[:self.Ridx]
        self.L = self.L[:self.Ridx]
        self.Cs = self.Cs[:self.Ridx]

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
            AR1 = np.max(np.abs(self.PE)*(10**(-G/20)), axis=0)[:self.Ridx]
            Aref = np.mean(self.Arefs[:4]*(10**(-self.ArefsGains[:4]/20)))
        else:
            idx = np.where(self.ArefsGains==G)[0][0]
            Aref = self.Arefs[idx]
            AR1 = np.max(np.abs(self.PE), axis=0)[:self.Ridx]
            
        Zw = 1.48e6 # acoustic impedance of water (N.s/m)
        d = Zw / self.CL * (Aref + AR1) / (Aref - AR1) # density (kg/m^3)
        self.density = d * 1e-3 # density (g/cm^3)

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
            row = 'scanpos,L,CL,Cs,density'
            f.write(row+'\n')
        with open(self.Results_path, _amode) as f:
            for i in range(self.N_acqs):
                row = f'{self.scanpos[i]},{self.L[i]},{self.CL[i]},{self.Cs[i]},{self.density[i]}'
                f.write(row+'\n')

    def winTT(self, WindowTTshear, Loc_TT):
        '''
        Window TT pulses for shear velocity.

        Parameters
        ----------
        WindowTTshear : bool
            If True, window TT. Else, self.windowTT = self.TT.copy().
        Loc_TT : int, optional
            Location of the TT window.
            
        Returns
        -------
        None.

        Arnau, 23/03/23
        '''
        self.windowedTT = self.TT.copy()
        if WindowTTshear:
            ScanLen = self.config_dict['Smax1'] - self.config_dict['Smin1']
            # Loc_TT = 1175 # educated guess: Epoxy Resin
            WinLen_TT = 50
            MyWin_TT = USF.makeWindow(SortofWin='tukey', WinLen=WinLen_TT,
                            param1=0.25, param2=1, Span=ScanLen, Delay=Loc_TT - int(WinLen_TT/2))
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