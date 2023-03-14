# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:09:23 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import os
import scipy.signal as scsig

import src.ultrasound as US

#%%
class ExperimentSP:
    def __init__(self, Path):
        self.Path = Path
        self.name = Path.split('\\')[-1]
        
        self._paths() # set paths
        self._load() # load data
        self._angleAndScanpos() # get rotation angle and scanner position vector
        self.LPFtemperature() # Low-Pass filter temperature
        self.TOF() # Compute Time-of-Flights
        self.results() # Compute results
    
    def _paths(self):
        self.Experiment_config_file_name = 'config.txt' # Without Backslashes
        self.Experiment_PEref_file_name = 'PEref.bin'
        self.Experiment_WP_file_name = 'WP.bin'
        self.Experiment_acqdata_file_name = 'acqdata.bin'
        self.Experiment_Temperature_file_name = 'temperature.txt'
        self.Experiment_scanpath_file_name = 'scanpath.txt'
        
        self.Config_path = os.path.join(self.Path, self.Experiment_config_file_name)
        self.PEref_path = os.path.join(self.Path, self.Experiment_PEref_file_name)
        self.WP_path = os.path.join(self.Path, self.Experiment_WP_file_name)
        self.Acqdata_path = os.path.join(self.Path, self.Experiment_acqdata_file_name)
        self.Temperature_path = os.path.join(self.Path, self.Experiment_Temperature_file_name)
        self.Scanpath_path = os.path.join(self.Path, self.Experiment_scanpath_file_name)
    
    def _load(self):
        # Config
        self.config_dict = US.load_config(self.Config_path)
        self.Fs = self.config_dict['Fs']
        self.N_acqs = self.config_dict['N_acqs']
        
        # Data
        self.TT, self.PE = US.load_bin_acqs(self.Acqdata_path, self.config_dict['N_acqs'])
        
        # Temperature and CW
        self.temperature_dict = US.load_columnvectors_fromtxt(self.Temperature_path)
        self.temperature = self.temperature_dict['Inside']
        self.Cw = self.temperature_dict['Cw']
        
        # Scan pattern
        self.scanpattern = US.load_columnvectors_fromtxt(self.Scanpath_path, delimiter=',', header=False, dtype=str)
        
        # WP
        with open(self.WP_path, 'rb') as f:
            self.WP = np.fromfile(f)
        
        # PE ref
        with open(self.PEref_path, 'rb') as f:
            self.PEref = np.fromfile(f)

    def _angleAndScanpos(self):
        self.Ridx = [np.where(self.scanpattern == s)[0][0] for s in self.scanpattern if 'R' in s][0] + 1
        self.theta = float(self.scanpattern[self.Ridx-1][1:]) * np.pi / 180
        step = float(self.scanpattern[0][1:])
        self.scanpos = np.arange(self.Ridx)*step # mm

    def LPFtemperature(self):
        self.lpf_order = 2
        self.lpf_fc = 100e3 # Hz
        if self.Fs < 2*self.lpf_fc:
            print(f'Signal does not have frequency components beyond {self.lpf_fc} Hz, therefore it is not filtered.')
        else:
            b_IIR, a_IIR = scsig.iirfilter(self.lpf_order, 2*self.lpf_fc/self.Fs, btype='lowpass')
            self.temperature_lpf = scsig.filtfilt(b_IIR, a_IIR, self.temperature)
            self.Cw_lpf = US.speedofsound_in_water(self.temperature_lpf)
    
    def TOF(self):
        ScanLen = self.config_dict['Smax1'] - self.config_dict['Smin1']
        Loc_TT = 1140
        WinLen_TT = 80
        MyWin_TT = US.makeWindow(SortofWin='tukey', WinLen=WinLen_TT,
                       param1=0.25, param2=1, Span=ScanLen, Delay=Loc_TT - int(WinLen_TT/2))
        self.windowedTT = self.TT.copy()
        for i in range(self.Ridx, self.N_acqs):
            self.windowedTT[:,i] = self.TT[:,i] * MyWin_TT

        def TOF2(x, y):
            # m1 = US.CosineInterpMax(x, xcor=False)
            # m2 = US.CosineInterpMax(y, xcor=False)
            # return m1 - m2
            
            # xh = np.absolute(scsig.hilbert(x))
            # yh = np.absolute(scsig.hilbert(y))
            # return US.CalcToFAscanCosine_XCRFFT(xh, yh, UseHilbEnv=False, UseCentroid=False)[0]

            return US.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=True, UseCentroid=False)[0]

        def ID(x, y):
            # xh = np.absolute(scsig.hilbert(x))
            # yh = np.absolute(scsig.hilbert(y))
            # return US.deconvolution(xh, yh, stripIterNo=2, UseHilbEnv=False, Extend=True, Same=False)[0]
            
            return US.deconvolution(x, y, stripIterNo=2, UseHilbEnv=False, Extend=True, Same=False)[0]

        # ToF_TW = np.apply_along_axis(TOF, 0, TT, WP)
        self.ToF_TW = np.apply_along_axis(TOF2, 0, self.windowedTT, self.WP)
        self.ToF_RW = np.apply_along_axis(ID, 0, self.PE, self.PEref)
        self.ToF_R21 = self.ToF_RW[1] - self.ToF_RW[0]

    def results(self, mode='mean'):
        if type(mode) is not str:
            cw = self.config_dict['Cw']
        else:
            if mode.lower() not in ['mean', 'lpf', 'temperature']:
                cw = self.config_dict['Cw']
            elif mode.lower()=='temperature':
                cw = self.Cw
            elif mode.lower()=='mean':
                cw = np.mean(self.Cw)
            elif mode.lower()=='lpf':
                cw = self.Cw_lpf
        cw_aux = np.asarray([cw]).flatten()[::-1]

        self.L = cw/2*(2*np.abs(self.ToF_TW) + self.ToF_R21)/self.Fs # thickness - m
        self.CL = cw*(2*np.abs(self.ToF_TW)/self.ToF_R21 + 1) # longitudinal velocity - m/s
        self.Cs = cw_aux / np.sqrt(np.sin(self.theta)**2 + (cw_aux * np.abs(self.ToF_TW[::-1]) / (self.L * self.Fs) + np.cos(self.theta))**2) # shear velocity - m/s

        self.CL = self.CL[:self.Ridx]
        self.L = self.L[:self.Ridx]
        self.Cs = self.Cs[:self.Ridx]



#%%
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\EpoxyResin'
Batches = ['A', 'B', 'C']
experiments = {}
for b in Batches:
    for i in range(1, 11): # 10 specimens for every batch
        Experiment_folder_name = f'{b}{i}' # Without Backslashes
        MyDir = os.path.join(Path, Experiment_folder_name)
        experiments[Experiment_folder_name] = ExperimentSP(MyDir)



