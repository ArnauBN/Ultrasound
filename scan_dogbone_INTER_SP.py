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
        self.computeTOF() # Compute Time-of-Flights
        self.computeResults() # Compute results
        self.computeDensity() # Compute density
    
    def _paths(self):
        self.Experiment_config_file_name = 'config.txt' # Without Backslashes
        self.Experiment_PEref_file_name = 'PEref.bin'
        self.Experiment_WP_file_name = 'WP.bin'
        self.Experiment_acqdata_file_name = 'acqdata.bin'
        self.Experiment_Temperature_file_name = 'temperature.txt'
        self.Experiment_scanpath_file_name = 'scanpath.txt'
        self.Experiment_img_file_name = 'img.jpg'
        
        self.Config_path = os.path.join(self.Path, self.Experiment_config_file_name)
        self.PEref_path = os.path.join(self.Path, self.Experiment_PEref_file_name)
        self.WP_path = os.path.join(self.Path, self.Experiment_WP_file_name)
        self.Acqdata_path = os.path.join(self.Path, self.Experiment_acqdata_file_name)
        self.Temperature_path = os.path.join(self.Path, self.Experiment_Temperature_file_name)
        self.Scanpath_path = os.path.join(self.Path, self.Experiment_scanpath_file_name)
        self.Img_path = os.path.join(self.Path, self.Experiment_img_file_name)
    
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
        
        # Image
        if os.path.exists(self.Img_path):
            self.img = mpimg.imread(self.Img_path)
        else:
            self.img = None

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
    
    def computeTOF(self):
        ScanLen = self.config_dict['Smax1'] - self.config_dict['Smin1']
        Loc_TT = 1140
        WinLen_TT = 80
        MyWin_TT = US.makeWindow(SortofWin='tukey', WinLen=WinLen_TT,
                       param1=0.25, param2=1, Span=ScanLen, Delay=Loc_TT - int(WinLen_TT/2))
        self.windowedTT = self.TT.copy()
        for i in range(self.Ridx, self.N_acqs):
            self.windowedTT[:,i] = self.TT[:,i] * MyWin_TT

        def TOF(x, y):
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

        self.ToF_TW = np.apply_along_axis(TOF, 0, self.TT, self.WP)
        # self.ToF_TW = np.apply_along_axis(TOF, 0, self.windowedTT, self.WP)
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
        
        
        

    def computeResults(self, mode='mean'):
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

    def computeDensity(self, UseAvgAcrossGains=False):
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

    def plotGUI(self, ptxt='northwest'):
        Smin = self.config_dict['Smin1']
        Smax = self.config_dict['Smax1']
        t = np.arange(Smin, Smax) / self.Fs * 1e6 # us
        US.pltGUI(self.scanpos, t, self.CL, self.Cs, self.L*1e3, self.PE, self.TT, self.img, ptxt=ptxt)

    def plotTemperature(self):
        ax1, ax2 = plt.subplots(2)[1]
        ax1.scatter(np.arange(self.N_acqs), self.temperature, marker='.', color='k')
        ax1.plot(self.temperature_lpf, 'r', lw=3)
        ax1.set_ylabel('Temperature (\u2103)')
        
        ax2.scatter(np.arange(self.N_acqs), self.Cw, marker='.', color='k')
        ax2.plot(self.Cw_lpf, 'r', lw=3)
        ax2.set_ylabel('Cw (m/s)')
        
        plt.tight_layout()
        
        return ax1, ax2


if __name__ == '__main__':
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS) # 10 colors
    
    # --------
    # Set path
    # --------
    Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\EpoxyResin'
    Batches = ['A', 'B', 'C']
    
    
    # ---------
    # Load data
    # ---------
    N = len(Batches)
    experiments = {}
    for b in Batches:
        for i in range(1, 11): # 10 specimens for every batch
            Experiment_folder_name = f'{b}{i}' # Without Backslashes
            MyDir = os.path.join(Path, Experiment_folder_name)
            experiments[Experiment_folder_name] = ExperimentSP(MyDir)
    N_acqs = experiments['A1'].N_acqs


    # ------------------------------
    # Compute statistics (CL and Cs)
    # ------------------------------
    velocities_array = np.array([(v.CL, v.Cs) for v in experiments.values()]) # Size of this array is: 10*N x 2 x N_acqs
    
    # Min and Max of ALL specimens (just a float)
    CLmin = np.min(velocities_array[:,0])
    CLmax = np.max(velocities_array[:,0])
    Csmin = np.min(velocities_array[:,1])
    Csmax = np.max(velocities_array[:,1])
    
    # Mean of every specimen. Size is: (10*N,)
    CLmeans = np.mean(velocities_array[:,0], axis=1)
    Csmeans = np.mean(velocities_array[:,1], axis=1)
    
    # Group by batch
    CLbatches = velocities_array[:,0].reshape(N, 10, N_acqs)
    Csbatches = velocities_array[:,1].reshape(N, 10, N_acqs)
    
    # CL and Cs mean and std of every batch
    CLbatches_means = np.zeros(N)
    Csbatches_means = np.zeros(N)
    CLbatches_stds = np.zeros(N)
    Csbatches_stds = np.zeros(N)
    for i, (CLbatch, Csbatch) in enumerate(zip(CLbatches, Csbatches)):
        CLbatches_means[i] = np.mean(CLbatch)
        Csbatches_means[i] = np.mean(Csbatch)
        CLbatches_stds[i] = np.std(CLbatch)
        Csbatches_stds[i] = np.std(CLbatch)


    # --------
    # Plotting
    # --------
    ax1, ax2 = plt.subplots(2)[1]
    ax1.set_ylabel('Longitudinal velocity (m/s)')
    ax1.set_xlabel('Specimen')
    ax2.set_ylabel('Shear velocity (m/s)')
    ax2.set_xlabel('Specimen')
    
    # Draw batch name and vertical lines as separators
    for i,batch in enumerate(Batches):
        ax1.text(i*10 + 5, CLmin - (CLmax*0.1 - CLmin) * 0.1, batch)
        ax2.text(i*10 + 5, Csmin - (Csmax*0.1 - Csmin) * 0.1, batch)
        if i != len(Batches) - 1:
            ax1.axvline(i*10 + 10.5, c='gray')
            ax2.axvline(i*10 + 10.5, c='gray')

    # Plot CL and Cs
    colors = ['k'] * N if  N > len(colors) else colors[:N]
    for i,(k,v) in enumerate(experiments.items()):
        ax1.scatter([i]*len(v.CL), v.CL, c=colors[i], marker='.')
        ax2.scatter([i]*len(v.Cs), v.Cs, c=colors[i], marker='.')
    ax1.plot(CLmeans, c='k')
    ax2.plot(Csmeans, c='k')
    
    #TODO: remove outliers
    #TODO: histogram with normal distribution