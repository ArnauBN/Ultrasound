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

class ExperimentSP:
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
        
        self.Config_path = os.path.join(self.Path, self.Experiment_config_file_name)
        self.PEref_path = os.path.join(self.Path, self.Experiment_PEref_file_name)
        self.WP_path = os.path.join(self.Path, self.Experiment_WP_file_name)
        self.Acqdata_path = os.path.join(self.Path, self.Experiment_acqdata_file_name)
        self.Temperature_path = os.path.join(self.Path, self.Experiment_Temperature_file_name)
        self.Scanpath_path = os.path.join(self.Path, self.Experiment_scanpath_file_name)
        self.Img_path = os.path.join(self.Path, self.Experiment_img_file_name)
    
    def _load(self):
        '''
        Load all data.

        Returns
        -------
        None.

        Arnau, 16/03/2023
        '''
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
        and cutoff frequency 100 kHz.

        Returns
        -------
        None.

        Arnau, 16/03/2023
        '''
        self.lpf_order = 2
        self.lpf_fc = 100e3 # Hz
        if self.Fs < 2*self.lpf_fc:
            print(f'Signal does not have frequency components beyond {self.lpf_fc} Hz, therefore it is not filtered.')
        else:
            b_IIR, a_IIR = scsig.iirfilter(self.lpf_order, 2*self.lpf_fc/self.Fs, btype='lowpass')
            self.temperature_lpf = scsig.filtfilt(b_IIR, a_IIR, self.temperature)
            self.Cw_lpf = US.speedofsound_in_water(self.temperature_lpf)
    
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

            return US.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=UseHilbEnv, UseCentroid=UseCentroid)[0]

        def ID(x, y):
            # xh = np.absolute(scsig.hilbert(x))
            # yh = np.absolute(scsig.hilbert(y))
            # return US.deconvolution(xh, yh, stripIterNo=2, UseHilbEnv=False, Extend=True, Same=False)[0]
            
            return US.deconvolution(x, y, stripIterNo=2, UseHilbEnv=UseHilbEnv, Extend=True, Same=False)[0]

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
            MyWin_TT = US.makeWindow(SortofWin='tukey', WinLen=WinLen_TT,
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
        US.dogboneGUI(self.scanpos, t, self.CL, self.Cs, self.L*1e3, self.PE, self.TT, self.img, length=length, ptxt=ptxt)

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

def loadBatches(Path: str, Batches: list[str], Nspecimens: int=10, Verbose: bool=False, **kwargs) -> dict:
    '''
    Load the specified batches. Each batch should have the number of specimens
    specified by Nspecimens.
    
    The names of the specimens folders should be the batch name followed by a
    number (starting at 1). For example: 'A1' for batch 'A' or 'test10' for 
    batch 'test'. If this is not true, use loadExperiments function instead.
    
    This can take a lot of time.

    Parameters
    ----------
    Path : str
        The absolute path of the folder containing all the batches.
    Batches : list[str]
        List of the batches names (folder names).
    Nspecimens : int, optional
        Number of specimens (experiments) in each batch. The default is 10.
    Verbose : bool, optional
        If True, print something every time a specimen is finished. Default is
        False.
    **kwargs : keyword args
        Keyword arguments for ExperimentSP's class constructor.
        
    Returns
    -------
    experiments : dict{ExperimentSP}
        Dictionary containing all the experiments.

    Arnau, 23/03/2023
    '''
    experiments = {}
    for b in Batches:
        BatchPath = os.path.join(Path, b)
        for i in range(1, Nspecimens + 1):
            Experiment_folder_name = f'{b}{i}'
            MyDir = os.path.join(BatchPath, Experiment_folder_name)
            experiments[Experiment_folder_name] = ExperimentSP(MyDir, **kwargs)
            if Verbose: print(f'Specimen {Experiment_folder_name} done.')
    return experiments

def loadExperiments(Path: str, Names: list[str], Verbose: bool=False, **kwargs) -> dict:
    '''
    Load the specified experiments. All experiments should be contained in the
    path specified by {Path}. The folder name of all experiments should be
    specified in the {Names} list.
    
    This can take a lot of time.

    Parameters
    ----------
    Path : str
        The absolute path of the folder containing all the experiments.
    Names : list[str]
        List of the experiments names (folder names).
    Verbose : bool, optional
        If True, print something every time a specimen is finished. Default is
        False.
    **kwargs : keyword args
        Keyword arguments for ExperimentSP's class constructor.
        
    Returns
    -------
    experiments : dict{ExperimentSP}
        Dictionary containing all the experiments.

    Arnau, 05/04/2023
    '''
    experiments = {}
    for e in Names:
        ExperimentName = os.path.join(Path, e)
        experiments[e] = ExperimentSP(ExperimentName, **kwargs)
        if Verbose: print(f'Specimen {e} done.')
    return experiments


#%%
if __name__ == '__main__':
    # -------
    # Imports
    # -------
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import seaborn as sns
    import matplotlib.colors as mcolors
    colors = list(mcolors.TABLEAU_COLORS) # 10 colors
    
    import src.ultrasound as US
    

    # ---------------------
    # Modifiable parameters
    # ---------------------
    Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\EpoxyResin'
    Batches = ['Anosonic']
    Batches = ['Anosonicreversed']
    Nspecimens = 10 # per batch
    Boxplot = False
    Violinplot = False
    RejectOutliers = False
    UseMedianForCs = True
    UseHilbEnv = False
    WindowTTshear = False
    Loc_TT = 1175 # Location of the TT shear window
    m_cl = 2.5 # Outlier detection threshold for CL
    m_cs = 0.6745 # Outlier detection threshold for Cs
    Verbose = True
    
    
    #%%
    # ---------
    # Load data
    # ---------
    experiments = loadBatches(Path, Batches, Nspecimens=Nspecimens, Verbose=Verbose, compute=not (UseHilbEnv or WindowTTshear))
    N = len(Batches)
    N_acqs = len(experiments[list(experiments.keys())[0]].scanpos)
    
    
    #%%
    # Envelope
    if UseHilbEnv or WindowTTshear:
        for v in experiments.values():
            v.computeTOF(UseHilbEnv=UseHilbEnv, WindowTTshear=WindowTTshear, Loc_TT=Loc_TT) # Compute Time-of-Flights
            v.computeResults()
            v.computeDensity()
            print(f'Specimen {v.name} done.')
    velocities_array = np.array([(v.CL, v.Cs) for v in experiments.values()]) # Size of this array is: len(experiments) x 2 x N_acqs


    # -----------------
    # Outlier detection
    # -----------------
    CLall = velocities_array[:,0]
    Csall = velocities_array[:,1]
    dall = np.array([v.density for v in experiments.values()]) # Size of this array is: len(experiments) x N_acqs
    if RejectOutliers:
        CLmasks = np.zeros([1, N_acqs])
        Csmasks = np.zeros_like(CLmasks)
        for cl, cs in zip(velocities_array[:,0], velocities_array[:,1]):
            if UseMedianForCs:
                med = (cs - np.median(cs))/np.std(cs)
                new_data = cs[np.abs(med)<m_cs]
                outliers = cs[np.abs(med)>=m_cs]
                Cs_outliers_indexes = np.where(np.abs(med)>=m_cs)[0]
            else:
                Cs_outliers_indexes = US.reject_outliers(cs, m=m_cs)[2]
            CL_outliers_indexes = US.reject_outliers(cl, m=m_cl)[2]
            
            _temp = np.zeros(len(cl))
            _temp[CL_outliers_indexes] = CL_outliers_indexes
            CLmask = np.array(_temp, dtype=bool)

            _temp = np.zeros(len(cs))
            _temp[Cs_outliers_indexes] = Cs_outliers_indexes
            Csmask = np.array(_temp, dtype=bool)
            
            CLmasks = np.vstack([CLmasks, CLmask])
            Csmasks = np.vstack([Csmasks, Csmask])
            
        del _temp
        
        CLmasks = CLmasks[1:,:]
        Csmasks = Csmasks[1:,:]
        
        CLall = np.ma.masked_array(velocities_array[:,0], mask=CLmasks)
        Csall = np.ma.masked_array(velocities_array[:,1], mask=Csmasks)
        dall = np.ma.masked_array(dall, mask=CLmasks)
    
    # ------------------------------
    # Compute statistics (CL and Cs)
    # ------------------------------
    # Min and Max of ALL specimens (just a float)
    CLmin = CLall.min()
    Csmin = Csall.min()
    dmin  = dall.min()
    CLmax = CLall.max()
    Csmax = Csall.max()
    dmax  = dall.max()
    
    # Mean of every specimen. Size is: (10*N,)
    CLmeans = CLall.mean(axis=1)
    Csmeans = Csall.mean(axis=1)
    dmeans  = dall.mean(axis=1)
    
    # Group by batch
    CLbatches = CLall.reshape(N, Nspecimens, N_acqs)
    Csbatches = Csall.reshape(N, Nspecimens, N_acqs)
    dbatches  = dall.reshape(N, Nspecimens, N_acqs)
    
    # CL and Cs mean and std of every batch
    CLbatches_means = np.array([x.mean() for x in CLbatches])
    Csbatches_means = np.array([x.mean() for x in Csbatches])
    dbatches_means  = np.array([x.mean() for x in dbatches])
    CLbatches_stds  = np.array([x.std() for x in CLbatches])
    Csbatches_stds  = np.array([x.std() for x in Csbatches])
    dbatches_stds   = np.array([x.std() for x in dbatches])

    # Compute Normal distribution of every batch
    Nsigmas = 5
    Npoints = 1000
    CLbatches_aux   = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(CLbatches_means, CLbatches_stds)])
    Csbatches_aux   = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(Csbatches_means, Csbatches_stds)])
    dbatches_aux    = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(dbatches_means, dbatches_stds)])
    CLbatches_gauss = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(CLbatches_aux, CLbatches_means, CLbatches_stds)])
    Csbatches_gauss = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(Csbatches_aux, Csbatches_means, Csbatches_stds)])
    dbatches_gauss  = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(dbatches_aux, dbatches_means, dbatches_stds)])


    #%%
    # --------
    # Plotting
    # --------
    ax1, ax2 = plt.subplots(2)[1]
    ax1.set_ylabel('Longitudinal velocity (m/s)')
    ax1.set_xlabel('Specimen')
    ax2.set_ylabel('Shear velocity (m/s)')
    ax2.set_xlabel('Specimen')
    
    # Plot CL and Cs
    if  N > len(colors):
        colors = mcolors.CSS4_COLORS
        repcolors = [c for c in colors for _ in range(len(experiments))]
    else:
        repcolors = [c for c in colors for _ in range(len(experiments))]
    if Boxplot:
        # Boxplot mean does not take into accoun the mask, so the mean is performed over ALL data points
        sns.boxplot(data=CLall.T, ax=ax1, palette=repcolors, flierprops=dict(marker='.'), showmeans=True, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
        sns.boxplot(data=Csall.T, ax=ax2, palette=repcolors, flierprops=dict(marker='.'), showmeans=True, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
        # sns.stripplot(data=CLall.T, ax=ax1)
        # sns.stripplot(data=Csall.T, ax=ax2)
    elif Violinplot:
        sns.violinplot(data=CLall.T, ax=ax1, palette=repcolors, flierprops=dict(marker='.'), inner=None, showmeans=True)
        sns.violinplot(data=Csall.T, ax=ax2, palette=repcolors, flierprops=dict(marker='.'), inner=None)
    else:
        for i, v in enumerate(experiments.values()):
            ax1.scatter([i]*len(v.CL), v.CL, c=repcolors[i], marker='.')
            ax2.scatter([i]*len(v.Cs), v.Cs, c=repcolors[i], marker='.')
    ax1.plot(CLmeans, c='k')
    ax2.plot(Csmeans, c='k')
    
    # Draw batch name and vertical lines as separators
    ylim1, ylimmax1 = ax1.get_ylim()
    ylim2, ylimmax2 = ax2.get_ylim()
    ax1.set_ylim([ylim1 - (ylimmax1 - ylim1)*0.1, ylimmax1])
    ax2.set_ylim([ylim2 - (ylimmax2 - ylim2)*0.1, ylimmax2])
    ylim1, ylimmax1 = ax1.get_ylim()
    ylim2, ylimmax2 = ax2.get_ylim()    
    for i, batch in enumerate(Batches):
        ax1.text(i*Nspecimens + Nspecimens/2, ylim1 + (ylimmax1 - ylim1)*0.025, batch)
        ax2.text(i*Nspecimens + Nspecimens/2, ylim2 + (ylimmax2 - ylim2)*0.025, batch)
        if i != len(Batches) - 1:
            ax1.axvline(i*Nspecimens + Nspecimens + Nspecimens/2, c='gray')
            ax2.axvline(i*Nspecimens + Nspecimens + Nspecimens/2, c='gray')
    plt.tight_layout()
    
    
    # Plot each batch separately   
    for b, cl, cs in zip(Batches, CLbatches, Csbatches):
        ax1, ax2, ax3 = plt.subplots(3)[1]
        ax1.set_title(f'Batch {b}')
        ax1.set_ylabel('Long. vel. (m/s)')
        ax2.set_ylabel('Shear vel. (m/s)')
        ax3.set_ylabel('Thickness (mm)')
        ax3.set_xlabel('Position (mm)')
        for i in range(Nspecimens):
            v = experiments[f'{b}{i+1}']
            ax1.plot(v.scanpos, cl[i], c=colors[i], lw=2)
            ax2.plot(v.scanpos, cs[i], c=colors[i], lw=2)
            if np.ma.is_masked(cl):
                auxcl, auxcs = cl[i].copy(), cs[i].copy()
                auxcl.mask = ~cl[i].mask
                auxcs.mask = ~cs[i].mask
                maskedL = np.ma.masked_array(v.L, mask=cl[i].mask)
                
                ax1.scatter(v.scanpos, auxcl, c='k', marker='.')
                ax2.scatter(v.scanpos, auxcs, c='k', marker='.')
                ax3.plot(v.scanpos, maskedL*1e3, c=colors[i])
                
                notmaskedL = np.ma.masked_array(v.L, mask=~cl[i].mask)
                ax3.scatter(v.scanpos, notmaskedL*1e3, c='k', marker='.')
            else:
                ax3.plot(v.scanpos, v.L*1e3, c=colors[i])
        plt.tight_layout()            
    
    # Histogram of every batch
    CLhists = [US.hist(x.flatten(), density=True) for x in CLbatches] # h, b, width
    Cshists = [US.hist(x.flatten(), density=True) for x in Csbatches]
    dhists = [US.hist(x.flatten(), density=True) for x in dbatches]
    for i, (CLh, Csh) in enumerate(zip(CLhists, Cshists)):
        ax1, ax2 = plt.subplots(2)[1]
        plt.title(Batches[i])
        US.plot_hist(*CLh, ax=ax1, xlabel='Longitudinal velocity (m/s)', ylabel='pdf', edgecolor='k')
        US.plot_hist(*Csh, ax=ax2, xlabel='Shear velocity (m/s)', ylabel='pdf', edgecolor='k')
        ax1.plot(CLbatches_aux[i], CLbatches_gauss[i], c='r')
        ax2.plot(Csbatches_aux[i], Csbatches_gauss[i], c='r')
        plt.tight_layout()
    
    
    for j, (db, dh) in enumerate(zip(dbatches, dhists)):
        ax1, ax2 = plt.subplots(2)[1]
        plt.title(Batches[j])
        ax1.set_ylabel('Density ($g/cm^3$)')
        ax1.set_xlabel('Position (mm)')
        US.plot_hist(*dh, ax=ax2, xlabel='Density ($g/cm^3$)', ylabel='pdf', edgecolor='k')
        ax2.plot(dbatches_aux[j], dbatches_gauss[j], c='r')
        
        for i in range(Nspecimens):
            v = experiments[f'{b}{i+1}']
            ax1.plot(v.scanpos, db[i], c=colors[i], lw=2)
            if np.ma.is_masked(db[i]):
                auxdb = db[i].copy()
                auxdb.mask = ~db[i].mask
                ax1.scatter(v.scanpos, auxdb, c='k', marker='.')
        plt.tight_layout()
        
        
    #%%
    def fun(x):
        return US.CosineInterpMax(x, xcor=False)
    
    PEmax = np.zeros([len(experiments), experiments['A1'].Ridx])
    for i, v in enumerate(experiments.values()):
        PEmax[i,:] = np.apply_along_axis(fun, 0, v.PE)[:v.Ridx]
    
    sorted_idxs = np.argsort(PEmax, axis=1)
    PEmax_sorted = np.take_along_axis(PEmax, sorted_idxs, axis=1)
        
    sorted_idxsbatches = sorted_idxs.reshape(N, Nspecimens, N_acqs)
    dball_sorted = np.take_along_axis(dall, sorted_idxs, axis=1)
    dbatches_sorted = dball_sorted.reshape(N, Nspecimens, N_acqs)
    
    for j, (db, dbs) in enumerate(zip(dbatches, dbatches_sorted)):
        ax1, ax2 = plt.subplots(2)[1]
        plt.title(Batches[j])
        ax1.set_ylabel('Density ($g/cm^3$)')
        ax1.set_xlabel('Position (mm)')
        ax2.set_ylabel('Density ($g/cm^3$)')
        ax2.set_xlabel('Position (mm)')
        
        for i in range(Nspecimens):
            v = experiments[f'{b}{i+1}']
            ax1.plot(v.scanpos, db[i], c=colors[i], lw=2)
            if np.ma.is_masked(db[i]):
                auxdb = db[i].copy()
                auxdb.mask = ~db[i].mask
                ax1.scatter(v.scanpos, auxdb, c='k', marker='.')
            
            ax2.plot(v.scanpos, dbs[i], c=colors[i], lw=2)
            if np.ma.is_masked(dbs[i]):
                auxdbs = dbs[i].copy()
                auxdbs.mask = ~dbs[i].mask
                ax2.scatter(v.scanpos, auxdbs, c='k', marker='.')            
        plt.tight_layout()