# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:08:05 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import os
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.colors as mcolors

import src.ultrasound as US

def get_experiment_paths(Path: str=None, Batches: list[str]=None, sort: bool=True) -> list[str]:
    '''
    Returns a list of experiment paths inside the given path.

    Parameters
    ----------
    Path : str, optional
        Directory path that contains all batches. The default is None (current
        working directory).
    Batches : list[str], optional
        List of the batches names (folder names). The default is None (search
        all directories in Path).
    sort : bool, optional
        If True, sort specimen names by their number. Default is True.

    Returns
    -------
    Paths : list[str]
        List of all experiments paths.

    Arnau, 14/08/2023
    '''
    p = os.getcwd() if Path is None else Path
    btchs = US.get_dir_names(p) if Batches is None else Batches
    Paths = []
    for b in btchs:
        batch_path = os.path.join(p, b)
        batch_specimens = US.popBadSpecimens(b, US.get_dir_names(Path=batch_path))
        if sort: batch_specimens = US.sort_strings_by_number(batch_specimens)
        for name in batch_specimens:
            exp_path = os.path.join(batch_path, name)
            Paths.append(exp_path)
    return Paths       


def loadDogBoneExperiments(Paths: list[str], Verbose: bool=False, **kwargs) -> dict[US.DogboneSP]:
    '''
    Load the experiments specified by their paths.

    Parameters
    ----------
    Paths : list[str]
        List of all experiments paths.
    Verbose : bool, optional
        If True, print something every time a specimen is finished. Default is
        False.
    **kwargs : keyword args
        Keyword arguments for DogboneSP's class constructor.

    Returns
    -------
    experiments : dict[US.DogboneSP]
        Dictionary containing all the experiments.

    Arnau, 14/08/2023
    '''
    pths = [Paths] if isinstance(Paths, str) else Paths
    experiments = {}
    for path in pths:
        name = os.path.basename(path)
        experiments[name] = US.DogboneSP(path, **kwargs)
        if Verbose: print(f'Specimen {name} done.')
    return experiments



class Batch:
    colors = list(mcolors.TABLEAU_COLORS) + ['k'] # 11 colors
    def __init__(self, CL, Cs, d, shear_modulus, young_modulus, bulk_modulus,
                 poisson_ratio, L, scanpos, archdensity, name=None,
                 UseMedian_CL=False, m_CL=0.6745,
                 UseMedian_Cs=False, m_Cs=0.6745,
                 UseMedian_d=False, m_d=0.6745,
                 UseMedian_shear_modulus=False, m_shear_modulus=0.6745,
                 UseMedian_young_modulus=False, m_young_modulus=0.6745,
                 UseMedian_bulk_modulus=False, m_bulk_modulus=0.6745,
                 UseMedian_poisson_ratio=False, m_poisson_ratio=0.6745,
                 UseMedian_L=False, m_L=0.6745):
        self.archdensity = archdensity
        self.name = name
        self.Nspecimens = CL.shape[0]
        self.scanpos = scanpos
        self.CL = BatchData(CL, UseMedian_CL, m_CL)
        self.Cs = BatchData(Cs, UseMedian_Cs, m_Cs)
        self.d  = BatchData(d,  UseMedian_d,  m_d)
        self.L  = BatchData(L,  UseMedian_L,  m_L)
        self.shear_modulus = BatchData(shear_modulus, UseMedian_shear_modulus, m_shear_modulus)
        self.young_modulus = BatchData(young_modulus, UseMedian_young_modulus, m_young_modulus)
        self.bulk_modulus  = BatchData(bulk_modulus,  UseMedian_bulk_modulus,  m_bulk_modulus)
        self.poisson_ratio = BatchData(poisson_ratio, UseMedian_poisson_ratio, m_poisson_ratio)
        
    
    def plot(self, RejectOutliers_CL=False, RejectOutliers_Cs=False, 
             RejectOutliers_d=False, RejectOutliers_shear_modulus=False, 
             RejectOutliers_young_modulus=False, RejectOutliers_bulk_modulus=False, 
             RejectOutliers_poisson_ratio=False, RejectOutliers_L=False, PlotOutliers=False):
        ax1, ax2, ax3 = plt.subplots(3)[1]
        ax1.set_title(f'Batch {self.name}')
        ax1.set_ylabel('Long. vel. (m/s)')
        ax2.set_ylabel('Shear vel. (m/s)')
        ax3.set_ylabel('Thickness (mm)')
        ax3.set_xlabel('Position (mm)')
        
        CLdata = self.CL.masked.data if RejectOutliers_CL else self.CL.data
        Csdata = self.Cs.masked.data if RejectOutliers_Cs else self.Cs.data
        # ddata = self.d.masked.data if RejectOutliers_d else self.d.data
        Ldata = self.L.masked.data if RejectOutliers_L else self.L.data
        shear_modulusdata = self.shear_modulus.masked.data if RejectOutliers_shear_modulus else self.shear_modulus.data
        young_modulusdata = self.young_modulus.masked.data if RejectOutliers_young_modulus else self.young_modulus.data
        bulk_modulusdata = self.bulk_modulus.masked.data if RejectOutliers_bulk_modulus else self.bulk_modulus.data
        poisson_ratiodata = self.poisson_ratio.masked.data if RejectOutliers_poisson_ratio else self.poisson_ratio.data
        
        for i in range(self.Nspecimens):
            ax1.plot(self.scanpos, CLdata[i], c=Batch.colors[i], lw=2)
            ax2.plot(self.scanpos, Csdata[i], c=Batch.colors[i], lw=2)
            ax3.plot(self.scanpos, Ldata[i]*1e3, c=Batch.colors[i], lw=2)
            
            if PlotOutliers:
                if RejectOutliers_CL:
                    auxcl = CLdata[i].copy()
                    auxcl.mask = ~CLdata[i].mask
                    ax1.scatter(self.scanpos, auxcl, c='k', marker='.')
                if RejectOutliers_Cs:
                    auxcs = Csdata[i].copy()
                    auxcs.mask = ~Csdata[i].mask
                    ax2.scatter(self.scanpos, auxcs, c='k', marker='.')
                if RejectOutliers_L:
                    auxl = Ldata[i].copy()
                    auxl.mask = ~Ldata[i].mask
                    ax3.scatter(self.scanpos, auxl*1e3, c='k', marker='.')
        plt.tight_layout()        
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
        fig.suptitle(f'Batch {self.name}')
        ax1.set_ylabel('Shear Modulus (GPa)')
        ax2.set_ylabel('Young Modulus (GPa)')
        ax3.set_ylabel('Bulk Modulus (GPa)')
        ax4.set_ylabel('Poisson Ratio')
        ax4.set_xlabel('Position (mm)')
        for i in range(self.Nspecimens):
            ax1.plot(self.scanpos, shear_modulusdata[i]*1e-6, c=Batch.colors[i], lw=2)
            ax2.plot(self.scanpos, young_modulusdata[i]*1e-6, c=Batch.colors[i], lw=2)
            ax3.plot(self.scanpos, bulk_modulusdata[i]*1e-6, c=Batch.colors[i], lw=2)
            ax4.plot(self.scanpos, poisson_ratiodata[i], c=Batch.colors[i], lw=2)
            
            if PlotOutliers:
                if RejectOutliers_shear_modulus:
                    auxs = shear_modulusdata[i].copy()
                    auxs.mask = ~shear_modulusdata[i].mask
                    ax1.scatter(self.scanpos, auxs*1e-6, c='k', marker='.')
                if RejectOutliers_young_modulus:
                    auxy = young_modulusdata[i].copy()
                    auxy.mask = ~young_modulusdata[i].mask
                    ax2.scatter(self.scanpos, auxy*1e-6, c='k', marker='.')
                if RejectOutliers_bulk_modulus:
                    auxb = bulk_modulusdata[i].copy()
                    auxb.mask = ~bulk_modulusdata[i].mask
                    ax3.scatter(self.scanpos, auxb*1e-6, c='k', marker='.')
                if RejectOutliers_poisson_ratio:
                    auxp = poisson_ratiodata[i].copy()
                    auxp.mask = ~poisson_ratiodata[i].mask
                    ax4.scatter(self.scanpos, auxp, c='k', marker='.')
        plt.tight_layout()   
        
        
    def plothist(self, RejectOutliers_CL=False, RejectOutliers_Cs=False, 
             RejectOutliers_d=False, RejectOutliers_shear_modulus=False, 
             RejectOutliers_young_modulus=False, RejectOutliers_bulk_modulus=False, 
             RejectOutliers_poisson_ratio=False, RejectOutliers_L=False, PlotOutliers=False):

        CL = self.CL.masked if RejectOutliers_CL else self.CL
        Cs = self.Cs.masked if RejectOutliers_Cs else self.Cs
        d = self.d.masked if RejectOutliers_d else self.d
        # L = self.L.masked if RejectOutliers_L else self.L
        shear_modulus = self.shear_modulus.masked if RejectOutliers_shear_modulus else self.shear_modulus
        young_modulus = self.young_modulus.masked if RejectOutliers_young_modulus else self.young_modulus
        bulk_modulus = self.bulk_modulus.masked if RejectOutliers_bulk_modulus else self.bulk_modulus
        poisson_ratio = self.poisson_ratio.masked if RejectOutliers_poisson_ratio else self.poisson_ratio

        ax1, ax2 = plt.subplots(2)[1]
        US.plot_hist(*CL.hist, ax=ax1, xlabel='Longitudinal velocity (m/s)', ylabel='pdf', edgecolor='k')
        US.plot_hist(*Cs.hist, ax=ax2, xlabel='Shear velocity (m/s)', ylabel='pdf', edgecolor='k')
        ax1.plot(CL.aux, CL.gauss, c='r')
        ax2.plot(Cs.aux, Cs.gauss, c='r')
        ax1.set_title(f'Batch {self.name}')
        plt.tight_layout()
        
        ax1, ax2 = plt.subplots(2)[1]
        ax1.set_title(f'Batch {self.name}')
        ax1.set_ylabel('Density ($g/cm^3$)')
        ax1.set_xlabel('Position (mm)')
        US.plot_hist(*d.hist, ax=ax2, xlabel='Density ($g/cm^3$)', ylabel='pdf', edgecolor='k')
        ax2.plot(d.aux, d.gauss, c='r')
        for i in range(self.Nspecimens):
            ax1.plot(self.scanpos, d.data[i], c=Batch.colors[i], lw=2)
            if PlotOutliers and RejectOutliers_d:
                auxdb = d.data[i].copy()
                auxdb.mask = ~d.data[i].mask
                ax1.scatter(self.scanpos, auxdb, c='k', marker='.')
        plt.tight_layout()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
        US.plot_hist(shear_modulus.hist[0]*1e6, 
                     shear_modulus.hist[1]*1e-6, 
                     shear_modulus.hist[2]*1e-6, 
                     ax=ax1, xlabel='Shear modulus (GPa)', ylabel='pdf', edgecolor='k')
        US.plot_hist(young_modulus.hist[0]*1e6, 
                     young_modulus.hist[1]*1e-6, 
                     young_modulus.hist[2]*1e-6, 
                     ax=ax2, xlabel='Young modulus (GPa)', ylabel='pdf', edgecolor='k')
        US.plot_hist(bulk_modulus.hist[0]*1e6, 
                     bulk_modulus.hist[1]*1e-6, 
                     bulk_modulus.hist[2]*1e-6, 
                     ax=ax3, xlabel='Bulk modulus (GPa)', ylabel='pdf', edgecolor='k')
        US.plot_hist(*poisson_ratio.hist, ax=ax4, xlabel='Poisson ratio', ylabel='pdf', edgecolor='k')
        ax1.plot(shear_modulus.aux*1e-6, shear_modulus.gauss*1e6, c='r')
        ax2.plot(young_modulus.aux*1e-6, young_modulus.gauss*1e6, c='r')
        ax3.plot(bulk_modulus.aux*1e-6, bulk_modulus.gauss*1e6, c='r')
        ax4.plot(poisson_ratio.aux, poisson_ratio.gauss, c='r')
        fig.suptitle(f'Batch {self.name}')
        plt.tight_layout()



class BatchData:
    Nsigmas = 5
    Npoints = 1000
    def __init__(self, x, UseMedian=False, m=0.6745):
        self.data = x
        self.masked = MaskedBatchData(x, UseMedian=UseMedian, m=m)
        
        self.min = self.data.min()
        self.max = self.data.max()
        self.mean = self.data.mean()
        self.std = self.data.std()
        
        self.mins = self.data.min(axis=1)
        self.maxs = self.data.max(axis=1)
        self.means = self.data.mean(axis=1)
        self.stds = self.data.std(axis=1)
        
        self.aux = np.linspace(self.mean - BatchData.Nsigmas*self.std, self.mean + BatchData.Nsigmas*self.std, BatchData.Npoints)
        self.gauss = np.exp(-((self.aux - self.mean) / self.std)**2 / 2) / (self.std*np.sqrt(2*np.pi))
        self.hist = US.hist(self.data.flatten(), density=True) # h, b, width


class MaskedBatchData():
    def __init__(self, x, UseMedian=False, m=0.6745):
        self.data = US.maskOutliers(x, m=m, UseMedian=UseMedian)

        self.min = self.data.min()
        self.max = self.data.max()
        self.mean = self.data.mean()
        self.std = self.data.std()
        
        self.mins = self.data.min(axis=1)
        self.maxs = self.data.max(axis=1)
        self.means = self.data.mean(axis=1)
        self.stds = self.data.std(axis=1)
        
        self.aux = np.linspace(self.mean - BatchData.Nsigmas*self.std, self.mean + BatchData.Nsigmas*self.std, BatchData.Npoints)
        self.gauss = np.exp(-((self.aux - self.mean) / self.std)**2 / 2) / (self.std*np.sqrt(2*np.pi))
        self.hist = US.hist(self.data.flatten(), density=True) # h, b, width




#%%
# ---------------------
# Modifiable parameters
# ---------------------
Path = r'..\Data\Scanner\EpoxyResin\DogboneScan'
# Batches = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'Ans', 'Bns', 'Cns', 'Dns', 'Ens', 'Fns', 'Gns','Hns', 'Ins', 'Jns', 'Kns']
Batches = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
# Batches = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'Ans', 'Bns', 'Cns', 'Dns', 'Ens', 'Fns', 'Gns','Hns', 'Ins', 'Jns', 'Kns']
# Batches = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
Batches = ['Ans', 'Bns', 'Cns', 'Dns', 'Ens', 'Fns', 'Gns','Hns', 'Ins', 'Jns', 'Kns']
# Batches = ['Ans', 'Dns', 'Ens', 'Hns', 'Ins']
# Batches = ['Ans']
# Batches = ['Ens']
Verbose = True

RejectOutliers = {'RejectOutliers_CL': True,
                  'RejectOutliers_Cs': True,
                  'RejectOutliers_d': True,
                  'RejectOutliers_L': False,
                  'RejectOutliers_shear_modulus': True,
                  'RejectOutliers_young_modulus': True,
                  'RejectOutliers_bulk_modulus': True,
                  'RejectOutliers_poisson_ratio': True,}
# RejectOutliers = {'RejectOutliers_CL': False,
#                   'RejectOutliers_Cs': False,
#                   'RejectOutliers_d': False,
#                   'RejectOutliers_L': False,
#                   'RejectOutliers_shear_modulus': False,
#                   'RejectOutliers_young_modulus': False,
#                   'RejectOutliers_bulk_modulus': False,
#                   'RejectOutliers_poisson_ratio': False,}
UseMedian = {'UseMedian_CL': False,
             'UseMedian_Cs': True,
             'UseMedian_d': False,
             'UseMedian_L': False,
             'UseMedian_shear_modulus': True,
             'UseMedian_young_modulus': True,
             'UseMedian_bulk_modulus': True,
             'UseMedian_poisson_ratio': True,}
m = {'m_CL': 2,
     'm_Cs': 0.6745,
     'm_d': 0.6745,
     'm_L': 0.6745,
     'm_shear_modulus': 0.6745,
     'm_young_modulus': 0.6745,
     'm_bulk_modulus': 0.6745,
     'm_poisson_ratio': 0.6745,}


#%%
# ---------------------
# Compute and save data
# ---------------------
UseHilbEnv = False
WindowTTshear = True
# Loc_TT = 2470 # Location of the TT shear window
# Loc_TT = 2410
Loc_TT = None

experiments = loadDogBoneExperiments(get_experiment_paths(Path, Batches), Verbose=Verbose, 
                                     compute=True, UseHilbEnv=UseHilbEnv, WindowTTshear=WindowTTshear, Loc_TT=Loc_TT)
# for e in experiments.values():
#     e.computeResults()
#%%
for e in experiments.values():
    e.saveResults()

#%%
# ---------
# Load data
# ---------
experiments = loadDogBoneExperiments(get_experiment_paths(Path, Batches), Verbose=Verbose, compute=False)
# for e in experiments.values():
#     e.computeModuli(UseArchDensity=True)

Nbatches = len(Batches)
N = len(experiments)
scanpos = experiments[list(experiments.keys())[0]].scanpos
idxsmap = ['CL', 'Cs', 'd', 'L', 'shear_modulus', 'young_modulus', 'bulk_modulus', 'poisson_ratio', 'archdensity']

Batch.colors = list(mcolors.TABLEAU_COLORS) + ['k']
if len(Batches) == 10:
    concentrations = [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Batch.colors.pop(3)
else:
    concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = Batch.colors

Batches_objs = {}
for b in Batches:
    CL = np.array([v.CL for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    Cs = np.array([v.Cs for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    d  = np.array([v.density  for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    L = np.array([v.L for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    shear_modulus = np.array([v.shear_modulus for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    young_modulus = np.array([v.young_modulus for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    bulk_modulus  = np.array([v.bulk_modulus  for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    poisson_ratio = np.array([v.poisson_ratio for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    archdensity = np.array([v.archdensity for k,v in experiments.items() if US.isGoodSpecimen(b, k)])
    
    Batches_objs[b] = Batch(CL, Cs, d, shear_modulus, young_modulus, 
                            bulk_modulus, poisson_ratio, L, scanpos, archdensity, b,
                            **UseMedian, **m)


#%% Initialize variables
CLidx = idxsmap.index('CL')
Csidx = idxsmap.index('Cs')
didx = idxsmap.index('d')
Lidx = idxsmap.index('L')
shear_modulusidx = idxsmap.index('shear_modulus')
young_modulusidx = idxsmap.index('young_modulus')
bulk_modulusidx = idxsmap.index('bulk_modulus')
poisson_ratioidx = idxsmap.index('poisson_ratio')
archdensityidx = idxsmap.index('archdensity')

data_all = [[None]*Nbatches for _ in range(len(idxsmap))]
data_means = [[None]*Nbatches for _ in range(len(idxsmap))]
data_mean = [[None]*Nbatches for _ in range(len(idxsmap))]
data_means_flat = [None]*len(idxsmap)
data_all_flat = [None]*len(idxsmap)
Nspecimens = [None]*Nbatches
for i,v in enumerate(Batches_objs.values()):    
    data_all[CLidx][i] = v.CL.masked.data.copy() if RejectOutliers['RejectOutliers_CL'] else v.CL.data.copy()
    data_all[Csidx][i] = v.Cs.masked.data.copy() if RejectOutliers['RejectOutliers_Cs'] else v.Cs.data.copy()
    data_all[didx][i]  = v.d.masked.data.copy()  if RejectOutliers['RejectOutliers_d']  else v.d.data.copy()
    data_all[Lidx][i]  = v.L.masked.data.copy()  if RejectOutliers['RejectOutliers_L']  else v.L.data.copy()
    data_all[shear_modulusidx][i] = v.shear_modulus.masked.data.copy() if RejectOutliers['RejectOutliers_shear_modulus'] else v.shear_modulus.data.copy()
    data_all[young_modulusidx][i] = v.young_modulus.masked.data.copy() if RejectOutliers['RejectOutliers_young_modulus'] else v.young_modulus.data.copy()
    data_all[bulk_modulusidx][i]  = v.bulk_modulus.masked.data.copy()  if RejectOutliers['RejectOutliers_bulk_modulus']  else v.bulk_modulus.data.copy()
    data_all[poisson_ratioidx][i] = v.poisson_ratio.masked.data.copy() if RejectOutliers['RejectOutliers_poisson_ratio'] else v.poisson_ratio.data.copy()
    data_all[archdensityidx][i] = v.archdensity
    
    
    data_means[CLidx][i] = v.CL.masked.means.copy() if RejectOutliers['RejectOutliers_CL'] else v.CL.means.copy()
    data_means[Csidx][i] = v.Cs.masked.means.copy() if RejectOutliers['RejectOutliers_Cs'] else v.Cs.means.copy()
    data_means[didx][i]  = v.d.masked.means.copy()  if RejectOutliers['RejectOutliers_d']  else v.d.means.copy()
    data_means[Lidx][i]  = v.L.masked.means.copy()  if RejectOutliers['RejectOutliers_L']  else v.L.means.copy()
    data_means[shear_modulusidx][i] = v.shear_modulus.masked.means.copy() if RejectOutliers['RejectOutliers_shear_modulus'] else v.shear_modulus.means.copy()
    data_means[young_modulusidx][i] = v.young_modulus.masked.means.copy() if RejectOutliers['RejectOutliers_young_modulus'] else v.young_modulus.means.copy()
    data_means[bulk_modulusidx][i]  = v.bulk_modulus.masked.means.copy()  if RejectOutliers['RejectOutliers_bulk_modulus']  else v.bulk_modulus.means.copy()
    data_means[poisson_ratioidx][i] = v.poisson_ratio.masked.means.copy() if RejectOutliers['RejectOutliers_poisson_ratio'] else v.poisson_ratio.means.copy()
    
    data_mean[CLidx][i] = v.CL.masked.mean.copy() if RejectOutliers['RejectOutliers_CL'] else v.CL.mean.copy()
    data_mean[Csidx][i] = v.Cs.masked.mean.copy() if RejectOutliers['RejectOutliers_Cs'] else v.Cs.mean.copy()
    data_mean[didx][i]  = v.d.masked.mean.copy()  if RejectOutliers['RejectOutliers_d']  else v.d.mean.copy()
    data_mean[Lidx][i]  = v.L.masked.mean.copy()  if RejectOutliers['RejectOutliers_L']  else v.L.mean.copy()
    data_mean[shear_modulusidx][i] = v.shear_modulus.masked.mean.copy() if RejectOutliers['RejectOutliers_shear_modulus'] else v.shear_modulus.mean.copy()
    data_mean[young_modulusidx][i] = v.young_modulus.masked.mean.copy() if RejectOutliers['RejectOutliers_young_modulus'] else v.young_modulus.mean.copy()
    data_mean[bulk_modulusidx][i]  = v.bulk_modulus.masked.mean.copy()  if RejectOutliers['RejectOutliers_bulk_modulus']  else v.bulk_modulus.mean.copy()
    data_mean[poisson_ratioidx][i] = v.poisson_ratio.masked.mean.copy() if RejectOutliers['RejectOutliers_poisson_ratio'] else v.poisson_ratio.mean.copy()
    
    Nspecimens[i] = v.Nspecimens
    
    

data_all_flat[CLidx] = [item for sublist in data_all[CLidx] for item in sublist]
data_all_flat[Csidx] = [item for sublist in data_all[Csidx] for item in sublist]
data_all_flat[didx]  = [item for sublist in data_all[didx]  for item in sublist]
data_all_flat[Lidx]  = [item for sublist in data_all[Lidx]  for item in sublist]
data_all_flat[shear_modulusidx] = [item for sublist in data_all[shear_modulusidx] for item in sublist]
data_all_flat[young_modulusidx] = [item for sublist in data_all[young_modulusidx] for item in sublist]
data_all_flat[bulk_modulusidx]  = [item for sublist in data_all[bulk_modulusidx]  for item in sublist]
data_all_flat[poisson_ratioidx] = [item for sublist in data_all[poisson_ratioidx] for item in sublist]
data_all_flat[archdensityidx]   = [item for sublist in data_all[archdensityidx]   for item in sublist]

data_means_flat[CLidx] = [item for sublist in data_means[CLidx] for item in sublist]
data_means_flat[Csidx] = [item for sublist in data_means[Csidx] for item in sublist]
data_means_flat[didx]  = [item for sublist in data_means[didx]  for item in sublist]
data_means_flat[Lidx]  = [item for sublist in data_means[Lidx]  for item in sublist]
data_means_flat[shear_modulusidx] = [item for sublist in data_means[shear_modulusidx] for item in sublist]
data_means_flat[young_modulusidx] = [item for sublist in data_means[young_modulusidx] for item in sublist]
data_means_flat[bulk_modulusidx]  = [item for sublist in data_means[bulk_modulusidx]  for item in sublist]
data_means_flat[poisson_ratioidx] = [item for sublist in data_means[poisson_ratioidx] for item in sublist]

if Nbatches > len(Batch.colors):
    colors = list(mcolors.CSS4_COLORS)
    repcolors = []
    for i,n in enumerate(Nspecimens):
        repcolors.extend([colors[i]]*n)
else:
    colors = Batch.colors
    repcolors = []
    for i,n in enumerate(Nspecimens):
        repcolors.extend([colors[i]]*n)


def getUnmaskedValues(x):
    try:
        return x.compressed()
    except AttributeError:
        return x
data_all_without_outliers = US.apply2listElements(data_all, getUnmaskedValues)
data_all_flat_without_outliers = US.apply2listElements(data_all_flat, getUnmaskedValues)

#%% Compare Batches
# ax1, ax2 = plt.subplots(2, figsize=(10,8))[1]
fig1, ax1 = plt.subplots(1, figsize=(10,4))
fig2, ax2 = plt.subplots(1, figsize=(10,4))
ax1.set_ylabel('Longitudinal velocity (m/s)')
ax2.set_ylabel('Shear velocity (m/s)')
# ax1.set_xlabel('Batch (graphite concentration)')
# ax2.set_xlabel('Batch (graphite concentration)')
ax1.set_xlabel('Graphite concentration (wt%)')
ax2.set_xlabel('Graphite concentration (wt%)')

Niter = iter(range(N))
for cl, cs in zip(data_all[CLidx], data_all[Csidx]):
    for l,s in zip(cl, cs):
        i = next(Niter)
        ax1.scatter([i]*len(l), l, c=repcolors[i], marker='.')
        ax2.scatter([i]*len(s), s, c=repcolors[i], marker='.')

ax1.set_xticks([])
ax2.set_xticks([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax1.plot(data_means_flat[CLidx], c='k', zorder=3)
ax2.plot(data_means_flat[Csidx], c='k', zorder=3)

# Draw batch name and vertical lines as separators
ylim1, ylimmax1 = ax1.get_ylim()
ylim2, ylimmax2 = ax2.get_ylim()
ax1.set_ylim([ylim1 - (ylimmax1 - ylim1)*0.1, ylimmax1])
ax2.set_ylim([ylim2 - (ylimmax2 - ylim2)*0.1, ylimmax2])
prevNspecimens = 0
hdataprev = ax1.get_xlim()[0]
for i,(k,v) in enumerate(Batches_objs.items()):
    if i != len(Batches) - 1:
        h = ax1.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        ax2.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        hdata = h.get_xdata()[0]
    else:
        hdata = ax1.get_xlim()[1]
    # ax1.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax1.get_xaxis_transform())
    # ax2.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax2.get_xaxis_transform())
    ax1.text((hdataprev + hdata)/2, 0.025, f'{concentrations[i]}', horizontalalignment='center', transform=ax1.get_xaxis_transform())
    ax2.text((hdataprev + hdata)/2, 0.025, f'{concentrations[i]}', horizontalalignment='center', transform=ax2.get_xaxis_transform())
    hdataprev = hdata
    prevNspecimens += Nspecimens[i]
fig1.tight_layout()
fig2.tight_layout()

#%% Moduli
fig1, ax1 = plt.subplots(1, figsize=(10,4))
fig2, ax2 = plt.subplots(1, figsize=(10,4))
fig3, ax3 = plt.subplots(1, figsize=(10,4))
fig4, ax4 = plt.subplots(1, figsize=(10,4))

ax1.set_ylabel('Shear modulus (GPa)')
ax2.set_ylabel('Young modulus (GPa)')
ax3.set_ylabel('Bulk modulus (GPa)')
ax4.set_ylabel('Poisson ratio')
ax4.set_xlabel('Batch (graphite concentration)')

Niter = iter(range(N))
for sall, yall, ball, pall in zip(data_all[shear_modulusidx], data_all[young_modulusidx], data_all[bulk_modulusidx], data_all[poisson_ratioidx]):
    for s, y, b, p in zip(sall, yall, ball, pall):
        i = next(Niter)
        ax1.scatter([i]*len(s), s*1e-6, c=repcolors[i], marker='.')
        ax2.scatter([i]*len(y), y*1e-6, c=repcolors[i], marker='.')
        ax3.scatter([i]*len(b), b*1e-6, c=repcolors[i], marker='.')
        ax4.scatter([i]*len(p), p, c=repcolors[i], marker='.')

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])
ax1.plot([i * 1e-6 for i in data_means_flat[shear_modulusidx]], c='k', zorder=3)
ax2.plot([i * 1e-6 for i in data_means_flat[young_modulusidx]], c='k', zorder=3)
ax3.plot([i * 1e-6 for i in data_means_flat[bulk_modulusidx]], c='k', zorder=3)
ax4.plot(data_means_flat[poisson_ratioidx], c='k')

# Draw batch name and vertical lines as separators
ylim1, ylimmax1 = ax1.get_ylim()
ylim2, ylimmax2 = ax2.get_ylim()
ylim3, ylimmax3 = ax3.get_ylim()
ylim4, ylimmax4 = ax4.get_ylim()
ax1.set_ylim([ylim1 - (ylimmax1 - ylim1)*0.1, ylimmax1])
ax2.set_ylim([ylim2 - (ylimmax2 - ylim2)*0.1, ylimmax2])
ax3.set_ylim([ylim3 - (ylimmax3 - ylim3)*0.1, ylimmax3])
ax4.set_ylim([ylim4 - (ylimmax4 - ylim4)*0.1, ylimmax4])
prevNspecimens = 0
hdataprev = ax1.get_xlim()[0]
for i,(k,v) in enumerate(Batches_objs.items()):    
    if i != len(Batches) - 1:
        h = ax1.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        ax2.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        ax3.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        ax4.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        hdata = h.get_xdata()[0]
    else:
        hdata = ax1.get_xlim()[1]
    ax1.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax1.get_xaxis_transform())
    ax2.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax2.get_xaxis_transform())
    ax3.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax3.get_xaxis_transform())
    ax4.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax4.get_xaxis_transform())
    hdataprev = hdata
    prevNspecimens += Nspecimens[i]
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()

#%% Density
ax1 = plt.subplots(1, figsize=(10,4))[1]
ax1.set_ylabel('Density (g/cm$^3$)')
ax1.set_xlabel('Graphite concentration (wt%)')

Niter = iter(range(N))
for db in data_all[didx]:
    for d in db:
        i = next(Niter)
        ax1.scatter([i]*len(d), d, c=repcolors[i], marker='.')

ax1.set_xticks([])
ax1.set_xticklabels([])
ax1.plot(data_means_flat[didx], c='k', zorder=3)
ax1.plot(data_all_flat[archdensityidx], c='r', zorder=3)

# Draw batch name and vertical lines as separators
ylim1, ylimmax1 = ax1.get_ylim()
ax1.set_ylim([ylim1 - (ylimmax1 - ylim1)*0.1, ylimmax1])
prevNspecimens = 0
hdataprev = ax1.get_xlim()[0]
for i,(k,v) in enumerate(Batches_objs.items()):
    if i != len(Batches) - 1:
        h = ax1.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        hdata = h.get_xdata()[0]
    else:
        hdata = ax1.get_xlim()[1]
    ax1.text((hdataprev + hdata)/2, 0.025, f'{concentrations[i]}', horizontalalignment='center', transform=ax1.get_xaxis_transform())
    hdataprev = hdata
    prevNspecimens += Nspecimens[i]
plt.tight_layout()

#%% Seaborn per-batch
ax1, ax2 = plt.subplots(2, figsize=(10,8))[1]
ax1.set_ylabel('Longitudinal velocity (m/s)')
ax1.set_xlabel('Batch (graphite concentration)')
ax2.set_ylabel('Shear velocity (m/s)')
ax2.set_xlabel('Batch (graphite concentration)')

# sns.boxplot(data=data_all_without_outliers[CLidx], ax=ax1, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=data_all_without_outliers[Csidx], ax=ax2, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.violinplot(data=data_all_without_outliers[CLidx], ax=ax1, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.violinplot(data=data_all_without_outliers[Csidx], ax=ax2, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

# ax1.set_xticks([])
# ax2.set_xticks([])
ax1.set_xticklabels(Batches)
ax2.set_xticklabels(Batches)
ax1.plot(data_mean[CLidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot(data_mean[Csidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
plt.tight_layout()
#%% Density
ax1 = plt.subplots(1, figsize=(10,4))[1]
ax1.set_ylabel('Density (g/cm$^3$)')
ax1.set_xlabel('Batch (graphite concentration)')

# sns.boxplot(data=data_all_without_outliers[didx], ax=ax1, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.violinplot(data=data_all_without_outliers[didx], ax=ax1, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

# ax1.set_xticks([])
ax1.set_xticklabels(Batches)
ax1.plot(data_mean[didx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
plt.tight_layout()
#%% Moduli
fig1, ax1 = plt.subplots(1, figsize=(10,4))
fig2, ax2 = plt.subplots(1, figsize=(10,4))
fig3, ax3 = plt.subplots(1, figsize=(10,4))
fig4, ax4 = plt.subplots(1, figsize=(10,4))

ax1.set_ylabel('Shear modulus (GPa)')
ax2.set_ylabel('Young modulus (GPa)')
ax3.set_ylabel('Bulk modulus (GPa)')
ax4.set_ylabel('Poisson ratio')
ax4.set_xlabel('Batch (graphite concentration)')

# new_shear = US.apply2listElements(data_all_without_outliers[shear_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
# new_young = US.apply2listElements(data_all_without_outliers[young_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
# new_bulk = US.apply2listElements(data_all_without_outliers[bulk_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
new_shear = US.apply2listElements(data_all_without_outliers[shear_modulusidx], lambda x: x*1e-6)
new_young = US.apply2listElements(data_all_without_outliers[young_modulusidx], lambda x: x*1e-6)
new_bulk = US.apply2listElements(data_all_without_outliers[bulk_modulusidx], lambda x: x*1e-6)


# sns.boxplot(data=new_shear, ax=ax1, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=new_young, ax=ax2, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=new_bulk, ax=ax3, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=data_all_without_outliers[poisson_ratioidx], ax=ax4, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

sns.violinplot(data=new_shear, ax=ax1, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.violinplot(data=new_young, ax=ax2, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.violinplot(data=new_bulk, ax=ax3, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.violinplot(data=data_all_without_outliers[poisson_ratioidx], ax=ax4, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

# ax1.set_xticks([])
# ax2.set_xticks([])
# ax3.set_xticks([])
# ax4.set_xticks([])
ax1.set_xticklabels(Batches)
ax2.set_xticklabels(Batches)
ax3.set_xticklabels(Batches)
ax4.set_xticklabels(Batches)
ax1.plot([i * 1e-6 for i in data_mean[shear_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot([i * 1e-6 for i in data_mean[young_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax3.plot([i * 1e-6 for i in data_mean[bulk_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax4.plot(data_mean[poisson_ratioidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()


#%% Seaborn per-specimen
ax1, ax2 = plt.subplots(2, figsize=(10,8))[1]
ax1.set_ylabel('Longitudinal velocity (m/s)')
ax1.set_xlabel('Batch (graphite concentration)')
ax2.set_ylabel('Shear velocity (m/s)')
ax2.set_xlabel('Batch (graphite concentration)')

# sns.boxplot(data=data_all_flat_without_outliers[CLidx], ax=ax1, palette=repcolors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=data_all_flat_without_outliers[Csidx], ax=ax2, palette=repcolors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.violinplot(data=data_all_flat_without_outliers[CLidx], ax=ax1, palette=repcolors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10), linewidth=0)
sns.violinplot(data=data_all_flat_without_outliers[Csidx], ax=ax2, palette=repcolors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10), linewidth=0)


ax1.set_xticks([])
ax2.set_xticks([])
ax1.plot(data_means_flat[CLidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot(data_means_flat[Csidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)

# Draw batch name and vertical lines as separators
ylim1, ylimmax1 = ax1.get_ylim()
ylim2, ylimmax2 = ax2.get_ylim()
ax1.set_ylim([ylim1 - (ylimmax1 - ylim1)*0.1, ylimmax1])
ax2.set_ylim([ylim2 - (ylimmax2 - ylim2)*0.1, ylimmax2]) 
prevNspecimens = 0  
hdataprev = ax1.get_xlim()[0]
for i,(k,v) in enumerate(Batches_objs.items()):
    if i != len(Batches) - 1:
        h = ax1.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        ax2.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        hdata = h.get_xdata()[0]
    else:
        hdata = ax1.get_xlim()[1]
    ax1.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax1.get_xaxis_transform())
    ax2.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax2.get_xaxis_transform())
    hdataprev = hdata
    prevNspecimens += Nspecimens[i]
plt.tight_layout()
#%% Moduli
fig1, ax1 = plt.subplots(1, figsize=(10,4))
fig2, ax2 = plt.subplots(1, figsize=(10,4))
fig3, ax3 = plt.subplots(1, figsize=(10,4))
fig4, ax4 = plt.subplots(1, figsize=(10,4))

ax1.set_ylabel('Shear modulus (GPa)')
ax2.set_ylabel('Young modulus (GPa)')
ax3.set_ylabel('Bulk modulus (GPa)')
ax4.set_ylabel('Poisson ratio')
ax4.set_xlabel('Batch (graphite concentration)')

# new_shear = US.apply2listElements(data_all_flat[shear_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
# new_young = US.apply2listElements(data_all_flat[young_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
# new_bulk = US.apply2listElements(data_all_flat[bulk_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
new_shear = US.apply2listElements(data_all_flat_without_outliers[shear_modulusidx], lambda x: x*1e-6)
new_young = US.apply2listElements(data_all_flat_without_outliers[young_modulusidx], lambda x: x*1e-6)
new_bulk = US.apply2listElements(data_all_flat_without_outliers[bulk_modulusidx], lambda x: x*1e-6)

sns.boxplot(data=new_shear, ax=ax1, palette=repcolors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.boxplot(data=new_young, ax=ax2, palette=repcolors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.boxplot(data=new_bulk, ax=ax3, palette=repcolors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.boxplot(data=data_all_flat_without_outliers[poisson_ratioidx], ax=ax4, palette=repcolors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

# sns.violinplot(data=new_shear, ax=ax1, palette=repcolors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.violinplot(data=new_young, ax=ax2, palette=repcolors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.violinplot(data=new_bulk, ax=ax3, palette=repcolors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.violinplot(data=data_all_flat_without_outliers[poisson_ratioidx], ax=ax4, palette=repcolors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax1.plot([i * 1e-6 for i in data_means_flat[shear_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot([i * 1e-6 for i in data_means_flat[young_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax3.plot([i * 1e-6 for i in data_means_flat[bulk_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax4.plot(data_means_flat[poisson_ratioidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
plt.tight_layout()

# Draw batch name and vertical lines as separators
ylim1, ylimmax1 = ax1.get_ylim()
ylim2, ylimmax2 = ax2.get_ylim()
ax1.set_ylim([ylim1 - (ylimmax1 - ylim1)*0.1, ylimmax1])
ax2.set_ylim([ylim2 - (ylimmax2 - ylim2)*0.1, ylimmax2]) 
prevNspecimens = 0  
hdataprev = ax1.get_xlim()[0]
for i,(k,v) in enumerate(Batches_objs.items()):    
    if i != len(Batches) - 1:
        h = ax1.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        ax2.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        ax3.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        ax4.axvline(prevNspecimens + Nspecimens[i] - 0.5, c='gray')
        hdata = h.get_xdata()[0]
    else:
        hdata = ax1.get_xlim()[1]
    ax1.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax1.get_xaxis_transform())
    ax2.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax2.get_xaxis_transform())
    ax3.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax3.get_xaxis_transform())
    ax4.text((hdataprev + hdata)/2, 0.025, f'{k} ({concentrations[i]}%)', horizontalalignment='center', transform=ax4.get_xaxis_transform())
    hdataprev = hdata
    prevNspecimens += Nspecimens[i]
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()




#%% FFT
# lims = [2125, 2300]

# if Nbatches > len(Batch.colors):
#     colors = list(mcolors.CSS4_COLORS)
#     repcolors = []
#     for i,n in enumerate(Nspecimens):
#         repcolors.extend([colors[i]]*n)
# else:
#     repcolors = []
#     for i,n in enumerate(Nspecimens):
#         repcolors.extend([Batch.colors[i]]*n)

# ax1 = plt.subplots(1)[1]
# ax1.set_xlim([0,4])
# ax1.set_xlabel('Frequency (MHz)')
# ax1.set_ylabel('|$\cal{F}$(TT)|$^2$')

# Fs = experiments[list(experiments.keys())[0]].Fs
# nfft = 2**13
# f = np.linspace(0,Fs/2, nfft)
# for i,(k,v) in enumerate(experiments.items()):
#     tts = v.TT[lims[0]:lims[1],:]
#     tt_fft = np.fft.fft(tts, nfft, axis=0).mean(axis=1)
    
#     if i>0:
#         lbl = k[0] if repcolors[i] != repcolors[i-1] else None
#     else:
#         lbl = k[0]
#     ax1.plot(f*1e-6, np.abs(tt_fft)**2, repcolors[i], label=lbl)

# plt.legend()
# plt.tight_layout()






#%% Sort specimens in each batch
sorted_data_means_flat = [None]*len(idxsmap)
sorted_data_means = data_means.copy()
sorted_data_all = data_all.copy()
for i in range(len(idxsmap)):
    for j,(sorted_batch_means,batch_means) in enumerate(zip(sorted_data_means[i], data_means[i])):
        sorted_batch_means.sort()
        for k,bm in enumerate(sorted_batch_means):
            idx = np.where(batch_means == bm)[0][0]
            sorted_data_all[i][j][k] = data_all[i][j][idx]

sorted_data_means_flat[CLidx] = [item for sublist in sorted_data_means[CLidx] for item in sublist]
sorted_data_means_flat[Csidx] = [item for sublist in sorted_data_means[Csidx] for item in sublist]
sorted_data_means_flat[didx]  = [item for sublist in sorted_data_means[didx]  for item in sublist]
sorted_data_means_flat[Lidx]  = [item for sublist in sorted_data_means[Lidx]  for item in sublist]
sorted_data_means_flat[shear_modulusidx] = [item for sublist in sorted_data_means[shear_modulusidx] for item in sublist]
sorted_data_means_flat[young_modulusidx] = [item for sublist in sorted_data_means[young_modulusidx] for item in sublist]
sorted_data_means_flat[bulk_modulusidx]  = [item for sublist in sorted_data_means[bulk_modulusidx]  for item in sublist]
sorted_data_means_flat[poisson_ratioidx] = [item for sublist in sorted_data_means[poisson_ratioidx] for item in sublist]

#%% Curve Fit function
def CurveFitData(x, data, guess, x_model):
    cf = US.CurveFit(x, data, guess, func=2, errorfunc='L2', nparams=4, npredictions=10_000)
    cfu = US.CurveFit(x, cf.u, guess, func=2, errorfunc='L2', nparams=4, npredictions=10_000)
    cfl = US.CurveFit(x, cf.l, guess, func=2, errorfunc='L2', nparams=4, npredictions=10_000)
    model_u = cfu.func(cfu.params_opt, x_model)
    model_l = cfl.func(cfl.params_opt, x_model)
    return cf, cfu, cfl, model_u, model_l
    
    
#%% Curve Fit
fitted_data = data_mean

x = np.arange(0, len(fitted_data[CLidx]))
x_model = np.arange(np.min(x), np.max(x), 0.001)
cf_cl, cfu_cl, cfl_cl, model_u_cl, model_l_cl = CurveFitData(x, fitted_data[CLidx], [2730,0.1,0.01], x_model)
cf_cs, cfu_cs, cfl_cs, model_u_cs, model_l_cs = CurveFitData(x, fitted_data[Csidx], [1250,0.1,0.01], x_model)


#%% Curve fit Density
fitted_data = data_mean

x = np.arange(0, len(fitted_data[didx]))
x_model = np.arange(np.min(x), np.max(x), 0.001)
cf_d, cfu_d, cfl_d, model_u_d, model_l_d = CurveFitData(x, fitted_data[didx], [1.2,0.1,0.01], x_model)


#%% Curve fit Moduli
fitted_data = data_mean

x = np.arange(0, len(fitted_data[shear_modulusidx]))
x_model = np.arange(np.min(x), np.max(x), 0.001)
cf_shear_modulus, cfu_shear_modulus, cfl_shear_modulus, model_u_shear_modulus, model_l_shear_modulus = CurveFitData(x, np.array(fitted_data[shear_modulusidx])*1e-6, [1.8,0.1,0.01], x_model)

x = np.arange(0, len(fitted_data[young_modulusidx]))
x_model = np.arange(np.min(x), np.max(x), 0.001)
cf_young_modulus, cfu_young_modulus, cfl_young_modulus, model_u_young_modulus, model_l_young_modulus = CurveFitData(x, np.array(fitted_data[young_modulusidx])*1e-6, [4.9,0.1,0.01], x_model)


x = np.arange(0, len(fitted_data[bulk_modulusidx]))
x_model = np.arange(np.min(x), np.max(x), 0.001)
cf_bulk_modulus, cfu_bulk_modulus, cfl_bulk_modulus, model_u_bulk_modulus, model_l_bulk_modulus = CurveFitData(x, np.array(fitted_data[bulk_modulusidx])*1e-6, [6.2,0.1,0.01], x_model)

x = np.arange(0, len(fitted_data[poisson_ratioidx]))
x_model = np.arange(np.min(x), np.max(x), 0.001)
cf_poisson_ratio, cfu_poisson_ratio, cfl_poisson_ratio, model_u_poisson_ratio, model_l_poisson_ratio = CurveFitData(x, fitted_data[poisson_ratioidx], [0.42,0.1,0.01], x_model)


#%% Plot Curve Fit
print('--- CL ---')
print(cf_cl)
print(f'R2: {cf_cl.r2}')
print('--- Cs ---')
print(cf_cs)
print(f'R2: {cf_cs.r2}')

# ax1, ax2 = plt.subplots(2)[1]
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
ax1.plot(x, fitted_data[CLidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax1.plot(x_model, cf_cl.func(cf_cl.params_opt, x_model), c='r')
ax1.plot(x_model, cfu_cl.func(cfu_cl.params_opt, x_model), c='r', ls='--')
ax1.plot(x_model, cfl_cl.func(cfl_cl.params_opt, x_model), c='r', ls='--')

ax2.plot(x, fitted_data[Csidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot(x_model, cf_cs.func(cf_cs.params_opt, x_model), c='r')
ax2.plot(x_model, cfu_cs.func(cfu_cs.params_opt, x_model), c='r', ls='--')
ax2.plot(x_model, cfl_cs.func(cfl_cs.params_opt, x_model), c='r', ls='--')

ax1.set_xticks(x)
# ax1.set_xticks([])
ax2.set_xticks(x)
ax1.set_xticklabels(concentrations)
ax2.set_xticklabels(concentrations)

ax2.set_xlabel('Concentration (wt%)')

ax1.set_ylabel('CL (m/s)')
ax2.set_ylabel('Cs (m/s)')
fig1.tight_layout()
fig2.tight_layout()

#%% Plot Curve Fit Density
print('--- d ---')
print(cf_d)
print(f'R2: {cf_d.r2}')

ax1 = plt.subplots(1)[1]
ax1.plot(x, fitted_data[didx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax1.plot(x_model, cf_d.func(cf_d.params_opt, x_model), c='r')
ax1.plot(x_model, cfu_d.func(cfu_d.params_opt, x_model), c='r', ls='--')
ax1.plot(x_model, cfl_d.func(cfl_d.params_opt, x_model), c='r', ls='--')

ax1.set_xticks(x)
ax1.set_xticklabels(concentrations)
ax1.set_xlabel('Concentration (wt%)')
ax1.set_ylabel('Density (g/cm$^3$)')
plt.tight_layout()


#%% Plot Seaborn per-batch and Curve Fit
# ax1, ax2 = plt.subplots(2, figsize=(10,8))[1]
fig1, ax1 = plt.subplots(1, figsize=(10,4))
fig2, ax2 = plt.subplots(1, figsize=(10,4))
ax1.set_ylabel('Longitudinal velocity (m/s)')
ax1.set_xlabel('Graphite concentration (wt%)')
ax2.set_ylabel('Shear velocity (m/s)')
ax2.set_xlabel('Graphite concentration (wt%)')

# sns.boxplot(data=data_all_without_outliers[CLidx], ax=ax1, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=data_all_without_outliers[Csidx], ax=ax2, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sv1 = sns.violinplot(data=data_all_without_outliers[CLidx], ax=ax1, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10), alpha=0.5)
sv2 = sns.violinplot(data=data_all_without_outliers[Csidx], ax=ax2, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10), alpha=0.5)

for violin1, violin2 in zip(sv1.collections[::2], sv2.collections[::2]):
    violin1.set_alpha(0.5)
    violin2.set_alpha(0.5)

# ax1.set_xticks(concentrations)
# ax2.set_xticks(concentrations)
ax1.set_xticklabels(concentrations)
ax2.set_xticklabels(concentrations)
ax1.plot(data_mean[CLidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot(data_mean[Csidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)

ax1.plot(x_model, cf_cl.func(cf_cl.params_opt, x_model), c='r', zorder=3)
ax1.plot(x_model, cfu_cl.func(cfu_cl.params_opt, x_model), c='r', ls='--', zorder=3)
ax1.plot(x_model, cfl_cl.func(cfl_cl.params_opt, x_model), c='r', ls='--', zorder=3)

ax2.plot(x_model, cf_cs.func(cf_cs.params_opt, x_model), c='r', zorder=3)
ax2.plot(x_model, cfu_cs.func(cfu_cs.params_opt, x_model), c='r', ls='--', zorder=3)
ax2.plot(x_model, cfl_cs.func(cfl_cs.params_opt, x_model), c='r', ls='--', zorder=3)


fig1.tight_layout()
fig2.tight_layout()


#%% Plot Seaborn per-batch and Curve Fit --- Density
ax1 = plt.subplots(1, figsize=(10,4))[1]
ax1.set_ylabel('Density (g/cm$^3$)')
ax1.set_xlabel('Batch (graphite concentration)')

# sns.boxplot(data=data_all_without_outliers[didx], ax=ax1, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.violinplot(data=data_all_without_outliers[didx], ax=ax1, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

# ax1.set_xticks([])
ax1.set_xticklabels(Batches)
ax1.plot(data_mean[didx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)

ax1.plot(x_model, cf_d.func(cf_d.params_opt, x_model), c='r', zorder=3)
ax1.plot(x_model, cfu_d.func(cfu_d.params_opt, x_model), c='r', ls='--', zorder=3)
ax1.plot(x_model, cfl_d.func(cfl_d.params_opt, x_model), c='r', ls='--', zorder=3)

plt.tight_layout()

#%% Plot Seaborn per batch and Curve Fit --- Moduli
fig1, ax1 = plt.subplots(1, figsize=(10,4))
fig2, ax2 = plt.subplots(1, figsize=(10,4))
fig3, ax3 = plt.subplots(1, figsize=(10,4))
fig4, ax4 = plt.subplots(1, figsize=(10,4))

ax1.set_xlabel('Graphite concentration (wt%)')
ax2.set_xlabel('Graphite concentration (wt%)')
ax3.set_xlabel('Graphite concentration (wt%)')
ax4.set_xlabel('Graphite concentration (wt%)')

ax1.set_ylabel('Shear modulus (GPa)')
ax2.set_ylabel('Young modulus (GPa)')
ax3.set_ylabel('Bulk modulus (GPa)')
ax4.set_ylabel('Poisson ratio')

# new_shear = US.apply2listElements(data_all_without_outliers[shear_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
# new_young = US.apply2listElements(data_all_without_outliers[young_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
# new_bulk = US.apply2listElements(data_all_without_outliers[bulk_modulusidx], US.multiplyMaskedArraybyScalar, 1e-6)
new_shear = US.apply2listElements(data_all_without_outliers[shear_modulusidx], lambda x: x*1e-6)
new_young = US.apply2listElements(data_all_without_outliers[young_modulusidx], lambda x: x*1e-6)
new_bulk = US.apply2listElements(data_all_without_outliers[bulk_modulusidx], lambda x: x*1e-6)


# sns.boxplot(data=new_shear, ax=ax1, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=new_young, ax=ax2, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=new_bulk, ax=ax3, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
# sns.boxplot(data=data_all_without_outliers[poisson_ratioidx], ax=ax4, palette=colors, showfliers=False, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

sv1 = sns.violinplot(data=new_shear, ax=ax1, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sv2 = sns.violinplot(data=new_young, ax=ax2, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sv3 = sns.violinplot(data=new_bulk, ax=ax3, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sv4 = sns.violinplot(data=data_all_without_outliers[poisson_ratioidx], ax=ax4, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))


for violin1, violin2, violin3, violin4 in zip(sv1.collections[::2], sv2.collections[::2], sv3.collections[::2], sv4.collections[::2]):
    violin1.set_alpha(0.5)
    violin2.set_alpha(0.5)
    violin3.set_alpha(0.5)
    violin4.set_alpha(0.5)


# ax1.set_xticks([])
# ax2.set_xticks([])
# ax3.set_xticks([])
# ax4.set_xticks([])
ax1.set_xticklabels(concentrations)
ax2.set_xticklabels(concentrations)
ax3.set_xticklabels(concentrations)
ax4.set_xticklabels(concentrations)
ax1.plot([i * 1e-6 for i in data_mean[shear_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot([i * 1e-6 for i in data_mean[young_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax3.plot([i * 1e-6 for i in data_mean[bulk_modulusidx]], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax4.plot(data_mean[poisson_ratioidx], c='k', marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)



ax1.plot(x_model, cf_shear_modulus.func(cf_shear_modulus.params_opt, x_model), c='r', zorder=3)
ax1.plot(x_model, cfu_shear_modulus.func(cfu_shear_modulus.params_opt, x_model), c='r', ls='--', zorder=3)
ax1.plot(x_model, cfl_shear_modulus.func(cfl_shear_modulus.params_opt, x_model), c='r', ls='--', zorder=3)

ax2.plot(x_model, cf_young_modulus.func(cf_young_modulus.params_opt, x_model), c='r', zorder=3)
ax2.plot(x_model, cfu_young_modulus.func(cfu_young_modulus.params_opt, x_model), c='r', ls='--', zorder=3)
ax2.plot(x_model, cfl_young_modulus.func(cfl_young_modulus.params_opt, x_model), c='r', ls='--', zorder=3)

ax3.plot(x_model, cf_bulk_modulus.func(cf_bulk_modulus.params_opt, x_model), c='r', zorder=3)
ax3.plot(x_model, cfu_bulk_modulus.func(cfu_bulk_modulus.params_opt, x_model), c='r', ls='--', zorder=3)
ax3.plot(x_model, cfl_bulk_modulus.func(cfl_bulk_modulus.params_opt, x_model), c='r', ls='--', zorder=3)

ax4.plot(x_model, cf_poisson_ratio.func(cf_poisson_ratio.params_opt, x_model), c='r', zorder=3)
ax4.plot(x_model, cfu_poisson_ratio.func(cfu_poisson_ratio.params_opt, x_model), c='r', ls='--', zorder=3)
ax4.plot(x_model, cfl_poisson_ratio.func(cfl_poisson_ratio.params_opt, x_model), c='r', ls='--', zorder=3)

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()


#%% Histograms
def plothist(dataName: str, label: str, RejectOutliers: bool, xlim=None):
    fig, axs = plt.subplots(4,3, figsize=(10,8))
    fig.suptitle(label)
    # fig.delaxes(axs[3,2])
    axs[3,2].set_yticks([])
    axs[3,2].set_xlim(xlim)
    axs[3,2].set_title('All')
    
    for i,bo in enumerate(Batches_objs.values()):
        n, m = np.unravel_index(i, (4,3))
        data = eval(f'bo.{dataName}.masked') if RejectOutliers else eval(f'bo.{dataName}')
        edgecolor = 'gray' if colors[i]=='k' else 'k'
        US.plot_hist(*data.hist, ax=axs[n,m], edgecolor=edgecolor, color=colors[i])
        axs[n,m].plot(data.aux, data.gauss, c='r')
        axs[n,m].set_title(f'{bo.name} ({concentrations[i]}%)')
        axs[n,m].set_yticks([])
        axs[n,m].set_xlim(xlim)
        axs[3,2].plot(data.aux, data.gauss, c=colors[i])
    plt.tight_layout()

#%%
plothist('CL', 'Longitudinal velocity (m/s)', RejectOutliers['RejectOutliers_CL'], [2700, 2780])
plothist('Cs', 'Shear velocity (m/s)', RejectOutliers['RejectOutliers_Cs'], [1220, 1280])
plothist('d', 'Density (g/cm$^3$)', RejectOutliers['RejectOutliers_d'], [0.95, 1.35])

#%% Average Stds
CL_mean_ = 0
Cs_mean_ = 0
shear_modulus_mean_ = 0
young_modulus_mean_ = 0
bulk_modulus_mean_ = 0
poisson_ratio_mean_ = 0
print('\nAverage Standard deviations')
print('===========================')
print('Batch |    CL    |   Cs    |  shear  |  young  |  bulk   | poisson')
print('-----------------+---------+---------+---------+---------+--------')
for k,v in Batches_objs.items():
    _tempCL = v.CL.masked if RejectOutliers['RejectOutliers_CL'] else v.CL
    _tempCs = v.Cs.masked if RejectOutliers['RejectOutliers_Cs'] else v.Cs
    _tempshear_modulus = v.shear_modulus.masked if RejectOutliers['RejectOutliers_shear_modulus'] else v.shear_modulus
    _tempyoung_modulus = v.young_modulus.masked if RejectOutliers['RejectOutliers_young_modulus'] else v.young_modulus
    _tempbulk_modulus = v.bulk_modulus.masked if RejectOutliers['RejectOutliers_bulk_modulus'] else v.bulk_modulus
    _temppoisson_ratio = v.poisson_ratio.masked if RejectOutliers['RejectOutliers_poisson_ratio'] else v.poisson_ratio
    CL_mean_ += _tempCL.std
    Cs_mean_ += _tempCs.std
    shear_modulus_mean_ += _tempshear_modulus.std
    young_modulus_mean_ += _tempyoung_modulus.std
    bulk_modulus_mean_ += _tempbulk_modulus.std
    poisson_ratio_mean_ += _temppoisson_ratio.std
    print(f' {k}  -> {_tempCL.std:.5f} | {_tempCs.std:.5f} | {_tempshear_modulus.std*1e-6:.5f} | {_tempyoung_modulus.std*1e-6:.5f} | {_tempbulk_modulus.std*1e-6:.5f} | {_temppoisson_ratio.std:.5f}')
CL_mean_ /= len(Batches_objs)
Cs_mean_ /= len(Batches_objs)
shear_modulus_mean_ /= len(Batches_objs)
young_modulus_mean_ /= len(Batches_objs)
bulk_modulus_mean_ /= len(Batches_objs)
poisson_ratio_mean_ /= len(Batches_objs)
print('-----------------+---------+---------+---------+---------+--------')
print(f' Avg  -> {CL_mean_:.5f} | {Cs_mean_:.5f} | {shear_modulus_mean_*1e-6:.5f} | {young_modulus_mean_*1e-6:.5f} | {bulk_modulus_mean_*1e-6:.5f} | {poisson_ratio_mean_:.5f}')


