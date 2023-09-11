# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 18:44:41 2023
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
Batches_ns = ['Ans', 'Bns', 'Cns', 'Dns', 'Ens', 'Fns', 'Gns','Hns', 'Ins', 'Jns', 'Kns']
Batches_s = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
Batches_s = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
Batches = Batches_s + Batches_ns
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
# ---------
# Load data
# ---------
experiments = loadDogBoneExperiments(get_experiment_paths(Path, Batches), Verbose=Verbose, compute=False)
for e in experiments.values():
    e.computeModuli(UseArchDensity=True)

# concentrations_s = [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
concentrations_s = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
concentrations_ns = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

exps_ns = {}
exps_s = {}
for k,e in experiments.items():
    if 'ns' in k:
        exps_ns[k] = e
    else:
        exps_s[k] = e

Nbatches_s = len(Batches_s)
Nbatches_ns = len(Batches_ns)
N = len(experiments)
scanpos = experiments[list(experiments.keys())[0]].scanpos
idxsmap = ['CL', 'Cs', 'd', 'L', 'shear_modulus', 'young_modulus', 'bulk_modulus', 'poisson_ratio', 'archdensity']

Batches_objs_s = {}
Batches_objs_ns = {}
for b in Batches:
    if 'ns' in b:
        CL = np.array([v.CL for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        Cs = np.array([v.Cs for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        d  = np.array([v.density  for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        L = np.array([v.L for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        shear_modulus = np.array([v.shear_modulus for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        young_modulus = np.array([v.young_modulus for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        bulk_modulus  = np.array([v.bulk_modulus  for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        poisson_ratio = np.array([v.poisson_ratio for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        archdensity = np.array([v.archdensity for k,v in exps_ns.items() if US.isGoodSpecimen(b, k)])
        
        Batches_objs_ns[b] = Batch(CL, Cs, d, shear_modulus, young_modulus, 
                                bulk_modulus, poisson_ratio, L, scanpos, archdensity, b,
                                **UseMedian, **m)
    else:
        CL = np.array([v.CL for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        Cs = np.array([v.Cs for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        d  = np.array([v.density  for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        L = np.array([v.L for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        shear_modulus = np.array([v.shear_modulus for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        young_modulus = np.array([v.young_modulus for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        bulk_modulus  = np.array([v.bulk_modulus  for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        poisson_ratio = np.array([v.poisson_ratio for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        archdensity = np.array([v.archdensity for k,v in exps_s.items() if US.isGoodSpecimen(b, k)])
        
        Batches_objs_s[b] = Batch(CL, Cs, d, shear_modulus, young_modulus, 
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

data_mean_s = [[None]*Nbatches_s for _ in range(len(idxsmap))]
data_mean_ns = [[None]*Nbatches_ns for _ in range(len(idxsmap))]
for i,v in enumerate(Batches_objs_s.values()): 
    data_mean_s[CLidx][i] = v.CL.masked.mean.copy() if RejectOutliers['RejectOutliers_CL'] else v.CL.mean.copy()
    data_mean_s[Csidx][i] = v.Cs.masked.mean.copy() if RejectOutliers['RejectOutliers_Cs'] else v.Cs.mean.copy()
    data_mean_s[didx][i]  = v.d.masked.mean.copy()  if RejectOutliers['RejectOutliers_d']  else v.d.mean.copy()
    data_mean_s[Lidx][i]  = v.L.masked.mean.copy()  if RejectOutliers['RejectOutliers_L']  else v.L.mean.copy()
    data_mean_s[shear_modulusidx][i] = v.shear_modulus.masked.mean.copy() if RejectOutliers['RejectOutliers_shear_modulus'] else v.shear_modulus.mean.copy()
    data_mean_s[young_modulusidx][i] = v.young_modulus.masked.mean.copy() if RejectOutliers['RejectOutliers_young_modulus'] else v.young_modulus.mean.copy()
    data_mean_s[bulk_modulusidx][i]  = v.bulk_modulus.masked.mean.copy()  if RejectOutliers['RejectOutliers_bulk_modulus']  else v.bulk_modulus.mean.copy()
    data_mean_s[poisson_ratioidx][i] = v.poisson_ratio.masked.mean.copy() if RejectOutliers['RejectOutliers_poisson_ratio'] else v.poisson_ratio.mean.copy()

for i,v in enumerate(Batches_objs_ns.values()):    
    data_mean_ns[CLidx][i] = v.CL.masked.mean.copy() if RejectOutliers['RejectOutliers_CL'] else v.CL.mean.copy()
    data_mean_ns[Csidx][i] = v.Cs.masked.mean.copy() if RejectOutliers['RejectOutliers_Cs'] else v.Cs.mean.copy()
    data_mean_ns[didx][i]  = v.d.masked.mean.copy()  if RejectOutliers['RejectOutliers_d']  else v.d.mean.copy()
    data_mean_ns[Lidx][i]  = v.L.masked.mean.copy()  if RejectOutliers['RejectOutliers_L']  else v.L.mean.copy()
    data_mean_ns[shear_modulusidx][i] = v.shear_modulus.masked.mean.copy() if RejectOutliers['RejectOutliers_shear_modulus'] else v.shear_modulus.mean.copy()
    data_mean_ns[young_modulusidx][i] = v.young_modulus.masked.mean.copy() if RejectOutliers['RejectOutliers_young_modulus'] else v.young_modulus.mean.copy()
    data_mean_ns[bulk_modulusidx][i]  = v.bulk_modulus.masked.mean.copy()  if RejectOutliers['RejectOutliers_bulk_modulus']  else v.bulk_modulus.mean.copy()
    data_mean_ns[poisson_ratioidx][i] = v.poisson_ratio.masked.mean.copy() if RejectOutliers['RejectOutliers_poisson_ratio'] else v.poisson_ratio.mean.copy()
    

#%% Curve Fit function
def CurveFitData(x, data, guess, x_model):
    cf = US.CurveFit(x, data, guess, func=2, errorfunc='L2', nparams=4, npredictions=10_000)
    cfu = US.CurveFit(x, cf.u, guess, func=2, errorfunc='L2', nparams=4, npredictions=10_000)
    cfl = US.CurveFit(x, cf.l, guess, func=2, errorfunc='L2', nparams=4, npredictions=10_000)
    model_u = cfu.func(cfu.params_opt, x_model)
    model_l = cfl.func(cfl.params_opt, x_model)
    return cf, cfu, cfl, model_u, model_l
    





#%%
# ==========
# SONICATION
# ==========

#%% Curve Fit
x_s = np.arange(0, len(data_mean_s[CLidx]))
x_model_s = np.arange(np.min(x_s), np.max(x_s), 0.001)
cf_cl_s, cfu_cl_s, cfl_cl_s, model_u_cl_s, model_l_cl_s = CurveFitData(x_s, data_mean_s[CLidx], [2730,0.1,0.01], x_model_s)
cf_cs_s, cfu_cs_s, cfl_cs_s, model_u_cs_s, model_l_cs_s = CurveFitData(x_s, data_mean_s[Csidx], [1250,0.1,0.01], x_model_s)

x_s = np.arange(0, len(data_mean_s[shear_modulusidx]))
x_model_s = np.arange(np.min(x_s), np.max(x_s), 0.001)
cf_shear_modulus_s, cfu_shear_modulus_s, cfl_shear_modulus_s, model_u_shear_modulus_s, model_l_shear_modulus_s = CurveFitData(x_s, np.array(data_mean_s[shear_modulusidx])*1e-6, [1.8,0.1,0.01], x_model_s)

x_s = np.arange(0, len(data_mean_s[young_modulusidx]))
x_model_s = np.arange(np.min(x_s), np.max(x_s), 0.001)
cf_young_modulus_s, cfu_young_modulus_s, cfl_young_modulus_s, model_u_young_modulus_s, model_l_young_modulus_s = CurveFitData(x_s, np.array(data_mean_s[young_modulusidx])*1e-6, [4.9,0.1,0.01], x_model_s)


x_s = np.arange(0, len(data_mean_s[bulk_modulusidx]))
x_model_s = np.arange(np.min(x_s), np.max(x_s), 0.001)
cf_bulk_modulus_s, cfu_bulk_modulus_s, cfl_bulk_modulus_s, model_u_bulk_modulus_s, model_l_bulk_modulus_s = CurveFitData(x_s, np.array(data_mean_s[bulk_modulusidx])*1e-6, [6.2,0.1,0.01], x_model_s)

x_s = np.arange(0, len(data_mean_s[poisson_ratioidx]))
x_model_s = np.arange(np.min(x_s), np.max(x_s), 0.001)
cf_poisson_ratio_s, cfu_poisson_ratio_s, cfl_poisson_ratio_s, model_u_poisson_ratio_s, model_l_poisson_ratio_s = CurveFitData(x_s, data_mean_s[poisson_ratioidx], [0.42,0.1,0.01], x_model_s)








#%%
# =============
# NO SONICATION
# =============

#%% Curve Fit
x_ns = np.arange(0, len(data_mean_ns[CLidx]))
x_model_ns = np.arange(np.min(x_ns), np.max(x_ns), 0.001)
cf_cl_ns, cfu_cl_ns, cfl_cl_ns, model_u_cl_ns, model_l_cl_ns = CurveFitData(x_ns, data_mean_ns[CLidx], [2730,0.1,0.01], x_model_ns)
cf_cs_ns, cfu_cs_ns, cfl_cs_ns, model_u_cs_ns, model_l_cs_ns = CurveFitData(x_ns, data_mean_ns[Csidx], [1250,0.1,0.01], x_model_ns)

x_ns = np.arange(0, len(data_mean_ns[shear_modulusidx]))
x_model_ns = np.arange(np.min(x_ns), np.max(x_ns), 0.001)
cf_shear_modulus_ns, cfu_shear_modulus_ns, cfl_shear_modulus_ns, model_u_shear_modulus_ns, model_l_shear_modulus_ns = CurveFitData(x_ns, np.array(data_mean_ns[shear_modulusidx])*1e-6, [1.8,0.1,0.01], x_model_ns)

x_ns = np.arange(0, len(data_mean_ns[young_modulusidx]))
x_model_ns = np.arange(np.min(x_ns), np.max(x_ns), 0.001)
cf_young_modulus_ns, cfu_young_modulus_ns, cfl_young_modulus_ns, model_u_young_modulus_ns, model_l_young_modulus_ns = CurveFitData(x_ns, np.array(data_mean_ns[young_modulusidx])*1e-6, [4.9,0.1,0.01], x_model_ns)


x_ns = np.arange(0, len(data_mean_ns[bulk_modulusidx]))
x_model_ns = np.arange(np.min(x_ns), np.max(x_ns), 0.001)
cf_bulk_modulus_ns, cfu_bulk_modulus_ns, cfl_bulk_modulus_ns, model_u_bulk_modulus_ns, model_l_bulk_modulus_ns = CurveFitData(x_ns, np.array(data_mean_ns[bulk_modulusidx])*1e-6, [6.2,0.1,0.01], x_model_ns)

x_ns = np.arange(0, len(data_mean_ns[poisson_ratioidx]))
x_model_ns = np.arange(np.min(x_ns), np.max(x_ns), 0.001)
cf_poisson_ratio_ns, cfu_poisson_ratio_ns, cfl_poisson_ratio_ns, model_u_poisson_ratio_ns, model_l_poisson_ratio_ns = CurveFitData(x_ns, data_mean_ns[poisson_ratioidx], [0.42,0.1,0.01], x_model_ns)



#%%
x_s = np.arange(0, len(data_mean_s[CLidx]))
x_ns = np.arange(0, len(data_mean_ns[CLidx]))




#%% Plot velocities
# CL
concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

fig1, ax1 = plt.subplots(1)
ax2 = ax1.twinx()

ax1.plot([0,1,2,4,5,6,7,8,9,10], data_mean_s[CLidx], c='k', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot(x_ns, data_mean_ns[CLidx], c='r', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)

ax1.set_xticks(x_s)
ax2.set_xticks(x_ns)

ax1.set_ylim([2725, 2760])
ax2.set_ylim([2725, 2760])
ax2.set_yticks([])

ax1.set_xticklabels(concentrations)

ax1.set_ylabel('Longitudinal velocity (m/s)')
ax1.set_xlabel('Graphite concentration (wt%)')
fig1.tight_layout()


# Cs
fig1, ax1 = plt.subplots(1)
ax2 = ax1.twinx()

ax1.plot([0,1,2,4,5,6,7,8,9,10], data_mean_s[Csidx], c='k', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot(x_ns, data_mean_ns[Csidx], c='r', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)

ax1.set_xticks(x_s)
ax2.set_xticks(x_ns)

ax1.set_ylim([1240, 1265])
ax2.set_ylim([1240, 1265])
ax2.set_yticks([])

ax1.set_xticklabels(concentrations)

ax1.set_ylabel('Shear velocity (m/s)')
ax1.set_xlabel('Graphite concentration (wt%)')
fig1.tight_layout()

#%% Plot moduli
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)
fig4, ax4 = plt.subplots(1)

ax12 = ax1.twinx()
ax22 = ax2.twinx()
ax32 = ax3.twinx()
ax42 = ax4.twinx()

ax1.plot([0,1,2,4,5,6,7,8,9,10], np.array(data_mean_s[shear_modulusidx])*1e-6, c='k', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax2.plot([0,1,2,4,5,6,7,8,9,10], np.array(data_mean_s[young_modulusidx])*1e-6, c='k', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax3.plot([0,1,2,4,5,6,7,8,9,10], np.array(data_mean_s[bulk_modulusidx])*1e-6,  c='k', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax4.plot([0,1,2,4,5,6,7,8,9,10], data_mean_s[poisson_ratioidx], c='k', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax12.plot(x_ns, np.array(data_mean_ns[shear_modulusidx])*1e-6, c='r', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax22.plot(x_ns, np.array(data_mean_ns[young_modulusidx])*1e-6, c='r', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax32.plot(x_ns, np.array(data_mean_ns[bulk_modulusidx])*1e-6,  c='r', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)
ax42.plot(x_ns, data_mean_ns[poisson_ratioidx], c='r', lw=2, marker='o', markerfacecolor='w', markeredgecolor='k', zorder=3)

ax1.set_xticks(x_s)
ax2.set_xticks(x_s)
ax3.set_xticks(x_s)
ax4.set_xticks(x_s)
ax12.set_xticks(x_ns)
ax22.set_xticks(x_ns)
ax32.set_xticks(x_ns)
ax42.set_xticks(x_ns)

ax1.set_ylim([1.77, 1.85])
ax2.set_ylim([4.86, 5.06])
ax3.set_ylim([6.150, 6.36])
ax4.set_ylim([0.418, 0.432])

ax12.set_ylim([1.77, 1.85])
ax22.set_ylim([4.86, 5.06])
ax32.set_ylim([6.150, 6.36])
ax42.set_ylim([0.418, 0.432])

ax12.set_yticks([])
ax22.set_yticks([])
ax32.set_yticks([])
ax42.set_yticks([])

ax1.set_xticklabels(concentrations)
ax2.set_xticklabels(concentrations)
ax3.set_xticklabels(concentrations)
ax4.set_xticklabels(concentrations)

ax1.set_ylabel('Shear modulus (GPa)')
ax2.set_ylabel('Young modulus (GPa)')
ax3.set_ylabel('Bulk modulus (GPa)')
ax4.set_ylabel('Poisson ratio')
ax1.set_xlabel('Graphite concentration (wt%)')
ax2.set_xlabel('Graphite concentration (wt%)')
ax3.set_xlabel('Graphite concentration (wt%)')
ax4.set_xlabel('Graphite concentration (wt%)')
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()

#%%
Batch.colors = list(mcolors.TABLEAU_COLORS) + ['k']
if len(Batches) == 10:
    concentrations = [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    Batch.colors.pop(3)
else:
    concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = Batch.colors

def plothist(dataName: str, label: str, RejectOutliers: bool, xlim=None):
    fig, axs = plt.subplots(4,3, figsize=(14,9))
    fig.suptitle(label)
    # fig.delaxes(axs[3,2])
    axs[3,2].set_yticks([])
    axs[3,2].set_xlim(xlim)
    axs[3,2].set_title('All')

    for i,bo in enumerate(Batches_objs_ns.values()):
        n, m = np.unravel_index(i, (4,3))
        data = eval(f'bo.{dataName}.masked') if RejectOutliers else eval(f'bo.{dataName}')
        US.plot_hist(*data.hist, ax=axs[n,m], edgecolor='pink', color='crimson', lw=0.3)
        # US.plot_hist(*data.hist, ax=axs[n,m], color='crimson')
        axs[n,m].plot(data.aux, data.gauss, c='r')
        axs[n,m].set_title(f'{concentrations[i]} wt%')
        axs[n,m].set_yticks([])
        axs[n,m].set_xlim(xlim)
        axs[3,2].plot(data.aux, data.gauss, c='r')
    plt.tight_layout()

    for i,bo in enumerate(Batches_objs_s.values()):
        n, m = np.unravel_index(i, (4,3))
        data = eval(f'bo.{dataName}.masked') if RejectOutliers else eval(f'bo.{dataName}')
        US.plot_hist(*data.hist, ax=axs[n,m], edgecolor='lightgray', color='dimgray', lw=0.3)
        # US.plot_hist(*data.hist, ax=axs[n,m], color='dimgray')
        axs[n,m].plot(data.aux, data.gauss, c='k')
        axs[n,m].set_title(f'{concentrations[i]} wt%')
        axs[n,m].set_yticks([])
        axs[n,m].set_xlim(xlim)
        axs[3,2].plot(data.aux, data.gauss, c='k')
    plt.tight_layout()
    


#%%
plothist('CL', 'Longitudinal velocity (m/s)', RejectOutliers['RejectOutliers_CL'], [2700, 2780])
# plothist('Cs', 'Shear velocity (m/s)', RejectOutliers['RejectOutliers_Cs'], [1220, 1280])
# plothist('d', 'Density (g/cm$^3$)', RejectOutliers['RejectOutliers_d'], [0.95, 1.35])