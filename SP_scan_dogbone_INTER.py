# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:09:23 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import os
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS) # 10 colors

import src.ultrasound as US

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
            experiments[Experiment_folder_name] = US.DogboneSP(MyDir, **kwargs)
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
        experiments[e] = US.DogboneSP(ExperimentName, **kwargs)
        if Verbose: print(f'Specimen {e} done.')
    return experiments


#%%
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
    CLall = US.maskOutliers(CLall, m=m_cl, UseMedian=False)
    Csall = US.maskOutliers(Csall, m=m_cs, UseMedian=UseMedianForCs)
    dall = np.ma.masked_array(dall, mask=CLall.mask)

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