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
        batch_specimens = US.get_dir_names(Path=batch_path)
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
        Keyword arguments for DogboneSP's class constructor.
        
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
        Keyword arguments for DogboneSP's class constructor.
        
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
# Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\EpoxyResin\DogboneScan'
Batches = ['A25nosonic']
Batches = ['A25nosonicreversed']
# Batches = ['A', 'B', 'C', 'D', 'E']
Nspecimens = 10 # per batch
Boxplot = False
Violinplot = False
RejectOutliers = True
UseMedianForCs = True
UseMedianForModuli = True
UseHilbEnv = False
WindowTTshear = False
Loc_TT = 1175 # Location of the TT shear window
m_cl = 2.5 # Outlier detection threshold for CL
m_cs = 0.6745 # Outlier detection threshold for Cs
m_moduli = 0.6745 # Outlier detection threshold for moduli
Verbose = True


#%%
# ---------
# Load data
# ---------
# experiments = loadDogBoneExperiments(get_experiment_paths(Path, Batches), Verbose=Verbose,
#                                        compute=True, UseHilbEnv=UseHilbEnv, WindowTTshear=WindowTTshear, Loc_TT=Loc_TT)
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
        v.computeModuli()
        print(f'Specimen {v.name} done.')
velocities_array = np.array([(v.CL, v.Cs) for v in experiments.values()]) # Size of this array is: len(experiments) x 2 x N_acqs
moduli_array = np.array([(v.shear_modulus, v.young_modulus, v.bulk_modulus, v.poisson_ratio) 
                         for v in experiments.values()]) # Size of this array is: len(experiments) x 4 x N_acqs
shear_modulus_all = moduli_array[:,0]
young_modulus_all = moduli_array[:,1]
bulk_modulus_all  = moduli_array[:,2]
poisson_ratio_all = moduli_array[:,3]

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
    
    shear_modulus_all = US.maskOutliers(shear_modulus_all, m=m_moduli, UseMedian=UseMedianForModuli)
    young_modulus_all = US.maskOutliers(young_modulus_all, m=m_moduli, UseMedian=UseMedianForModuli)
    bulk_modulus_all  = US.maskOutliers(bulk_modulus_all,  m=m_moduli, UseMedian=UseMedianForModuli)
    poisson_ratio_all = US.maskOutliers(poisson_ratio_all, m=m_moduli, UseMedian=UseMedianForModuli)

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

shear_modulus_min = shear_modulus_all.min()
young_modulus_min = young_modulus_all.min()
bulk_modulus_min  = bulk_modulus_all.min()
poisson_ratio_min = poisson_ratio_all.min()
shear_modulus_max = shear_modulus_all.max()
young_modulus_max = young_modulus_all.max()
bulk_modulus_max  = bulk_modulus_all.max()
poisson_ratio_max = poisson_ratio_all.max()

# Mean of every specimen. Size is: (10*N,)
CLmeans = CLall.mean(axis=1)
Csmeans = Csall.mean(axis=1)
dmeans  = dall.mean(axis=1)

shear_modulus_means = shear_modulus_all.mean(axis=1)
young_modulus_means = young_modulus_all.mean(axis=1)
bulk_modulus_means  = bulk_modulus_all.mean(axis=1)
poisson_ratio_means = poisson_ratio_all.mean(axis=1)


# Group by batch
CLbatches = CLall.reshape(N, Nspecimens, N_acqs)
Csbatches = Csall.reshape(N, Nspecimens, N_acqs)
dbatches  = dall.reshape(N, Nspecimens, N_acqs)

shear_modulus_batches = shear_modulus_all.reshape(N, Nspecimens, N_acqs)
young_modulus_batches = young_modulus_all.reshape(N, Nspecimens, N_acqs)
bulk_modulus_batches  = bulk_modulus_all.reshape(N, Nspecimens, N_acqs)
poisson_ratio_batches = poisson_ratio_all.reshape(N, Nspecimens, N_acqs)


# CL and Cs mean and std of every batch
CLbatches_means = np.array([x.mean() for x in CLbatches])
Csbatches_means = np.array([x.mean() for x in Csbatches])
dbatches_means  = np.array([x.mean() for x in dbatches])
CLbatches_stds  = np.array([x.std() for x in CLbatches])
Csbatches_stds  = np.array([x.std() for x in Csbatches])
dbatches_stds   = np.array([x.std() for x in dbatches])

shear_modulus_batches_means = np.array([x.mean() for x in shear_modulus_batches])
young_modulus_batches_means = np.array([x.mean() for x in young_modulus_batches])
bulk_modulus_batches_means  = np.array([x.mean() for x in bulk_modulus_batches])
poisson_ratio_batches_means = np.array([x.mean() for x in poisson_ratio_batches])
shear_modulus_batches_stds = np.array([x.std() for x in shear_modulus_batches])
young_modulus_batches_stds = np.array([x.std() for x in young_modulus_batches])
bulk_modulus_batches_stds  = np.array([x.std() for x in bulk_modulus_batches])
poisson_ratio_batches_stds = np.array([x.std() for x in poisson_ratio_batches])


# Compute Normal distribution of every batch
Nsigmas = 5
Npoints = 1000
CLbatches_aux   = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(CLbatches_means, CLbatches_stds)])
Csbatches_aux   = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(Csbatches_means, Csbatches_stds)])
dbatches_aux    = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(dbatches_means, dbatches_stds)])
CLbatches_gauss = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(CLbatches_aux, CLbatches_means, CLbatches_stds)])
Csbatches_gauss = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(Csbatches_aux, Csbatches_means, Csbatches_stds)])
dbatches_gauss  = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(dbatches_aux, dbatches_means, dbatches_stds)])

shear_modulus_batches_aux   = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(shear_modulus_batches_means, shear_modulus_batches_stds)])
young_modulus_batches_aux   = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(young_modulus_batches_means, young_modulus_batches_stds)])
bulk_modulus_batches_aux    = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(bulk_modulus_batches_means,  bulk_modulus_batches_stds)])
poisson_ratio_batches_aux   = np.array([np.linspace(m - Nsigmas*s, m + Nsigmas*s, Npoints)     for m, s      in zip(poisson_ratio_batches_means, poisson_ratio_batches_stds)])
shear_modulus_batches_gauss = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(shear_modulus_batches_aux, shear_modulus_batches_means, shear_modulus_batches_stds)])
young_modulus_batches_gauss = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(young_modulus_batches_aux, young_modulus_batches_means, young_modulus_batches_stds)])
bulk_modulus_batches_gauss  = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(bulk_modulus_batches_aux,  bulk_modulus_batches_means,  bulk_modulus_batches_stds)])
poisson_ratio_batches_gauss = np.array([np.exp(-((aux - m) / s)**2 / 2) / (s*np.sqrt(2*np.pi)) for aux, m, s in zip(poisson_ratio_batches_aux, poisson_ratio_batches_means, poisson_ratio_batches_stds)])


# Flat per batch
CL_flatperbatch = np.zeros([N, N_acqs*Nspecimens])
Cs_flatperbatch = np.zeros([N, N_acqs*Nspecimens])
for i, (cl, cs) in enumerate(zip(CLbatches, Csbatches)):
    CL_flatperbatch[i] = cl.flatten()
    Cs_flatperbatch[i] = cs.flatten()
if RejectOutliers:
    CL_flatperbatch = US.maskOutliers(CL_flatperbatch, m=m_cl, UseMedian=False)
    Cs_flatperbatch = US.maskOutliers(Cs_flatperbatch, m=m_cl, UseMedian=UseMedianForCs)

shear_modulus_flatperbatch = np.zeros([N, N_acqs*Nspecimens])
young_modulus_flatperbatch = np.zeros([N, N_acqs*Nspecimens])
bulk_modulus_flatperbatch = np.zeros([N, N_acqs*Nspecimens])
poisson_ratio_flatperbatch = np.zeros([N, N_acqs*Nspecimens])
for i, (s, y, b, p) in enumerate(zip(shear_modulus_batches, young_modulus_batches, bulk_modulus_batches, poisson_ratio_batches)):
    shear_modulus_flatperbatch[i] = s.flatten()
    young_modulus_flatperbatch[i] = y.flatten()
    bulk_modulus_flatperbatch[i]  = b.flatten()
    poisson_ratio_flatperbatch[i] = p.flatten()
if RejectOutliers:
    shear_modulus_flatperbatch = US.maskOutliers(shear_modulus_flatperbatch, m=m_moduli, UseMedian=UseMedianForModuli)
    young_modulus_flatperbatch = US.maskOutliers(young_modulus_flatperbatch, m=m_moduli, UseMedian=UseMedianForModuli)
    bulk_modulus_flatperbatch  = US.maskOutliers(bulk_modulus_flatperbatch,  m=m_moduli, UseMedian=UseMedianForModuli)
    poisson_ratio_flatperbatch = US.maskOutliers(poisson_ratio_flatperbatch, m=m_moduli, UseMedian=UseMedianForModuli)



#%%
concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]*len(Batches)
concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# --------
# Plotting
# --------
ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Longitudinal velocity (m/s)')
ax1.set_xlabel('Batch (graphite concentration)')
ax2.set_ylabel('Shear velocity (m/s)')
ax2.set_xlabel('Batch (graphite concentration)')

# Plot CL and Cs
if  N > len(colors):
    colors = list(mcolors.CSS4_COLORS)
    repcolors = [c for c in colors for _ in range(int(len(experiments)/N))]
else:
    repcolors = [c for c in colors for _ in range(int(len(experiments)/N))]
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
ax1.set_xticks([])
ax2.set_xticks([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
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
    ax1.text(i*Nspecimens + Nspecimens/2 - 1, ylim1 + (ylimmax1 - ylim1)*0.025, f'{batch} ({concentrations[i]}%)')
    ax2.text(i*Nspecimens + Nspecimens/2 - 1, ylim2 + (ylimmax2 - ylim2)*0.025, f'{batch} ({concentrations[i]}%)')
    if i != len(Batches) - 1:
        ax1.axvline(i*Nspecimens + Nspecimens - 0.5, c='gray')
        ax2.axvline(i*Nspecimens + Nspecimens - 0.5, c='gray')
plt.tight_layout()

#%% Plot concentration
ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Longitudinal velocity (m/s)')
ax1.set_xlabel('Graphite concentration (%)')
ax2.set_ylabel('Shear velocity (m/s)')
ax2.set_xlabel('Graphite concentration (%)')

sns.boxplot(data=CL_flatperbatch.T, ax=ax1, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.boxplot(data=Cs_flatperbatch.T, ax=ax2, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
ax1.plot(CLbatches_means, marker='o', markerfacecolor='w', c='k', zorder=3)
ax2.plot(Csbatches_means, marker='o', markerfacecolor='w', c='k', zorder=3)

ax1.set_xticks(np.arange(0, N))
ax2.set_xticks(np.arange(0, N))
ax1.set_xticklabels(concentrations[:N])
ax2.set_xticklabels(concentrations[:N])

plt.tight_layout()


#%% Plot each batch separately   
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
    US.plot_hist(*CLh, ax=ax1, xlabel='Longitudinal velocity (m/s)', ylabel='pdf', edgecolor='k')
    US.plot_hist(*Csh, ax=ax2, xlabel='Shear velocity (m/s)', ylabel='pdf', edgecolor='k')
    ax1.plot(CLbatches_aux[i], CLbatches_gauss[i], c='r')
    ax2.plot(Csbatches_aux[i], Csbatches_gauss[i], c='r')
    ax1.set_title(f'Batch {Batches[i]}')
    plt.tight_layout()


for j, (db, dh) in enumerate(zip(dbatches, dhists)):
    ax1, ax2 = plt.subplots(2)[1]
    ax1.set_title(f'Batch {Batches[j]}')
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
    
    
#%% I don't know what this is
# def fun(x):
#     return US.CosineInterpMax(x, xcor=False)

# PEmax = np.zeros([len(experiments), experiments['A1'].Ridx])
# for i, v in enumerate(experiments.values()):
#     PEmax[i,:] = np.apply_along_axis(fun, 0, v.PE)[:v.Ridx]

# sorted_idxs = np.argsort(PEmax, axis=1)
# PEmax_sorted = np.take_along_axis(PEmax, sorted_idxs, axis=1)
    
# sorted_idxsbatches = sorted_idxs.reshape(N, Nspecimens, N_acqs)
# dball_sorted = np.take_along_axis(dall, sorted_idxs, axis=1)
# dbatches_sorted = dball_sorted.reshape(N, Nspecimens, N_acqs)

# for j, (db, dbs) in enumerate(zip(dbatches, dbatches_sorted)):
#     ax1, ax2 = plt.subplots(2)[1]
#     plt.title(Batches[j])
#     ax1.set_ylabel('Density ($g/cm^3$)')
#     ax1.set_xlabel('Position (mm)')
#     ax2.set_ylabel('Density ($g/cm^3$)')
#     ax2.set_xlabel('Position (mm)')
    
#     for i in range(Nspecimens):
#         v = experiments[f'{b}{i+1}']
#         ax1.plot(v.scanpos, db[i], c=colors[i], lw=2)
#         if np.ma.is_masked(db[i]):
#             auxdb = db[i].copy()
#             auxdb.mask = ~db[i].mask
#             ax1.scatter(v.scanpos, auxdb, c='k', marker='.')
        
#         ax2.plot(v.scanpos, dbs[i], c=colors[i], lw=2)
#         if np.ma.is_masked(dbs[i]):
#             auxdbs = dbs[i].copy()
#             auxdbs.mask = ~dbs[i].mask
#             ax2.scatter(v.scanpos, auxdbs, c='k', marker='.')            
#     plt.tight_layout()



#%% Select and plot one batch
batch = 'D'
idx = Batches.index(batch)
cl = CLbatches[idx]
cs = Csbatches[idx]
db = dbatches[idx]
# exps = experiments['D1']

ax1, ax2, ax3 = plt.subplots(3)[1]
ax1.set_title(f'Batch {batch}')
ax1.set_ylabel('Long. vel. (m/s)')
ax2.set_ylabel('Shear vel. (m/s)')
ax3.set_ylabel('Thickness (mm)')
ax3.set_xlabel('Position (mm)')
for i in range(Nspecimens):
    v = experiments[f'{batch}{i+1}']
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
CLh = US.hist(cl.flatten(), density=True) # h, b, width
Csh = US.hist(cs.flatten(), density=True)
dh = US.hist(db.flatten(), density=True)

ax1, ax2 = plt.subplots(2)[1]
US.plot_hist(*CLh, ax=ax1, xlabel='Longitudinal velocity (m/s)', ylabel='pdf', edgecolor='k')
US.plot_hist(*Csh, ax=ax2, xlabel='Shear velocity (m/s)', ylabel='pdf', edgecolor='k')
ax1.plot(CLbatches_aux[idx], CLbatches_gauss[idx], c='r')
ax2.plot(Csbatches_aux[idx], Csbatches_gauss[idx], c='r')
ax1.set_title(f'Batch {batch}')
plt.tight_layout()



ax1, ax2 = plt.subplots(2)[1]
ax1.set_title(f'Batch {batch}')
ax1.set_ylabel('Density ($g/cm^3$)')
ax1.set_xlabel('Position (mm)')
US.plot_hist(*dh, ax=ax2, xlabel='Density ($g/cm^3$)', ylabel='pdf', edgecolor='k')
ax2.plot(dbatches_aux[idx], dbatches_gauss[idx], c='r')

for i in range(Nspecimens):
    v = experiments[f'{batch}{i+1}']
    ax1.plot(v.scanpos, db[i], c=colors[i], lw=2)
    if np.ma.is_masked(db[i]):
        auxdb = db[i].copy()
        auxdb.mask = ~db[i].mask
        ax1.scatter(v.scanpos, auxdb, c='k', marker='.')
plt.tight_layout()










#%% 
# =====================
# MECHANICAL PROPERTIES
# =====================
concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# --------
# Plotting
# --------
(ax1, ax2), (ax3, ax4) = plt.subplots(2,2, figsize=(10,8))[1]
ax1.set_xlabel('Batch (graphite concentration)')
ax2.set_xlabel('Batch (graphite concentration)')
ax3.set_xlabel('Batch (graphite concentration)')
ax4.set_xlabel('Batch (graphite concentration)')
ax1.set_ylabel('Shear modulus (GPa)')
ax2.set_ylabel('Young modulus (GPa)')
ax3.set_ylabel('Bulk modulus (GPa)')
ax4.set_ylabel('Poisson ratio')

if  N > len(colors):
    colors = mcolors.CSS4_COLORS
    repcolors = [c for c in colors for _ in range(int(len(experiments)/N))]
else:
    repcolors = [c for c in colors for _ in range(int(len(experiments)/N))]
if Boxplot:
    # Boxplot mean does not take into accoun the mask, so the mean is performed over ALL data points
    sns.boxplot(data=shear_modulus_all.T*1e-6, ax=ax1, palette=repcolors, flierprops=dict(marker='.'), showmeans=True, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
    sns.boxplot(data=young_modulus_all.T*1e-6, ax=ax2, palette=repcolors, flierprops=dict(marker='.'), showmeans=True, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
    sns.boxplot(data=bulk_modulus_all.T*1e-6,  ax=ax3, palette=repcolors, flierprops=dict(marker='.'), showmeans=True, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
    sns.boxplot(data=poisson_ratio_all.T,      ax=ax4, palette=repcolors, flierprops=dict(marker='.'), showmeans=True, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
    # sns.stripplot(data=shear_modulus_all.T*1e-6, ax=ax1)
    # sns.stripplot(data=young_modulus_all.T*1e-6, ax=ax2)
    # sns.stripplot(data=bulk_modulus_all.T*1e-6, ax=ax1)
    # sns.stripplot(data=poisson_ratio_all.T, ax=ax2)
elif Violinplot:
    sns.violinplot(data=shear_modulus_all.T*1e-6, ax=ax1, palette=repcolors, flierprops=dict(marker='.'), inner=None, showmeans=True)
    sns.violinplot(data=young_modulus_all.T*1e-6, ax=ax2, palette=repcolors, flierprops=dict(marker='.'), inner=None, showmeans=True)
    sns.violinplot(data=bulk_modulus_all.T*1e-6,  ax=ax3, palette=repcolors, flierprops=dict(marker='.'), inner=None, showmeans=True)
    sns.violinplot(data=poisson_ratio_all.T,      ax=ax4, palette=repcolors, flierprops=dict(marker='.'), inner=None, showmeans=True)
else:
    for i, v in enumerate(experiments.values()):
        ax1.scatter([i]*len(v.shear_modulus), v.shear_modulus*1e-6, c=repcolors[i], marker='.')
        ax2.scatter([i]*len(v.young_modulus), v.young_modulus*1e-6, c=repcolors[i], marker='.')
        ax3.scatter([i]*len(v.bulk_modulus),  v.bulk_modulus*1e-6,  c=repcolors[i], marker='.')
        ax4.scatter([i]*len(v.poisson_ratio), v.poisson_ratio, c=repcolors[i], marker='.')
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])
ax1.plot(shear_modulus_means*1e-6, c='k')
ax2.plot(young_modulus_means*1e-6, c='k')
ax3.plot(bulk_modulus_means*1e-6, c='k')
ax4.plot(poisson_ratio_means, c='k')

# Draw batch name and vertical lines as separators
ylim1, ylimmax1 = ax1.get_ylim()
ylim2, ylimmax2 = ax2.get_ylim()
ylim3, ylimmax3 = ax3.get_ylim()
ylim4, ylimmax4 = ax4.get_ylim()
ax1.set_ylim([ylim1 - (ylimmax1 - ylim1)*0.1, ylimmax1])
ax2.set_ylim([ylim2 - (ylimmax2 - ylim2)*0.1, ylimmax2])
ax3.set_ylim([ylim3 - (ylimmax3 - ylim3)*0.1, ylimmax3])
ax4.set_ylim([ylim4 - (ylimmax4 - ylim4)*0.1, ylimmax4])
ylim1, ylimmax1 = ax1.get_ylim()
ylim2, ylimmax2 = ax2.get_ylim()
ylim3, ylimmax3 = ax3.get_ylim()
ylim4, ylimmax4 = ax4.get_ylim()  
for i, batch in enumerate(Batches):
    ax1.text(i*Nspecimens + Nspecimens/2 - 1, ylim1 + (ylimmax1 - ylim1)*0.025, f'{batch} ({concentrations[i]}%)')
    ax2.text(i*Nspecimens + Nspecimens/2 - 1, ylim2 + (ylimmax2 - ylim2)*0.025, f'{batch} ({concentrations[i]}%)')
    ax3.text(i*Nspecimens + Nspecimens/2 - 1, ylim3 + (ylimmax3 - ylim3)*0.025, f'{batch} ({concentrations[i]}%)')
    ax4.text(i*Nspecimens + Nspecimens/2 - 1, ylim4 + (ylimmax4 - ylim4)*0.025, f'{batch} ({concentrations[i]}%)')
    if i != len(Batches) - 1:
        ax1.axvline(i*Nspecimens + Nspecimens - 0.5, c='gray')
        ax2.axvline(i*Nspecimens + Nspecimens - 0.5, c='gray')
        ax3.axvline(i*Nspecimens + Nspecimens - 0.5, c='gray')
        ax4.axvline(i*Nspecimens + Nspecimens - 0.5, c='gray')
plt.tight_layout()

#%% Plot concentration
(ax1, ax2), (ax3, ax4) = plt.subplots(2,2, figsize=(10,8))[1]
ax1.set_xlabel('Graphite concentration (%)')
ax2.set_xlabel('Graphite concentration (%)')
ax3.set_xlabel('Graphite concentration (%)')
ax4.set_xlabel('Graphite concentration (%)')
ax1.set_ylabel('Shear modulus (GPa)')
ax2.set_ylabel('Young modulus (GPa)')
ax3.set_ylabel('Bulk modulus (GPa)')
ax4.set_ylabel('Poisson ratio')

if np.ma.is_masked(shear_modulus_flatperbatch):
    s = shear_modulus_flatperbatch.data
    y = young_modulus_flatperbatch.data
    b = bulk_modulus_flatperbatch.data
    p = poisson_ratio_flatperbatch.data
else:
    s = shear_modulus_flatperbatch
    y = young_modulus_flatperbatch
    b = bulk_modulus_flatperbatch
    p = poisson_ratio_flatperbatch

sns.boxplot(data=s.T*1e-6, ax=ax1, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.boxplot(data=y.T*1e-6, ax=ax2, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.boxplot(data=b.T*1e-6, ax=ax3, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))
sns.boxplot(data=p.T,      ax=ax4, palette=colors, flierprops=dict(marker='.'), showmeans=False, meanprops=dict(marker='.', markerfacecolor='w', markeredgecolor='k', markersize=10))

ax1.plot(shear_modulus_batches_means*1e-6, marker='o', markerfacecolor='w', c='k', zorder=3)
ax2.plot(young_modulus_batches_means*1e-6, marker='o', markerfacecolor='w', c='k', zorder=3)
ax3.plot(bulk_modulus_batches_means*1e-6,  marker='o', markerfacecolor='w', c='k', zorder=3)
ax4.plot(poisson_ratio_batches_means,      marker='o', markerfacecolor='w', c='k', zorder=3)

ax1.set_xticks(np.arange(0, N))
ax2.set_xticks(np.arange(0, N))
ax3.set_xticks(np.arange(0, N))
ax4.set_xticks(np.arange(0, N))
ax1.set_xticklabels(concentrations[:N])
ax2.set_xticklabels(concentrations[:N])
ax3.set_xticklabels(concentrations[:N])
ax4.set_xticklabels(concentrations[:N])

plt.tight_layout()

#%% Plot each batch separately   
for bat, s, y, b, p in zip(Batches, shear_modulus_batches, young_modulus_batches, bulk_modulus_batches, poisson_ratio_batches):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
    fig.suptitle(f'Batch {bat}')
    ax1.set_ylabel('Shear (GPa)')
    ax2.set_ylabel('Young (GPa)')
    ax3.set_ylabel('Bulk (GPa)')
    ax4.set_ylabel('Poisson')
    ax4.set_xlabel('Position (mm)')
    for i in range(Nspecimens):
        v = experiments[f'{bat}{i+1}']
        ax1.plot(v.scanpos, s[i]*1e-6, c=colors[i], lw=2)
        ax2.plot(v.scanpos, y[i]*1e-6, c=colors[i], lw=2)
        ax3.plot(v.scanpos, b[i]*1e-6, c=colors[i], lw=2)
        ax4.plot(v.scanpos, p[i], c=colors[i], lw=2)
        if np.ma.is_masked(s):
            auxs, auxy, auxb, auxp = s[i].copy(), y[i].copy(), b[i].copy(), p[i].copy()
            auxs.mask = ~s[i].mask
            auxy.mask = ~y[i].mask
            auxb.mask = ~b[i].mask
            auxp.mask = ~p[i].mask
            
            ax1.scatter(v.scanpos, auxs*1e-6, c='k', marker='.')
            ax2.scatter(v.scanpos, auxy*1e-6, c='k', marker='.')
            ax3.scatter(v.scanpos, auxb*1e-6, c='k', marker='.')
            ax4.scatter(v.scanpos, auxp, c='k', marker='.')
    plt.tight_layout()            

# Histogram of every batch
shear_modulus_hists = [US.hist(x.flatten(), density=True) for x in shear_modulus_batches] # h, b, width
young_modulus_hists = [US.hist(x.flatten(), density=True) for x in young_modulus_batches]
bulk_modulus_hists = [US.hist(x.flatten(), density=True) for x in bulk_modulus_batches]
poisson_ratio_hists = [US.hist(x.flatten(), density=True) for x in poisson_ratio_batches]
for i, (sh, yh, bh, ph) in enumerate(zip(shear_modulus_hists, young_modulus_hists, bulk_modulus_hists, poisson_ratio_hists)):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,8))
    US.plot_hist(sh[0]*1e6, sh[1]*1e-6, sh[2]*1e-6, ax=ax1, xlabel='Shear modulus (GPa)', ylabel='pdf', edgecolor='k')
    US.plot_hist(yh[0]*1e6, yh[1]*1e-6, yh[2]*1e-6, ax=ax2, xlabel='Young modulus (GPa)', ylabel='pdf', edgecolor='k')
    US.plot_hist(bh[0]*1e6, bh[1]*1e-6, bh[2]*1e-6, ax=ax3, xlabel='Bulk modulus (GPa)', ylabel='pdf', edgecolor='k')
    US.plot_hist(*ph, ax=ax4, xlabel='Poisson ratio', ylabel='pdf', edgecolor='k')
    ax1.plot(shear_modulus_batches_aux[i]*1e-6, shear_modulus_batches_gauss[i]*1e6, c='r')
    ax2.plot(young_modulus_batches_aux[i]*1e-6, young_modulus_batches_gauss[i]*1e6, c='r')
    ax3.plot(bulk_modulus_batches_aux[i]*1e-6, bulk_modulus_batches_gauss[i]*1e6, c='r')
    ax4.plot(poisson_ratio_batches_aux[i], poisson_ratio_batches_gauss[i], c='r')
    fig.suptitle(f'Batch {Batches[i]}')
    plt.tight_layout()

