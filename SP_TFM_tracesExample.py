# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:42:24 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
from matplotlib.widgets import CheckButtons
import os
import scipy.signal as scsig
import seaborn as sns

import src.ultrasound as US

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
        If True, print something every time an experiment is finished. Default
        is False.
    **kwargs : keyword args
        Keyword arguments for RealtimeSP's class constructor.
        
    Returns
    -------
    experiments : dict{RealtimeSP}
        Dictionary containing all the experiments.

    Arnau, 16/05/2023
    '''
    experiments = {}
    for e in Names:
        ExperimentName = os.path.join(Path, e)
        experiments[e] = US.RealtimeSP(ExperimentName, **kwargs)
        if Verbose: print(f'Experiment {e} done.')
    return experiments

#%% Load data
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Deposition'
Names = ['Rb0_20_M_rt1']
experiments = loadExperiments(Path, Names, Verbose=True, Cw_material='resin')
e = experiments['Rb0_20_M_rt1']
Time_axis = np.arange(0, len(e.WPraw))/e.Fs


#%%
ax1, ax2, ax3 = plt.subplots(3)[1]
ax1.set_ylabel('PE (V)')
ax2.set_ylabel('TT (V)')
ax3.set_ylabel('WP (V)')
ax3.set_xlabel('Time ($\mu$s)')

ax1.plot(Time_axis*1e6, e.PEraw[:,0], c='k', lw=1)
ax2.plot(Time_axis*1e6, e.TTraw[:,0], c='k', lw=1)
ax3.plot(Time_axis*1e6, e.WPraw, c='k', lw=1)

ax1.set_ylim([-0.5, 0.5])
ax2.set_ylim([-0.5, 0.5])
ax3.set_ylim([-0.5, 0.5])

ax1.set_yticks([-0.5, 0, 0.5])
ax2.set_yticks([-0.5, 0, 0.5])
ax3.set_yticks([-0.5, 0, 0.5])

ax1.set_xticklabels([])
ax2.set_xticklabels([])


ax1ins = ax1.inset_axes([0.65, 0.1, 0.3, 0.3])
ax1ins.plot(Time_axis*1e6, e.PEraw[:,0], c='k', lw=1)
ax1ins.set_xlim([34, 39])
ax1ins.set_ylim([-0.02, 0.02])
ax1.indicate_inset_zoom(ax1ins, edgecolor="gray")

ax2ins = ax2.inset_axes([0.65, 0.1, 0.3, 0.3])
ax2ins.plot(Time_axis*1e6, e.TTraw[:,0], c='k', lw=1)
ax2ins.set_xlim([28, 32])
ax2ins.set_ylim([-0.05, 0.05])
ax2.indicate_inset_zoom(ax2ins, edgecolor="gray")

ax1ins.set_yticks([])
ax2ins.set_yticks([])
ax1ins.set_xticks([])
ax2ins.set_xticks([])

plt.tight_layout()