# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:30:30 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
from matplotlib.widgets import CheckButtons
import os
import scipy.signal as scsig
import seaborn as sns
import matplotlib.colors as mcolors
colors = list(mcolors.TABLEAU_COLORS) # 10 colors
colors.append('k')
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

def LPF2mHz(x, Fs):
    '''
    Filter a signal with a Low-Pass IIR filter of order 2 and cutoff frequency
    2 mHz.

    Parameters
    ----------
    x : ndarray
        Input signal.
    Fs : float
        Sampling frequency in Hz.
    
    Returns
    -------
    x_lpf : ndarray
        The filtered signal.

    Arnau, 16/05/2023
    '''
    lpf_order = 2
    lpf_fc = 2e-3 # Hz
    if Fs < 2*lpf_fc:
        print(f'Signal does not have frequency components beyond {lpf_fc*1e3} mHz, therefore it is not filtered.')
        return x
    else:
        b_IIR, a_IIR = scsig.iirfilter(lpf_order, 2*lpf_fc/Fs, btype='lowpass')
        x_lpf = scsig.filtfilt(b_IIR, a_IIR, x)
        return x_lpf


#%% Load data
Path = r'..\Data\Deposition'
Names_ns = ['Rb0_0_M_rt1', 'Rb01_0_M_rtw', 'Rb02_0_M_rtw', 'Rb03_0_M_rtw', 'Rb04_0_M_rtw', 'Rb05_0_M_rtw', 'Rb06_0_M_rtw', 'Rb07_0_M_rtw', 'Rb08_0_M_rtw', 'Rb09_0_M_rtw', 'Rb10_0_M_rtw']
Names_sonic = ['Rb0_20_M_rt1', 'Rb01_20_M_rt', 'Rb02_20_M_rt', 'Rb03_20_M_rt', 'Rb04_20_M_rt', 'Rb05_20_M_rt', 'Rb06_20_M_rt', 'Rb07_20_M_rt', 'Rb08_20_M_rt', 'Rb09_20_M_rt', 'Rb10_20_M_rt']

Names = Names_sonic + Names_ns
concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lbls = [f'{c} wt%' for c in concentrations]
experiments = loadExperiments(Path, Names, Verbose=True, Cw_material='resin')
Time_axis = np.arange(0, len(experiments[list(experiments.keys())[0]].C))*experiments[list(experiments.keys())[0]].Ts/60




c_ = ['k','r']
repcolors = []
for i in range(2):
    repcolors.extend([c_[i]]*int(len(Names)/2))


#%% Outlier detection
UseMedian = False
m_c = 0.675 # Outlier detection threshold for C
m_l = 0.675 # Outlier detection threshold for L

Call = np.array([e.C for e in experiments.values()]) # Size of this array is: len(experiments) x N_acqs
Lall = np.array([e.L for e in experiments.values()]) # Size of this array is: len(experiments) x N_acqs
L_lpf = np.array([LPF2mHz(e.L, 1/e.Ts) for e in experiments.values()])
Call_masked = US.maskOutliers(Call, m=m_c, UseMedian=UseMedian)
Lall_masked = US.maskOutliers(Lall, m=m_l, UseMedian=UseMedian)

_tempLall = [l for l in Lall_masked]
_tempCall = [l for l in Call_masked]

Call_without_outliers = US.apply2listElements(_tempCall, lambda x: x.compressed())
Lall_without_outliers = US.apply2listElements(_tempLall, lambda x: x.compressed())

L_lpf_from_masked = [LPF2mHz(l, 1/e.Ts) for l,e in zip(Lall_without_outliers, experiments.values())]

Call_diffs = np.array([e.C - US.temp2sos(e.temperature_lpf, material='resin') for e in experiments.values()]) # Size of this array is: len(experiments) x N_acqs
Call_masked_diffs = US.maskOutliers(Call_diffs, m=m_c, UseMedian=UseMedian)
Call_without_outliers_diffs = Call_masked_diffs.compressed()

Lall_without_outliers_flat = np.array([elem for l in Lall_without_outliers for elem in l])

#%% Plot
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)
fig4, ax4 = plt.subplots(1)
fig5, ax5 = plt.subplots(1)
fig6, ax6 = plt.subplots(1)
ax1.set_xlabel('Time (min)')
ax2.set_xlabel('Time (min)')
ax3.set_xlabel('Time (min)')
ax4.set_xlabel('Time (min)')
ax5.set_xlabel('Time (min)')
ax6.set_xlabel('Time (min)')

ax1.set_ylabel('Velocity (m/s)')
ax2.set_ylabel("Temperature model's velocity (m/s)")
ax3.set_ylabel('Differential velocity (m/s)')
ax4.set_ylabel('Thickness ($\mu$m)')
ax5.set_ylabel('Diameter (mm)')
ax6.set_ylabel('Temperature (celsius)')

for i,e in enumerate(experiments.values()):
    if i==0:
        lbl = 'with sonication'
    elif i==len(experiments)-1:
        lbl = 'without sonication'
    else:
        lbl = ''
    L_lpf_ = LPF2mHz(e.L, 1/e.Ts)
    C_lpf = LPF2mHz(e.C, 1/e.Ts)
    c = e.C - US.temp2sos(e.temperature_lpf, material='resin')
    c_lpf = C_lpf - US.temp2sos(e.temperature_lpf, material='resin')
    
    # _p = ax1.plot(Time_axis, e.C, c=repcolors[i], alpha=0.4)
    ax1.plot(Time_axis, C_lpf, label=lbl, c=repcolors[i], alpha=1, zorder=3)    
    
    ax2.plot(Time_axis, e.Cw_lpf, c=repcolors[i], label=lbl)
    
    # _p = ax3.plot(Time_axis, c, c=repcolors[i], alpha=0.4)
    ax3.plot(Time_axis, c_lpf, c=repcolors[i], label=lbl, alpha=1, zorder=3)
    
    ax4.plot(Time_axis, e.Lc*1e6, c=repcolors[i], label=lbl)
    
    # _p = ax5.plot(Time_axis, e.L*1e3, c=repcolors[i], alpha=0.4)
    ax5.plot(Time_axis, L_lpf_*1e3, label=lbl, c=repcolors[i], alpha=1, zorder=3)
    
    ax6.plot(Time_axis, e.temperature_lpf, c=repcolors[i], label=lbl)

ax2.legend()
plt.tight_layout()
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()

US.movefig('northwest', fig1)
US.movefig('northeast', fig2)
US.movefig('southwest', fig3)
US.movefig('southeast', fig4)
US.movefig('south', fig5)
US.movefig('north', fig6)










#%%
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)
fig4, ax4 = plt.subplots(1)
fig5, ax5 = plt.subplots(1)
fig6, ax6 = plt.subplots(1)
ax1.set_xlabel('Time (min)')
ax2.set_xlabel('Time (min)')
ax3.set_xlabel('Time (min)')
ax4.set_xlabel('Time (min)')
ax5.set_xlabel('Time (min)')
ax6.set_xlabel('Time (min)')

ax1.set_ylabel('Velocity (m/s)')
ax2.set_ylabel("Temperature model's velocity (m/s)")
ax3.set_ylabel('Differential velocity (m/s)')
ax4.set_ylabel('Thickness ($\mu$m)')
ax5.set_ylabel('Diameter (mm)')
ax6.set_ylabel('Temperature ($^\circ$C)')

for i,e in enumerate(experiments.values()):
    if i==0:
        lbl = 'with sonication'
    elif i==len(experiments)-1:
        lbl = 'without sonication'
    else:
        lbl = ''
    L_lpf_ = LPF2mHz(e.L, 1/e.Ts)
    C_lpf = LPF2mHz(e.C, 1/e.Ts)
    
    cw = US.temp2sos(e.temperature_lpf, material='resin', deg=1)
    c = e.C - cw
    c_lpf = C_lpf - cw
    
    # _p = ax1.plot(Time_axis, e.C, c=repcolors[i], alpha=0.4)
    ax1.plot(Time_axis, C_lpf, label=lbl, c=repcolors[i], alpha=1, zorder=3)    
    
    ax2.plot(Time_axis, e.Cw_lpf, c=repcolors[i], label=lbl)
    
    # _p = ax3.plot(Time_axis, c, c=repcolors[i], alpha=0.4)
    ax3.plot(Time_axis, c_lpf, c=repcolors[i], label=lbl, alpha=1, zorder=3)
    
    ax4.plot(Time_axis, e.Lc*1e6, c=repcolors[i], label=lbl)
    
    # _p = ax5.plot(Time_axis, e.L*1e3, c=repcolors[i], alpha=0.4)
    ax5.plot(Time_axis, L_lpf_*1e3, label=lbl, c=repcolors[i], alpha=1, zorder=3)
    
    ax6.plot(Time_axis, e.temperature_lpf, c=repcolors[i], label=lbl)

ax2.legend()
plt.tight_layout()
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()

US.movefig('northwest', fig1)
US.movefig('northeast', fig2)
US.movefig('southwest', fig3)
US.movefig('southeast', fig4)
US.movefig('south', fig5)
US.movefig('north', fig6)