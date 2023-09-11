# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:47:22 2023
Python version: Python 3.8

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


# ax1, ax2, _ = US.histGUI(Time_axis, C, xlabel='Time (h)', ylabel='Velocity (m/s)')


#%% Load experiments
Path = r'..\Data\Deposition'
Names = ['W0_0', 'W0_0_2', 'W0_0_3', 'W0_0_4', 'W0_0_5', 'W0_0_6', 
         'W0_0_7a', 'W0_0_7b', 'W0_0_7c', 'W0_0_7d', 'W0_0_7e', 'W0_0_7f', 'W0_0_7g', 'W0_0_7h', 'W0_0_7i', 
         'W01_0', 'W02_0', 'W03_0', 'W05_0', 'W06_0', 'W07_0', 'W08_0', 'W09_0', 'W10_0']
Names = ['W0_0', 'W01_0', 'W02_0', 'W03_0', 'W05_0', 'W06_0', 'W07_0', 'W08_0', 'W09_0', 'W10_0']
Names = ['W0_0_M', 'W01_0_M', 'W02_0_M', 'W03_0_M', 'W04_0_M', 'W05_0_M', 'W06_0_M', 'W07_0_M', 'W08_0_M', 'W09_0_M', 'W10_0_M']
Names = ['W0_0_M', 'W01_0_M', 'W02_0_M', 'W03_0_M', 'W04_0_M', 'W05_0_M', 'W06_0_M', 'W07_0_M', 'W08_0_M', 'W09_0_M', 'W10_0_M', 'W20_0_M']
Names = ['W0_20_M', 'W0_20_M_2', 'W01_20_M', 'W02_20_M', 'W03_20_M', 'W20_20_M']
Names = ['W0_0_M_F', 'W01_0_M_F', 'W02_0_M_F', 'W03_0_M_F']
Names = ['R0_0_M', 'R0_0_M_2', 'R0_0_M_3', 'R0_0_M_4', 'R0_0_M_5']
Names = ['R0_20_M', 'R0_20_M_2', 'R0_20_M_3']
Names = ['R0_0_M_t1', 'R0_0_M_t2', 'R0_20_M', 'R0_20_M_2', 'R0_20_M_3', 'R10_20_M']
Names = ['R0_0_M_rt1m', 'R0_0_M_rt2m', 'R0_0_M_rt3m', 'R0_0_M_rt4m']
Names = ['R0_0_M_rt1m', 'R0_0_M_rt2m', 'R0_0_M_rt3m', 'R0_0_M_rt4m', 'R0_20_M_rt1', 'R0_0_M_rt1', 'R0_0_M_rt2', 'Rb0_0_M_rt1', 'Rb0_0_M_rt2', 'Rb0_0_M_rt3', 'Rb0_0_M_rt4', 'Rb0_20_M_rt1']
# Names = ['R0_0_M_rt1m', 'R0_0_M_rt2m', 'R0_0_M_rt3m', 'R0_0_M_rt4m', 'R0_0_M_rt1', 'R0_0_M_rt2', 'Rb0_0_M_rt1', 'Rb0_0_M_rt2', 'Rb0_0_M_rt3', 'Rb0_0_M_rt4']
# Names = ['R0_0_M_rt1', 'R0_0_M_rt2', 'R0_20_M_rt1']
# Names = ['Rb0_0_M_rt1', 'Rb0_0_M_rt2', 'Rb0_0_M_rt3', 'Rb0_20_M_rt1', 'Rb01_20_M_rt', 'Rb02_20_M_rt', 'Rb03_20_M_rt', 'Rb04_20_M_rt', 'Rb05_20_M_rt', 'Rb06_20_M_rt', 'Rb07_20_M_rt', 'Rb08_20_M_rt', 'Rb09_20_M_rt', 'Rb10_20_M_rt']
# Names = ['Rb0_20_M_rt1', 'Rb0_20_M_rt2', 'Rb01_20_M_rt', 'Rb02_20_M_rt', 'Rb03_20_M_rt', 'Rb04_20_M_rt', 'Rb05_20_M_rt', 'Rb06_20_M_rt', 'Rb07_20_M_rt', 'Rb08_20_M_rt', 'Rb09_20_M_rt', 'Rb10_20_M_rt']
# Names += Names2
# Names = ['Rb0_0_M_rt1', 'Rb01_0_M_rtw', 'Rb02_0_M_rtw', 'Rb03_0_M_rtw', 'Rb04_0_M_rtw', 'Rb05_0_M_rtw', 'Rb06_0_M_rtw', 'Rb07_0_M_rtw', 'Rb08_0_M_rtw', 'Rb09_0_M_rtw', 'Rb10_0_M_rtw']
# Names = ['Rb0_0_M_rt_vh', 'Rb0_0_M_rt_vh2']
concentrations = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Cw20 = US.temp2sos(20, material='water')
experiments = loadExperiments(Path, Names, Verbose=True, Cw_material='resin')
# Error in speed of sound will be at least 0.1 m/s due to the temperature resolution of 0.032 ºC
# Prediction error in resin temperature model is +/- 4.52 m/s
#%%
Cc = 2732 # methacrylate
Cc = 2726 # methacrylate (small container)
# Cc = 2327 # plastic
# Cc = 5490 # glass
cws = np.zeros(len(experiments))
for i,k in enumerate(Names):
    print(f'--------- {k} ---------')
    if 'M' in k and 'W' in k:
        experiments[k].windowAscans(Loc_WP=3200, Loc_TT=2970, Loc_PER=1180, Loc_PETR=5500, 
                                    WinLen_WP=1000, WinLen_TT=1000, WinLen_PER=1300, WinLen_PETR=1000)
    elif 'M' in k and 'R' in k:
        experiments[k].windowAscans(Loc_WP=3200, Loc_TT=2850, Loc_PER=1180, Loc_PETR=3650, 
                                    WinLen_WP=1000, WinLen_TT=1000, WinLen_PER=1000, WinLen_PETR=1000)
    else:
        experiments[k].windowAscans()
    experiments[k].computeTOF(windowXcor=False, correction=True, filter_tofs_pe=True, UseHilbEnv=False)
    
    cws[i] = US.temp2sos(experiments[k].config_dict['WP_temperature'], material='water') if 'rt' in k else None
    if 'rt' in k: experiments[k].LPFtemperature()
    
    experiments[k].computeResults(Cc=Cc, charac_container=False, cw=cws[i])
    
    experiments[k].saveResults()
    experiments[k].saveCw()
    
#%%
plt.figure()
for e in experiments.values():
    plt.plot(e.temperature_lpf)

#%% Plotting
# for e in experiments.values():
#     e.computeResults(Cc=Cc, charac_container=False, cw=experiments[list(experiments.keys())[0]].Cw[0])
#     # e.computeResults(Cc=Cc, charac_container=False)

# -----------------
# Outlier detection
# -----------------
UseMedian = False
m_c = 0.6745 # Outlier detection threshold for C
m_l = 0.6745 # Outlier detection threshold for L

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

# --------
# Plotting
# --------
fig, axs = plt.subplots(2, 2, figsize=(10,8))
axs[0,0].set_xlabel('Time (min)')
axs[0,1].set_xlabel('Time (min)')
axs[1,0].set_xlabel('Time (min)')
axs[1,1].set_xlabel('Time (min)')

axs[0,0].set_ylabel('Velocity (m/s)')
axs[0,1].set_ylabel('Cw_lpf (m/s)')
axs[1,0].set_ylabel('Thickness (um)')
axs[1,1].set_ylabel('Diameter (mm)')

Time_axis = np.arange(0, len(experiments[list(experiments.keys())[0]].C))*experiments[list(experiments.keys())[0]].Ts/60
idx = np.where(Time_axis<=15)[0][-1]

plot_diffs = True
for e in experiments.values():
    L_lpf_ = LPF2mHz(e.L, 1/e.Ts)
    c = e.C - US.temp2sos(e.temperature_lpf, material='resin') if plot_diffs else e.C
    axs[0,0].plot(Time_axis, c, label=e.name)
    axs[0,1].plot(Time_axis, e.Cw_lpf, label=e.name)
    axs[1,0].plot(Time_axis, e.Lc*1e6, label=e.name)
    _p = axs[1,1].plot(Time_axis, e.L*1e3, label=e.name, alpha=0.5)
    axs[1,1].plot(Time_axis, L_lpf_*1e3, label=e.name, c=_p[-1].get_color(), alpha=1, zorder=3)
axs[0,1].legend()
plt.tight_layout()

rax = fig.add_axes([0.093, 0.9665, 0.16, 0.03])
check = CheckButtons(
    ax=rax,
    labels=['Remove Outliers'],
    actives=[False],
)

def callback(label):
    axs[0,0].clear()
    axs[0,0].set_xlabel('Time (min)')
    axs[0,0].set_ylabel('Velocity (m/s)')
    
    axs[1,1].clear()
    axs[1,1].set_xlabel('Time (min)')
    axs[1,1].set_ylabel('Diameter (mm)')
    
    _check = check.get_status()[0]
    if (not _check) and (not plot_diffs):
        data = zip(Call, Lall, L_lpf)
    elif (not _check) and plot_diffs:
        data = zip(Call_diffs, Lall, L_lpf)
    elif _check and not plot_diffs:
        data = zip(Call_masked, Lall_masked, L_lpf)
    elif _check and plot_diffs:
        data = zip(Call_masked_diffs, Lall_masked, L_lpf)
    
    for i,(c, l, l_lpf) in enumerate(data):
        axs[0,0].plot(Time_axis, c, label=Names[i])
        _p = axs[1,1].plot(Time_axis, l*1e3, label=Names[i], alpha=0.5)
        axs[1,1].plot(Time_axis, l_lpf*1e3, label=Names[i], c=_p[-1].get_color(), alpha=1, zorder=3)
    axs[0,1].legend()
    fig.canvas.draw_idle()

check.on_clicked(callback)

plt.show()

#%% Violin plot
idx = np.where(Time_axis<=5)[0][-1]

ax = plt.subplots(1, figsize=(10,8))[1]
ax.set_xlabel('Mass concentration (%)')
ax.set_ylabel('Velocity (m/s)')

# C_violin = np.array([e.Cw_lpf[:idx] for e in experiments.values()]) 
# C_violin = np.array([e.C[:idx] - US.temp2sos(e.temperature_lpf, material='resin')[:idx] - ref[:idx] for e in experiments.values()]) 

# C_violin = np.array([e.Cw_lpf[-idx:] for e in experiments.values()]) 
C_violin = np.array([e.C[-idx:] - US.temp2sos(e.temperature_lpf, material='resin')[-idx:] - ref[-idx:] for e in experiments.values()]) 

# sns.violinplot(data=C_violin.T, ax=ax, flierprops=dict(marker='.'), inner=None, showmeans=True)
sns.boxplot(data=C_violin.T, ax=ax, flierprops=dict(marker='.'))
ax.set_xticklabels(concentrations);

plt.tight_layout()


#%%
ax1, ax2, ax3 = plt.subplots(3, figsize=(10,8))[1]
ax1.set_xlabel('Time (min)')
ax2.set_xlabel('Time (min)')
ax3.set_xlabel('Time (min)')

ax1.set_ylabel('Corrected velocity (m/s)')
ax2.set_ylabel('Uncorrected Velocity (m/s)')
ax3.set_ylabel('Cw (m/s)')

for e in experiments.values():
    ctemp = US.temp2sos(e.temperature_lpf, material='resin')
    ax1.plot(Time_axis, e.C - ctemp, label=e.name)
    ax2.plot(Time_axis, e.C, label=e.name)
    ax3.plot(Time_axis, e.Cw_lpf, label=e.name)
ax1.set_xlim([0,20])
ax2.set_xlim([0,20])
ax3.set_xlim([0,20])
ax1.legend(loc='upper right')
plt.tight_layout()



#%% FFT
UseMean = False
lims = [2750, 3070]
drawlims = [0,3] # MHz

N = len(Names)
cm = plt.get_cmap('gist_rainbow')

ax1 = plt.subplots(1)[1]
ax1.set_xlim(drawlims)
ax1.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('|$\cal{F}$(TT)|$^2$')
ax1.set_prop_cycle(color=[cm(1.*i/N) for i in range(N)])

Fs = experiments[list(experiments.keys())[0]].Fs
nfft = 2**13
f = np.linspace(0, Fs/2, nfft)
fstep = f[1] - f[0]
idx_inf = int(drawlims[0]*1e6 // fstep)
idx_sup = int(drawlims[1]*1e6 // fstep)
maxs = np.zeros(N)
idxs_maxs = np.zeros(N)
for i,(k,v) in enumerate(experiments.items()):
    if UseMean:
        tts = v.TTraw[lims[0]:lims[1],:]
        tt_fft = np.fft.fft(tts, nfft, axis=0).mean(axis=1)
    else:
        tts = v.TTraw[lims[0]:lims[1],0]
        tt_fft = np.fft.fft(tts, nfft)
    
    maxs[i] = max(np.abs(tt_fft[idx_inf:idx_sup])**2)
    idxs_maxs[i] = np.where(np.abs(tt_fft[idx_inf:idx_sup])**2==maxs[i])[0][0]
    ax1.plot(f*1e-6, np.abs(tt_fft)**2, label=k)

plt.legend()
plt.tight_layout()
#%% Only 1 experiment
e = experiments[list(experiments.keys())[-1]] # select experiment


lims = [2750, 3070]
drawlims = [0,3] # MHz

N = e.N_acqs
cm = plt.get_cmap('gist_rainbow')

ax1 = plt.subplots(1)[1]
ax1.set_xlim(drawlims)
ax1.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('|$\cal{F}$(TT)|$^2$')
ax1.set_prop_cycle(color=[cm(1.*i/N) for i in range(N)])

Fs = e.Fs
nfft = 2**13
f = np.linspace(0, Fs/2, nfft)
fstep = f[1] - f[0]
idx_inf = int(drawlims[0]*1e6 // fstep)
idx_sup = int(drawlims[1]*1e6 // fstep)
maxs = np.zeros(N)
idxs_maxs = np.zeros(N)

ttraw = e.TTraw[lims[0]:lims[1],:]

for i,tt in enumerate(ttraw.T):
    tt_fft = np.fft.fft(tt, nfft)
    
    maxs[i] = max(np.abs(tt_fft[idx_inf:idx_sup])**2)
    idxs_maxs[i] = np.where(np.abs(tt_fft[idx_inf:idx_sup])**2==maxs[i])[0][0]
    ax1.plot(f*1e-6, np.abs(tt_fft)**2)

plt.tight_layout()
#%%
ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Maxs')
ax2.set_ylabel('Freqs (MHz)')

ax1.plot(maxs)
ax2.plot(idxs_maxs*fstep*1e-6)

plt.tight_layout()
#%%
cw_firsts = [exp.Cw_lpf[0] for exp in experiments.values()]
ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Maxs')
ax2.set_ylabel('Freqs (MHz)')

ax1.plot(maxs)
ax2.plot(idxs_maxs*fstep*1e-6)


ax1twin = ax1.twinx()
ax1twin.set_ylabel('Cw[0] (m/s)', c='r')
ax1twin.plot(cw_firsts, c='r')

ax2twin = ax2.twinx()
ax2twin.set_ylabel('Cw[0] (m/s)', c='r')
ax2twin.plot(cw_firsts, c='r')

plt.tight_layout()
#%% maxs and idxs_maxs for all experiments
lims = [2750, 3070]
drawlims = [0,3] # MHz
nfft = 2**13
maxs = []
idxs_maxs = []
for e in experiments.values():
    N = e.N_acqs
    Fs = e.Fs
    f = np.linspace(0, Fs/2, nfft)
    fstep = f[1] - f[0]
    idx_inf = int(drawlims[0]*1e6 // fstep)
    idx_sup = int(drawlims[1]*1e6 // fstep)
    ttraw = e.TTraw[lims[0]:lims[1],:]
    _temp_maxs = np.zeros(N)
    _temp_idxs_maxs = np.zeros(N)
    for i,tt in enumerate(ttraw.T):
        tt_fft = np.fft.fft(tt, nfft)
        _temp_maxs[i] = max(np.abs(tt_fft[idx_inf:idx_sup])**2)
        _temp_idxs_maxs[i] = np.where(np.abs(tt_fft[idx_inf:idx_sup])**2==_temp_maxs[i])[0][0]
    maxs.append(_temp_maxs)
    idxs_maxs.append(_temp_idxs_maxs)

maxs = np.array(maxs)
idxs_maxs = np.array(idxs_maxs)

#%%
N = len(Names)
cm = plt.get_cmap('gist_rainbow')
ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Maxs')
ax2.set_ylabel('Freqs (MHz)')
ax1.set_prop_cycle(color=[cm(1.*i/N) for i in range(N)])
ax2.set_prop_cycle(color=[cm(1.*i/N) for i in range(N)])

midpoints = np.zeros(N)
for i,(m,idx) in enumerate(zip(maxs, idxs_maxs)):
    ax1.plot(m)
    ax2.plot(idx*fstep*1e-6, label=Names[i])
    midpoints[i] = idx[np.where(US.find_nearest(m, m.max()/2))[0][0]]

plt.legend()
plt.tight_layout()


ax1 = plt.subplots(1)[1]
ax1.set_ylabel('midpoints')
ax1.plot(midpoints*fstep*1e-6)


#%% Lc to Cc
ccdiffs = np.zeros(len(experiments))
ax1 = plt.subplots(1)[1]
for i,e in enumerate(experiments.values()):
    cc = 2*e.Fs*np.mean(e.Lc)/e.ToF_R21
    
    ccmax = np.max(cc)
    ccmin = np.min(cc)
    ccdiffs[i] = ccmax - ccmin
    
    ax1.plot(Time_axis, cc)
print(np.max(ccdiffs))

#%%
ax2, ax3 = plt.subplots(2)[1]
for i,e in enumerate(experiments.values()):
    
    cw = US.temp2sos(e.config_dict['WP_temperature'], material='water')
    
    e.computeResults(Cc=Cc, cw=cw)
    C_ = e.C
    e.computeResults(Cc=Cc+30, cw=cw)
    
    ax2.plot(Time_axis, C_, c='k')
    ax2.plot(Time_axis, e.C, c='r')
    
    ax3.plot(Time_axis, C_ - e.C)
plt.tight_layout()

#%%
Time_axis = np.arange(0, len(experiments[list(experiments.keys())[0]].C))*experiments[list(experiments.keys())[0]].Ts/60

plt.figure()
for i,e in enumerate(experiments.values()):
    c = e.C - US.temp2sos(e.temperature_lpf, material='resin')
    
    c = np.mean(c[:int(10/e.Ts*60)])
    
    
    plt.scatter(i, c, label=e.name)


















































#%% Plot ToFs
figtofs, axstofs = plt.subplots(2, 2, figsize=(10,8))
axstofs[0,0].set_xlabel('Time (h)')
axstofs[0,1].set_xlabel('Time (h)')
axstofs[1,0].set_xlabel('Time (h)')
axstofs[1,1].set_xlabel('Time (h)')

axstofs[0,0].set_ylabel('ToF_TW')
axstofs[0,1].set_ylabel('tofs_pe_lpf')
axstofs[1,0].set_ylabel('ToF_R21')
axstofs[1,1].set_ylabel('ToF_TR1R2')

for e in experiments.values():
    axstofs[0,0].plot(np.arange(0, len(e.C))*e.Ts/3600, np.abs(e.ToF_TW), label=e.name)
    # axstofs[0,1].plot(np.arange(0, len(e.C))*e.Ts/3600, e.ToF_TRW[1], label=e.name)
    axstofs[0,1].plot(np.arange(0, len(e.C))*e.Ts/3600, e.tofs_pe_lpf, label=e.name)
    axstofs[1,0].plot(np.arange(0, len(e.C))*e.Ts/3600, e.ToF_R21, label=e.name)
    axstofs[1,1].plot(np.arange(0, len(e.C))*e.Ts/3600, e.ToF_TR1R2, label=e.name)

plt.tight_layout()

#%%
N_acqs = experiments['R0_0_M_rt1'].N_acqs
temps = np.c_[experiments['R0_0_M_rt1m'].temperature_lpf, 
              experiments['R0_0_M_rt2m'].temperature_lpf, 
              experiments['R0_0_M_rt3m'].temperature_lpf, 
              experiments['R0_0_M_rt4m'].temperature_lpf]
vels = np.c_[experiments['R0_0_M_rt1m'].C, 
              experiments['R0_0_M_rt2m'].C, 
              experiments['R0_0_M_rt3m'].C, 
              experiments['R0_0_M_rt4m'].C]

temps = np.c_[experiments['R0_0_M_rt1m'].temperature_lpf, 
              experiments['R0_0_M_rt2m'].temperature_lpf, 
              experiments['R0_0_M_rt3m'].temperature_lpf, 
              experiments['R0_0_M_rt4m'].temperature_lpf,
              experiments['R0_0_M_rt1'].temperature_lpf,
              experiments['R0_0_M_rt2'].temperature_lpf,
              experiments['Rb0_0_M_rt1'].temperature_lpf,
              experiments['Rb0_0_M_rt2'].temperature_lpf,
              experiments['Rb0_0_M_rt3'].temperature_lpf]
vels = np.c_[experiments['R0_0_M_rt1m'].C, 
              experiments['R0_0_M_rt2m'].C, 
              experiments['R0_0_M_rt3m'].C, 
              experiments['R0_0_M_rt4m'].C,
              experiments['R0_0_M_rt1'].C,
              experiments['R0_0_M_rt2'].C,
              experiments['Rb0_0_M_rt1'].C,
              experiments['Rb0_0_M_rt2'].C,
              experiments['Rb0_0_M_rt3'].C]


# temps = np.c_[experiments['R0_0_M_rt1m'].temperature_lpf, 
#               experiments['R0_0_M_rt2m'].temperature_lpf, 
#               experiments['R0_0_M_rt3m'].temperature_lpf, 
#               experiments['R0_0_M_rt4m'].temperature_lpf,
#               experiments['R0_0_M_rt1'].temperature_lpf,
#               experiments['R0_0_M_rt2'].temperature_lpf,
#               experiments['Rb0_0_M_rt1'].temperature_lpf,
#               experiments['Rb0_0_M_rt2'].temperature_lpf,
#               experiments['Rb0_0_M_rt3'].temperature_lpf,
#               experiments['Rb0_0_M_rt4'].temperature_lpf]
# vels = np.c_[experiments['R0_0_M_rt1m'].C, 
#               experiments['R0_0_M_rt2m'].C, 
#               experiments['R0_0_M_rt3m'].C, 
#               experiments['R0_0_M_rt4m'].C,
#               experiments['R0_0_M_rt1'].C,
#               experiments['R0_0_M_rt2'].C,
#               experiments['Rb0_0_M_rt1'].C,
#               experiments['Rb0_0_M_rt2'].C,
#               experiments['Rb0_0_M_rt3'].C,
#               experiments['Rb0_0_M_rt4'].C]

# temps2 = np.c_[experiments['Rb0_20_M_rt1'].temperature_lpf, 
#               experiments['Rb01_20_M_rt'].temperature_lpf, 
#               experiments['Rb02_20_M_rt'].temperature_lpf, 
#               experiments['Rb03_20_M_rt'].temperature_lpf,
#               experiments['Rb04_20_M_rt'].temperature_lpf,
#               experiments['Rb05_20_M_rt'].temperature_lpf,
#               experiments['Rb06_20_M_rt'].temperature_lpf,
#               experiments['Rb07_20_M_rt'].temperature_lpf,
#               experiments['Rb08_20_M_rt'].temperature_lpf,
#               experiments['Rb09_20_M_rt'].temperature_lpf,
#               experiments['Rb10_20_M_rt'].temperature_lpf]
# vels2 = np.c_[experiments['Rb0_20_M_rt1'].C, 
#               experiments['Rb01_20_M_rt'].C, 
#               experiments['Rb02_20_M_rt'].C, 
#               experiments['Rb03_20_M_rt'].C,
#               experiments['Rb04_20_M_rt'].C,
#               experiments['Rb05_20_M_rt'].C,
#               experiments['Rb06_20_M_rt'].C,
#               experiments['Rb07_20_M_rt'].C,
#               experiments['Rb08_20_M_rt'].C,
#               experiments['Rb09_20_M_rt'].C,
#               experiments['Rb10_20_M_rt'].C]


# --------
# Outliers
# --------
vels_mask = np.zeros_like(vels, dtype=bool)
vels_mask[4,3] = 1
vels_mask[16,3] = 1
vels_mask[81,3] = 1
vels_mask[3,2] = 1
vels_mask[4,2] = 1
vels_mask[5,2] = 1
vels_mask[18,2] = 1
vels_mask[129,2] = 1
vels_mask[122,8] = 1
masked_vels = np.ma.masked_array(vels, mask=vels_mask)
masked_temps = np.ma.masked_array(temps, mask=vels_mask)

cf = US.CurveFit(masked_temps.flatten(), masked_vels.flatten(), [1820,-1,-0.1, -0.01], func=3, errorfunc='L2', nparams=4, npredictions=10_000)
cfu = US.CurveFit(temps.flatten(), cf.u, [1820,-1,-0.1, -0.01], func=3, errorfunc='L2', nparams=4, npredictions=10_000)
cfl = US.CurveFit(temps.flatten(), cf.l, [1820,-1,-0.1, -0.01], func=3, errorfunc='L2', nparams=4, npredictions=10_000)
temp_model = np.arange(np.min(temps), np.max(temps), 0.001)
model_u = cfu.func(cfu.params_opt, temp_model)
model_l = cfl.func(cfl.params_opt, temp_model)
print(cf.params_opt)
print(cf.r2)

ax1, ax2 = plt.subplots(2)[1]
ax1.plot(temps, vels, c='k')
ax1.plot(temps2, vels2)
ax1.plot(temp_model, cf.func(cf.params_opt, temp_model), c='k')
ax1.plot(temp_model, cfu.func(cfu.params_opt, temp_model), c='k', ls='--')
ax1.plot(temp_model, cfl.func(cfl.params_opt, temp_model), c='k', ls='--')
ax2.plot(temp_model, (model_l - model_u)/2)
ax1.set_ylabel('C (m/s)')
ax2.set_ylabel('Prediction Interval')
ax2.set_xlabel('Temperature (celsius)')
plt.tight_layout()

#%% Linear fit
linreg = US.SimpleLinReg(temps.flatten(), vels.flatten())
m = linreg.slope.value
c = linreg.intercept.value
parabolas = linreg.parabolas
predictionIntervals = linreg.predictionIntervals
r2 = linreg.r**2
m_95interval = linreg.slope.confidenceInterval
c_95interval = linreg.intercept.confidenceInterval

temp_model = np.arange(np.min(temps), np.max(temps), 0.001)
print(f'r2 = {round(r2, 4)}')
print(f'm = {round(m, 2)} \u00b1 {round(m_95interval, 2)}')
print(f'c = {round(c, 2)} \u00b1 {round(c_95interval, 2)}')

plt.figure()
plt.plot(temps, vels)
# plt.plot(experiments['Rb0_0_M_rt1'].temperature_lpf, experiments['Rb0_0_M_rt1'].C, label=experiments['Rb0_0_M_rt1'].name)
# plt.plot(experiments['Rb0_0_M_rt2'].temperature_lpf, experiments['Rb0_0_M_rt2'].C, label=experiments['Rb0_0_M_rt2'].name)
# plt.plot(experiments['Rb0_0_M_rt3'].temperature_lpf, experiments['Rb0_0_M_rt3'].C, label=experiments['Rb0_0_M_rt3'].name)
# plt.plot(experiments['Rb0_0_M_rt4'].temperature_lpf, experiments['Rb0_0_M_rt4'].C, label=experiments['Rb0_0_M_rt4'].name)
# plt.plot(temps, np.sort(vels, axis=0), c='k')
plt.plot(temp_model, m*temp_model + c, c='r')
plt.plot(temp_model, parabolas[0](temp_model), ls='--', c='gray')
plt.plot(temp_model, parabolas[1](temp_model), ls='--', c='gray');
plt.plot(temp_model, predictionIntervals[0](temp_model), ls='--', c='k');
plt.plot(temp_model, predictionIntervals[1](temp_model), ls='--', c='k');
# plt.legend()

plt.figure()
plt.plot(temp_model, (predictionIntervals[0](temp_model) - predictionIntervals[1](temp_model))/2, ls='--', c='k');






























#%% Velocity and thickness plot
ax1, ax2 = plt.subplots(2, figsize=(10,8))[1]
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Velocity (m/s)')
ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Thickness (um)')
for e in experiments.values():
    e.computeResults(Cc=2732, charac_container=False, cw=None)
    ax1.plot(np.arange(0, len(e.C))*e.Ts/3600, e.C - (e.Cw_lpf - e.Cw_lpf[0]))
    _temp = np.mean(e.C)
    print(f'{_temp} m/s')
    
    e.computeResults(Cc=2732, charac_container=False, cw=US.speedofsound_in_water(US.speedofsound2temperature(e.Cw[0]) + 0.032))
    ax1.plot(np.arange(0, len(e.C))*e.Ts/3600, e.C - (e.Cw_lpf - e.Cw_lpf[0]))
    print(f'{np.mean(e.C)} m/s')
    print(f'{np.mean(e.C) - _temp} m/s')
    # ax1.plot(np.arange(0, len(e.C))*e.Ts/3600, e.C - e.Cw_lpf + Cw20)
    # ax1.plot(np.arange(0, len(e.C))*e.Ts/3600, e.C)
    ax2.plot(np.arange(0, len(e.Lc))*e.Ts/3600, e.Lc*1e6, label=e.name)
    # ax2.plot(np.arange(0, len(e.L))*e.Ts/3600, e.L*1e3, label=e.name)
plt.legend()
plt.tight_layout()

#%%
ax1 = plt.subplots(1)[1]
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('ToF_R21 (samples)')
for e in experiments.values():
    ax1.plot(np.arange(0, len(e.C))*e.Ts/3600, e.ToF_R21, label=e.name)
    ax1.plot(np.arange(0, len(e.C))*e.Ts/3600, e.ToF_R21, label=e.name)
plt.legend()
plt.tight_layout()

#%% Thermal expansion coeff
ax1, ax2 = plt.subplots(2, figsize=(10,8))[1]
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Thickness (um)')
ax2.set_xlabel('Time (h)')
ax2.set_ylabel('Temperature (celsius)')
for e in experiments.values():
    ax1.plot(np.arange(0, len(e.Lc))*e.Ts/3600, e.Lc*1e6, label=e.name)
    
    b_IIR, a_IIR = scsig.iirfilter(e.lpf_order, 2*e.lpf_fc*e.Ts, btype='lowpass')
    e.Lc_lpf = scsig.filtfilt(b_IIR, a_IIR, e.Lc)
    
    ax1.plot(np.arange(0, len(e.Lc))*e.Ts/3600, e.Lc_lpf*1e6, label=e.name)
    
    ax2.plot(np.arange(0, len(e.temperature))*e.Ts/3600, e.temperature, label=e.name)    
    ax2.plot(np.arange(0, len(e.temperature))*e.Ts/3600, e.temperature_lpf, label=e.name)
plt.tight_layout()

alpha = (e.Lc_lpf[-1] - e.Lc_lpf[0]) / e.Lc_lpf[0] / (e.temperature_lpf[-1] - e.temperature_lpf[0])

alpha = (e.Lc_lpf[1:] - e.Lc_lpf[0]) / e.Lc_lpf[0] / (e.temperature_lpf[1:] - e.temperature_lpf[0])
plt.figure()
plt.plot(alpha*1e6)
plt.ylabel('Thermal expansion coeff (x10^-6 1/Cº)')
plt.tight_layout()
print(f'Thermal expansion coeff = {alpha*1e6} x 10^-6 1/Cº')

#%% Differential velocity comparison
meansdiffC = []
for e in experiments.values():
    idx = int(15*60/e.Ts)
    diffC = np.diff(e.C[:idx] - (e.Cw[:idx] - e.Cw[0]))
    meansdiffC.append(np.mean(diffC)) 

ax = plt.subplots(1)[1]
ax.set_xlabel('Dispersed phase mass concentration (%)')
ax.set_ylabel('Differential Velocity (m/s)')
ax.scatter(concentrations, meansdiffC, c='k')
for i, txt in enumerate(Names):
    ax.annotate(txt, (concentrations[i], meansdiffC[i]))
plt.tight_layout()


#%% Plot Cw
Equalize = False

N = len(experiments)
Colors = np.linspace([0.2, 0.2, 0], [0.8, 0.8, 0], N, axis=0) # red

plt.figure()
plt.xlabel('Time (h)')
plt.ylabel('Cw (m/s)')
for i,e in enumerate(experiments.values()):
    cw = e.Cw_lpf - e.Cw_lpf[0] + Cw20 if Equalize else e.Cw_lpf
    plt.plot(np.arange(0, len(e.C))*e.Ts/3600, cw, label=e.name)
    # plt.plot(np.arange(0, len(e.C))*e.Ts/3600, cw, label=e.name, c=Colors[i])
plt.legend()
plt.tight_layout()


#%% Plot C with temperature equalization
plt.figure()
plt.xlabel('Time (h)')
plt.ylabel('Velocity (m/s)')
for e in experiments.values():
    c = e.C - e.Cw_lpf + Cw20 # C - (Cw_lpf - Cw_lpf[0]) - Cw_lpf[0] + Cw20
    plt.plot(np.arange(0, len(e.C))*e.Ts/3600, c, label=e.name)
plt.legend()
plt.tight_layout()

#%%
plt.figure()
plt.xlabel('Time (h)')
plt.ylabel('Diameter (mm)')
for e in experiments.values():
    plt.plot(np.arange(0, len(e.C))*e.Ts/3600, (e.L - e.L[0])*1e3, label=e.name)
plt.legend()
plt.tight_layout()

#%% Equalize C (assuming water inside)
N = len(experiments)
Colors = np.linspace([0.2, 0.5, 0], [0.8, 0.5, 0], N, axis=0) # red

plt.figure()
plt.xlabel('Time (h)')
plt.ylabel('Velocity (m/s)')
for i,e in enumerate(experiments.values()):
    c = e.C - (e.Cw_lpf - e.Cw_lpf[0])
    # c = e.C - e.Cw_lpf + Cw20
    # c = e.C
    # c = e.C - experiments['W0_20_M'].C - (e.Cw_lpf - e.Cw_lpf[0])
    # plt.plot(np.arange(0, len(e.C))*e.Ts/3600, c - c[0] + Cw20, label=e.name, c=Colors[i])
    plt.plot(np.arange(0, len(e.C))*e.Ts/3600, c - c[0] + Cw20, label=e.name)
    # plt.plot(np.arange(0, len(e.C))*e.Ts/3600, c - c[-1] + Cw20, label=e.name)
    # plt.plot(np.arange(0, len(e.C))*e.Ts/3600, c, label=e.name)
plt.legend()
plt.tight_layout()


#%% Test UseHilbEnv on one experiment
e = experiments['W0_0_M']
e.windowAscans()
e.computeTOF(windowXcor=False, correction=True, filter_tofs_pe=True, UseHilbEnv=False)
e.computeResults(Cc=Cc, charac_container=False)
C = e.C
Cw_lpf = e.Cw_lpf

plt.figure()
plt.plot(C - (Cw_lpf - Cw_lpf[0]))

e.windowAscans()
e.computeTOF(windowXcor=False, correction=True, filter_tofs_pe=True, UseHilbEnv=True)
e.computeResults(Cc=Cc, charac_container=False)

plt.plot(e.C - (e.Cw_lpf - e.Cw_lpf[0]))

# ===============================
# For creating copies of objects:
# -------------------------------
# import copy
# e2 = copy.deepcopy(e)
# ===============================

# ===============================
# For saving results:
# -------------------------------
# for e in experiments.values():
#     e.saveResults()
# ===============================


#%% Violin plot
ax1, ax2 = plt.subplots(2, figsize=(10,8))[1]
ax1.set_xlabel('Mass concentration (%)')
ax1.set_ylabel('Thickness (um)')
ax2.set_xlabel('Mass concentration (%)')
ax2.set_ylabel('Velocity (m/s)')

Lc_all = np.array([e.Lc for e in experiments.values()])
L_all = np.array([e.L for e in experiments.values()]) 
C_all = np.array([e.C - (e.Cw_lpf - e.Cw_lpf[0]) - e.Cw_lpf[0] + Cw20 for e in experiments.values()]) 

# sns.violinplot(data=Lc_all*1e6, ax=ax1, flierprops=dict(marker='.'), inner=None, showmeans=True)
# ax1.set_xticklabels(concentrations);
# sns.violinplot(data=L_all*1e3, ax=ax1, flierprops=dict(marker='.'), inner=None, showmeans=True)
# ax1.set_xticklabels(concentrations);
# ax1.set_ylabel('Inner diameter (mm)')
sns.violinplot(data=(L_all + 2*Lc_all).T*1e3, ax=ax1, flierprops=dict(marker='.'), inner=None, showmeans=True)
# sns.boxplot(data=(L_all + 2*Lc_all).T*1e3, ax=ax1, flierprops=dict(marker='.'))
ax1.set_xticklabels(concentrations);
ax1.set_ylabel('Diameter (mm)')

sns.violinplot(data=C_all.T, ax=ax2, flierprops=dict(marker='.'), inner=None, showmeans=True)
# sns.boxplot(data=C_all.T, ax=ax2, flierprops=dict(marker='.'))
ax2.set_xticklabels(concentrations);

plt.tight_layout()

#%% Hysteresis
Nexps = len(Names)
newcolor = 1/Nexps

ax1 = plt.subplots(1)[1]
ax1.set_ylabel('Velocity (m/s)')
ax1.set_xlabel('Temperature (celsius)')
for i,k in enumerate(Names):
    N = len(experiments[k].temperature_lpf)
    # Colors = np.linspace([0, 0, 0.2], [0, 0, 0.8], N, axis=0) # blue
    # Colors = np.linspace([0, 0.2, 0], [0, 0.8, 0], N, axis=0) # green
    Colors = np.linspace([0, newcolor*i, 0], [1, newcolor*i, 0], N, axis=0) # red
    if (Colors[:,1] > 1).any():
        Colors[:,1] = 1
    
    # ax1.scatter(experiments[k].temperature_lpf, experiments[k].C - (experiments[k].Cw_lpf - experiments[k].Cw_lpf[0]), c=Colors)
    ax1.scatter(experiments[k].temperature_lpf, experiments[k].C - experiments[k].Cw_lpf + Cw20, c=Colors)
    # ax1.scatter(experiments[k].temperature_lpf, experiments[k].C, c=Colors)

temperature = np.linspace(0, 100, 10_000)
Cw = US.speedofsound_in_water(temperature, 'Abdessamad', 148)
# ax1.plot(temperature, Cw)








#%% Container characterization
expkeys = ['W0_0', 'W0_0_2', 'W0_0_3', 'W0_0_4', 'W0_0_5', 'W0_0_6', 
           'W0_0_7a', 'W0_0_7b', 'W0_0_7c', 'W0_0_7d', 'W0_0_7e', 'W0_0_7f', 'W0_0_7g', 'W0_0_7h', 'W0_0_7i']
others = [x for x in Names if x not in expkeys]
for k in expkeys:
    print(f'--------- {k} ---------')
    experiments[k].windowAscans()
    experiments[k].computeTOF(windowXcor=True, correction=True, filter_tofs_pe=True, UseHilbEnv=False)
    experiments[k].computeResults(charac_container=True)
    

#%%
ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Thickness (um)')
ax2.set_ylabel('Velocity (m/s)')
for k in expkeys:
    c = 'k' if '5' in experiments[k].name or '6' in experiments[k].name or experiments[k].name=='W0_0' else None
    ax1.plot(experiments[k].Lc*1e6, c=c)
    ax2.plot(experiments[k].Cc, label=experiments[k].name, c=c)
plt.legend()
plt.tight_layout()

ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Cw (m/s)')
ax2.set_ylabel('Temperature (celsius)')
for k in expkeys:
    c = 'k' if '5' in experiments[k].name or '6' in experiments[k].name or experiments[k].name=='W0_0' else None
    ax1.plot(experiments[k].Cw_lpf, c=c)
    ax2.plot(experiments[k].temperature_lpf, label=experiments[k].name, c=c)
plt.legend()
plt.tight_layout()

#%% 1 bottle
labls = ['W0_0_7a', 'W0_0_7b', 'W0_0_7c', 'W0_0_7d', 'W0_0_7e', 'W0_0_7f', 'W0_0_7g', 'W0_0_7h', 'W0_0_7i']
ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Thickness (um)')
ax2.set_ylabel('Velocity (m/s)')
for k in labls:
    ax1.plot(experiments[k].Lc*1e6)
    ax2.plot(experiments[k].Cc, label=experiments[k].name)
plt.legend()
plt.tight_layout()

ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Cw (m/s)')
ax2.set_ylabel('Temperature (celsius)')
for k in labls:
    ax1.plot(experiments[k].Cw_lpf)
    ax2.plot(experiments[k].temperature_lpf, label=experiments[k].name)
plt.legend()
plt.tight_layout()


#%% Bottle characterization
obj = US.RealtimeSP(r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Deposition\B6')
obj.windowAscans()
obj.computeTOF(windowXcor=True, correction=True, filter_tofs_pe=True, UseHilbEnv=False)
obj.computeResults(charac_container=True)

# ax1, ax2 = plt.subplots(2)[1]
# ax1.set_ylabel('Thickness (um)')
# ax2.set_ylabel('Velocity (m/s)')
# ax1.plot(obj.Lc*1e6)
# ax2.plot(obj.Cc)
# plt.tight_layout()
print(f'--------- {obj.name} ---------')
print(f'Thickness: {np.mean(obj.Lc)*1e6} um')
print(f'Velocity: {np.mean(obj.Cc)} m/s')




#%%
ax1 = plt.subplots(1)[1]
ax1.set_ylabel('Velocity (m/s)')
for k in expkeys:    
    # ax1.plot(experiments[k].Cc - experiments[k].Cc[0])
    # ax1.scatter(experiments[k].temperature, experiments[k].Cc)
    ax1.scatter(experiments[k].temperature, experiments[k].Cc - (experiments[k].Cw_lpf - experiments[k].Cw_lpf[0]))

ax1 = plt.subplots(1)[1]
ax1.set_ylabel('Cw (m/s)')
for k in expkeys:
    ax1.plot(experiments[k].Cw_lpf - experiments[k].Cw_lpf[0])


ax1 = plt.subplots(1)[1]
ax1.set_ylabel('Temperature (celsius)')
for k in expkeys:
    ax1.plot(experiments[k].temperature)

#%%
Nexps = len(expkeys)
newcolor = 1/Nexps

ax1 = plt.subplots(1)[1]
ax1.set_ylabel('Velocity (m/s)')
ax1.set_xlabel('Temperature (celsius)')
for i,k in enumerate(expkeys):
    N = len(experiments[k].temperature_lpf)
    # Colors = np.linspace([0, 0, 0.2], [0, 0, 0.8], N, axis=0) # blue
    # Colors = np.linspace([0, 0.2, 0], [0, 0.8, 0], N, axis=0) # green
    Colors = np.linspace([0, newcolor*i, 0], [1, newcolor*i, 0], N, axis=0) # red
    if Colors[:,1].any() > 1:
        Colors[:,1] = 1
    
    ax1.scatter(experiments[k].temperature_lpf, experiments[k].Cc - (experiments[k].Cw_lpf - experiments[k].Cw_lpf[0]), c=Colors)


N = len(others)
newcolor = 1/N
ax1 = plt.subplots(1)[1]
ax1.set_ylabel('Velocity (m/s)')
ax1.set_xlabel('Temperature (celsius)')
for i,k in enumerate(others):
    N = len(experiments[k].temperature_lpf)
    # Colors = np.linspace([0, 0, 0.2], [0, 0, 0.8], N, axis=0) # blue
    # Colors = np.linspace([0, 0.2, 0], [0, 0.8, 0], N, axis=0) # green
    Colors = np.linspace([0, newcolor*i, 0], [1, newcolor*i, 0], N, axis=0) # red
    if Colors[:,1].any() > 1:
        Colors[:,1] = 1
    
    ax1.scatter(experiments[k].temperature_lpf, experiments[k].C - (experiments[k].Cw_lpf - experiments[k].Cw_lpf[0]), c=Colors)


#%%
e = experiments['W0_0_3']
# e.windowAscans()
# e.computeTOF(windowXcor=True, correction=True, filter_tofs_pe=True, UseHilbEnv=False)
e.computeResults(Cc=6179, charac_container=False)

ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Velocity (m/s)')
ax1.set_xlabel('Temperature (\u2103)')
ax2.set_ylabel('Temperature (\u2103)')
ax2.set_xlabel('Time (h)')

Colors = np.linspace([0.2, 0, 0], [1, 0, 0], N, axis=0) # red
ax1.scatter(e.temperature_lpf, e.C - (e.Cw_lpf - e.Cw_lpf[0]), c=Colors)
ax2.scatter(np.arange(0, len(e.C))*e.Ts/3600, e.temperature_lpf, c=Colors)
plt.tight_layout()



ax1 = plt.subplots(1)[1]
ax1.set_ylabel('Velocity (m/s)')
ax1.set_xlabel('Time (h)')

Colors = np.linspace([0.2, 0, 0], [1, 0, 0], N, axis=0) # red
ax1.scatter(np.arange(0, len(e.C))*e.Ts/3600, e.Cw_lpf, c=Colors)
ax1.scatter(np.arange(0, len(e.C))*e.Ts/3600, e.C - (e.Cw_lpf - e.Cw_lpf[0]), c=Colors)
plt.tight_layout()


#%% Hysteresis measurement
ax1, ax2 = plt.subplots(2)[1]
ax1.set_xlabel('Time (h)')
ax1.set_ylabel('Temperature (\u2103)')
ax2.set_ylabel('Diff (\u2103)')
ax2.set_xlabel('Time (h)')

tempdiffs = np.diff(e.temperature_lpf)
idxs_inf = np.where(tempdiffs<0)[0]
idxs_sup = np.where(tempdiffs>=0)[0]
tempdiffs_inf = tempdiffs[idxs_inf]
tempdiffs_sup = tempdiffs[idxs_sup]
Colors_inf = np.linspace([0, 0, 0.2], [0, 0, 1], len(tempdiffs_inf), axis=0) # blue
Colors_sup = np.linspace([0, 0.2, 0], [0, 1, 0], len(tempdiffs_sup), axis=0) # green
Colors_diffs = np.r_[Colors_inf, Colors_sup]
Colors = np.linspace([0.2, 0, 0], [1, 0, 0], N, axis=0) # red
ax1.scatter(np.arange(0, len(e.C))*e.Ts/3600, e.temperature_lpf, c=Colors)
ax2.scatter(np.arange(0, len(tempdiffs))*e.Ts/3600, tempdiffs, c=Colors_diffs)
plt.tight_layout()


hyst = np.zeros(len(idxs_inf))
idxs = np.zeros(len(idxs_inf))
for i, val_inf in enumerate(e.temperature_lpf[idxs_inf]):
    idx, val_sup = US.find_nearest(e.temperature_lpf[idxs_sup][::-1], val_inf)
    hyst[i] = val_inf - val_sup
    idxs[i] = idx

ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Hysteresis (\u2103)')
ax1.set_xlabel('Time (h)')
ax2.set_ylabel('Original indices used')
ax2.set_xlabel('Time (h)')
ax1.plot(np.arange(0, len(hyst))*e.Ts/3600, hyst)
ax2.plot(np.arange(0, len(hyst))*e.Ts/3600, idxs + len(idxs_inf))
plt.tight_layout()




#%% Container characterization
Path2 = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Deposition'
Names2 = ['BP1a', 'BP1b', 'BP1c', 'BP1d', 'BP1e', 'BP1f', 'BP1g', 'BP1h', 'BP1i']
Names2 = ['BM1a', 'BM1b', 'BM1c', 'BM1d', 'BM1e']
Names2 = ['BM2a', 'BM2b', 'BM2c', 'BM2d', 'BM2e', 'BM2f']
Cw20 = US.speedofsound_in_water(20)
experiments2 = loadExperiments(Path2, Names2, Verbose=True)
for e in experiments2.values():
    print(f'--------- {e.name} ---------')
    if e.name in ['BM1a', 'BM1c', 'BM1e']:
        e.windowAscans(Loc_WP=3200, Loc_TT=2970, Loc_PER=1180, Loc_PETR=5500,
                       WinLen_WP=1000, WinLen_TT=1000, WinLen_PER=1300, WinLen_PETR=1000)
    elif 'BM2' in e.name:
        e.windowAscans(Loc_WP=3200, Loc_TT=3000, Loc_PER=1180, Loc_PETR=3900, 
                                    WinLen_WP=1000, WinLen_TT=1000, WinLen_PER=1000, WinLen_PETR=1000)
    else:
        e.windowAscans(Loc_WP=3200, Loc_TT=2970, Loc_PER=1180, Loc_PETR=6600,
                       WinLen_WP=1000, WinLen_TT=1000, WinLen_PER=1300, WinLen_PETR=600)        
    e.computeTOF(windowXcor=False, correction=True, filter_tofs_pe=True, UseHilbEnv=False)
    e.computeResults(charac_container=True)
    e.saveResults()


#%%
e = experiments2['BM1a']
plt.figure()
plt.plot(e.TTraw, c='k')
plt.plot(e.TT, c='r')

#%% 1 bottle
ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Thickness (um)')
ax2.set_ylabel('Velocity (m/s)')
for e in experiments2.values():
    ax1.plot(e.Lc*1e6)
    ax2.plot(e.Cc, label=e.name)
plt.legend()
plt.tight_layout()

ax1, ax2 = plt.subplots(2)[1]
ax1.set_ylabel('Cw (m/s)')
ax2.set_ylabel('Temperature (celsius)')
for e in experiments2.values():
    ax1.plot(e.Cw_lpf)
    ax2.plot(e.temperature_lpf, label=e.name)
plt.legend()
plt.tight_layout()

#%%
mylabs = ['BM1a', 'BM1c', 'BM1e']
mylabs = ['BM2d']
c = []
l = []
for e in experiments2.values():
    if e.name in mylabs:
        c.append(e.Cc)
        l.append(e.Lc)
c = np.array(c).flatten()
l = np.array(l).flatten()
print(np.mean(c))
print(np.mean(l)*1e6)














#%%
# def MAX(x):
#     return US.CosineInterpMax(x, xcor=False, UseHilbEnv=False)

# tofs_r1 = np.apply_along_axis(MAX, 0, PE_R)
# tofs_r1 = tofs_r1 + Smin2

# N = len(tofs_r1)
# all_diffs = np.zeros([len(tofs_r1), N])
# for n in range(N):
#     # np.random.seed(0)
#     idxs = np.random.permutation(range(len(tofs_r1)))
#     shuffled_tofs = tofs_r1[idxs]
#     shuffled_Cw = Cw[idxs]
    
#     divtof = tofs_r1 / shuffled_tofs
#     divcw = Cw / shuffled_Cw
#     diff = divtof - divcw
    
#     all_diffs[:,n] = diff

# avg_diff = np.mean(all_diffs, axis=0)

# # tofs_r1_2 = tofs_r1 - tofs_pe # correction

# # shuffled_tofs_2 = tofs_r1_2[idxs]

# # divtof_2 = tofs_r1_2 / shuffled_tofs_2
# # divcw_2 = Cw / shuffled_Cw
# # diff_2 = np.abs(divtof_2 - divcw_2)

# plt.figure()
# # plt.plot(np.arange(0, N_acqs)*Ts/3600, diff)
# plt.plot(np.arange(0, N_acqs)*Ts/3600, avg_diff)
# # plt.plot(np.arange(0, N_acqs)*Ts/3600, diff_2)
# plt.xlabel('Time (h)')

# #%%
# # tofs_pe
# # tofs_r1

# d = tofs_r1/Fs * Cw[0] / 2

# # d = tofs_r1/Fs * Cw / 2

# t = d / Cw
# sample_t = t * Fs
# tofs_t = sample_t - sample_t[0]

# plt.figure()
# plt.plot(tofs_pe)
# plt.plot(-2*tofs_t)

# plt.figure()
# plt.plot(tofs_pe + 2*tofs_t)



# ToF_TW_2 = ToF_TW + 2*tofs_t
# Cc2 = Cw*(np.abs(ToF_TW_2)/ToF_R21 + 1) # container speed - m/s
# Lc2 = Cw/2*(np.abs(ToF_TW_2) + ToF_R21)/Fs # container thickness - m

# plt.figure()
# # plt.plot(Cc2)
# plt.plot(Lc2)

# #%%
# plt.figure()
# plt.plot(tofs_pe/Fs*Cw*1e6)
# plt.ylabel('Distance (um)')


# #%%
# tempdiff = temperature - temperature[0]
# alpha = 3.3e-6
# expansion = Lc[0] * alpha * tempdiff


# t2 = tofs_r1[0]/Fs/2
# d2 = Cw * t2

# plt.figure()
# plt.plot((d2 - d2[0])*1e6)
# plt.plot(expansion*1e6)
# plt.ylabel('Distance (um)')

# #%%
# tofs_exp = 2*expansion/Cw*Fs

# plt.figure()
# plt.plot(tofs_exp)