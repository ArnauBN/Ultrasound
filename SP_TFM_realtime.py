# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 16:53:27 2023

@author: arnau
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
Names = ['Rb0_0_M_rt1', 'Rb01_0_M_rtw', 'Rb02_0_M_rtw', 'Rb03_0_M_rtw', 'Rb04_0_M_rtw', 'Rb05_0_M_rtw', 'Rb06_0_M_rtw', 'Rb07_0_M_rtw', 'Rb08_0_M_rtw', 'Rb09_0_M_rtw', 'Rb10_0_M_rtw']
Names = ['Rb0_0_M_rt1', 'Rb01_0_M_rt', 'Rb02_0_M_rt', 'Rb03_0_M_rt', 'Rb04_0_M_rt', 'Rb05_0_M_rt', 'Rb06_0_M_rt', 'Rb07_0_M_rt', 'Rb08_0_M_rt', 'Rb09_0_M_rtw', 'Rb10_0_M_rtw']

# Names = ['Rb0_20_M_rt1', 'Rb01_20_M_rt', 'Rb02_20_M_rt', 'Rb03_20_M_rt', 'Rb04_20_M_rt', 'Rb05_20_M_rt', 'Rb06_20_M_rt', 'Rb07_20_M_rt', 'Rb08_20_M_rt', 'Rb09_20_M_rt', 'Rb10_20_M_rt']
concentrations = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lbls = [f'{c} wt%' for c in concentrations]
experiments = loadExperiments(Path, Names, Verbose=True, Cw_material='resin')
Time_axis = np.arange(0, len(experiments[list(experiments.keys())[0]].C))*experiments[list(experiments.keys())[0]].Ts/60


#%% Compute ToFs
Cc = 2726 # methacrylate (small container)
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
    # experiments[k].computeTOF(windowXcor=False, correction=True, filter_tofs_pe=True, UseHilbEnv=False)
    experiments[k].computeTOF(windowXcor=False, correction=False, filter_tofs_pe=True, UseHilbEnv=False)
    
    cws[i] = US.temp2sos(experiments[k].config_dict['WP_temperature'], material='water') if 'rt' in k else None
    if 'rt' in k: experiments[k].LPFtemperature()
    
    # experiments[k].computeResults(Cc=Cc, charac_container=False, cw=cws[i])
    experiments[k].computeResultsFinal(Cc=Cc, lpf_temperature=True)


#%% Outlier detection
UseMedian = False
m_c = 3 # Outlier detection threshold for C
m_l = 3 # Outlier detection threshold for L

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

ax2, ax3 = plt.subplots(2)[1]
for i,e in enumerate(experiments.values()):
    e.computeResults(Cc=Cc, cw=cws[i])
    C_ = e.C
    e.computeResults(Cc=Cc+30, cw=cws[i])
    
    ax2.plot(Time_axis, C_, c='k')
    ax2.plot(Time_axis, e.C, c='r')
    
    ax3.plot(Time_axis, C_ - e.C)
plt.tight_layout()


#%% Plot
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)
fig4, ax4 = plt.subplots(1)
fig5, ax5 = plt.subplots(1)
ax1.set_xlabel('Time (min)')
ax2.set_xlabel('Time (min)')
ax3.set_xlabel('Time (min)')
ax4.set_xlabel('Time (min)')
ax5.set_xlabel('Time (min)')

ax1.set_ylabel('Velocity (m/s)')
ax2.set_ylabel("Temperature model's velocity (m/s)")
ax3.set_ylabel('Differential velocity (m/s)')
ax4.set_ylabel('Thickness ($\mu$m)')
ax5.set_ylabel('Diameter (mm)')

for i,e in enumerate(experiments.values()):
    L_lpf_ = LPF2mHz(e.L, 1/e.Ts)
    C_lpf = LPF2mHz(e.C, 1/e.Ts)
    c = e.C - US.temp2sos(e.temperature_lpf, material='resin')
    c_lpf = C_lpf - US.temp2sos(e.temperature_lpf, material='resin')
    
    _p = ax1.plot(Time_axis, e.C, c=colors[i], alpha=0.4)
    ax1.plot(Time_axis, C_lpf, label=lbls[i], c=_p[-1].get_color(), alpha=1, zorder=3)    
    
    ax2.plot(Time_axis, e.Cw_lpf, c=colors[i], label=lbls[i])
    
    _p = ax3.plot(Time_axis, c, c=colors[i], alpha=0.4)
    ax3.plot(Time_axis, c_lpf, c=_p[-1].get_color(), label=lbls[i], alpha=1, zorder=3)
    
    ax4.plot(Time_axis, e.Lc*1e6, c=colors[i], label=lbls[i])
    
    _p = ax5.plot(Time_axis, e.L*1e3, c=colors[i], alpha=0.4)
    ax5.plot(Time_axis, L_lpf_*1e3, label=lbls[i], c=_p[-1].get_color(), alpha=1, zorder=3)

ax3.legend()
plt.tight_layout()
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()

US.movefig('northwest', fig1)
US.movefig('northeast', fig2)
US.movefig('southwest', fig3)
US.movefig('southeast', fig4)
US.movefig('south', fig5)


#%% Histogram of L
data = Lall_without_outliers_flat*1e3
# data = Lall.flatten()*1e3

Nsigmas = 5
Npoints = 1000

hL, bL, wL = US.hist(Lall.flatten()*1e3, density=True)
# hL, bL, wL = US.hist(data, density=True)
m = np.mean(data)
std = np.std(data)

aux = np.linspace(m - Nsigmas*std, m + Nsigmas*std, Npoints)
gauss = np.exp(-((aux - m) / std)**2 / 2) / (std*np.sqrt(2*np.pi))

ax = plt.subplots(1)[1]
US.plot_hist(hL, bL, wL, ax=ax, xlabel='Diameter (mm)', ylabel='pdf', edgecolor='k')
ax.plot(aux, gauss, c='r')
ax.set_xlim([18.2, 19.2])




#%% Order
c10 = np.zeros(len(experiments))
cw10 = np.zeros(len(experiments))
fig1, (ax1, ax2) = plt.subplots(2)
fig2, ax3 = plt.subplots(1)
for i,e in enumerate(experiments.values()):
    c = e.C - US.temp2sos(e.temperature_lpf, material='resin')
    c10[i] = np.mean(c[:int(10/e.Ts*60)])
    cw10[i] = np.mean(e.Cw_lpf[:int(10/e.Ts*60)])
    ax1.scatter(concentrations[i], c10[i], c=colors[i], label=lbls[i])
    ax2.scatter(concentrations[i], cw10[i], c=colors[i], label=lbls[i])
    ax3.scatter(cw10[i], c10[i], c=colors[i], label=lbls[i])
ax1.set_xticks(concentrations)
ax1.set_xlabel('Concentration (wt%)')
ax1.set_ylabel('Differential velocity (m/s)')

ax2.set_xticks(concentrations)
ax2.set_xlabel('Concentration (wt%)')
ax2.set_ylabel("Temperature model's velocity (m/s)")

ax3.set_xlabel("Temperature model's velocity (m/s)")
ax3.set_ylabel('Differential velocity (m/s)')
fig1.tight_layout()
fig2.tight_layout()



fitted_data = c10
x = np.array(concentrations)
x_model = np.arange(np.min(x), np.max(x), 0.001)

linreg = US.SimpleLinReg(x, fitted_data)
m = linreg.slope.value
c = linreg.intercept.value
parabolas = linreg.parabolas
predictionIntervals = linreg.predictionIntervals
r2 = linreg.r**2
m_95interval = linreg.slope.confidenceInterval
c_95interval = linreg.intercept.confidenceInterval

print(f'r2 = {round(r2, 4)}')
print(f'm = {round(m, 2)} \u00b1 {round(m_95interval, 2)}')
print(f'c = {round(c, 2)} \u00b1 {round(c_95interval, 2)}')

ax1.plot(x, fitted_data, c='k')
ax1.scatter(x, fitted_data, marker='o', c=colors, zorder=3)
ax1.plot(x_model, m*x_model + c, c='r', ls='--')
plt.tight_layout()

# -----------------------------------------

fitted_data = cw10
x = np.array(concentrations)
x_model = np.arange(np.min(x), np.max(x), 0.001)

linreg = US.SimpleLinReg(x, fitted_data)
m = linreg.slope.value
c = linreg.intercept.value
parabolas = linreg.parabolas
predictionIntervals = linreg.predictionIntervals
r2 = linreg.r**2
m_95interval = linreg.slope.confidenceInterval
c_95interval = linreg.intercept.confidenceInterval

print(f'r2 = {round(r2, 4)}')
print(f'm = {round(m, 2)} \u00b1 {round(m_95interval, 2)}')
print(f'c = {round(c, 2)} \u00b1 {round(c_95interval, 2)}')

ax2.plot(x, fitted_data, c='k')
ax2.scatter(x, fitted_data, marker='o', c=colors, zorder=3)
ax2.plot(x_model, m*x_model + c, c='r', ls='--')
plt.tight_layout()

# -----------------------------------------

fitted_data = c10
x = cw10
x_model = np.arange(np.min(cw10), np.max(cw10), 0.001)

linreg = US.SimpleLinReg(x, fitted_data)
m = linreg.slope.value
c = linreg.intercept.value
parabolas = linreg.parabolas
predictionIntervals = linreg.predictionIntervals
r2 = linreg.r**2
m_95interval = linreg.slope.confidenceInterval
c_95interval = linreg.intercept.confidenceInterval

print(f'r2 = {round(r2, 4)}')
print(f'm = {round(m, 2)} \u00b1 {round(m_95interval, 2)}')
print(f'c = {round(c, 2)} \u00b1 {round(c_95interval, 2)}')

# ax3.plot(x, fitted_data, c='k')
ax3.scatter(x, fitted_data, marker='o', c=colors, zorder=3)
ax3.plot(x_model, m*x_model + c, c='r', ls='--')
plt.tight_layout()



#%% Plot without outliers
plot_diffs = True

fig1, ax1 = plt.subplots(1)
fig4, ax4 = plt.subplots(1)
ax1.set_xlabel('Time (min)')
ax4.set_xlabel('Time (min)')

ax1lbl = 'Differential velocity (m/s)' if plot_diffs else 'Velocity (m/s)'

ax1.set_ylabel(ax1lbl)
ax4.set_ylabel('Diameter (mm)')

zipdata = zip(Call_masked_diffs, Lall_masked, L_lpf, experiments.values()) if plot_diffs else zip(Call_masked, Lall_masked, L_lpf, experiments.values())
for i, (call, lall, lall_lpf, e) in enumerate(zipdata):
    C_lpf = LPF2mHz(e.C, 1/e.Ts)
    c_lpf = C_lpf - US.temp2sos(e.temperature_lpf, material='resin')

    _p = ax1.plot(Time_axis, call, c=colors[i], alpha=0.4)
    ax1.plot(Time_axis, c_lpf, label=lbls[i], c=_p[-1].get_color(), alpha=1, zorder=3)
    
    
    
    _p = ax4.plot(Time_axis, lall*1e3, c=colors[i], label=lbls[i], alpha=0.4)
    ax4.plot(Time_axis, lall_lpf*1e3, label=lbls[i], c=_p[-1].get_color(), alpha=1, zorder=3)
ax1.legend()
fig1.tight_layout()
fig4.tight_layout()




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

tt_ffts = np.zeros([ttraw.shape[1],nfft])
for i,tt in enumerate(ttraw.T):
    tt_fft = np.fft.fft(tt, nfft)
    tt_ffts[i] = np.abs(tt_fft)**2
    
    maxs[i] = max(np.abs(tt_fft[idx_inf:idx_sup])**2)
    idxs_maxs[i] = np.where(np.abs(tt_fft[idx_inf:idx_sup])**2==maxs[i])[0][0]
    ax1.plot(f*1e-6, np.abs(tt_fft)**2)

plt.tight_layout()

#%% Surface plot
# X = np.arange(-5, 5, 0.25)
# Y = np.arange(-5, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, lw=0, antialiased=False)

yticks = Time_axis
idx = np.where(f*1e-6 >= 3)[0][0]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim(drawlims)
ax.set_prop_cycle(color=[cm(1.*i/N) for i in range(len(Time_axis))])
for tt_fft, yt in zip(tt_ffts, yticks):
    ax.plot(f[:idx]*1e-6, np.abs(tt_fft[:idx])**2, zs=yt, zdir='y')

plt.tight_layout()


#%% Gradient
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
fig3, ax3 = plt.subplots(1)
# fig4, ax4 = plt.subplots(1)

ax1.set_xlabel('Time (min)')
ax2.set_xlabel('Time (min)')
ax3.set_xlabel('Time (min)')

ax1.set_ylabel('Gradient of Velocity (m/s/min)')
ax2.set_ylabel("Gradient of Temperature model's velocity (m/s/min)")
ax3.set_ylabel('Gradient of Differential velocity (m/s/min)')

for i, e in enumerate(experiments.values()):
    C_lpf = LPF2mHz(e.C, 1/e.Ts)
    c_lpf = C_lpf - US.temp2sos(e.temperature_lpf, material='resin')
    
    step = Time_axis[1] - Time_axis[0]
    
    ax1.plot(Time_axis, np.abs(np.gradient(C_lpf, step)), c=colors[i], label=lbls[i])
    ax2.plot(Time_axis, np.abs(np.gradient(e.Cw_lpf, step)), c=colors[i], label=lbls[i])
    ax3.plot(Time_axis, np.abs(np.gradient(c_lpf, step)), c=colors[i], label=lbls[i])
    # ax4.plot(Time_axis[:-1], np.diff(c_lpf), c=colors[i], label=lbls[i])
    
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
# fig4.tight_layout()
    
    
    
    
    