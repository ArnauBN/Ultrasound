# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 09:26:58 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import os
import scipy.signal as scsig

import src.ultrasound as US

# --------------------------------
# Expected values for methacrylate
# --------------------------------
# Longitudinal velocity: 2730 m/s
# Shear velocity: 1430 m/s
# Density: PMMA -> 1.18 g/cm^3


#%%
########################################################
# Paths and file names to use
########################################################
Path = r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\Methacrylate'
Experiment_folder_name = 'test_50us' # Without Backslashes
Experiment_config_file_name = 'config.txt' # Without Backslashes
Experiment_results_file_name = 'results.txt'
Experiment_PEref_file_name = 'PEref.bin'
Experiment_WP_file_name = 'WP.bin'
Experiment_acqdata_file_name = 'acqdata.bin'
Experiment_Temperature_file_name = 'temperature.txt'
Experiment_scanpath_file_name = 'scanpath.txt'
Experiment_description_file_name = 'Experiment_description.txt'
Experiment_PEforCW_file_name = 'PEat0_PEat10mm.txt'

MyDir = os.path.join(Path, Experiment_folder_name)
Config_path = os.path.join(MyDir, Experiment_config_file_name)
Results_path = os.path.join(MyDir, Experiment_results_file_name)
PEref_path = os.path.join(MyDir, Experiment_PEref_file_name)
WP_path = os.path.join(MyDir, Experiment_WP_file_name)
Acqdata_path = os.path.join(MyDir, Experiment_acqdata_file_name)
Temperature_path = os.path.join(MyDir, Experiment_Temperature_file_name)
Experiment_description_path = os.path.join(MyDir, Experiment_description_file_name)
Scanpath_path = os.path.join(MyDir, Experiment_scanpath_file_name)
PEforCW_path = os.path.join(MyDir, Experiment_PEforCW_file_name)


#%%
########################################################
# Load data
########################################################
# Config
config_dict = US.load_config(Config_path)
N_acqs = config_dict['N_acqs']
Fs = config_dict['Fs']

# Data
TT, PE = US.load_bin_acqs(Acqdata_path, N_acqs)

# Temperature and CW
temperature_dict = US.load_columnvectors_fromtxt(Temperature_path)
temperature = temperature_dict['Inside']
Cw = temperature_dict['Cw']

# Scan pattern
scanpattern = US.load_columnvectors_fromtxt(Scanpath_path, delimiter=',', header=False, dtype=str)

# WP
with open(WP_path, 'rb') as f:
    WP = np.fromfile(f)

# PE ref
with open(PEref_path, 'rb') as f:
    PEref = np.fromfile(f)


#%% Parameters
Ridx = [np.where(scanpattern == s)[0][0] for s in scanpattern if 'R' in s][0] + 1
theta_deg = float(scanpattern[Ridx-1][1:])
theta = theta_deg * np.pi / 180

step = float(scanpattern[0][1:])

x = np.arange(Ridx)*step # mm


#%% Plot temperature
lpf_order = 2
lpf_fc = 100e3 # Hz

if Fs < 2*lpf_fc:
    print(f'Signal does not have frequency components beyond {lpf_fc} Hz, therefore it is not filtered.')
else:
    # Create filter:
    # Calculate the coefficients
    b_IIR, a_IIR = scsig.iirfilter(lpf_order, 2*lpf_fc/Fs, btype='lowpass')
       
    # Filter signal
    temperature_lpf = scsig.filtfilt(b_IIR, a_IIR, temperature)
    
    Cw2 = US.speedofsound_in_water(temperature_lpf)
    
    ax1, ax2 = plt.subplots(2)[1]
    ax1.scatter(np.arange(N_acqs), temperature, marker='.', color='k')
    ax1.plot(temperature_lpf, 'r', lw=3)
    ax1.set_ylabel('Temperature (\u2103)')
    
    ax2.scatter(np.arange(N_acqs), Cw, marker='.', color='k')
    ax2.plot(Cw2, 'r', lw=3)
    ax2.set_ylabel('Cw (m/s)')
    
    plt.tight_layout()


#%% First plot
nfft = 2**(int(np.ceil(np.log2(np.abs(len(WP))))) + 1) # Number of FFT points (power of 2)

US.multiplot_tf(np.column_stack((PE[:,0], TT[:,0], WP, PEref)).T, Fs=Fs, nfft=nfft, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 20]), f_ylims=None, f_units='MHz', f_Norm=False,
            PSD=False, dB=False, label=['PE', 'TT', 'WP', 'PEref'], Independent=True, FigNum='Signals', FgSize=(6.4,4.8))


#%% TOF computations
def TOF(x, y):
    # m1 = US.CosineInterpMax(x, xcor=False)
    # m2 = US.CosineInterpMax(y, xcor=False)
    # return m1 - m2
    
    xh = np.absolute(scsig.hilbert(x))
    yh = np.absolute(scsig.hilbert(y))
    return US.CalcToFAscanCosine_XCRFFT(xh, yh, UseHilbEnv=False, UseCentroid=False)[0]

    # return US.CalcToFAscanCosine_XCRFFT(x, y, UseHilbEnv=True, UseCentroid=False)[0]

def ID(x, y):
    xh = np.absolute(scsig.hilbert(x))
    yh = np.absolute(scsig.hilbert(y))
    return US.deconvolution(xh, yh, stripIterNo=2, UseHilbEnv=False, Extend=True, Same=False)[0]
    
    # return US.deconvolution(x, y, stripIterNo=2, UseHilbEnv=True, Extend=True, Same=False)[0]

ToF_TW = np.apply_along_axis(TOF, 0, TT, WP)
ToF_RW = np.apply_along_axis(ID, 0, PE, PEref)
ToF_R21 = ToF_RW[1] - ToF_RW[0]


#%% Velocity and thickness computations
cw = np.mean(Cw)
# cw = config_dict['Cw']
# cw = Cw
# cw = Cw2
cw_aux = np.asarray([cw]).flatten()[::-1]


L = cw/2*(2*np.abs(ToF_TW) + ToF_R21)/Fs # thickness - m    
CL = cw*(2*np.abs(ToF_TW)/ToF_R21 + 1) # longitudinal velocity - m/s
Cs = cw_aux / np.sqrt(np.sin(theta)**2 + (cw_aux * np.abs(ToF_TW[::-1]) / (L * Fs) + np.cos(theta))**2) # shear velocity - m/s

CL = CL[:Ridx]
L = L[:Ridx]
Cs = Cs[:Ridx]


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



#%% Plot results
ax1, ax2 = plt.subplots(2)[1]

ax1.set_ylabel('Longitudinal velocity (m/s)', color='b')
ax1.plot(x, CL, 'b')

ax1twin = ax1.twinx()
ax1twin.set_ylabel('Shear velocity (m/s)', color='r')
ax1twin.plot(x, Cs, 'r')

ax2.set_ylabel('Thickness (mm)')
ax2.plot(x, L*1e3, 'k')
ax2.set_xlabel('Position (mm)')

plt.tight_layout()


#%% Density
# Recorded Arefs
Aref_50us = 0.2965645596590909
Gain_Aref_50us = 15 # dB

Mean_Arefs_50us = 0.3098931403882576
Gain_Means_Arefs_50us = 15 # dB


# Select Aref
Gain_Aref = Gain_Means_Arefs_50us
Aref = Mean_Arefs_50us



Gain_Ch2 = config_dict['Gain_Ch2']
AR1 = np.max(np.abs(PE)*(10**(-Gain_Ch2/20)), axis=0)[:Ridx]
Aref = Aref*(10**(-Gain_Aref/20)) # Reference amplitude (V)
Zw = 1.48e6 # acoustic impedance of water (N.s/m)

'''
------------------
Archimedean method
------------------
EpoxyResin Densities (g/cm^3):
1 --> 1.1363733011405817
2 --> 1.1621696788915479
3 --> 1.1522852023883876
4 --> 1.1367729721550683
5 --> 1.147496912592075

Methacrylate Density (g/cm^3):
6 --> 1.165269741337367
'''


d = Zw / CL * (Aref + AR1) / (Aref - AR1) # density (kg/m^3)
d *= 1e-3 # density (g/cm^3)

ax1, ax2, _ = US.histGUI(x, d, xlabel='Position (mm)', ylabel='Density (g/cm^3)')


if '1' in Experiment_folder_name:
    archval = 1.1363733011405817
elif '2' in Experiment_folder_name:
    archval = 1.1621696788915479
elif '3' in Experiment_folder_name:
    archval = 1.1522852023883876
elif '4' in Experiment_folder_name:
    archval = 1.1367729721550683
elif '5' in Experiment_folder_name:
    archval = 1.147496912592075
else:
    archval = 1.165269741337367 # methacrylate
theoval = 1.18 if 'Methacrylate' in Path else 1.15
ax1.axhline(theoval, ls='--', c='r', label='Theoretical')
ax1.axhline(archval, ls='--', c='b', label='Archimedean')
ax1.axhline(np.mean(d), ls='-', c='g', label='mean')
ax1.legend()


#%%
img = mpimg.imread(r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\Methacrylate\methacrylate_photo.jpg')

Smin = config_dict['Smin1']
Smax = config_dict['Smax1']
t = np.arange(Smin, Smax) / Fs * 1e6 # us

US.pltGUI(x, t, CL, Cs, L*1e3, PE, TT, img, ptxt='northwest')




#%%
img = mpimg.imread(r'G:\Unidades compartidas\Proyecto Cianocrilatos\Data\Scanner\Methacrylate\methacrylate_photo.jpg')

Smin = config_dict['Smin1']
Smax = config_dict['Smax1']
t = np.arange(Smin, Smax) / Fs * 1e6 # us
xcors = np.apply_along_axis(US.fastxcorr, 0, TT, WP)

def xcorGUI(x, x4, CL, Cs, WP, PE, TT, xcors, img, ptxt='northwest'):
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(4,3)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 1:])
    ax3 = fig.add_subplot(gs[2, 1:])
    ax4 = fig.add_subplot(gs[3, 1:])
    
    ax0.imshow(img, extent=[0, img.shape[1], x[-1], x[0]], aspect="auto")
    ax0.get_xaxis().set_ticks([])
    line1, = ax1.plot(CL, 'b')
    line2, = ax2.plot(Cs, 'r')
    line3, = ax3.plot(xcors[:,0])
    line4_PE, = ax4.plot(x4, PE[:,0])
    line4_TT, = ax4.plot(x4, TT[:,0])
    ax4.plot(x4, WP)
    ax4.set_ylim([np.min(np.c_[PE,TT,WP].flatten())*1.1, np.max(np.c_[PE,TT,WP].flatten())*1.1])
    
    ax0.set_ylabel('Position (mm)')
    ax1.set_xlabel('Position (mm)')
    ax2.set_xlabel('Position (mm)')
    ax3.set_xlabel('Samples')
    ax4.set_xlabel('Time (\u03bcs)')
    
    ax1.set_ylabel('Long. vel. (m/s)')
    ax2.set_ylabel('Shear vel. (m/s)')
    ax3.set_ylabel('Xcorr')
    ax4.set_ylabel('Amplitude (V)')
    
    hline0, = ax0.plot([0, img.shape[1]], [x[0], x[0]], c='k')
    vline1 = ax1.axvline(x[0], c='k')
    vline2 = ax2.axvline(x[0], c='k')
      
    d = US.text_position(ptxt)
    txt = ax1.text(d['x'], d['y'], f"{round(x[0], 2)}, {round(CL[0], 2)}", ha=d['ha'], va=d['va'], transform=ax1.transAxes)
    txt.set_bbox(dict(facecolor='w', edgecolor='k', alpha=0.5))
    
    plt.tight_layout()
    
    def update(idx, l, shear):
        line1.set_marker('')
        line2.set_marker('')
        
        x, y = l.get_data()
        txt.set_text(f"{round(x[idx], 2)}, {round(y[idx], 2)}")
        
        hline0.set_ydata(x[idx])
        vline1.set_xdata(x[idx])
        vline2.set_xdata(x[idx])
        
        if shear:
            auxPE = PE[:,::-1]
            auxTT = TT[:,::-1]
            auxxcors = xcors[:,::-1]
            line4_PE.set_ydata(auxPE[:,idx])
            line4_TT.set_ydata(auxTT[:,idx])
            line3.set_ydata(auxxcors[:,idx])
        else:
            line4_PE.set_ydata(PE[:,idx])
            line4_TT.set_ydata(TT[:,idx])
            line3.set_ydata(xcors[:,idx])
        
        l.set_marker('.')
        l.set_markerfacecolor('k')
        l.set_markeredgecolor('k')
        l.set_markevery([idx])
    
    
    def hover(event):
        shear = False
        if event.inaxes == ax1:
            l = line1
        elif event.inaxes == ax2:
            l = line2
            shear = True
        else:
            return
        cont, ind = l.contains(event)
        if cont:
            if len(ind["ind"]) != 0:
                update(ind["ind"][0], l, shear)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.show()

xcorGUI(x, t, CL, Cs, WP, PE, TT, xcors, img, ptxt='northwest')


#%%
n1 = 900 # tof=22
n2 = 700 # tof=8

n1 = 1000
n2 = 900

_y1 = xcors[:,n1]
_y2 = xcors[:,n2]
_envy1 = np.absolute(scsig.hilbert(_y1))
_envy2 = np.absolute(scsig.hilbert(_y2))
N = len(_y1)

_maxy1 = US.CosineInterpMax(_y1, xcor=True, UseHilbEnv=False)
_maxy2 = US.CosineInterpMax(_y2, xcor=True, UseHilbEnv=False)
_maxenvy1 = US.CosineInterpMax(_envy1, xcor=True, UseHilbEnv=False)
_maxenvy2 = US.CosineInterpMax(_envy2, xcor=True, UseHilbEnv=False)

print(_maxy1)
print(_maxy2)
print(_maxenvy1)
print(_maxenvy2)


_newmaxy1 = _maxy1
_newmaxy2 = _maxy2
_newmaxenvy1 = _maxenvy1
_newmaxenvy2 = _maxenvy2
if _maxy1 < 0:
    _newmaxy1 = N + _maxy1
if _maxy2 < 0:
    _newmaxy2 = N + _maxy2
if _maxenvy1 < 0:
    _newmaxenvy1 = N + _maxenvy1
if _maxenvy2 < 0:
    _newmaxenvy2 = N + _maxenvy2


ax1, ax2 = plt.subplots(2)[1]
ax1.plot(_y1)
ax1.plot(_y2)
ax1.plot(_envy1, c='b')
ax1.plot(_envy2, c='r')
ax1.axvline(_newmaxenvy1, c='k')
ax1.axvline(_newmaxenvy2, c='k')
ax1.axvline(_newmaxy1, c='k', ls='--')
ax1.axvline(_newmaxy2, c='k', ls='--')


ax2.plot(np.roll(_y1, 500))
ax2.plot(np.roll(_y2, 500))
ax2.plot(np.roll(_envy1, 500), c='b')
ax2.plot(np.roll(_envy2, 500), c='r')
ax2.axvline(_maxenvy1 + 500, c='k')
ax2.axvline(_maxenvy2 + 500, c='k')
ax2.axvline(_maxy1 + 500, c='k', ls='--')
ax2.axvline(_maxy2 + 500, c='k', ls='--')


#%%
_a = np.array([1,2,3,4])
_b = np.array([1,2,3,4])
_a = TT[:,900]
_b = WP
nfft = len(_a) + len(_b) - 1

_xcorab = np.fft.ifft(np.fft.fft(_a, nfft) * np.conj(np.fft.fft(_b, nfft)))
print(np.real(_xcorab))

_xcorabh = np.fft.ifft(np.fft.fft(np.imag(scsig.hilbert(_a)), nfft) * np.conj(np.fft.fft(np.imag(scsig.hilbert(_b)), nfft)))
print(np.real(_xcorabh))

_hxcorab = scsig.hilbert(np.real(_xcorab))

plt.figure()
plt.plot(np.real(_xcorab))
# plt.plot(np.real(_xcorab) - np.mean(np.real(_xcorab)))
plt.plot(np.real(_xcorabh))
# plt.plot(np.abs(_xcorabh) - np.mean(np.abs))
# plt.plot(np.imag(_hxcorab))

print(np.mean(np.real(_xcorab)))
print(np.mean(np.real(_xcorabh)))

print(np.real(_xcorab) - np.mean(np.real(_xcorab)))

#%%
_xcorabh = np.fft.ifft(np.fft.fft(_a, nfft) * np.conj(np.fft.fft(np.imag(scsig.hilbert(_b)), nfft)))

plt.figure()
plt.plot(np.real(_xcorabh))
plt.plot(np.real(_xcorab))
plt.plot(-np.imag(scsig.hilbert(np.real(_xcorab))))