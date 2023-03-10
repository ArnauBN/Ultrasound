# -*- coding: utf-8 -*-.
# pylint: disable=E501
"""
Created on Thu Nov  5 06:17:23 2020.

@author: alrom
"""

from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import numpy as np
from scipy import signal
import functools

from IPython.display import clear_output

import matplotlib.pylab as plt
from matplotlib import colors
from matplotlib.widgets import Slider, Button, SpanSelector

from . import US_Functions as USF

# from skimage.measure import marching_cubes_lewiner, marching_cubes_classic
#import plotly.graph_objects as go
# import plotly.io as pio
# from plotly.offline import plot as plyplot
#pio.renderers.default='browser'


# variable used for frequency units and scaling
FreqDict = {'hz': [1, 'Hz'], 'khz': [1e-3, 'KHz'], 'mhz': [1e-6, 'MHz'],
            'ghz': [1e-9, 'GHz']}

# variable used for time units and scaling
TimeDict = {'samples': [1, 'samples'], 'm': [1, 'm'], 'mm': [1e3, 'mm'],
            'um': [1e6, 'μm'], 'sec': [1, 'sec'], 'ms': [1e3, 'ms'],
            'us': [1e6, 'μs'], 'ns': [1e9, 'ns'], 'ps': [1e12, 'ps']}


def plot_tf(Data1, Data2=None, Fs=1, nfft=None, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 0]), f_ylims=None, f_units='Hz', f_Norm=False,
            PSD=False, dB=False, Phase=False, D1label='Ascan',
            D2label='Ascan2', FigNum=None, FgSize=None):
    """
    Plot 1 or 2 signals in time (samples or distance) and frequency.

    Parameters
    ----------
    Data1 : 1D np.array, main Ascan to plot
    Data2 : 1D np.array, secondary Ascan to plot, default None
    Fs : +float, Sampling frequency in Hz, default 1 Hz
    nfft : +int, number of points in FFT, default is None-> length of Ascan.
    Cs : +float, speed of sound, default 340 m/s
    t_units : str, units of time axis acording to TimeDict, def. 'samples'
    t_ylabel : str, time plot y label, default is 'amplitude'.
    t_Norm : Boolean, If True, plot normalized amplitude, default is False.
    t_xlims : [float, float], set xlims to time axis. The default is None.
    t_ylims : [float, float], set ylims to time axis. The default is None.
    f_xlims : [min,max] frequency in F_axis, default is [0,0]->[0, Fs/2]
    f_ylims : [float, float], set ylims to time axis. The default is None.
    f_units : str, units of frequency axis acording to FrecDict, def. 'Hz'
    f_Norm : Boolean, if True plot norm. mag. espectrum default is False.
    PSD : Boolean, if True plot power spectral density, default is True.
    dB : Boolean, if True plot freq. mag. axis in dB, default is False.
    Phase : Boolean, if true include phase in freq. plot, default is False.
    D1label : label 1 for freq. legend, default is 'Ascan'.
    D2label : label 2 for freq. legend, default is 'Ascan2'.
    FigNum : int or str, name of figure, default is None.
    FgSize = tupple, figure size in inches, if None (6,3)

    Returns
    -------
    None.

    """
    t_Scale = TimeDict[t_units.lower()][0]
    if t_units.lower() in {'m', 'mm', 'um'}:
        t_Scale = t_Scale * Cs * 1/Fs
    elif t_units.lower() in {'s', 'ms', 'us',  'ns',  'ps'}:
        t_Scale = t_Scale * 1/Fs
    t_xlabel = TimeDict[t_units.lower()][1]
    time_axis = np.arange(len(Data1)) * t_Scale
    if nfft is None:
        nfft = len(Data1)
    f_Scale = FreqDict[f_units.lower()][0]
    freq_axis = np.arange(nfft) * Fs/nfft * f_Scale
    f_xlabel = 'frequency ({})'.format(FreqDict[f_units.lower()][1])
    if Data2 is not None:
        if len(Data2) < len(Data1):
            Data2 = np.concatenate((Data2, np.zeros(len(Data1)-len(Data2))))
        else:
            Data2 = Data2[0:len(Data1)]
    if t_Norm:
        Data1 = Data1/np.max(np.abs(Data1))
        if Data2 is not None:
            Data2 = Data2/np.max(np.abs(Data2))
    X = np.fft.fft(Data1, nfft)
    XM = np.abs(X)
    if f_Norm:
        XM = XM / np.amax(XM)
    if Data2 is not None:
        Y = np.fft.fft(Data2, nfft)
        YM = np.abs(Y)
        if f_Norm:
            YM = YM / np.amax(YM)
    f_ylabel = 'Magnitude'
    if PSD:
        XM = XM**2
        f_ylabel = 'PSD'
        if Data2 is not None:
            YM = YM**2
    if dB:
        XM = 10*np.log10(XM)
        f_ylabel = f_ylabel + ' (dB)'
        if Data2 is not None:
            YM = 10*np.log10(YM)
    if FigNum is None:
        FigNum = 1
    if FgSize is None:
        FgSize=(6,3)
    fig = plt.figure(num=FigNum, clear=True, figsize=FgSize)
    if Data2 is not None:
        axs1 = fig.add_subplot(221)
        axs1b = fig.add_subplot(223)
        axs1b.plot(time_axis, Data2, 'b')
        axs1b.set_xlabel(t_xlabel)
        axs1b.set_ylabel(D2label)
        t_ylabel = D1label
        axs2 = fig.add_subplot(122)
    else:
        axs1 = fig.add_subplot(211)
        axs2 = fig.add_subplot(212)
    axs1.plot(time_axis, Data1, 'k')
    axs1.set_xlabel(t_xlabel)
    axs1.set_ylabel(t_ylabel)
    if t_xlims is not None:
        axs1.set_xlim(t_xlims)
        if Data2 is not None:
            axs1b.set_xlim(t_xlims)
    else:
        axs1.set_xlim(time_axis[0], time_axis[-1])
        if Data2 is not None:
            axs1b.set_xlim(time_axis[0], time_axis[-1])
    if t_ylims is not None:
        axs1.set_xlim(t_ylims)
        if Data2 is not None:
            axs1b.set_ylim(t_ylims)
    axs2.plot(freq_axis, XM, 'k', label=D1label)
    if Data2 is not None:
        axs2.plot(freq_axis, YM, 'b', label=D2label)
        axs2.legend()
    axs2.set_xlabel(f_xlabel)
    axs2.set_ylabel(f_ylabel)
    if Phase:
        axs2b = axs2.twinx()
        axs2b.plot(freq_axis, np.angle(X), 'k--')
        axs2b.set_ylabel('phase')
        axs2b.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axs2b.set_yticklabels([r'-$\pi$', r'-$\pi$/2', '0', r'$\pi$/2', r'$\pi$'])
        if Data2 is not None:
            axs2b.plot(freq_axis, np.angle(X), 'b--')
    if f_xlims[1] == 0:
        axs2.set_xlim(f_xlims[0], freq_axis[-1]/2)
    else:
        axs2.set_xlim(np.array(f_xlims))
    fig.tight_layout()
    plt.show()

def multiplot_tf(Data, Fs=1, nfft=None, Cs=343, t_units='samples',
            t_ylabel='amplitude', t_Norm=False, t_xlims=None, t_ylims=None,
            f_xlims=([0, 0]), f_ylims=None, f_units='Hz', f_Norm=False,
            PSD=False, dB=False, label=None, Independent=False, FigNum=None,
            FgSize=None):
    """
    Plot multiple signals in time (samples or distance) and frequency.

    Parameters
    ----------
    Data : np.array, Ascan to plot in a matrix, (N Signals x Signal Length)    
    Fs : +float, Sampling frequency in Hz, default 1 Hz
    nfft : +int, number of points in FFT, default is None-> length of Ascan.
    Cs : +float, speed of sound, default 340 m/s
    t_units : str, units of time axis acording to TimeDict, def. 'samples'
    t_ylabel : str, time plot y label, default is 'amplitude'.
    t_Norm : Boolean, If True, plot normalized amplitude, default is False.
    t_xlims : [float, float], set xlims to time axis. The default is None.
    t_ylims : [float, float], set ylims to time axis. The default is None.
    f_xlims : [min,max] frequency in F_axis, default is [0,0]->[0, Fs/2]
    f_ylims : [float, float], set ylims to time axis. The default is None.
    f_units : str, units of frequency axis acording to FrecDict, def. 'Hz'
    f_Norm : Boolean, if True plot norm. mag. espectrum default is False.
    PSD : Boolean, if True plot power spectral density, default is True.
    dB : Boolean, if True plot freq. mag. axis in dB, default is False.    
    label : label legend, default is 'Ascan'.   
    Indpendent : Boolean, if False plot all togeter, if True plot separated
    FigNum : int or str, name of figure, default is None.

    Returns
    -------
    None.

    """
    [NumAscans, AscanLen] = Data.shape
    t_Scale = TimeDict[t_units.lower()][0]
    if t_units.lower() in {'m', 'mm', 'um'}:
        t_Scale = t_Scale * Cs * 1/Fs
    elif t_units.lower() in {'s', 'ms', 'us',  'ns',  'ps'}:
        t_Scale = t_Scale * 1/Fs
    t_xlabel = TimeDict[t_units.lower()][1]
    time_axis = np.arange(AscanLen) * t_Scale
    if nfft is None:
        nfft = AscanLen
    f_Scale = FreqDict[f_units.lower()][0]
    freq_axis = np.arange(nfft) * Fs/nfft * f_Scale
    f_xlabel = 'frequency ({})'.format(FreqDict[f_units.lower()][1])    
    if t_Norm:
        for i in np.arange(NumAscans):
            Data[i,:] = Data[i,:]/np.max(np.abs(Data[i,:]))        
    X = np.fft.fft(Data, nfft)
    XM = np.abs(X)
    if f_Norm:
        for i in np.arange(NumAscans):
            XM[i,:] = XM[i,:] / np.amax(XM[i,:])    
    f_ylabel = 'Magnitude'
    if PSD:
        XM = XM**2
        f_ylabel = 'PSD'        
    if dB:
        XM = 10*np.log10(XM)
        f_ylabel = f_ylabel + ' (dB)'

    if FigNum is None:
        FigNum = 1
    if FgSize is None:
        FgSize=(6,3)
    fig = plt.figure(num=FigNum, clear=True, figsize=FgSize)
    cmap = plt.get_cmap("tab10")
    if Independent:
        for i in np.arange(NumAscans):
            axs1 = fig.add_subplot(NumAscans, 2, 2*i+1)
            axs1.plot(time_axis, Data[i, :].T, color=cmap(i))
            axs1.set_xlabel(t_xlabel)
            axs1.set_ylabel(t_ylabel)
            if t_xlims is not None:
                axs1.set_xlim(t_xlims)
            else:
                axs1.set_xlim(time_axis[0], time_axis[-1])
            if t_ylims is not None:
                axs1.set_xlim(t_ylims)
        axs2 = fig.add_subplot(122)
    else:
        axs1 = fig.add_subplot(211)    
        axs1.plot(time_axis, Data.T)
        axs1.set_xlabel(t_xlabel)
        axs1.set_ylabel(t_ylabel)
        if t_xlims is not None:
            axs1.set_xlim(t_xlims)
        else:
            axs1.set_xlim(time_axis[0], time_axis[-1])
        if t_ylims is not None:
            axs1.set_xlim(t_ylims)  
        if label is not None:
            axs1.legend(label)
        axs2 = fig.add_subplot(212)
        
    # axs2 = fig.add_subplot(212)
    axs2.plot(freq_axis, XM.T)    
    axs2.set_xlabel(f_xlabel)
    axs2.set_ylabel(f_ylabel)

    if f_xlims[1] == 0:
        axs2.set_xlim(f_xlims[0], freq_axis[-1]/2)
    else:
        axs2.set_xlim(np.array(f_xlims))
    if label is not None:        
        axs2.legend(label)
    fig.tight_layout()
    plt.show()
    
    
def Plot_Specgram(x, Fs=1.0, window=('tukey', 0.25), WinSize=None,
                  Overlap=None, nfft=None, detrend='constant', 
                  return_onesided=True, scaling='density', Units='dB',axis=-1,
                  mode='psd', t_units='samples', Time_OffSet=0, f_units='Hz',
                  f_xlims=([0, 0]), Cs=343, FigNum=1, 
                  Filled=True, Imshow=False, ColorMap='plasma', Shading='flat',
                  vmin=None, vmax=None):
    """
    Calculate and plot spectrogramo af a 1D signal.

    Parameters
    ----------
        x =input 1D signal
        fs = sampling frecuency in Hz
        window = (sort of window in text, parameter), defsault ('tukey', 0.25)
        WinSize = window size in samples, default 256 to None
        Overlap = windows overlap in number of samples, default WinSize//8 to None
        nfft = number of points of the fft, default WinSize to None
        detrend = function for detren segments {'linear','constant',False}, default 'constant', which removes mean of data
        return_onesided = if True return one-sided spectrum, if false two sided (unless complex data, then always twosided)
        scaling = {‘density’, ‘spectrum’ }, default density as V**2/Hz
        Units = {'dB','linear'}, units of the output, default dB
        axis = axis along which spectrogram is computed, default last axis (-1)
        mode =  [‘psd’, ‘complex’, ‘magnitude’, ‘angle’, ‘phase’]
         ‘complex’ is equivalent to the output of stft with no padding or boundary extension. 
         ‘magnitude’ returns the absolute magnitude of the STFT. 
         ‘angle’ and ‘phase’ return the complex angle of the STFT, with and without unwrapping, respectively.
        t_units : str, units of time axis acording to TimeDict, def. 'samples'
        Time_OffSet = time offset to be substracted from time axis, in same units, default 0
        f_units : str, units of frequency axis acording to FrecDict, def. 'Hz'
        f_xlims : [min,max] frequency in F_axis, default is [0,0]->[0, Fs/2]
        Cs : speed of wave in media m/s, default 343, sound in air
        FigNum = figure number, default 1
        Filled = Boolean, to fill in both sides to align both figures
        ImShow = boolean, True to plot imshow, False (default) to plot pcolormesh
        ColorMap = text, colormap to use, default 'plasma'
        Shading = {'flat','gouraud'}, shading interpolation method.
        
    Returns
    -------
    f = array of sample frequencies
    t = array of segment times
    Sxx = Spectrogram of x.
    """
    # FreqDict = {'hz': [1, 'Hz'], 'khz': [1e-3, 'KHz'], 'mhz': [1e-6, 'MHz'], 'ghz': [1e-9, 'GHz']}
    f, t, Sxx = signal.spectrogram(x, fs=Fs, window=window,
                                   nperseg=WinSize, noverlap=Overlap, nfft=nfft,
                                   detrend=detrend, return_onesided=return_onesided,
                                   scaling=scaling, axis=axis, mode=mode)
    # next is used to fill in both sides the spectrogram to align to plot in time
    # Filled = True
    cero = np.min(np.min(Sxx)) * np.ones((Sxx.shape[0], 1))
    if Filled:
        # t = np.insert(t, 0, 0)
        # Sxx = np.concatenate((cero, Sxx, cero), axis=1)
        t = np.append(t, t[-1]+t[1])        
        Sxx = np.concatenate((Sxx, cero), axis=1)
    cblabel = mode
    if Units.lower() == 'db':
        Sxx = np.log10(Sxx)
        cblabel = cblabel + ' (dB)'
    f = f * FreqDict[f_units.lower()][0]
    freq_label = 'frequency ({})'.format(FreqDict[f_units.lower()][1])
    F1 = USF.findIndexInAxis(f, f_xlims[0])
    if f_xlims[1] == 0:
        F2 = USF.findIndexInAxis(f, Fs/2)
    else:
        F2 = USF.findIndexInAxis(f, f_xlims[1])
    t_Scale = TimeDict[t_units.lower()][0]
    if t_units.lower() in {'m', 'mm', 'um'}:
        t_Scale = t_Scale * Cs * 1/Fs
    elif t_units.lower() in {'s', 'ms', 'us',  'ns',  'ps'}:
        t_Scale = t_Scale * 1/Fs
    t_xlabel = TimeDict[t_units.lower()][1]
    time_axis = np.arange(len(x)) * t_Scale - Time_OffSet
    fig = plt.figure(num=FigNum, clear=True)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    if Imshow:
        ax1.imshow(Sxx[F1:F2, :], cmap=ColorMap, interpolation='bilinear')
    else:
        im = ax1.pcolormesh(t * Fs * t_Scale - Time_OffSet, f[F1:F2],
                            Sxx[F1:F2, :], cmap=ColorMap, shading=Shading,
                            vmin=vmin, vmax=vmax)
        # axins1 = inset_axes(ax1, width="2%", height="100%", loc='lower left',
        #                     bbox_to_anchor=(1.01, -0.01, 1, 1), bbox_transform=ax1.transAxes)
        # fig.colorbar(im, cax=axins1, orientation="vertical")
        # ax1.set_xlim(Time_Axis[0], Time_Axis[-1])
#        axins1.xaxis.set_ticks_position("right")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(im, cax=cax, label=cblabel)
        plt.tight_layout()

    ax1.set_ylabel(freq_label)
    ax1.set_xlabel(t_xlabel)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.plot(time_axis, x)
    ax2.set_xlim(time_axis[0], time_axis[-1])
    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel(t_xlabel)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", "5%", pad="3%")
    cax2.set_visible(False)
    plt.tight_layout()
    plt.show()


def Plot_Bscan(Bscan, Xtoplot=None, Fs=1, t_units='samples', Time_OffSet=0,
               Cs=343, Xaxis_Scale=1, TScal=0, FigNum=1, X_Label='X axis',
               cblabel='Amp', y_label='Ascan',
               SelfNorm = False, ColorMap='seismic', Shading='gouraud'):
    """
    Plot Bscan and Ascan. Ascan can be selected with mouse.

    Parameters
    ----------
        Bscan, matrix with Bscan data
        Xtoplot = select Ascan from matrix, integer
        Fs : +float, sampling frequency in Hz, default 1 Hz
        t_units : str, units of time axis acording to TimeDict, def. 'samples'
        Time_OffSet : float, offset of time axis in same units , defaul 0
        Cs : speed of sound in media in m/s, default 343 m/s
        Xaxis_Scale, scale factor to apply to Xaxis
        TScal, scale factor to apply to t_axis, if zero, use units
        FigNum = figure number
        X_Label = text for X axis
        SelfNorm : Boolean, if True normalize each scan
        clabel = str, lbel of colormap
        ColorMap : str, colormap, default 'seismic'
        Shading = {'flat','gouraud'}, shading interpolation method.

    Fcuntions
    ---------
        handlerBscan(fig, ax1, ax2, Bscan, Xaxis, time_axis, event)
            Used to updeate plot on click.
    """
    if SelfNorm:
        Bscan=USF.normalizeAscans(Bscan)
    if TScal == 0:
        t_Scale = TimeDict[t_units.lower()][0]
        if t_units.lower() in {'m', 'mm', 'um'}:
            t_Scale = t_Scale * Cs * 1/Fs
        elif t_units.lower() in {'s', 'ms', 'us',  'ns',  'ps'}:
            t_Scale = t_Scale * 1/Fs
        t_xlabel = TimeDict[t_units.lower()][1]
        time_axis = np.arange(Bscan.shape[1]) * t_Scale - Time_OffSet
    else:
        t_xlabel = y_label
        time_axis = np.arange(Bscan.shape[1]) * TScal - Time_OffSet
    Xaxis = np.arange(0, Bscan.shape[0]) * Xaxis_Scale
    fig = plt.figure(num=FigNum, clear=True, figsize=[6.4 , 4.8])
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    im = ax1.pcolormesh(time_axis, Xaxis, Bscan, cmap=ColorMap, shading=Shading)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax, label=cblabel)
    plt.tight_layout()
    ax1.set_ylabel(X_Label)
    ax1.set_xlabel('depth ('+t_xlabel+')')

    # plot a line in the selected X position
    if Xtoplot is None:
        Xtoplot = int(np.round(Bscan.shape[0]/2))
    X = Xaxis[Xtoplot] + (Xaxis[1]-Xaxis[0])/2
    ax1.plot([time_axis[0], time_axis[-1]], [X, X], 'k')

    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.plot(time_axis, Bscan[Xtoplot, :], 'k')
    plt.xlim(time_axis[0],time_axis[-1])
    ax2.set_ylabel(y_label)
    ax2.set_xlabel('depth ('+t_xlabel+')')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", "5%", pad="3%")
    cax2.set_visible(False)
    plt.tight_layout()
    # handler para manejar click de ratón sobre gráfica
    handler_wrapper = functools.partial(handlerBscan, fig, ax1, ax2, Bscan, Xaxis, time_axis)
    fig.canvas.mpl_connect('button_press_event', handler_wrapper)
    plt.show()


# Used to handle mouse clicking on axis
def handlerBscan(fig, ax1, ax2, Bscan, Xaxis, time_axis, event):
    # Verify click is within the axes of interest
    if ax1.in_axes(event):
        ax1.lines[0].remove()
        ax2.lines[0].remove()
        Xtoplot = USF.findIndexInAxis(Xaxis, event.ydata)
        X = Xaxis[Xtoplot] + (Xaxis[1]-Xaxis[0])/2
        ax2.plot(time_axis, Bscan[Xtoplot, :], 'k')
        ax1.plot([time_axis[0], time_axis[-1]], [X, X], 'k')
        fig.canvas.draw()


def wireFrame_Bscan(Bscan, cstride=1, rstride=0, t_units='samples', Time_OffSet=0,
                    SelfNorm = False, Cs=343, Fs=1,
                    Xaxis_Scale=1, FigNum=1, X_Label='X axis'):
    """
    Plot Bscan in wire frame mode.
    
    Parameters
    ----------
        Bscan, matrix with Bscan data
        t_units : str, units of time axis acording to TimeDict, def. 'samples'
        FigNum = figure number
        X_Label = text for X axis
        cstride : +int, separation between lines in X axis, default 1
        rstride : +int, separation between line in time Y, default 0
        Cs : +float speed of soun in media, default 343, epeed in air
        Fs : +float sampling frequency in Hz, default 1 Hz

    Returns
    -------
    None.

    """
    if SelfNorm:
        Bscan=USF.normalizeAscans(Bscan)
    t_Scale = TimeDict[t_units.lower()][0]
    if t_units.lower() in {'m', 'mm', 'um'}:
        t_Scale = t_Scale * Cs * 1/Fs
    elif t_units.lower() in {'s', 'ms', 'us',  'ns',  'ps'}:
        t_Scale = t_Scale * 1/Fs
    t_xlabel = TimeDict[t_units.lower()][1]
    time_axis = np.arange(Bscan.shape[1]) * t_Scale - Time_OffSet
    Xaxis = np.arange(0, Bscan.shape[0]) * Xaxis_Scale
    fig = plt.figure(num=FigNum, clear=True)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(Xaxis, time_axis)
    ax.plot_wireframe(X, Y, Bscan.T, rstride=0, cstride=cstride)
    ax.set_ylabel(t_xlabel)
    ax.set_xlabel(X_Label)
    plt.tight_layout()


def surf_Bscan(Bscan, cstride=1, rstride=2, t_units='samples', Cs=343, Fs=1,
               Time_OffSet=0,Xaxis_Scale=1, FigNum=1, time_Label='samples', SelfNorm = False,
               X_Label='X axis', ColorMap='seismic', linewidth=0, antialiased=True):
    """
    Plot Bscan in surf mode.

    Parameters
    ----------
        Bscan, matrix with Bscan data
        t_units : str, units of time axis acording to TimeDict, def. 'samples'
        Time_Offset : +float, time offset in same units, default 0
        Cs : +float speed of soun in media, default 343, epeed in air
        Fs : +float sampling frequency in Hz, default 1 Hz
        Xaxis_Scale, scale factor to apply to Xaxis
        FigNum = figure number
        time_Label = text for time axis label
        X_Label = text for X axis
        cstride : +int, separation between lines in X axis, default 1
        rstride : +int, separation between line in time Y, default 0
        Colormap to use
        linewidth, default 0
        antialiased, default True.

    Returns
    -------
    None.

    """
    if SelfNorm:
        Bscan=USF.normalizeAscans(Bscan)
    t_Scale = TimeDict[t_units.lower()][0]
    if t_units.lower() in {'m', 'mm', 'um'}:
        t_Scale = t_Scale * Cs * 1/Fs
    elif t_units.lower() in {'s', 'ms', 'us',  'ns',  'ps'}:
        t_Scale = t_Scale * 1/Fs
    t_xlabel = TimeDict[t_units.lower()][1]
    time_axis = np.arange(Bscan.shape[-1]) * t_Scale - Time_OffSet
    Xaxis = np.arange(0, Bscan.shape[0]) * Xaxis_Scale
    fig = plt.figure(num = FigNum, clear=True)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(Xaxis, time_axis)
    surf = ax.plot_surface(X, Y, Bscan.T,rstride=2, cstride=1, cmap=ColorMap,
                        linewidth=linewidth, antialiased=antialiased)
    ax.set_ylabel(time_Label)
    ax.set_xlabel(X_Label)
    plt.tight_layout()
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# def ply_surf_Bscan(Bscan, autosize=True, width=1200, height=800,
#                    aspectmode='manual',aspectratio=dict(x=1.8, y=1, z=1),
#                    t_units='samples', t_offset=0,
#                    Xaxis_scale=1, X_label='X axis', Z_label='Amplitude',
#                    Cs=343, Fs=1, FigNum=1, SelfNorm=False,
#                    ):
#     """
#     Plot Bscan in surf mode.

#     Parameters
#     ----------
#         Bscan : +float array, matrix with 2D data
#         SelfNorm : boolean to normalize each Ascan by its maximum
#         autosize=True
#         width : figure width in case autosize False, default 1200
#         height : figure height in case autosize False, default 800
#         aspectmode : aspect of axis, default 'manual'
#         aspectratio : , ratio between axis default, dict(x=1.8, y=1, z=1),
#         t_units : str, units of time axis acording to TimeDict, def. 'samples'
#         t_offset : +float, time offset in same units, default 0
#         Cs : +float speed of soun in media, default 343, epeed in air
#         Fs : +float sampling frequency in Hz, default 1 Hz
#         X_scale, scale factor to apply to X_axis
#         FigNum = figure number or name
#         X_label = text for X axis, default 'X axis'
#         Z_label = text for Z axis, default 'amplitude'
#     Returns
#     -------
#     None.

#     """
#     if SelfNorm:
#         Bscan = USF.normalizeAscans(Bscan)
#     t_scale = TimeDict[t_units.lower()][0]
#     if t_units.lower() in {'m', 'mm', 'um'}:
#         t_scale = t_scale * Cs * 1/Fs
#     elif t_units.lower() in {'s', 'ms', 'us',  'ns',  'ps'}:
#         t_scale = t_scale * 1/Fs
#     t_label = TimeDict[t_units.lower()][1]
#     t_axis = np.arange(Bscan.shape[-1]) * t_scale - t_offset
#     X_axis = np.arange(0, Bscan.shape[0]) * Xaxis_scale

#     fig = go.Figure(data=[go.Surface(z=Bscan, x=t_axis, y=X_axis)])
#     fig.update_layout(autosize=True,# width=1200, height=800,
#                   scene=dict(aspectmode='manual',
#                              aspectratio=dict(x=1.8, y=1, z=1),
#                              xaxis_title=t_label,
#                              yaxis_title=X_label,
#                              zaxis_title=Z_label,
#                              )
#                   )
#     plyplot(fig)



def Plot_Cscan(Cscan, Slice=None, SoP='std', XaxisStep=1, YaxisStep=1, SelfNorm = False,
               Xlabel='X axis (mm)', Ylabel='Y axis (mm)', FigNum=1, cblabel='avg',
               Imshow=False, ColorMap='seismic', Shading='gouraud'):
    '''
    plots Cscan.

    Parameters
    ----------
        Cscan : np.array matrix with Cscan data
        Slice : +int, select slice to plot, avoit processing
        SoP : str, Sort of plot {'std', 'max', 'pow'}
        XaxisStep : +float, step size in X axis, mm
        YaxisStep : +float, step size in Y axis, mm
        Xlabel : str, label of X axis
        Ylabel : str, label of Y axis
        FigNum = num or str, figure number
        cblabel = str, label of colormap
        Imshow : Boolean, if True, plot imshow, if false plot pcolormesh.
        colormap : str, colormap to use. Default 'seismic'
        Shading = {'flat','gouraud'}, shading interpolation method.
        
    Functions
    ---------
    operaCscan(Data, SoP='std'), to process data
    '''
    if SelfNorm:
        Cscan=USF.normalizeAscans(Cscan)
    Xaxis = np.arange(Cscan.shape[0]) * XaxisStep
    Yaxis = np.arange(Cscan.shape[1]) * YaxisStep
    fig = plt.figure(num=FigNum, clear=True)
    ax1 = fig.add_subplot(111)
    if Slice is not None:
        Data = Cscan[:, :, Slice]
    else:
        Data = operaCscan(Cscan, SoP=SoP)  # process Cscan
    if Imshow:
        ax1.imshow(Data, aspect='auto', cmap='RdBu', interpolation='bilinear')
    else:
        im = ax1.pcolormesh(Yaxis, Xaxis, Data, cmap=ColorMap, shading=Shading)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(im, cax=cax, label=cblabel)
        plt.tight_layout()
        ax1.axis('image')
    plt.show()

def SliceCscan(Cscan, Ztoplot=None, Depth=None, Xtoplot=None, Ytoplot=None,
               SoP='avg', Z_axis=[], X_axis=[], Y_axis=[], Z_OffSet=0, y_Label='Ascan',
               X_Label='X axis', Y_Label='X axis', Z_Label='Z axis', cblabel='Amp',
               SelfNorm = False, FixScale = False,
               FigNum=1, ColorMap='jet', Shading='gouraud'):
    """
    Plot Cscan Slice.

    Parameters
    ----------
    Cscan : Cscan data
    Ztoplot : Z to plot, same units as Z_axis, if None operates on all matrix
    Depth : Depth to plot, same units as Z_axis, if None or 0, takes single slice
    Xtoplot : +float, X to plot, if None take the midle, same units as X_axis
    Ytoplot : +float, Y to plot, if None take the midle, same units as Y_axis
    SoP : str, sort of operation 'std', 'max', 'pow'
    Z_axis : array, Z axis 1D dimension data, if [], dimension of last axis
    X_axis : array, X axis 2D dimension data, if [], dimension of second axis
    Y_axis : array, Y axis 2D dimension data, if [], dimension of first axis
    Z_OffSet : ofsett of time axis, in same units as Z_axis    
    X_Label : str, label of X axis    
    Y_Label : str, Y axes label
    Z_Label : str, Z axes label
    y_Label : str, y axes label for 1D plot
    cblabel : label of colorbar
    FigNum : number or name of figure            
    FixScale : Boolean, True to fix scale fo colorbar 
    ColorMap : str, color map to apply, defaul 'jet'
    Shading : str, shading 'gouraud' o 'flat'
    
    Functions
    ---------
    handlerCscan(), used to handle mouse click

    Returns
    -------
    None.

    """
    if SelfNorm:
        Cscan=USF.normalizeAscans(Cscan)
    vmin = None  # limits of colormap, None -> auto
    vmax = None  # limits of colormap, None -> auto
    # build axes and check point to plot in each axes
    if Z_axis==[]:
        Z_axis = np.arange(Cscan.shape[2])
    if X_axis==[]:
        X_axis = np.arange(Cscan.shape[1])
    if Y_axis==[]:
        Y_axis = np.arange(Cscan.shape[0]) 
    
    # ind = np.unravel_index(np.argmax(Cscan, axis=None), Cscan.shape)    
    if Xtoplot is None:
        Xtoplot = USF.findIndexInAxis(X_axis, X_axis[-1]/2)
        # Xtoplot = ind[1]
    else:
        Xtoplot = int(USF.findIndexInAxis(X_axis, Xtoplot))  # find indexes
    if Ytoplot is None:
        Ytoplot = USF.findIndexInAxis(Y_axis, Y_axis[-1]/2)
        # Ytoplot = ind[0]
    else:
        Ytoplot = int(USF.findIndexInAxis(Y_axis, Ytoplot))  # find indexes

    if Ztoplot is not None:
        if Depth is None or Depth == 0:
            Z = USF.findIndexInAxis(Z_axis, Ztoplot)  # find indexes
            Data = Cscan[:, :, Z]  # use the specific slice
            if FixScale:
                vmin = np.amin(Cscan)
                vmax = np.amax(Cscan)
        else:
            if Ztoplot + Depth > Z_axis[-1]:
                Ztoplot = Z_axis[-1] - Depth
            Z = USF.findIndexInAxis(Z_axis, Ztoplot)  # find indexes
            Zplus = USF.findIndexInAxis(Z_axis, Ztoplot + Depth)
            if FixScale:
                v1=[]
                v2=[]
                for i in np.arange(0, len(Z_axis)-Zplus):#, (Zplus-Z)):
                    v1.append(np.amin(operaCscan(Cscan[:, :, i:i + Zplus], SoP=SoP)))
                    v2.append(np.amax(operaCscan(Cscan[:, :, i:i + Zplus], SoP=SoP)))
                vmin = np.amin(v1)
                vmax = np.amax(v2)
            Data = operaCscan(Cscan[:, :, Z:Z + Zplus], SoP=SoP)  # process slice
    else:
        Z = USF.findIndexInAxis(Z_axis, Ztoplot)
        Data = operaCscan(Cscan, SoP=SoP)  # process all data
    fig = plt.figure(num=FigNum, clear=True)  # create figure
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)  # make axis 1
    ax2 = plt.subplot2grid((3, 1), (2, 0))  # make axis 2
    im = ax1.pcolormesh(X_axis, Y_axis, Data, cmap=ColorMap, shading=Shading)    
    ax1.plot(X_axis[Xtoplot], Y_axis[Ytoplot], 'kx')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(im, cax=cax, label=cblabel)
    # im.set_clim(vmin, vmax)
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # im.set_norm(norm)
    plt.tight_layout()
    ax1.axis('image')
    ax1.set_ylabel(Y_Label)
    ax1.set_xlabel(X_Label)

    ax2 = plt.subplot2grid((3, 1), (2, 0))
    Ascan = Cscan[Ytoplot, Xtoplot, :]  # ascan to plot
    ax2.clear()
    ax2.plot(Z_axis, Ascan)
    if Ztoplot is not None:
        if Depth is None or Depth == 0:
            ax2.axvline(x=Z_axis[Z], color='r')  # plot vertical line
        else:
            # create and plot rectangle of the slice if any
            RectWidth = Z_axis[Zplus] - Z_axis[Z]
            RectHigh = abs((np.amax(Cscan))-(np.amin(Cscan)))
            rect = plt.Rectangle((Z_axis[Z], np.amin(Cscan)), RectWidth, RectHigh,
                                 color='b', alpha=0.3)
            ax2.add_patch(rect)
    plt.xlim(Z_axis[0], Z_axis[-1])
    ax2.set_ylabel(y_Label)
    ax2.set_xlabel(Z_Label)
    ax2.set_xlim([Z_axis[0], Z_axis[-1]])
    ax2.set_ylim([np.amin(Cscan), np.amax(Cscan)])
    # handler para manejar click de ratón sobre gráfica
    X2p = Xtoplot
    Y2p = Ytoplot
    handler_wrapper = functools.partial(handlerCscan, fig, ax1, ax2, cax, X2p, Y2p, 
                                        Cscan, Ascan, X_axis, Y_axis, Z_axis,
                                        Z, Depth, SoP, im, cblabel, ColorMap,
                                        vmin, vmax, Shading)
    fig.canvas.mpl_connect('button_press_event', handler_wrapper)
    plt.show()


# Used to handle mouse clicking on axis
def handlerCscan(fig, ax1, ax2, cax,  X2p, Y2p, Cscan, Ascan, X_axis, Y_axis, Z_axis, 
                 Z, Depth, SoP, im, cblabel, ColorMap, vmin, vmax, Shading, event):
    # Verify click is within the axes of interest
    if ax2.in_axes(event):
        Z = USF.findIndexInAxis(Z_axis, event.xdata)  # find indexes
        if Depth is None or Depth == 0:
            Data = Cscan[:, :, Z]  # use the specific slice
            ax2.lines[1].remove()
            ax2.axvline(x=Z_axis[Z], color='r')  # plot vertical line
        else:
            ax2.patches = []
            if event.xdata + Depth > Z_axis[-1]:
                Z = Z_axis[-1] - Depth
            Zplus = USF.findIndexInAxis(Z_axis, event.xdata + Depth)
            Data = operaCscan(Cscan[:, :, Z:Z + Zplus], SoP=SoP)  # use big slice
            # create and plot rectangle of the slice if any
            RectWidth = Z_axis[Zplus] - Z_axis[Z]
            RectHigh = abs((np.amax(Cscan))-(np.amin(Cscan)))
            rect = plt.Rectangle((Z_axis[Z], np.amin(Cscan)), RectWidth, RectHigh,
                                 color='b', alpha=0.3)
            ax2.add_patch(rect)
        Y = ax1.lines[0].get_ydata()[0]
        X = ax1.lines[0].get_xdata()[0]
        # print(Y)
        Y2p = USF.findIndexInAxis(Y_axis, Y)  # find indexes
        # print(X)
        X2p = USF.findIndexInAxis(X_axis, X)  # find indexes
        ax1.clear()
        cax.clear()
        im = ax1.pcolormesh(X_axis, Y_axis, Data, cmap=ColorMap, 
                            vmin=vmin, vmax=vmax, shading=Shading)
        plt.colorbar(im, cax=cax, label=cblabel)
        
        punto, = ax1.plot(X_axis[X2p], Y_axis[Y2p], 'kx')
        plt.tight_layout()
        ax1.axis('image')
        fig.canvas.draw()

    elif ax1.in_axes(event):  
        Z = USF.findIndexInAxis(Z_axis, ax2.lines[1].get_xdata()[0])  # find indexes
        Y2p = USF.findIndexInAxis(Y_axis, event.ydata)  # find indexes
        X2p = USF.findIndexInAxis(X_axis, event.xdata)  # find indexes 
        
        ax2.clear()
        
        ax2.plot(Z_axis, Cscan[Y2p, X2p, :], 'b')
        ax2.axvline(x=Z_axis[Z], color='r')  # plot vertical line
        ax2.set_xlim([Z_axis[0], Z_axis[-1]])
        ax2.set_ylim([np.amin(Cscan), np.amax(Cscan)])
        ax1.lines[0].remove()
        # if Depth is None or Depth == 0:
            # ax2.axvline(x=time_axis[Z], color='r')  # plot vertical line
            # Data = Cscan[:, :, Z]
        # else:
            # ax2.patches = []
            # if event.xdata + Depth > time_axis[-1]:
            #     Z = time_axis[-1] - Depth
            # Zplus = USF.findIndexInAxis(time_axis, event.xdata + Depth)
            # Data = operaCscan(Cscan[:, :, Z:Z + Zplus], SoP=SoP)  # use big slice
            # # create and plot rectangle of the slice if any
            # RectWidth = time_axis[Zplus] - time_axis[Z]
            # RectHigh = abs((np.max(Ascan))-(np.min(Ascan)))
            # rect = plt.Rectangle((time_axis[Z], np.min(Ascan)), RectWidth, RectHigh,
            #                       color='b', alpha=0.3)
            # ax2.add_patch(rect)
        # ax1.clear()
        # cax.clear()
        # im = ax1.pcolormesh(Yaxis, Xaxis, Data, cmap=ColorMap, 
        #                     vmin=vmin, vmax=vmax, shading=Shading)
        # plt.colorbar(im, cax=cax, label=cblabel)
        ax1.plot(X_axis[X2p], X_axis[Y2p], 'kx')
        plt.tight_layout()
        ax1.axis('image')
        fig.canvas.draw()


def operaCscan(Data, SoP='std'):
    """
    Operates a Cscan.

    Parameters
    ----------
    Data : Cscan to be processed, 3D matrix, (X, Y, Ascan)
    SoF : str, sort of processing to apply, 'std', 'max', 'pow'.

    Returns
    -------
    processed matrix as 2D (X, Y.)

    """
    if SoP.lower() == 'std':
        return np.std(Data, axis=2)
    elif SoP.lower() == 'max':
        return np.abs(Data).max(axis=2)
    elif SoP.lower() == 'pow':
        return np.sum(Data**2, axis=2) / Data.shape[2]
    elif SoP.lower() == 'avg':
        return np.mean(Data, axis=2)
    else:
        return None


def plot3Dvol(Data, threshold, space=0.1, step_size=2,
              Xaxis=None, Yaxis=None, Zaxis=None, 
              SoP='mesh', colormap='Spectral', FigName=1):
    """
    Plot 3D volume. To be used in processed Cscans

    Parameters
    ----------
    Data : 3D array to plot
    Threshold : +float 0<th<1, threshold to create volumes
    space : +float, spacing between vertices, default 0.1
    step_size : +int, step between surfaces, default 2
    SoP : str, dort of plot 'mesh' or 'tri_surf'
    FigName : int or str, name of figure.

    Returns
    -------
    None.

    """
    spacing = (space, space, space)
    Data = Data / np.max(Data)

    # threshold = 0.1
    verts, faces, _, _ = marching_cubes_lewiner(Data, threshold,
                                                spacing=spacing, step_size=step_size)
    fig = plt.figure(num=FigName, clear=True)
    ax = fig.add_subplot(111, projection='3d')
    if SoP.lower() == 'mesh':
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], linewidths=0, alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    else:
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        cmap=colormap, lw=1)
    ax.set_xlim(0, Data.shape[0]*space)
    ax.set_ylim(0, Data.shape[1]*space)
    ax.set_zlim(0, Data.shape[2]*space)
    plt.show()


def videoCscan(Cscan, Depth=None, Xtoplot=None, Ytoplot=None,
               SoP='std', t_units='samples', Time_OffSet=0, SelfNorm = False,
               Xaxis_Scale=1, X_Label='X axis', Yaxis_Scale=1,  Y_Label='X axis',
               FigNum=1, cblabel='Amp', Cs=343, Fs=1, FixScale = False,
               ColorMap='jet', Shading='gouraud'):
    """
    Make a video of Plot Cscan Slice.

    Parameters
    ----------
    Cscan : Cscan data
    Depth : Depth to plot, same units as time, if None or 0, takes single slice
    Xtoplot : +float, X to plot, if None take the midle
    Ytoplot : +float, Y to plot, if None take the midle
    SoP : str, sort of operation 'std', 'max', 'pow'
    t_units : str, units of time axis acording to TimeDict, def. 'samples'
    Time_OffSet : ofsett of time axis, if any
    Xaxis_Scale : +float, X axis scale
    X_Label : str, label of X axis
    Yaxis_Scale : +float, Y axes scale
    Y_Label : str, Y axes label
    FigNum : number or name of figure
    cblabel : label of colorbar
    Cs : +float speed of soun in media, default 343, epeed in air
    Fs : +float sampling frequency in Hz, default 1 Hz
    FixScale : Boolean, True to fix scale fo colorbar 
    ColorMap : str, color map to apply, defaul 'jet'
    Shading : str, shading 'gouraud' o 'flat'

    Returns
    -------
    None.

    """
    if SelfNorm:
        Cscan=USF.normalizeAscans(Cscan)
    vmin = None  # limits of colormap, None -> auto
    vmax = None  # limits of colormap, None -> auto
    # build axes and check point to plot in each axes
    t_Scale = TimeDict[t_units.lower()][0]
    if t_units.lower() in {'m', 'mm', 'um'}:
        t_Scale = t_Scale * Cs * 1/Fs
    elif t_units.lower() in {'s', 'ms', 'us',  'ns',  'ps'}:
        t_Scale = t_Scale * 1/Fs
    t_xlabel = TimeDict[t_units.lower()][1]
    time_axis = np.arange(Cscan.shape[2]) * t_Scale - Time_OffSet
    Xaxis = np.arange(0, Cscan.shape[0]) * Xaxis_Scale
    Yaxis = np.arange(0, Cscan.shape[1]) * Yaxis_Scale
    
    ind = np.unravel_index(np.argmax(Cscan, axis=None), Cscan.shape)
    if Xtoplot is None:
        # Xtoplot = int(Cscan.shape[0]/2)
        Xtoplot = ind[0]
    else:
        Xtoplot = int(USF.findIndexInAxis(Xaxis, Xtoplot))  # find indexes
    if Ytoplot is None:        
        # Ytoplot = int(Cscan.shape[1]/2)
        Ytoplot = ind[1]
    else:
        Ytoplot = int(USF.findIndexInAxis(Yaxis, Ytoplot))  # find indexes

    fig = plt.figure(num=FigNum, clear=True)  # create figure
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)  # make axis 1
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", "5%", pad="3%")
    ax2 = plt.subplot2grid((3, 1), (2, 0))  # make axis 2
    if Depth is not None:
        Segment = USF.findIndexInAxis(time_axis, Depth)
        if Segment == 0:
            Segment += 1
    else:
        Segment = 1
    if FixScale:
        v1 = []
        v2 = []
        for i in np.arange(0, len(time_axis), Segment):
            v1.append(np.amin(operaCscan(Cscan[:, :, i:i + Segment], SoP=SoP)))
            v2.append(np.amax(operaCscan(Cscan[:, :, i:i + Segment], SoP=SoP)))
        vmin = np.amin(v1)
        vmax = np.amax(v2)
    Ascan = Cscan[Xtoplot, Ytoplot, :]  # ascan to plot
    ax2.plot(time_axis, Ascan)
    plt.xlim(time_axis[0], time_axis[-1])
    ax2.set_ylabel('Ascan')
    ax2.set_xlabel(t_xlabel)
    for i in np.arange(0, len(time_axis), Segment):
        ax1.clear()
        cax.clear()
        if Segment == 1:
            Data = Cscan[:, :, i]
        else:
            Data = operaCscan(Cscan[:, :, i:i + Segment], SoP=SoP)
        im = ax1.pcolormesh(Yaxis, Xaxis, Data, cmap=ColorMap,
                            shading=Shading)
        plt.colorbar(im, cax=cax, label=cblabel)
        # im.set_clim(vmin, vmax)
        # norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # im.set_norm(norm)
        plt.tight_layout()
        ax1.axis('image')
        ax1.set_ylabel(X_Label)
        ax1.set_xlabel(Y_Label)

        if Segment == 1:
            ax2.lines[1].remove()
            ax2.axvline(x=time_axis[i], color='r')  # plot vertical line
        else:
            ax2.patches = []
            # create and plot rectangle of the slice if any
            RectWidth = time_axis[i+Segment-1] - time_axis[i]
            RectHigh = abs((np.max(Ascan))-(np.min(Ascan)))
            rect = plt.Rectangle((time_axis[i], np.min(Ascan)), RectWidth, RectHigh,
                                 color='b', alpha=0.3)
            ax2.add_patch(rect)

        plt.tight_layout()
        plt.pause(0.1)
        plt.show()


def live_plot(y1, y2, y3, FGR, TransferFunction, Faxis, Flim, NoItePulseMax, Normalize=True, Spectrum_Units_dB=True, figsize=(8,6)):
    """
    Plot partial outputs in APWP optimization code

    Parameters
    ----------
    y1 : array, RNLFM signal, transmitted
    y2 : array, received isgnal in time
    y3 : Original rx signal in time domain (ref), for comparison
    FGR: Flatness to Gain ratio, optimization parameter
    TransferFunction : array, spectrum transfer function
    Faxis : frequency axis
    Flim : Upper frequency for representation
    NoItePulseMax : integer, Nomber of sub-iterations
    Normalize : boolean, true (default) to normalize spectrums
    Spectrum_Units_dB : boolean, false (default) to plot in n.u., True in dB
    figsize : 2 elements, width and heigh of figure. The default is (8,6).

    Returns
    -------
    None.

    """

    clear_output(wait=True)
    fig=plt.figure(figsize=figsize) 
    
    axs1 = plt.subplot(221)
    axs2 = plt.subplot(223)
    axs3 = plt.subplot(122)  
    
    axs1.plot(y1, label='Original RNLFM')
    axs1.plot(y2, label='New APWP')
    axs1.set_title('APWP Optimization')
    axs1.set_xlabel('samples')
    axs1.set_ylabel('Amplitude')
    axs1.legend(loc='upper right')
    Y1 = np.abs(TransferFunction * np.fft.fft(y1,len(Faxis)))
    Y2 = np.abs(TransferFunction * np.fft.fft(y2,len(Faxis)))
    Y3 = np.abs(np.fft.fft(y3,len(Faxis)))
    axs3.set_ylabel('|X(F)|')
    if Normalize:
        Y1 = Y1 / np.amax(Y1)
        Y2 = Y2 / np.amax(Y2)
        Y3 = Y3 / np.amax(Y3)
    if Spectrum_Units_dB:
        Y1 = np.log10(Y1) 
        Y2 = np.log10(Y2) 
        Y3 = np.log10(Y3) 
        axs3.set_ylabel('|X(F)| (dB)')
    axs3.plot(Faxis, Y1, label='Spectr. RNLFM')
    axs3.plot(Faxis, Y2, label='Spectr. New APWP')
    axs3.plot(Faxis, Y3, label='Spectr. Original')
    axs3.set_xlabel('frequency (MHz)')  
    axs3.set_xlim(0, Flim)
    axs3.legend(loc='upper right')
        
    axs2.plot(FGR)
    axs2.set_xlim(0, NoItePulseMax)
    axs2.set_xlabel('Iterations')
    axs2.set_ylabel('Flatness to Energy Ratio')
    
    plt.show()
    
# plot gencode and rx signal
def plot_GenCod_Rx(OriginalRx, OriginalGenCode, Fs=100e6, OffSet = 0,
                   Xlim=None, Flim=[0,12]):
    """
    Plot GenCode excitation and received signal in time and frequency

    Parameters
    ----------
    OriginalRx : float array, gencode
    OriginalGenCode : float array, gencode
    Fs : Sampling frequency in Hz. The default is 100e6.
    OffSet : Offset in us. The default is 0.
    Xlim : array of 2 elements, xlims for time axis. The default is None.
    Flim : array of 2 elements, xlims for freq axis in MHz.The default is [0,12].

    Returns
    -------
    None.

    """
    N = int(np.ceil(np.log2(np.abs(len(OriginalRx)))))+2
    nfft = 2**N
    
    OriginalRx = OriginalRx / np.max(np.abs(OriginalRx))
    
    OriginalGenCode[OriginalGenCode==2] = -1
    OriginalGenCode = OriginalGenCode[0:-1:2]
    OriginalGenCode = OriginalGenCode - np.mean(OriginalGenCode)
    OriginalGenCode[np.abs(OriginalGenCode)<0.5] = 0
    OriginalGenCode = -np.append(OriginalGenCode, np.zeros(len(OriginalRx)-len(OriginalGenCode)))
    


    OffSet_samples = np.int(np.round(OffSet * Fs /1e6))
    OriginalGenCode = np.roll(OriginalGenCode, OffSet_samples)
    time_axis = np.arange(len(OriginalRx)) / Fs * 1e6 - OffSet
    plt.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure(num=1, clear=True)
    ax221 = plt.subplot(221)
    ax221.plot(time_axis,OriginalGenCode,'k')
    ax221.tick_params(labelsize=16)
    ax221.set_xlabel(r"time ($\mu$s)" + "\n"+"(a)",fontsize=20)
    ax221.set_ylabel('Amplitude',fontsize=20)
#    ax221.text(-0.065, 1.05, 'a', horizontalalignment='center',fontsize=32,
#                   verticalalignment='center', transform=ax221.transAxes)
    ax221.set_xlim(Xlim)
    
    ax222 = plt.subplot(223)
    ax222.plot(time_axis,OriginalRx,'b')
    ax222.tick_params(labelsize=16)
    ax222.set_xlabel(r"time ($\mu$s)" + "\n"+"(b)",fontsize=20)
    ax222.set_ylabel('Amplitude',fontsize=20)
#    ax222.text(-0.065, 1.05, 'b', horizontalalignment='center',fontsize=32,
#                   verticalalignment='center', transform=ax222.transAxes)
    ax222.set_xlim(Xlim)
    
    
    TF_OGC = np.abs(np.fft.fft(OriginalGenCode, nfft))
    TF_ORx = np.abs(np.fft.fft(OriginalRx, nfft))
    F_axis = np.arange(nfft) * Fs/nfft / 1e6 # frequency axis in MHz
    ax223 = plt.subplot(122)
    ax223.plot(F_axis,TF_OGC,'k', label='NLFM Rectangular Chirp')
    ax223.plot(F_axis,TF_ORx,'b', label='Received signal')
    ax223.tick_params(labelsize=16)
    ax223.set_xlabel(r"Frequency (MHz)" + "\n"+"(c)",fontsize=20)
    ax223.set_ylabel('Magnitude',fontsize=20)
#    ax223.text(-0.065, 1.05, 'c', horizontalalignment='center',fontsize=32,
#                   verticalalignment='center', transform=ax223.transAxes)
    ax223.set_xlim(Flim)
    ax223.legend(fontsize=20)



def movefig(location: str):
    '''
    Moves the current figure to the specified location on the screen.
    
    Accepted locations are:
        'northeast' or 'ne'
        'southeast' or 'se'
        'northwest' or 'nw'
        'southwest' or 'sw'
        'south'     or 's'
        'north'     or 'n'

    Parameters
    ----------
    location : str
        Location on the screen to move the figure to.

    Returns
    -------
    None.
    
    Arnau, 02/11/2022
    '''
    from win32api import GetSystemMetrics
    width = GetSystemMetrics(0)
    height = GetSystemMetrics(1)
    
    mngr = plt.get_current_fig_manager()
    
    geom = mngr.window.geometry()
    x,y,dx,dy = geom.getRect()

    if location.lower() in ['northeast', 'ne']:
        mngr.window.setGeometry(width-5-dx, 30, dx, dy)
    elif location.lower() in ['southeast', 'se']:
        mngr.window.setGeometry(width-5-dx, height-40-dy, dx, dy)
    elif location.lower() in ['northwest', 'nw']:
        mngr.window.setGeometry(5, 30, dx, dy)
    elif location.lower() in ['southwest', 'sw']:
        mngr.window.setGeometry(5, height-40-dy, dx, dy)
    elif location.lower() in ['north', 'n']:
        mngr.window.setGeometry((width-dx)//2, 30, dx, dy)
    elif location.lower() in ['south', 's']:
        mngr.window.setGeometry((width-dx)//2, height-40-dy, dx, dy)


def SliderWindow(data, SortofWin='boxcar', param1=1, param2=1, fignum='SliderWindow'):
    '''
    Plot the data and a moveable window. The figure includes 2 sliders (one for
    window length and another one for window location) and a Reset button. The
    function blocks the execution of the main program until any keyboard key is
    pressed, then the selected window is returned.

    Parameters
    ----------
    data : ndarray
        data to be plotted (and normalized).

    Returns
    -------
    new_window : ndarray.
        the window selected.

    Arnau, 16/11/2022
    '''
    ScanLen = len(data)
    
    # Define initial parameters
    init_WinLen = ScanLen//2
    init_Loc = ScanLen//2

    # Initial window
    init_win = USF.makeWindow(SortofWin=SortofWin, WinLen=init_WinLen,
                   param1=param1, param2=param2, Span=ScanLen, Delay=init_Loc - init_WinLen//2)

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(num=fignum)
    ax.set_title('Press any key to return to main program')
    ax.plot(USF.normalizeAscans(data), lw=2)
    line, = ax.plot(init_win, lw=2)
    
    # adjust the main plot to make room for the sliders
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the window length.
    axloc = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    Loc_slider = Slider(
        ax=axloc,
        label='Location',
        valmin=0,
        valmax=ScanLen,
        valinit=init_Loc,
        valstep=1,
    )

    # Make a vertically oriented slider to control the position of the window
    axwinlen = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    WinLen_slider = Slider(
        ax=axwinlen,
        label="Window Length",
        valmin=0,
        valmax=ScanLen,
        valinit=init_WinLen,
        valstep=1,
        orientation="vertical"
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        global new_window
        new_window = USF.makeWindow(SortofWin=SortofWin, WinLen=WinLen_slider.val,
                       param1=param1, param2=param2, Span=ScanLen, Delay=Loc_slider.val - WinLen_slider.val//2)
        line.set_ydata(new_window)
        fig.canvas.draw_idle()
    
    # register the update function with each slider
    Loc_slider.on_changed(update)
    WinLen_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(resetax, 'Reset', hovercolor='0.975')
    def reset(event):
        Loc_slider.reset()
        WinLen_slider.reset()
    reset_button.on_clicked(reset)
    resetax._reset_button = reset_button # dummy reference to avoid garbage collector

    # confirmax = fig.add_axes([0.5, 0.025, 0.1, 0.04])
    # confirm_button = Button(confirmax, 'Confirm', hovercolor='0.975')
    # def confirm(event):
    #     plt.close()
    #     return new_window
    # confirm_button.on_clicked(confirm)
    # confirmax._confirm_button = confirm_button # dummy reference to avoid garbage collector

    plt.show()
    while not plt.waitforbuttonpress():
        pass
    plt.close()
    return new_window



def plot_hist(h, b, width, ax=None, fignum=1, xlabel='', ylabel='', **kwargs):
    '''
    Plot a histogram given the histogram values (h), the bin edges (b), and the
    width of the bins (width).

    Parameters
    ----------
    h : ndarray
        The values of the histogram.
    b : ndarray
        Bin edges (length(hist)+1).
    width : float
        Width of the bins.
    ax : AxesSubplot, optional
        Axis handle to use for plotting. If None, create new figure. The
        default is None.
    fignum : int or str, optional
        The name or number of the new figure. Only used if ax is None. The
        default is 1.
    xlabel : str, optional
        X axis label. The default is ''.
    ylabel : str, optional
        Y axis label. The default is ''.
    **kwargs : Rectangle properties
        plt.bar aditional properties.

    Returns
    -------
    None.

    Arnau 23/12/2022
    '''
    if ax is None:
        plt.figure(fignum)
        plt.bar(b[:-1], h, width=width, **kwargs)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
    else:
        ax.bar(b[:-1], h, width=width, **kwargs)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

def animate_sliced_array(x, idxs, ax=None, fignum=1, pauseTime=0.3):
    '''
    Perform an animation of the plotting of x and the corresponding indices.
    This is intended to be used with the output of slice_array().

    Parameters
    ----------
    x : ndarray
        Matrix containing data row-wise.
    idxs : ndarray
        Matrix containing the indeces of the data in x.
    fignum : int or str, optional
        The name or number of the new figure. Only used if ax is None. The
        default is 1.
    pauseTime : float, optional
        Pause between frames in seconds. The default is 0.3.

    Returns
    -------
    None.

    Arnau 23/12/2022
    '''
    x_axis = np.arange(0, np.max(idxs))
    
    if ax is None:
        plt.figure(fignum)
        for idx, slc in zip(idxs, x):
            l = plt.scatter(idx, slc, color='r', marker='.')
            plt.xlim([0, x_axis[-1]])
            plt.pause(pauseTime)
            l.set_color('k')
    else:
        for idx, slc in zip(idxs, x):
            l = ax.scatter(idx, slc, color='r', marker='.')
            ax.set_xlim([0, x_axis[-1]])
            plt.pause(pauseTime)
            l.set_color('k')

def moving_hist(data, slicesize, overlap=0.5, fignum=1, pauseTime=0.3, **kwargs):  
    '''
    Animate the sliced histogram of data.

    Parameters
    ----------
    data : ndarray
        Input data.
    slicesize : int
        Window size in samples.
    overlap : float, optional
        Float between 0 and 1. Specifies the overlap amount. The default is 0.5.
    fignum : int or str, optional
        The name or number of the new figure. Only used if ax is None. The
        default is 1.
    pauseTime : float, optional
        Pause between frames in seconds. The default is 0.3.
    **kwargs : Rectangle properties
        plt.bar aditional properties.

    Returns
    -------
    None.

    Arnau, 23/12/2022
    '''
    slices, idxs = USF.slice_array(data, slicesize, overlap)
    
    h, b, width = USF.hist(slices, density=True)
    hfull, bfull, widthfull = USF.hist(data, density=True)
    
    x_axis = np.arange(len(data))
    
    axs = plt.subplots(2, num=fignum)[1]
    ax2 = axs[1].twinx()
    plot_hist(hfull, bfull, widthfull, ax=ax2, ylabel='Global histogram', alpha=0.5, **kwargs)
    
    axs[1].set_zorder(10)
    axs[1].patch.set_visible(False)
    axs[1].set_ylabel('Sliced histogram')
    axs[1].set_ylim([0, np.max(h)+1])
    
    axs[0].scatter(x_axis, data, color='k', marker='.')
    for i, j, k, slc, idx in zip(b, h, width, slices, idxs):
        l = axs[0].scatter(idx, slc, color='r', marker='.')
        lb = axs[1].bar(i[:-1], j, width=k, zorder=3, color=u'#1f77b4', **kwargs)
        plt.pause(pauseTime)
        lb.remove()
        l.remove()


def text_position(p):
    '''
    Returns a dictionary with the keys 'x', 'y', 'ha' and 'va'. These are used
    as parameters for the plt.text() function to determine the position of the
    text.

    Parameters
    ----------
    p : str
        Desired position of text. Available positions are:
            'northwest'
            'northeast'
            'southwest'
            'southeast'
            'north'
            'south'

    Returns
    -------
    d : dict
        Dictionary with 4 keys: 'x', 'y', 'ha' and 'va'. The values for these
        keys would generate some text in the desired position if given to
        plt.text().

    Arnau, 21/02/2023
    '''
    d = {}
    hoff = 0.02
    voff = 0.1
    if p.lower() == 'northwest':
        d['x'] = hoff
        d['y'] = 1 - voff
        d['ha'] = 'left'
        d['va'] = 'top'
    elif p.lower() == 'northeast':
        d['x'] = 1 - hoff
        d['y'] = 1 - voff
        d['ha'] = 'right'
        d['va'] = 'top'
    elif p.lower() == 'southwest':
        d['x'] = hoff
        d['y'] = voff
        d['ha'] = 'left'
        d['va'] = 'bottom'
    elif p.lower() == 'southeast':
        d['x'] = 1 - hoff
        d['y'] = voff
        d['ha'] = 'right'
        d['va'] = 'bottom'
    elif p.lower() == 'north':
        d['x'] = 0.5
        d['y'] = 1 - voff
        d['ha'] = 'center'
        d['va'] = 'top'
    elif p.lower() == 'south':
        d['x'] = 0.5
        d['y'] = voff
        d['ha'] = 'center'
        d['va'] = 'bottom'
    return d

def pltGUI(x, x4, CL, Cs, L, PE, TT, img, ptxt='northwest'):
    '''
    Matplotlib Graphical User Interface for dog-bone analysis.

    Parameters
    ----------
    x : ndarray
        Position array in millimeters.
    x4 : ndarray
        Time array in microseconds.
    CL : ndarray
        Longitudinal velocity in m/s.
    Cs : ndarray
        Shear velocity in m/s.
    L : ndarray
        Thickness in millimeters.
    PE : 2D-ndarray
        Pulse-eco traces in volts. Each column is an acq so that the shape of
        PE is (len(x4), len(x)).
    TT : 2D-ndarray
        Through-Transmission traces in volts. Each column is an acq so that the
        shape of TT is (len(x4), len(x)).
    img : 3D-ndarray
        RGB image as returned by matplotlib.image.imread().
    ptxt : str, optional
        Location of the text displaying the current point that the cursor is
        hovering over. The default is 'northwest'.

    Returns
    -------
    None.

    Arnau, 21/02/2023
    '''
    fig = plt.figure(figsize=(12,8))
    gs = fig.add_gridspec(4,3)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 1:])
    ax3 = fig.add_subplot(gs[2, 1:])
    ax4 = fig.add_subplot(gs[3, 1:])
    
    ax0.imshow(img, extent=[0, img.shape[1], x[-1], x[0]], aspect="auto")
    ax0.get_xaxis().set_ticks([])
    line1, = ax1.plot(x, CL, 'b')
    line2, = ax2.plot(x, Cs, 'r')
    line3, = ax3.plot(x, L)
    line4_PE, = ax4.plot(x4, PE[:,0])
    line4_TT, = ax4.plot(x4, TT[:,0])
    ax4.set_ylim([np.min([PE,TT])*1.1, np.max([PE,TT])*1.1])
    
    ax0.set_ylabel('Position (mm)')
    ax1.set_xlabel('Position (mm)')
    ax2.set_xlabel('Position (mm)')
    ax3.set_xlabel('Position (mm)')
    ax4.set_xlabel('Time (\u03bcs)')
    
    ax1.set_ylabel('Long. vel. (m/s)')
    ax2.set_ylabel('Shear vel. (m/s)')
    ax3.set_ylabel('Thickness (mm)')
    ax4.set_ylabel('Amplitude (V)')
    
    hline0, = ax0.plot([0, img.shape[1]], [x[0], x[0]], c='k')
    vline1 = ax1.axvline(x[0], c='k')
    vline2 = ax2.axvline(x[0], c='k')
    vline3 = ax3.axvline(x[0], c='k')
       
    d = text_position(ptxt)
    txt = ax1.text(d['x'], d['y'], f"{round(x[0], 2)}, {round(CL[0], 2)}", ha=d['ha'], va=d['va'], transform=ax1.transAxes)
    txt.set_bbox(dict(facecolor='w', edgecolor='k', alpha=0.5))
    
    plt.tight_layout()
    
    def update(idx, l, shear):
        line1.set_marker('')
        line2.set_marker('')
        line3.set_marker('')
        
        x, y = l.get_data()
        txt.set_text(f"{round(x[idx], 2)}, {round(y[idx], 2)}")
        
        hline0.set_ydata(x[idx])
        vline1.set_xdata(x[idx])
        vline2.set_xdata(x[idx])
        vline3.set_xdata(x[idx])
        
        if shear:
            auxPE = PE[:,::-1]
            auxTT = TT[:,::-1]
            line4_PE.set_ydata(auxPE[:,idx])
            line4_TT.set_ydata(auxTT[:,idx])
        else:
            line4_PE.set_ydata(PE[:,idx])
            line4_TT.set_ydata(TT[:,idx])
        
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
        elif event.inaxes == ax3:
            l = line3
        else:
            return
        cont, ind = l.contains(event)
        if cont:
            if len(ind["ind"]) != 0:
                update(ind["ind"][0], l, shear)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    plt.show()

def histGUI(x, y, xlabel='', ylabel='', **kwargs):
    '''
    Interactive histogram. Allows for the creation of a movable window by 
    clicking and dragging the cursor. The histogram of the data inside the
    window is computed and displayed.

    Parameters
    ----------
    x : ndarray
        X data.
    y : ndarray
        Input data.
    **kwargs : Rectangle properties
        plt.bar aditional properties.

    Returns
    -------
    ax1 : matplotlib.Axes
        Axes handle of the first subplot.
    ax2 : matplotlib.Axes
        Axes handle of the second subplot.
    span : matplotlib.widgets.SpanSelector
        Handle for the span selector. MUST be stored in a dummy variable to
        avoid being garbage-collected. For example:
            ax1, ax2, _ = histGUI(x, y)

    Arnau, 23/02/2023
    '''
    hfull, bfull, widthfull = USF.hist(y, density=True)
    
    fig, (ax1, ax2) = plt.subplots(2)
    plot_hist(hfull, bfull, widthfull, ax=ax2, alpha=0.5, edgecolor='k', **kwargs)
    
    ax2.set_zorder(10)
    ax2.set_ylim([0, np.max(hfull)*1.1])
    ax2.set_xlim([bfull[0]-widthfull, bfull[-1]+widthfull])
    
    ax1.plot(x, y, c='k')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax2.set_xlabel(ylabel)

    def onselect(xmin, xmax):
        indmin, indmax = np.searchsorted(x, (xmin, xmax))
        indmax = min(len(x) - 1, indmax)

        region_x = x[indmin:indmax]
        region_y = y[indmin:indmax]

        if len(region_x) >= 2:
            h, b, width = USF.hist(region_y, density=True)

            ax2.clear()
            plot_hist(hfull, bfull, widthfull, ax=ax2, alpha=0.5, edgecolor='k', **kwargs)
            ax2.set_zorder(10)
            ax2.set_xlim([bfull[0]-widthfull, bfull[-1]+widthfull])
            plot_hist(h, b, width, ax=ax2, alpha=0.7, edgecolor='k', **kwargs)
            ax2.set_ylim([0, np.max(np.r_[hfull, h])*1.1])
            ax2.set_xlabel(ylabel)
            fig.canvas.draw_idle()

    span = SpanSelector(
        ax1,
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.5, facecolor="tab:blue"),
        interactive=True,
        drag_from_anywhere=True
    )

    plt.tight_layout()
    plt.show()
    return ax1, ax2, span
