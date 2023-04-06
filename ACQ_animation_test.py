# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:54:34 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.devices import SeDaq as SD

#%% Parameters
RecLen = 32*1024                # Maximum range of ACQ - samples (max=32*1024)

# Smin = 0                     # starting point of the scan of each channel - samples
# Smax = RecLen                     # last point of the scan of each channel - samples

# Smin = 3400                     # starting point of the scan of each channel - samples
# Smax = 8300                     # last point of the scan of each channel - samples

Smin = 4000                     # starting point of the scan of each channel - samples
Smax = 6500                     # last point of the scan of each channel - samples

AvgSamplesNumber = 25           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels


Smin1, Smin2 = Smin, Smin       # starting point of the scan of each channel - samples
Smax1, Smax2 = Smax, Smax       # last point of the scan of each channel - samples
Smin_tuple = (Smin1, Smin2)     # starting points - samples
Smax_tuple = (Smax1, Smax2)     # last points - samples
ScanLen = Smax - Smin  # Total scan length for computations (zero padding is used) - samples

#%%
SeDaq = SD.SeDaqDLL()
SeDaq.AvgSamplesNumber = AvgSamplesNumber
SeDaq.Quantiz_Levels = Quantiz_Levels


#%%
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=np.array([Smin, Smax])/(100e6)*1e6, ylim=(-0.5, 0.5))
TT_line, = ax.plot([], [], lw=2)
PE_line, = ax.plot([], [], lw=2)
x = np.arange(Smin, Smax)/(100e6)*1e6
ax.axvline(50)

# initialization function: plot the background of each frame
def init():
    TT_line.set_data([], [])
    PE_line.set_data([], [])
    return TT_line, PE_line

# animation function. This is called sequentially
def animate(frame):
    TT_Ascan, PE_Ascan = SeDaq.GetAscan_Ch1_Ch2(Smin_tuple, Smax_tuple) #acq Ascan
    TT_line.set_data(x, TT_Ascan)
    PE_line.set_data(x, PE_Ascan)
    return TT_line, PE_line

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(fig, animate, init_func=init, frames=1, interval=0, blit=True, cache_frame_data=False, repeat=True)

plt.show()





    
    
#%%
Smin = 6500                     # starting point of the scan of each channel - samples
Smax = 7000                     # last point of the scan of each channel - samples
ScanLen = Smax - Smin  # Total scan length for computations (zero padding is used) - samples
N = 10_000

maxs = []
x = []

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
# ax = plt.axes(xlim=(Smin, Smax), ylim=(-0.5, 0.5))
ax = plt.axes(xlim=(0, N), ylim=(200,300))
WP_line, = ax.plot([], [], lw=2)
# x = np.arange(Smin, Smax)

# initialization function: plot the background of each frame
def init():
    WP_line.set_data([], [])
    return WP_line,

# animation function. This is called sequentially
def animate(frame):
    WP_Ascan = SeDaq.GetAscan_Ch1(Smin, Smax)
    
    x.append(frame)
    
    MaxLoc = np.argmax(np.abs(WP_Ascan))  # find index of maximum
    A = MaxLoc - 1  # left proxima
    B = MaxLoc + 1  # Right proxima
    
    # calculate interpolation maxima according to cosine interpolation
    Alpha = np.arccos((WP_Ascan[A] + WP_Ascan[B]) / (2 * WP_Ascan[MaxLoc]))
    Beta = np.arctan((WP_Ascan[A] - WP_Ascan[B]) / (2 * WP_Ascan[MaxLoc] * np.sin(Alpha)))
    Px = Beta / Alpha

    # Calculate ToF in samples
    DeltaToF = MaxLoc - Px

    maxs.append(DeltaToF)
    
    WP_line.set_data(x, maxs)
    
    # WP_line.set_data(x, WP_Ascan)
    return WP_line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = FuncAnimation(fig, animate, init_func=init, frames=N, interval=0, blit=True, cache_frame_data=False, repeat=False)

plt.show()

