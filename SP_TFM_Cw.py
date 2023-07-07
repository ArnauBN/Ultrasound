# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:28:02 2022
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
import src.ultrasound as US

#%% Parameters
temperature = np.linspace(0, 100, 10_000)
methods = ['Bilaniuk', 'Marczak', 'Lubbers', 'Abdessamad']
method_params_dict = {methods[0]: [36, 112, 148], methods[1]: None, methods[2]: ['15-35', '10-40'], methods[3]: [36, 148]}


#%% All methods
x1, x2, y1, y2 = 19.94, 20.06, 1482, 1482.5 # inset

ax = plt.subplots(1, num='Cw_all')[1]
ax.set_ylabel('Cw (m/s)')
ax.set_xlabel('Temperature (\u2103)')

# inset
axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# ax.indicate_inset_zoom(axins, edgecolor="black")

for k, v in method_params_dict.items():
    if k != methods[1]:
        for value in v:
            Cw = US.speedofsound_in_water(temperature, k, value)
            ax.plot(temperature, Cw, label=f'{k} - {value}')
            axins.plot(temperature, Cw)
    else:
        Cw = US.speedofsound_in_water(temperature, k)
        ax.plot(temperature, Cw, label=k)
        axins.plot(temperature, Cw)
plt.legend(loc='upper left', prop={'size': 8})
plt.grid()
plt.tight_layout()


#%% Chosen method
Cw = US.speedofsound_in_water(temperature, 'Abdessamad', 148)

ax = plt.subplots(1, num='Cw')[1]
ax.set_ylabel('Speed of sound in pure water (m/s)')
ax.set_xlabel('Temperature (\u2103)')
ax.plot(temperature, Cw)

x1, x2, y1, y2 = 16, 22, 1469, 1490 # inset
x1, x2, y1, y2 = 20, 30, 1480, 1510 # inset
axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
ax.indicate_inset_zoom(axins, edgecolor="black")

axins.plot(temperature, Cw)

plt.grid()
axins.minorticks_on()
axins.grid(which='both')
plt.tight_layout()