# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:21:36 2022

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>
"""

import matplotlib.pyplot as plt
import time

from src.devices import Ultratek # (32-bit python)

#%%
card = Ultratek.Ultratek()
handler = card.init()
if handler:
    card.setSamplingRate(6.25)
    card.setBufferLength(1800)
    card.setGain(40)
    card.setTriggerDelay(10)
    card.setADTriggerSource('software')

print("Single acquisition mode")

card.softwareTrigger()

while card.isDataReady(): # Wait for the data to be available
    start = time.time()
    data = card.getData(1800) # Discard initial garbage packet
    print((time.time() - start) * 1e3)
    data = card.getData(1800)

plt.figure()
plt.plot(data)