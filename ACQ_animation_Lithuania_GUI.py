# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 10:14:22 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:06:56 2023
Python version: Python 3.8

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

ICONS: https://p.yusukekamiyamane.com/

"""
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import QApplication

from src.devices import SeDaq as SD

#%% Initialize acq. system
SeDaqAnimation = SD.SeDaqDLL()


#%% Set acq. system parameters
AvgSamplesNumber = 1           # Number of traces to average to improve SNR
Quantiz_Levels = 1024           # Number of quantization levels
SeDaqAnimation.AvgSamplesNumber = AvgSamplesNumber
SeDaqAnimation.Quantiz_Levels = Quantiz_Levels


#%%
class MainWindow(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("USGUI")
        self.resize(QSize(1080,720))
        
        # Plot style
        self.pen = pg.mkPen(color='k', width=2)
        self.axisStyle = {"color": "k", "font-size": "12px"}
        self.setBackground("w")

        # Curve and plot objects
        self.p1 = self.addPlot(row=0, col=0)
        self.p2 = self.addPlot(row=1, col=0)
        self.curve1 = self.p1.plot([], [], pen=self.pen)
        self.curve2 = self.p2.plot([], [], pen=self.pen)
        self.p1.setYRange(-0.5, 0.5)
        self.p2.setYRange(-0.5, 0.5)
        self.p1.disableAutoRange(True)
        self.p2.disableAutoRange(True)
        
        # X data
        self.Fs = 100e6
        self.Smin = 5000
        self.Smax = 10000
        self.timeFactor = self.Fs*1e6
        self.xTop = np.arange(self.Smin, self.Smax)
        self.x = np.arange(self.Smin, self.Smax)/self.timeFactor
        
        # Plot configuration
        self.setAppearance(self.p1)
        self.setAppearance(self.p2)
        self.p1.setLabel("left", "Channel 1 (V)")
        self.p1.setLabel("bottom", "Time (us)")
        self.p1.setLabel("top", "Samples")
        self.p2.setLabel("left", "Channel 2 (V)")
        self.p2.setLabel("bottom", "Time (us)")
        self.p2.setLabel("top", "Samples")   
        
        # Start loop
        self.timer = QTimer(self)
        self.timer.start(10)
        self.timer.timeout.connect(self.uptade)

    def setAppearance(self, p):
        p.getAxis('left').setTextPen('k')
        p.getAxis('left').setPen('k')
        p.setLabel("left", "", **self.axisStyle)
        p.getAxis('left').enableAutoSIPrefix(False)
        
        p.getAxis('bottom').setTextPen('k')
        p.getAxis('bottom').setPen('k')
        p.setLabel("bottom", "", **self.axisStyle)
        self._temp = p.getAxis('bottom').tickValues(min(self.x), max(self.x), len(self.x))[1][1]
        p.getAxis('bottom').enableAutoSIPrefix(False)
        
        p.getAxis('top').setTextPen('k')
        p.getAxis('top').setPen('k')
        p.setLabel("top", "", **self.axisStyle)
        p.getAxis('top').setTicks([[(v, str(int(v*self.timeFactor))) for v in self._temp]])
        p.getAxis('top').enableAutoSIPrefix(False)
        
        p.getAxis('right').setTextPen('k')
        p.getAxis('right').setPen('k')
        p.setLabel("right", "", **self.axisStyle)
        p.getAxis('right').setTicks([])
        p.getAxis('right').enableAutoSIPrefix(False)
        
        p.showGrid(x=True, y=True, alpha=0.1)

    def uptade(self):
        Smin1, Smin2 = self.Smin, self.Smin       # starting point of the scan of each channel - samples
        Smax1, Smax2 = self.Smax, self.Smax       # last point of the scan of each channel - samples
        Smin_tuple = (Smin1, Smin2)     # starting points - samples
        Smax_tuple = (Smax1, Smax2)     # last points - samples
        
        TT_Ascan, PE_Ascan = SeDaqAnimation.GetAscan_Ch1_Ch2(Smin_tuple, Smax_tuple) #acq Ascan
        
        self.curve1.setData(self.x, TT_Ascan)
        self.curve2.setData(self.x, PE_Ascan)

app = QApplication([])
app.setQuitOnLastWindowClosed(True)

window = MainWindow()
window.show()