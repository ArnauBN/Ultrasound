# -*- coding: utf-8 -*-.
"""
Created on Thu Nov  5 09:20:41 2020

@author: alrom
"""

import scipy.io as sio
# import matplotlib.pyplot as plt
import numpy as np
# from scipy import signal


def loadFromMatlab(FileName, DataName):
    """
    Load file from matlab to numpy array.

    Parameters
    ----------
    FileName : str, full name of file including extension and path
    DataName : str, name of the data stored in .mat file

    Returns
    -------
    DataMatrix, np.array of data.

    """
    mat_contents = sio.loadmat(FileName, squeeze_me=True, mat_dtype=True)
    # a = mat_contents.keys()
    DataMatrix = mat_contents[DataName]
    return DataMatrix
