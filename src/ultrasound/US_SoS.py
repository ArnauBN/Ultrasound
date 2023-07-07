# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:05:50 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

===========================================
SoS (Speed of Sound) in different materials
===========================================
"""
import numpy as np
from scipy.optimize import fsolve

#%% METHODS FOR EACH MATERIAL
def water_temp2sos(T, method: str='Abdessamad', method_param=148):
    '''
    Compute speed of sound in pure water as a function of temperature with the
    given method. Some methods require parameters given in method_param.
    
    Available methods:
        'Bilaniuk'
        'Marczak'
        'Lubbers'
        'Abdessamad'
    
    Available method parameters:
        'Bilaniuk':
            112
            36
            148
        'Lubbers':
            '15-35'
            '10-40'
        'Abdessamad':
            148
            36

    Parameters
    ----------
    T : float or ndarray
        Temperature of water in Celsius.
    method : str, optional
        Method to use. The default is 'Abdessamad'.
    method_param : int or str, optional
        Parameter for the method. The default is 148.

    Returns
    -------
    c : float or ndarray
        the speed of sound in pure (distilled) water in m/s.

    Arnau, 10/01/2023
    '''
    if method.lower()=='bilaniuk':
        # Bilaniuk and Wong (0-100 ºC) - year 1993-1996
        if method_param==112:
            c = 1.40238742e3 + 5.03821344*T - 5.80539349e-2*(T**2) + \
                3.32000870e-4*(T**3) - 1.44537900e-6*(T**4) + 2.99402365e-9*(T**5)
        elif method_param==36:
            c = 1.40238677e3 + 5.03798765*T - 5.80980033e-2*(T**2) + \
                3.34296650e-4*(T**3) - 1.47936902e-6*(T**4) + 3.14893508e-9*(T**5)
        elif method_param==148:
            c = 1.40238744e3 + 5.03836171*T - 5.81172916e-2*(T**2) + \
                3.34638117e-4*(T**3) - 1.48259672e-6*(T**4) + 3.16585020e-9*(T**5)
    elif method.lower()=='marczak':
        # Marczak (0-95 ºC) - year 1997
        c = 1.402385e3 + 5.038813*T - 5.799136e-2*(T**2) + 3.287156e-4*(T**3) - \
            1.398845e-6*(T**4) + 2.787860e-9*(T**5)
    elif method.lower()=='lubbers':
        if method_param=='15-35':
            # Lubbers and Graaff (15-35 ºC) - year 1998
            c = 1404.3 + 4.7*T - 0.04*(T**2)
        elif method_param=='10-40':
            # Lubbers and Graaff (10-40 ºC) - year 1998
            c = 1405.03 + 4.62*T - 3.83e-2*(T**2)
    elif method.lower()=='abdessamad':
        # Abdessamad, Malaoui & Iqdour, Radouane & Ankrim, Mohammed & Zeroual, 
        # Abdelouhab & Benhayoun, Mohamed & Quotb, K.. (2005). 
        # New model for speed of sound with temperature in pure water. 
        # AMSE Review (Association for the Advancement of Modelling and Simulation
        # Techniques in Enterprises). 74. 12.10-12.13.
        if method_param==148:
            # (0.001-95.126 ºC)
            c = 1.569678141e3*np.exp(-((T-5.907868678e1)/(-3.443078912e2))**2) - \
                2.574064370e4*np.exp(-((T+3.705052160e2)/(-1.601257116e2))**2)
        elif method_param==36:
            # (0.056-74.022 ºC)
            c = 1.567302324e3*np.exp(-((T-6.101414576e1)/(-3.388027429e2))**2) - \
                1.468922269e4*np.exp(-((T+3.255477156e2)/(-1.478114724e2))**2)
    return c


def water_sos2temp(target_speed, temperature_guess=20, method: str='Abdessamad', method_param=148):
    '''
    Finds the water temperature that would induce the specified target speed
    of sound.

    Parameters
    ----------
    target_speed : float
        Target speed of sound in pure water in m/s.
    temperature_guess : float, optional
        Initial temperature guess in celsius. The default is 20.
    method : str, optional
        Method to use. The default is 'Abdessamad'.
    method_param : int or str, optional
        Parameter for the method. The default is 148.

    Returns
    -------
    temperature : float
        The water temperature that produces the desired speed of sound.

    Arnau, 23/02/2023
    '''
    return fsolve(lambda x: target_speed - water_temp2sos(x, method, method_param), temperature_guess)[0]

def resin_temp2sos(T):
    '''
    Compute the speed of sound in epoxy resin given a temperature in celsius.

    Parameters
    ----------
    T : float or ArrayLike
        Temperature in celsius.

    Returns
    -------
    C : float or ArrayLike
        Speed of sound in m/s.

    Arnau, 06/07/2023
    '''
    m = -6.578749669765215
    c = 1854.2829464167598
    return m*T + c

def resin_sos2temp(C):
    '''
    Compute the temperature that would produce the specified speed of sound.

    Parameters
    ----------
    C : float or ArrayLike
        Speed of sound in m/s.

    Returns
    -------
    T : float or ArrayLike
        Temperature in celsius.

    Arnau, 06/07/2023
    '''
    m = -6.578749669765215
    c = 1854.2829464167598
    return (C - c)/m


#%% ALL
def temp2sos(T, material: str='water', *args, **kwargs):
    '''
    Compute the speed of sound in the specified material given a temperature in
    celsius.

    Parameters
    ----------
    T : float or ArrayLike
        Temperature in celsius.
    material : str, optional
        Material. The default is 'water'.
    *args : args
        Arguments for computation. Depends on the material. Check their
        respective methods.
    **kwargs : kwargs
        keyword arguments for computation. Depends on the material. Check their
        respective methods.

    Raises
    ------
    NotImplementedError
        'Material not available'.

    Returns
    -------
    C : float or ArrayLike
        Speed of sound in m/s.

    Arnau, 06/07/2023
    '''
    if material.lower() in ['water', 'distilled water', 'deionized water']:
        return water_temp2sos(T, *args, **kwargs)
    elif material.lower() in ['resin', 'epoxy', 'epoxy resin', 'resin epoxy']:
        return resin_temp2sos(T, *args, **kwargs)
    else:
        raise NotImplementedError('Material not available.')

def sos2temp(C, material: str='water', *args, **kwargs):
    '''
    Compute the temperature in celsius in the specified material that would
    procude the specified speed of sound in m/s.

    Parameters
    ----------
    C : float or ArrayLike
        Speed of sound in m/s.
    material : str, optional
        Material. The default is 'water'.
    *args : args
        Arguments for computation. Depends on the material. Check their
        respective methods.
    **kwargs : kwargs
        keyword arguments for computation. Depends on the material. Check their
        respective methods.

    Raises
    ------
    NotImplementedError
        'Material not available'.

    Returns
    -------
    T : float or ArrayLike
        Temperature in celsius.

    Arnau, 06/07/2023
    '''
    if material.lower() in ['water', 'distilled water', 'deionized water']:
        return water_sos2temp(C, *args, **kwargs)
    elif material.lower() in ['resin', 'epoxy', 'epoxy resin', 'resin epoxy']:
        return resin_sos2temp(C, *args, **kwargs)
    else:
        raise NotImplementedError('Material not available.')