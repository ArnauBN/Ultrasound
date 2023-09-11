# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 11:53:43 2023
Python version: Python 3.9

@author: Arnau Busqué Nadal <arnau.busque@goumh.umh.es>

"""
import numpy as np
import matplotlib.pylab as plt
import os

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

#%% Load data
Path = r'..\Data\Deposition'
Names = ['R0_0_M_rt1m', 'R0_0_M_rt2m', 'R0_0_M_rt3m', 'R0_0_M_rt4m', 'R0_0_M_rt1', 'R0_0_M_rt2', 'Rb0_0_M_rt1', 'Rb0_0_M_rt2', 'Rb0_0_M_rt3', 'Rb0_0_M_rt_vh', 'Rb0_0_M_rt_vh2']
experiments = loadExperiments(Path, Names, Verbose=True)

# temps = np.c_[experiments['R0_0_M_rt1m'].temperature_lpf, 
#               experiments['R0_0_M_rt2m'].temperature_lpf, 
#               experiments['R0_0_M_rt3m'].temperature_lpf, 
#               experiments['R0_0_M_rt4m'].temperature_lpf,
#               experiments['R0_0_M_rt1'].temperature_lpf,
#               experiments['R0_0_M_rt2'].temperature_lpf,
#               experiments['Rb0_0_M_rt1'].temperature_lpf,
#               experiments['Rb0_0_M_rt2'].temperature_lpf,
#               experiments['Rb0_0_M_rt3'].temperature_lpf,
#               experiments['Rb0_0_M_rt_vh'].temperature_lpf,
#               experiments['Rb0_0_M_rt_vh2'].temperature_lpf]
# vels = np.c_[experiments['R0_0_M_rt1m'].C, 
#               experiments['R0_0_M_rt2m'].C, 
#               experiments['R0_0_M_rt3m'].C, 
#               experiments['R0_0_M_rt4m'].C,
#               experiments['R0_0_M_rt1'].C,
#               experiments['R0_0_M_rt2'].C,
#               experiments['Rb0_0_M_rt1'].C,
#               experiments['Rb0_0_M_rt2'].C,
#               experiments['Rb0_0_M_rt3'].C,
#               experiments['Rb0_0_M_rt_vh'].C,
#               experiments['Rb0_0_M_rt_vh2'].C]

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

#%%
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


# ----------
# Regression
# ----------
cf = US.CurveFit(masked_temps.flatten(), masked_vels.flatten(), [2100, -40, 1, -0.01], func=3, errorfunc='L2', nparams=4, npredictions=10_000)
cfu = US.CurveFit(temps.flatten(), cf.u, [2100, -40, 1, -0.01], func=3, errorfunc='L2', nparams=4, npredictions=10_000)
cfl = US.CurveFit(temps.flatten(), cf.l, [2100, -40, 1, -0.01], func=3, errorfunc='L2', nparams=4, npredictions=10_000)
temp_model = np.arange(np.min(temps), np.max(temps), 0.001)
model_u = cfu.func(cfu.params_opt, temp_model)
model_l = cfl.func(cfl.params_opt, temp_model)
print(cf)
print(f'R2 = {cf.r2}')
print(f'Prediction Interval = \u00b1 {np.mean((model_l - model_u)/2)}')


#%%
# --------
# Plotting
# --------
ax = plt.subplots(1)[1]
ax.scatter(masked_temps, masked_vels, c='C0', alpha=0.5, s=1, label='Data')
# ax.plot(masked_temps, masked_vels, c='C0', alpha=0.5, label='Data')
ax.plot(temp_model, cf.func(cf.params_opt, temp_model), c='k', label='Fit')
ax.plot(temp_model, cfu.func(cfu.params_opt, temp_model), c='k', ls='--', label='97.5% Prediction Interval')
ax.plot(temp_model, cfl.func(cfl.params_opt, temp_model), c='k', ls='--')
ax.set_ylabel('Speed of sound (m/s)')
ax.set_xlabel('Temperature (\u2103)')
plt.legend()
plt.tight_layout()
