"""

Main code for running fsca models

"""

# Import python packages
import sys 
import numpy as np

sys.path.append("../snow_dist")

from ssnowd import ssnowd
from debeer import debeer

def initialize_output_arrays(timesteps, zeros_array):
    """
    Initialize the output arrays.

    Args:
        timesteps (array-like): An array containing the timesteps.
        forcing_ds (xarray.Dataset): An xarray Dataset containing the 'hru' data variable.

    Returns:
        dict: A dictionary where each key is a variable name and each value is an empty numpy array
              with dimensions matching the number of timesteps and the length of 'hru'.
    """
    n_timesteps = len(timesteps)
    n_hrus = len(zeros_array)

    output_arrays = {
        "swe_output": np.empty((n_timesteps, n_hrus)),
        "melt": np.empty((n_timesteps, n_hrus)),
        "snowfall": np.empty((n_timesteps, n_hrus)),
        "snowfall_total": np.empty((n_timesteps, n_hrus)),
        "melt_total": np.empty((n_timesteps, n_hrus)),
        "melt_flux": np.empty((n_timesteps, n_hrus)),
        "fsca": np.empty((n_timesteps, n_hrus)),
        "residual_new_snow": np.empty((n_timesteps, n_hrus)),
    }

    return output_arrays

def initialize_model_arrays(zeros_array):
    """
    Initialize the output arrays.

    Args:
        timesteps (array-like): An array containing the timesteps.
        forcing_ds (xarray.Dataset): An xarray Dataset containing the 'hru' data variable.

    Returns:
        dict: A dictionary where each key is a variable name and each value is an empty numpy array
              with dimensions matching the number of timesteps and the length of 'hru'.
    """

    n_hrus = len(zeros_array)

    # Generate empty arrays for model state, flux, and output
    model_arrays = {
        "swe": np.zeros(n_hrus),
        "melt": np.zeros(n_hrus),
        "snowfall": np.zeros(n_hrus),
        "snowfall_total": np.zeros(n_hrus),
        "melt_total": np.zeros(n_hrus),
        "melt_flux": np.zeros(n_hrus),
        "fsca": np.zeros(n_hrus),
        "residual_new_snow": np.zeros(n_hrus),
    }


    return model_arrays

def compute_fsca_hru(model_array, cv, fsca_model = 'ssnowd'):
    """
    Compute SWE for each HRU.

    Args:
        swe_input (float): SWE input for the current HRU.
        cv (float): CV for the current HRU.

    Returns:
        Dictionary containing updated model values for the current HRU.
    """

    cv = float(cv)
    
    if model_array['swe'] <= 0.01:
        swe_output = 0 
        snowfall_total_output = 0#model_array['snowfall_total']
        melt_total_output = 0
        fsca_output = 0
        residual_new_snow = 0

        output_dict = {
        "swe_output": swe_output,
        "snowfall": model_array['snowfall'],
        "melt": model_array['melt'],
        "snowfall_total": snowfall_total_output,
        "melt_total": melt_total_output,
        "fsca": fsca_output,
        'residual_new_snow': residual_new_snow
        }
    
    elif fsca_model == 'ssnowd':

        """ Run implementation of ssnowd from Liston 2004"""
        fsca_output, swe_output, snowfall_total_output, melt_total_output, _ , _ = ssnowd(
            model_array['snowfall'],
            model_array['melt'],
            model_array['snowfall_total'],
            model_array['melt_total'],
            model_array['swe'],
            model_array['residual_new_snow'],
            cv
        )
        
        #TODO: Add residual_new_snow to ssnowd
        residual_new_snow = 0
        
        output_dict = {
        "swe_output": swe_output,
        "snowfall": model_array['snowfall'],
        "melt": model_array['melt'],
        "snowfall_total": snowfall_total_output,
        "melt_total": melt_total_output,
        "fsca": fsca_output,
        "residual_new_snow": residual_new_snow
        }

    elif fsca_model == 'debeer': 
        """ Run implementation of debeer from DeBeer and Pomeroy 2009 """
        fsca_output, snowfall_total_output, melt_total_output, residual_new_snow = debeer(
            model_array['snowfall'],
            model_array['melt'],
            model_array['snowfall_total'],
            model_array['melt_total'],
            model_array['fsca'],
            model_array['swe'],
            cv
        )
        
        output_dict = {
        'snowfall_total': snowfall_total_output,
        'melt_total':  melt_total_output,
        "fsca": fsca_output,
        "residual_new_snow": residual_new_snow
        }

    return output_dict

