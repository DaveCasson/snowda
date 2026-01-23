"""
This script is a Python implementation of the Helbig et al, 2021 seasonal algorithm of the snow-covered area algorithm.

Helbig, Nora, Michael Schirmer, Jan Magnusson, Flavia Mäder, Alec van Herwijnen, Louis Quéno, Yves Bühler, 
Jeff S. Deems, and Simon Gascoin. 2021. "A Seasonal Algorithm of the Snow-Covered Area Fraction for 
Mountainous Terrain." The Cryosphere 15 (9): 4607–24. https://doi.org/10.5194/tc-15-4607-2021.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
from numba import jit, float64, boolean
from concurrent.futures import ThreadPoolExecutor
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def validate_inputs(snow_depth_n, swe_tmp, swe_hist_n, buffer_length):
    """Validate input arrays and parameters"""
    if not all(isinstance(arr, np.ndarray) for arr in [snow_depth_n, swe_tmp, swe_hist_n]):
        raise TypeError("Input arrays must be numpy arrays")
    
    if not (len(snow_depth_n) == len(swe_tmp) == swe_hist_n.shape[0]):
        raise ValueError("Input arrays must have matching lengths")
    
    if buffer_length <= 0:
        raise ValueError("Buffer length must be positive")

@jit(nopython=True)
def _calculate_snow_cover_core(
    snow_depth_n, snow_depth_max_n, snow_depth_min_n,
    sd_snow_depth_0, d_snow_depth, d_snow_depth_max, d_snow_depth_recent,
    snow_depth_threshold
):
    """Numba-optimized core calculations for snow cover fraction"""
    
    # Calculate seasonal fSCA
    sd_snow_depth_0_dhs = d_snow_depth_max**0.84
    sd_snow_depth_0_dhs_recent = d_snow_depth_recent**0.84
    
    # Vectorized calculations
    valid_max = snow_depth_max_n > 0
    fsnow_season = np.where(valid_max,
                           np.tanh(1.3 * snow_depth_min_n / sd_snow_depth_0),
                           0.0)
    
    # Calculate new snow fSCA components
    fsnow_nsnow = np.where(d_snow_depth_max > 0,
                          np.tanh(1.3 * d_snow_depth / sd_snow_depth_0_dhs),
                          0.0)
    
    fsnow_nsnow_recent = np.where(d_snow_depth_recent > 0,
                                 np.tanh(1.3 * d_snow_depth_recent / sd_snow_depth_0_dhs_recent),
                                 0.0)
    
    # Combine results
    fsnow_nsnow = np.maximum(fsnow_nsnow, fsnow_nsnow_recent)
    
    # Apply thresholds
    fsnow_season = np.where(snow_depth_min_n < snow_depth_threshold, 0, fsnow_season)
    fsnow_nsnow = np.where(d_snow_depth < snow_depth_threshold, 0, fsnow_nsnow)
    
    # Calculate final fSCA
    fsnow_n = np.maximum(fsnow_season, fsnow_nsnow)
    fsnow_n = np.minimum(fsnow_n, 1.0)
    
    return fsnow_n

def snow_cover_fraction(
    snow_depth_n,
    swe_tmp,
    swe_hist_n,
    swe_min_n,
    swe_max_n,
    snow_depth_hist_n,
    snow_depth_min_n,
    snow_depth_max_n,
    slope_mu_n,
    xi_n,
    Ld_n,
    snow_depth_threshold,
    buffer_length
):
    """
    Calculate snow cover fraction for multiple HRUs using array operations
    
    Parameters
    ----------
    snow_depth_n : array-like
        Snow depth for each HRU (m)
    swe_tmp : array-like 
        Temporary SWE for each HRU (kg/m^2)
    swe_hist_n : array-like
        History of SWE for each HRU during buffer period (kg/m^2)
    swe_min_n : array-like
        Min SWE during season for each HRU (m)
    swe_max_n : array-like
        Max SWE during season for each HRU (m) 
    snow_depth_hist_n : array-like
        History of snow depth for each HRU during buffer period (m)
    snow_depth_min_n : array-like
        Min snow depth at time step swemin (m)
    snow_depth_max_n : array-like
        Max snow depth at time step swemax (m)
    slope_mu_n : array-like
        Squared slope related parameter for each HRU
    xi_n : array-like
        Terrain correlation length for each HRU
    Ld_n : array-like
        Coarse-grid cell size for each HRU
    snow_depth_threshold : float
        Threshold for minimum snow depth
    buffer_length : int
        Length of history buffer
        
    Returns
    -------
    tuple
        (fsnow_n, snow_depth_min_n, snow_depth_max_n, snow_depth_hist_n, 
         swe_min_n, swe_max_n, swe_hist_n, coeff_vari)
    """
    # Validate inputs
    validate_inputs(snow_depth_n, swe_tmp, swe_hist_n, buffer_length)
    
    n_hrus = len(snow_depth_n)
    
    # Pre-allocate arrays
    swe_buffer = np.zeros((n_hrus, buffer_length))
    snow_depth_buffer = np.zeros((n_hrus, buffer_length))
    
    # Fill buffers
    swe_buffer[:,0] = swe_tmp
    swe_buffer[:,1:] = swe_hist_n
    snow_depth_buffer[:,0] = snow_depth_n  
    snow_depth_buffer[:,1:] = snow_depth_hist_n
    
    # Update history
    swe_hist_n = swe_buffer[:,:-1]
    snow_depth_hist_n = snow_depth_buffer[:,:-1]
    
    # Calculate indices efficiently
    i_abs_max = np.argmax(swe_buffer, axis=1)
    i_abs_min = np.argmin(swe_buffer, axis=1)
    
    # Calculate diff vector and find recent min
    diff_swe_buffer = np.diff(swe_buffer, axis=1)
    i_break = np.argmax((diff_swe_buffer > 0.5), axis=1)
    i_break[i_break == 0] = buffer_length - 2
    
    i_recent_min = np.array([np.argmin(swe_buffer[i,:idx+1]) for i, idx in enumerate(i_break)])
    
    # Get snow depth values using indices
    snow_depth_min_buffer = np.take_along_axis(snow_depth_buffer, i_abs_min[:,None], axis=1).squeeze()
    snow_depth_max_buffer = np.take_along_axis(snow_depth_buffer, i_abs_max[:,None], axis=1).squeeze()
    snow_depth_min_recent = np.take_along_axis(snow_depth_buffer, i_recent_min[:,None], axis=1).squeeze()
    
    # Calculate snow depth differences
    d_snow_depth = np.maximum(snow_depth_n - snow_depth_min_buffer, 0)
    d_snow_depth_max = np.maximum(snow_depth_max_buffer - snow_depth_min_buffer, 0)
    d_snow_depth_max = np.maximum(d_snow_depth_max, d_snow_depth)
    d_snow_depth_recent = np.maximum(snow_depth_n - snow_depth_min_recent, 0)
    
    # Update state variables based on current values
    no_snow = swe_tmp == 0
    swe_max_n = np.where(no_snow, 0, swe_max_n)
    swe_min_n = np.where(no_snow, 0, swe_min_n)
    snow_depth_max_n = np.where(no_snow, 0, snow_depth_max_n)
    snow_depth_min_n = np.where(no_snow, 0, snow_depth_min_n)
    
    # Update max/min when current exceeds previous
    new_max = swe_tmp >= swe_max_n
    swe_max_n = np.where(new_max, swe_tmp, swe_max_n)
    swe_min_n = np.where(new_max, swe_tmp, swe_min_n)
    snow_depth_max_n = np.where(new_max, snow_depth_n, snow_depth_max_n)
    snow_depth_min_n = np.where(new_max, snow_depth_n, snow_depth_min_n)
    
    # Update min when current is lower
    new_min = (swe_tmp < swe_max_n) & (swe_tmp < swe_min_n)
    swe_min_n = np.where(new_min, swe_tmp, swe_min_n)
    snow_depth_min_n = np.where(new_min, snow_depth_n, snow_depth_min_n)
    
    # Calculate topo terms
    sd_snow_depth_1 = np.exp(-1 / (Ld_n / xi_n) ** 2)
    sd_snow_depth_2 = snow_depth_max_n ** (0.5330 * (Ld_n**0.0389))
    sd_snow_depth_3 = slope_mu_n ** (0.3193 * (Ld_n**0.1034))
    sd_snow_depth_0 = sd_snow_depth_1 * sd_snow_depth_2 * sd_snow_depth_3
    
    # Handle flat grid cells
    flat_cells = ~(slope_mu_n > np.finfo(float).eps)
    sd_snow_depth_0 = np.where(flat_cells, snow_depth_max_n**0.84, sd_snow_depth_0)
    
    # Calculate seasonal fSCA
    fsnow_season = np.zeros(n_hrus)
    valid_max = snow_depth_max_n > 0
    fsnow_season[valid_max] = np.tanh(1.3 * snow_depth_min_n[valid_max] / sd_snow_depth_0[valid_max])
    
    # Calculate coefficient of variation
    coeff_vari = np.where(snow_depth_max_n > 0, 
                         sd_snow_depth_0 / snow_depth_max_n,
                         np.nan)
    
    # Calculate new snow fSCA
    sd_snow_depth_0_dhs = d_snow_depth_max**0.84
    fsnow_nsnow = np.zeros(n_hrus)
    valid_d_max = d_snow_depth_max > 0
    fsnow_nsnow[valid_d_max] = np.tanh(1.3 * d_snow_depth[valid_d_max] / sd_snow_depth_0_dhs[valid_d_max])
    
    # Calculate recent new snow fSCA
    sd_snow_depth_0_dhs_recent = d_snow_depth_recent**0.84
    fsnow_nsnow_recent = np.zeros(n_hrus)
    valid_d_recent = d_snow_depth_recent > 0
    fsnow_nsnow_recent[valid_d_recent] = np.tanh(1.3 * d_snow_depth_recent[valid_d_recent] / 
                                                sd_snow_depth_0_dhs_recent[valid_d_recent])
    
    # Take maximum of new snow fSCAs
    fsnow_nsnow = np.maximum(fsnow_nsnow, fsnow_nsnow_recent)
    
    # Apply thresholds
    fsnow_season = np.where(snow_depth_min_n < snow_depth_threshold, 0, fsnow_season)
    fsnow_nsnow = np.where(d_snow_depth < snow_depth_threshold, 0, fsnow_nsnow)
    
    # Calculate final fSCA
    fsnow_n = np.maximum(fsnow_season, fsnow_nsnow)
    fsnow_n = np.where(snow_depth_n < np.finfo(float).eps, 0, np.minimum(fsnow_n, 1))
    
    return fsnow_n, snow_depth_min_n, snow_depth_max_n, snow_depth_hist_n, swe_min_n, swe_max_n, swe_hist_n, coeff_vari

def helbig_fsca_timestep(swe_timestep, snow_depth_timestep, hru_settings, state):
    """Calculate FSCA for all HRUs at a single timestep"""
    n_hrus = len(swe_timestep)
    
    updated_state = {
        'swe_min': np.zeros(n_hrus),
        'swe_max': np.zeros(n_hrus),
        'snow_depth_min': np.zeros(n_hrus),
        'snow_depth_max': np.zeros(n_hrus),
        'swe_hist': np.zeros((n_hrus, hru_settings['buffer_length']-1)),
        'snow_depth_hist': np.zeros((n_hrus, hru_settings['buffer_length']-1))
    }
    
    fsca_timestep, snow_depth_min, snow_depth_max, snow_depth_hist, swe_min, swe_max, swe_hist, _ = snow_cover_fraction(
        snow_depth_timestep,
        swe_timestep,
        state['swe_hist'],
        state['swe_min'],
        state['swe_max'],
        state['snow_depth_hist'],
        state['snow_depth_min'],
        state['snow_depth_max'],
        hru_settings['slope_mu'],
        hru_settings['xi'],
        hru_settings['Ld'],
        hru_settings['snow_depth_threshold'],
        hru_settings['buffer_length']
    )
    
    updated_state.update({
        'swe_min': swe_min,
        'swe_max': swe_max,
        'snow_depth_min': snow_depth_min,
        'snow_depth_max': snow_depth_max,
        'swe_hist': swe_hist,
        'snow_depth_hist': snow_depth_hist
    })
    
    return fsca_timestep, updated_state

@timing_decorator
def calculate_fsca(summa_output_ds, gdf, settings_df, global_settings):
    """
    Main function to calculate FSCA for all timesteps and HRUs
    
    Parameters
    ----------
    summa_output_ds : xarray.Dataset
        SUMMA output dataset containing SWE and snow depth
    gdf : GeoDataFrame
        Geodataframe containing HRU information
    settings_df : pandas.DataFrame
        DataFrame containing HRU-specific settings
    global_settings : dict
        Dictionary containing global parameters
        
    Returns
    -------
    tuple
        (fsca_results, state_results) where state_results contains arrays of state variables
    """
    n_timesteps = len(summa_output_ds.time)
    n_hrus = len(gdf)
    
    # Initialize state with zeros
    state = {
        'swe_min': np.zeros(n_hrus),
        'swe_max': np.zeros(n_hrus),
        'snow_depth_min': np.zeros(n_hrus),
        'snow_depth_max': np.zeros(n_hrus),
        'swe_hist': np.zeros((n_hrus, global_settings['buffer_length']-1)),
        'snow_depth_hist': np.zeros((n_hrus, global_settings['buffer_length']-1))
    }
    
    # Create HRU-specific settings
    hru_settings = {
        'slope_mu': settings_df['slope_mu'].values,
        'xi': settings_df['xi'].values,
        'Ld': settings_df['Ld'].values,
        'snow_depth_threshold': global_settings['snow_depth_threshold'],
        'buffer_length': global_settings['buffer_length']
    }
    
    # Initialize results arrays
    fsca_results = np.zeros((n_timesteps, n_hrus))
    state_results = {
        'swe_min': np.zeros((n_timesteps, n_hrus)),
        'swe_max': np.zeros((n_timesteps, n_hrus)), 
        'snow_depth_min': np.zeros((n_timesteps, n_hrus)),
        'snow_depth_max': np.zeros((n_timesteps, n_hrus))
    }
    
    # Process timesteps sequentially
    for t in range(n_timesteps):
        # Get data for current timestep
        swe_timestep = summa_output_ds['scalarSWE'][t].values
        snow_depth_timestep = summa_output_ds['scalarSnowDepth'][t].values * 1000  # Convert to mm
        
        # Calculate FSCA and update state
        fsca_timestep, state = helbig_fsca_timestep(
            swe_timestep,
            snow_depth_timestep,
            hru_settings,
            state
        )
        
        # Store results
        fsca_results[t] = fsca_timestep
        state_results['swe_min'][t] = state['swe_min']
        state_results['swe_max'][t] = state['swe_max']
        state_results['snow_depth_min'][t] = state['snow_depth_min']
        state_results['snow_depth_max'][t] = state['snow_depth_max']
        
        # Optional progress reporting for long runs
        if (t + 1) % 100 == 0:
            print(f"Processed {t + 1}/{n_timesteps} timesteps")
    
    return fsca_results, state_results

def plot_fsca_output(input_snow_data,fsnow_array,fsca_fortran_generated_data,fsca_original_data):

    fig, ax1 = plt.subplots(nrows=2,figsize=(8, 6))
    ax2 = ax1[0].twinx()
    ax3 = ax1[1].twinx()

    ax1[0].plot(fsca_fortran_generated_data, color='b', label="Fortran generated fSCA")
    ax1[0].plot(fsnow_array, color='r', label="Python generated fSCA", ls='--')
    ax2.plot(input_snow_data[1],color='k',ls='--')
    ax1[0].set_title('Comparison Fortran and Python Generated fSCA (Helbig et al, 2021)')
    ax1[0].set_xlabel('Day')
    ax1[0].set_ylabel('fSCA')
    ax1[0].get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylabel('SWE (mm)')
    ax1[0].legend(loc='upper right')

    ax1[1].plot(fsca_fortran_generated_data, color='b', label="Fortran generated fSCA") 
    ax1[1].plot(fsnow_array, color='r', label="Python generated fSCA", ls='--')
    ax3.plot(input_snow_data[0],color='k',ls='--')
    ax1[1].set_title('Comparison Fortran and Python Generated fSCA (Helbig et al, 2021)')
    ax1[1].set_xlabel('Day')
    ax1[1].set_ylabel('fSCA')
    ax3.set_ylabel('Snow Depth (mm)')
    ax1[1].legend(loc='upper right')

    plt.show()
def test_case():
    # Initialize settings
    global_settings = {
        'snow_depth_threshold': 0.005714286,
        'buffer_length': 15
    }

    # Create settings DataFrame with a single row
    settings_df = pd.DataFrame({
        'Ld': [1000],
        'slope_mu': [0.445], 
        'std_dem': [117.85],
        'xi': [2 * 117.85 / 0.445]  # Calculate xi
    })

    # Open input and output files
    input_hs_swe_path = '/Users/dcasson/GitHub/snow_dist/test_cases/helbig_2021/HS_SWE.txt'
    fsca_fortran_generated_path = '/Users/dcasson/GitHub/snow_dist/test_cases/helbig_2021/fSCA.txt'
    fsca_original_path = '/Users/dcasson/GitHub/snow_dist/test_cases/helbig_2021/fSCA_original.txt'
    input_snow_data = pd.read_csv(input_hs_swe_path, header=None, delimiter="\t")
    fsca_fortran_generated_data = pd.read_csv(fsca_fortran_generated_path, header=None, delimiter="\t")
    fsca_original_data = pd.read_csv(fsca_original_path, header=None, delimiter="\t")

    # Create a mock xarray dataset with the input data
    times = pd.date_range('2000-01-01', periods=len(input_snow_data), freq='D')
    ds = xr.Dataset(
        data_vars={
            'scalarSWE': (['time', 'hru'], input_snow_data[1].values.reshape(-1, 1)),
            'scalarSnowDepth': (['time', 'hru'], (input_snow_data[0].values/1000).reshape(-1, 1))  # Convert mm to m
        },
        coords={
            'time': times,
            'hru': [0]
        }
    )

    # Create mock geodataframe with one HRU
    gdf = gpd.GeoDataFrame({'geometry': [None]})

    # Calculate FSCA
    fsca_results, _ = calculate_fsca(ds, gdf, settings_df, global_settings)

    # Plot output
    plot_fsca_output(input_snow_data, fsca_results.squeeze(), fsca_fortran_generated_data, fsca_original_data)

def main():
    # File paths
    summa_output = '/Users/dcasson/Data/yukon_esp/summa_output/Chena_base_timestep.nc'
    output_shapefile_path = '/Users/dcasson/Data/YukonProject/domain_Chena/river_basins/output_shapefile_with_terrain_parameters_hrus.shp'
    
    # Load data
    summa_output_ds = xr.open_dataset(summa_output)
    
    # Clip dates
    start_time = pd.Timestamp('2005-09-01 00:00')
    end_time = pd.Timestamp('2009-10-01 23:00')
    summa_output_ds = summa_output_ds.sel(time=slice(start_time, end_time))
    
    # Resample to daily timesteps
    summa_output_ds = summa_output_ds.resample(time='D').mean()
    
    # Read shapefile
    gdf = gpd.read_file(output_shapefile_path)
    
    # Create settings DataFrame
    settings_df = pd.DataFrame({
        'Ld': gdf['Ld'],
        'slope_mu': gdf['mu'],
        'std_dem': gdf['std_dem'],
        'xi': gdf['xi']
    })
    
    # Global settings
    global_settings = {
        'snow_depth_threshold': 0.005714286,
        'buffer_length': 15
    }
    
    # Calculate FSCA and state variables
    fsca_results, state_results = calculate_fsca(summa_output_ds, gdf, settings_df, global_settings)
    
    # Create DataArrays for FSCA and state variables
    fsca_da = xr.DataArray(
        data=fsca_results,
        dims=['time', 'hru'],
        coords={
            'time': summa_output_ds.time,
            'hru': summa_output_ds.hru
        },
        attrs={
            'units': 'fraction',
            'long_name': 'Fractional Snow Covered Area',
            'description': 'FSCA calculated using Helbig method'
        }
    )
    
    # Create DataArrays for state variables
    state_das = {}
    for var_name, data in state_results.items():
        state_das[var_name] = xr.DataArray(
            data=data,
            dims=['time', 'hru'],
            coords={
                'time': summa_output_ds.time,
                'hru': summa_output_ds.hru
            },
            attrs={
                'units': 'mm' if 'snow_depth' in var_name else 'kg/m^2',
                'long_name': f'Helbig FSCA {var_name}',
                'description': f'State variable {var_name} from Helbig FSCA calculation'
            }
        )
    
    # Add to dataset
    summa_output_ds['FSCA'] = fsca_da
    for var_name, da in state_das.items():
        summa_output_ds[f'FSCA_{var_name}'] = da
    
    # Save dataset
    output_path = '/Users/dcasson/Data/yukon_esp/summa_output/Chena_base_timestep_with_fsca.nc'
    summa_output_ds.to_netcdf(output_path)
    print(f"Saved dataset with FSCA and state variables to {output_path}")
if __name__ == '__main__':

    #main()
    test_case()