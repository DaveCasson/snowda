import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import re
import os  # Added import for os module
"""
def plot_swe_and_csv_data(nc_files, csv_files, start_date, end_date):
    # Get unique HRU IDs from CSV files
    hru_ids = []
    for csv_file in csv_files:
        match = re.search(r'_HRU(\d+)', csv_file)
        if match:
            hru_ids.append(int(match.group(1)))
    hru_ids = sorted(list(set(hru_ids)))
    
    # Create subplots - one for each HRU
    n_hrus = len(hru_ids)
    fig, axes = plt.subplots(n_hrus, 1, figsize=(15, 2*n_hrus))
    if n_hrus == 1:
        axes = [axes]
    
    # For each HRU
    for idx, hru_id in enumerate(hru_ids):
        ax = axes[idx]
        
        # Initialize lists to store SWE data for each ensemble
        swe_ensembles = []
        individual_swe = []
        
        # Load and plot each NetCDF file for this HRU
        for nc_file_path in nc_files:
            # Load the NetCDF file
            ds = xr.open_dataset(nc_file_path)
            
            # Filter by the date range
            ds = ds.sel(time=slice(start_date, end_date))
            
            # Retrieve NetCDF data for the specific HRU
            swe_data = ds['scalarSWE'].sel(hru=hru_id, drop=True) / 1000
            
            # Check if the file is part of an ensemble
            match = re.search(r'_(\d+)_timestep', nc_file_path)
            if match:
                ensemble_num = int(match.group(1))
                swe_ensembles.append(swe_data)
            else:
                # Get label from filename
                base_name = os.path.basename(nc_file_path)
                label = os.path.splitext(base_name)[0]
                swe_data = swe_data * 1000000
                individual_swe.append((swe_data, label))
        
        # Convert list of SWE data to a single xarray DataArray
        if swe_ensembles:
            swe_ensembles = xr.concat(swe_ensembles, dim='ensemble')
            
            # Calculate mean, min, and max across ensembles
            swe_mean = swe_ensembles.mean(dim='ensemble')
            swe_min = swe_ensembles.min(dim='ensemble')
            swe_max = swe_ensembles.max(dim='ensemble')
            
            # Plot mean, min, and max envelopes
            ax.plot(swe_mean['time'], swe_mean, label='Scalar SWE (Mean)', color='blue')
            ax.fill_between(swe_mean['time'], swe_min, swe_max, color='blue', alpha=0.3, label='Scalar SWE (Min-Max)')
        
        # Plot individual SWE lines
        for swe_data, label in individual_swe:
            ax.plot(swe_data['time'], swe_data, label=label, linestyle='-', linewidth=1)
        
        # Load and plot CSV data for this HRU
        for csv_file in csv_files:
            # Check if this CSV file corresponds to current HRU
            match = re.search(r'_HRU(\d+)', csv_file)
            if match and int(match.group(1)) == hru_id:
                # Load the CSV data and filter by date range
                csv_data = pd.read_csv(csv_file, parse_dates=['Date'])
                csv_data = csv_data[(csv_data['Date'] >= start_date) & 
                                  (csv_data['Date'] <= end_date)]
                
                # Plot each variable from CSV
                for col in csv_data.columns:
                    if col != 'Date':
                        ax.plot(csv_data['Date'], csv_data[col], 
                               label=f'{col} (Observed)', 
                               linestyle='--', linewidth=2)
        
        # Adding plot details for each subplot
        ax.set_xlabel("Time")
        ax.set_ylabel("SWE (m)")
        ax.set_title(f"HRU {hru_id}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    plt.suptitle(f"SWE Ensemble and Observations\n{start_date} to {end_date}")
    plt.tight_layout()
    plt.show()

# Usage example
nc_files = [
    '/Users/dcasson/Data/yukon_esp/summa_output/Chena_001_timestep.nc',
    '/Users/dcasson/Data/yukon_esp/summa_output/Chena_002_timestep.nc',
    '/Users/dcasson/Data/yukon_esp/summa_output/Chena_003_timestep.nc',
    '/Users/dcasson/Data/yukon_esp/summa_output/Chena_004_timestep.nc',
    '/Users/dcasson/Data/yukon_esp/summa_output/Chena_005_timestep.nc',
    '/Users/dcasson/Data/yukon_esp/summa_output/Chena_006_timestep.nc',
    '/Users/dcasson/Data/yukon_esp/summa_output/Chena_007_timestep.nc',
    '/Users/dcasson/Data/yukon_esp/summa_output/Chena_008_timestep.nc',
    '/Users/dcasson/Data/yukon_esp/summa_output/ChenaRDRS_timestep.nc',
]
csv_files = [
    '/Users/dcasson/Data/yukon_esp/observations/Chena_Near_Two_Rivers_SWE_processed_952_HRU71.csv',
    '/Users/dcasson/Data/yukon_esp/observations/Chena_Near_Two_Rivers_SWE_processed_951_HRU133.csv',
    '/Users/dcasson/Data/yukon_esp/observations/Chena_Near_Two_Rivers_SWE_processed_948_HRU132.csv',
    '/Users/dcasson/Data/yukon_esp/observations/Chena_Near_Two_Rivers_SWE_processed_949_HRU138.csv',
    '/Users/dcasson/Data/yukon_esp/observations/Chena_Near_Two_Rivers_SWE_processed_950_HRU55.csv'
]
start_date = '2015-10-01'
end_date = '2018-09-30'

plot_swe_and_csv_data(nc_files, csv_files, start_date, end_date)


"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def plot_variable_comparison(nc_files, variable, hru_id, start_date=None, end_date=None, accumulate=False):
    """
    Generate comparative charts of a specified variable over time across multiple NetCDF files.
    
    Parameters:
    - nc_files (list of str): List of paths to NetCDF files.
    - variable (str): The name of the variable to plot.
    - hru_id (int): The HRU (Hydrologic Response Unit) ID to filter and plot.
    - start_date (str): Start date in the format 'YYYY-MM-DD'. Defaults to None.
    - end_date (str): End date in the format 'YYYY-MM-DD'. Defaults to None.
    - accumulate (bool): Whether to show the accumulation of the variable over time.
    """
    plt.figure(figsize=(12, 6))
    
    for nc_file in nc_files:
        # Load the NetCDF file
        ds = xr.open_dataset(nc_file)
        
        # Check if variable and HRU exist in the dataset
        if variable not in ds.variables:
            print(f"Variable '{variable}' not found in {nc_file}. Skipping.")
            continue
        if 'hru' not in ds.dims or hru_id not in ds['hru']:
            print(f"HRU {hru_id} not found in {nc_file}. Skipping.")
            continue
        
        # Extract the variable for the specified HRU
        data = ds[variable].sel(hru=hru_id, drop=True)
        
        # Apply date filtering if specified
        if start_date or end_date:
            data = data.sel(time=slice(start_date, end_date))
        
        # Optionally accumulate the variable over time
        if accumulate:
            data = data.cumsum(dim='time')
        
        # Plot the data
        plt.plot(data['time'], data, label=f"{nc_file.split('/')[-1]}")
    
    # Customize plot
    plt.xlabel("Time")
    plt.ylabel(f"{variable} (Accumulated)" if accumulate else f"{variable}")
    plt.title(f"Comparison of {variable} for HRU {hru_id}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
nc_files = [
    '/Users/dcasson/Data/yukon_esp/forcing/SUMMA/Chena_Near_Two_Rivers_remapped_2018-06-01-00-00-00.nc',
    '/Users/dcasson/Data/yukon_esp/forcing/RDRS/Chena_Near_Two_Rivers_rdrs_remapped_2018-06-01-00-00-00.nc',
    '/Users/dcasson/Data/gpep/chena/forcing/summa/007/rf_static_boxcox_low_predictors_ensMember_007.nc',
]
plot_variable_comparison(
    nc_files,
    variable='airtemp',
    hru_id=71,
    start_date='2018-06-01',
    end_date='2018-06-30',
    accumulate=False
)