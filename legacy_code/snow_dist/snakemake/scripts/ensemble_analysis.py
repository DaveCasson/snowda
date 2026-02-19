import xarray as xr
import pandas as pd
import numpy as np
import properscoring as ps

def read_ensemble_files(ens_files, var_name, basin_id):
    """
    Reads ensemble NetCDF files, selects a variable for the matching basinID (which is a variable itself), 
    and concatenates across ensembles.

    Parameters:
    - ens_files: list of strings. File paths to the ensemble NetCDF files.
    - var_name: string. The variable name to extract from each dataset.
    - basin_id: int or list of ints. The value(s) of the basinID(s) to select.

    Returns:
    - ens_ds: xarray.DataArray. Concatenated DataArray across ensembles for the selected basinID(s).
    """
    ens_ds_list = []
    for i, file in enumerate(ens_files):
        # Open the dataset
        ds = xr.open_dataset(file)
        
        # Round time to the nearest hour
        ds['time'] = ds['time'].dt.round('h')
        
        # Assign ensemble index as a new coordinate
        ds = ds.assign_coords(ens=i)
        # Find the index where basinID matches basin_id
        basin_index = ds.seg.where(ds.reachID == int(basin_id)).dropna(dim='seg').astype(int)
        
        # Select the variable for the matching basin_id
        selected_var = ds[var_name].isel(seg=basin_index[0].values)
        
        # Append the selected variable to the list
        ens_ds_list.append(selected_var)
    
    # Concatenate along the 'ens' dimension
    ens_ds = xr.concat(ens_ds_list, dim='ens')
    
    return ens_ds

def read_obs_data_matching_ens_dates(ens_ds, streamflow_obs, time_column='datetime', discharge_column='discharge_cms'):
    # Extract times from ens_ds
    ens_times = ens_ds['time'].values

    # Convert ens_times to pandas DatetimeIndex and round to the nearest hour
    ens_dates = pd.to_datetime(ens_times).round('h')

    # Use the dates to select corresponding values from streamflow_obs
    matched_obs = streamflow_obs[streamflow_obs[time_column].isin(ens_dates)]

    # Ensure the matched_obs is in the same order as ens_dates
    matched_obs = matched_obs.set_index(time_column).loc[ens_dates]

    # Create a new xarray DataArray with the matched observed values
    obs_da = xr.DataArray(
        data=matched_obs[discharge_column].values,
        dims=['time'],
        coords={'time': ens_ds['time']}
    )

    return obs_da

def match_ensemble_day_of_year(ens_ds, obs_df, start_year=None, end_year=None):
    """
    Matches the day of year from the ensemble data array to extract corresponding values from the observation dataframe,
    excluding years that correspond to the ensemble data array time period, and returns an xarray DataArray.

    Parameters:
    - ens_ds: xarray.DataArray with dimensions including 'time'.
    - obs_df: pandas DataFrame with index 'day_of_year' and columns for each year (e.g., 1992, 1994, ..., 2023).
    - start_year: int, optional. The start year for clipping the observation years.
    - end_year: int, optional. The end year for clipping the observation years.

    Returns:
    - An xarray DataArray with the observed values matching the day_of_year of ens_ds time, excluding years in ens_ds time.
      Dimensions are ('time', 'year').
    """
    # Extract times from ens_ds
    ens_times = ens_ds['time'].values
    ens_dates = pd.to_datetime(ens_times)

    # Normalize timestamps to the nearest hour to eliminate tiny deviations
    ens_dates = ens_dates.round('h')

    # Re-extract the years and day_of_year after rounding
    ens_years = ens_dates.year
    ens_day_of_year = ens_dates.dayofyear

    # Unique years in ens_ds times
    ens_years_unique = np.unique(ens_years)

    # Remove non-year columns from obs_df
    non_year_columns = ['mean_flow', 'std_dev', 'min_flow', 'max_flow', 'q25', 'q75', 'q10', 'q90']
    obs_df_years = obs_df.drop(columns=non_year_columns, errors='ignore')

    # Get all years available in obs_df
    obs_year_columns = obs_df_years.columns.astype(int)

    # Exclude years corresponding to ens_ds time
    years_to_use = [year for year in obs_year_columns if year not in ens_years_unique]

    # Clip the years based on start_year and end_year
    if start_year is not None:
        years_to_use = [year for year in years_to_use if year >= start_year]
    if end_year is not None:
        years_to_use = [year for year in years_to_use if year <= end_year]

    # Ensure that years_to_use is not empty
    if not years_to_use:
        raise ValueError("No observation years remain after applying exclusions and clipping.")

    # Filter the observation DataFrame to include only the selected years
    obs_df_filtered = obs_df_years[years_to_use]

    # For each day_of_year in ens_day_of_year, get the corresponding observed values
    observed_values_list = []
    for doy in ens_day_of_year:
        # Handle the case when day_of_year exceeds 365 in non-leap years
        if doy > 365 and not obs_df_filtered.index.isin([366]).any():
            obs_values = np.full(len(years_to_use), np.nan)
        else:
            # If day_of_year is in obs_df index
            if doy in obs_df_filtered.index:
                obs_values = obs_df_filtered.loc[doy].values
            else:
                # If day_of_year is not in index, fill with NaNs
                obs_values = np.full(len(years_to_use), np.nan)
        observed_values_list.append(obs_values)

    # Create a DataArray with observed values
    observed_values_array = np.array(observed_values_list)

    # Create coordinates for the DataArray
    coords = {
        'ens': years_to_use,
        'time': ens_dates
    }

    # Create the DataArray
    obs_values_da = xr.DataArray(
        data=observed_values_array.T,
        dims=['ens', 'time'],
        coords=coords,
        name='observed_values'
    )

    return obs_values_da

def continuous_rank_prob_score(Qobs, Qobs_ens, Qfc_ens, min_obs):
    # Ensure Qobs and Qfc_ens have the same time dimension
    assert Qobs_ens.time.equals(Qfc_ens.time), "Time dimensions must match"

    # Verify dimensions
    assert Qobs.dims == ('time',), "Qobs should be 1D with time dimension"
    #assert set(Qfc_ens.dims) == {'ens', 'time'}, "Qfc_ens should be 2D with ensemble and time dimensions"
    # Initialize arrays to store CRPS and CRPSS for each timestep
    CRPS_array = np.full(len(Qobs.time), np.nan)
    CRPSS_array = np.full(len(Qobs.time), np.nan)

    for i, time in enumerate(Qobs.time):

        Qobs_t = Qobs.sel(time=time)
        Qobs_ens_t = Qobs_ens.sel(time=time)
        Qfc_ens_t = Qfc_ens.sel(time=time)

        # Construct a baseline from the observations climatology
        baseline = Qobs_t

        if len(Qobs_ens_t.ens) >= min_obs:
            # Calculate CRPS and CRPSS for this timestep
            CRPS = ps.crps_ensemble(Qobs_t, Qfc_ens_t).mean()
            CRPS_baseline = ps.crps_ensemble(Qobs_t, Qobs_ens_t).mean()
            CRPSS = 1 - CRPS / CRPS_baseline

            CRPS_array[i] = CRPS
            CRPSS_array[i] = CRPSS

    return CRPS_array, CRPSS_array
