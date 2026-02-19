import pysumma as ps
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import logging
import subprocess
import calendar



def run_pysumma_esp(config, state_files, state_dates, start_times, end_times):
    for state_file, state_date in zip(state_files, state_dates):
        for start_time, end_time in zip(start_times, end_times):
            config['sim_start_time'] = start_time
            config['sim_end_time'] = end_time
            # Formay start time as yyyymmddHH using pandas
            start_time = pd.to_datetime(start_time).strftime("%Y")
            config['run_suffix'] = f"state{state_date}_year{start_time}"
            summa_exe = config['summa_exe']
            file_manager = config['file_manager']
            output_dir = Path(config['summa_output_path'], f"{config['case_name']}_{config['run_suffix']}")
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running SUMMA ESP for {config['run_suffix']}")
            state = Path(state_file[0]).name

            sim = ps.Simulation(summa_exe, file_manager)

            sim_config = {
                'manager': {
                    'simStartTime': config['sim_start_time'],
                    'simEndTime': config['sim_end_time'],
                    'initConditionFile': str(state),
                    'outputPath': str(output_dir) + '/'
                }
            }
            
            sim.apply_config(sim_config)

            sim.run(
                run_suffix=config['run_suffix'],
                write_config=True,
                write_manager=False,
            )
            print(sim.status)

    return


def update_mizuroute_control_file(input_file_path,output_file_path, settings_to_update=None):
    with open(input_file_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        if settings_to_update:
            for key, value in settings_to_update.items():
                if line.strip().startswith(f'<{key}>'):
                    padding = len(line.split('<')[0])  
                    padded_key = f"<{key}>".ljust(30)  # Pad the key to 20 characters
                    padded_value = _pad_string(value)
                    comment = line.split('!')[-1].strip() if '!' in line else ''
                    line = f"{' ' * padding}{padded_key}{padded_value}! {comment}\n"
                    break

        updated_lines.append(line)

    with open(output_file_path, 'w') as f:
        f.writelines(updated_lines)

def _pad_string(string, pad_to=20):
    return f"{str(string):{pad_to}}"

def update_mizuroute_run_settings(input_control_file, output_control_file, input_summa_file, input_mizuroute_state, case_name, sim_start, sim_end, ancil_dir, input_dir, output_dir, restart_dir, restart_write='monthly'):
    
    logging.info(f'Updating mizuroute run settings')
 
    sim_start = sim_start.strftime('%Y-%m-%d %H:%M:%S')
    sim_end = sim_end.strftime('%Y-%m-%d %H:%M:%S')
    settings_to_update= {
        "fname_qsim": input_summa_file,
        "fname_state_in" : input_mizuroute_state,
        "case_name": case_name,
        "sim_start" : sim_start,
        "sim_end" : sim_end,
        "ancil_dir":  str(ancil_dir) + '/', 
        "input_dir":  str(input_dir) + '/',
        "output_dir": str(output_dir) + '/',
        "restart_dir": str(restart_dir) + '/',
        "restart_write": restart_write
        }

    update_mizuroute_control_file(input_control_file,output_control_file, settings_to_update)

    return

def run_mizuroute(model_exe, control_file, log_file=None):
    """
    Run the mizuRoute model.

    This method executes the mizuRoute model,
    and handles any errors that occur during the run.
    """

    
    # Run mizuRoute
    mizu_command = f"{model_exe} {control_file}"

    try:
        result = subprocess.run(mizu_command, shell=True, check=True,text=True, capture_output=True)
        if "FATAL ERROR" in result.stderr:
            logging.error(f"mizuRoute run failed with internal error: {result.stderr}")
        else:
            logging.info("mizuRoute run completed successfully.")
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"mizuRoute run failed with error: {e}")

def update_netcdf_time(input, new_year, output_file, file=True):
    """
    Shift all dates in the time variable of a NetCDF file by the difference in years
    between the original start year and the new year.

    Parameters:
    - input: str or xarray.Dataset
        The input NetCDF file path (if `file=True`) or an xarray.Dataset.
    - new_year: str or int
        The new year to shift the dates to (e.g., '2013' or 2013).
    - output_file: str
        Path to the output NetCDF file.
    - file: bool, optional (default=True)
        Whether the input is a file path (True) or an xarray.Dataset (False).

    Raises:
    - ValueError: If the `time` variable is missing or if duplicates persist after handling.
    """

    # Ensure new_year is an integer
    new_year = int(new_year)
    
    # Open the NetCDF file or use the provided Dataset
    if file:
        ds = xr.open_dataset(input)
    else:
        ds = input
    
    # Check if 'time' variable exists
    if 'time' not in ds.variables:
        raise ValueError("The dataset does not contain a 'time' variable.")
    
    # Extract the current time variable
    time_var = ds['time']
    
    # Convert the time variable to a pandas DatetimeIndex
    original_dates = pd.to_datetime(time_var.values)
    
    if original_dates.empty:
        raise ValueError("The 'time' variable is empty.")
    
    # Determine the original start year from the first date
    original_start_year = original_dates[0].year
    
    # Calculate the difference in years
    year_diff = new_year - original_start_year
    
    if year_diff == 0:
        print("New year is the same as the original start year. No shift applied.")
        shifted_dates = original_dates
    else:
        # Function to shift a single date by year_diff years
        def shift_year(dt, years):
            try:
                return dt.replace(year=dt.year + years)
            except ValueError:
                # Handle February 29 on non-leap years by setting to February 28
                if dt.month == 2 and dt.day == 29:
                    return dt.replace(year=dt.year + years, day=28)
                else:
                    # For other invalid dates, adjust to the last valid day of the month
                    max_day = calendar.monthrange(dt.year + years, dt.month)[1]
                    return dt.replace(year=dt.year + years, day=max_day)
        
        # Apply the shift to all dates
        shifted_dates = original_dates.map(lambda dt: shift_year(dt, year_diff))
    
    # Check for duplicate times after shifting
    if shifted_dates.duplicated().any():
        print("Shifting dates resulted in duplicate 'time' coordinates. Removing duplicates.")
        # Identify duplicated indices (excluding the first occurrence)
        duplicated_indices = np.where(shifted_dates.duplicated())[0]
        
        # Remove duplicates by keeping only the first occurrence
        unique_dates, unique_indices = np.unique(shifted_dates, return_index=True)
        shifted_dates = shifted_dates[unique_indices]
        
        # Select only the unique time entries in the dataset
        ds = ds.isel(time=unique_indices)
        
        print(f"Removed {len(duplicated_indices)} duplicate 'time' coordinates.")
        
        # Verify no duplicates remain
        if shifted_dates.duplicated().any():
            raise ValueError("Shifting dates resulted in duplicate 'time' coordinates even after handling.")
        else:
            print("Duplicate 'time' coordinates have been adjusted successfully.")
    
    # Update the time variable in the dataset
    ds['time'] = shifted_dates
    
    # Add or update time encoding (adjust units if necessary)
    ds['time'].encoding = {
        'units': 'days since 1900-01-01',
        'calendar': 'standard'
    }
    
    # Update dataset's attributes to reflect the change
    if 'history' in ds.attrs:
        ds.attrs['history'] += f" | Time shifted by {year_diff} years to {new_year}"
    else:
        ds.attrs['history'] = f"Time shifted by {year_diff} years to {new_year}"
    
    # Save the updated dataset to a new NetCDF file
    ds.to_netcdf(output_file)
    
    # Close the dataset if it was opened from a file
    if file:
        ds.close()