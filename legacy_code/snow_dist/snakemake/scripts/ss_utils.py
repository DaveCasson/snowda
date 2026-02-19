# Read the configuration and resolve paths for simpler reading
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from datetime import datetime, timedelta
import re
from collections import defaultdict
from typing import List, Dict, Optional


def get_num_hrus_from_shp(config, input_shp):
    """ Read the number of HRUs from the catchment shapefile """
    shp_hru_attr = config['watershed_tools']["discretized_shp_attr"]["hru_id"]

    gdf = gpd.read_file(input_shp)

    hru_ids = gdf[shp_hru_attr].values.astype(int)
    num_hru = len(hru_ids)

    return num_hru, hru_ids

def get_num_hrus_from_forcing(config):
    """Read the number of HRUs from the forcing file"""

    # Read all forcing files in the forcing directory
    _, _, forcing_files = next(os.walk(config["summa_forcing_dir"]))
    first_forcing_file = Path(config["summa_forcing_dir"], forcing_files[0])

    try:
        # Read the first forcing file
        forc = xr.open_dataset(first_forcing_file)
    except:
        sys.exit("Could not read forcing file: {}".format(first_forcing_file))

    # Get the sorting order from the forcing file
    forcing_hruIds = forc["hruId"].values.astype(
        int
    )  # 'hruId' is prescribed by SUMMA so this variable must exist

    # Number of HRUs
    num_hru = len(forcing_hruIds)

    forc.close()

    return num_hru, forcing_hruIds


def read_cwarhm_control_file(workflow_control_file):
    """Read complete control data from the CWARHM (https://github.com/CH-Earth/CWARHM)
       format control file

    Parameters
    ----------
    workflow_control_file : SUMMAworkflow control file path
        i.e./summaWorkflow_public/0_control_files/control_active.txt

    Returns
    -------
    control_options --> dict
        dictionary containing all options from the control file

    """
    comment_char = "#"
    option_char = "|"
    control_options = {}
    f = open(workflow_control_file)
    for line in f:
        # First, remove comments:
        if comment_char in line:
            # split on comment char, keep only the part before
            line, comment = line.split(comment_char, 1)
        # Second, find lines with an option=value:
        if option_char in line:
            # split on option char:
            option, value = line.split(option_char, 1)
            # strip spaces:
            option = option.strip()
            value = value.strip()
            # store in dictionary:
            control_options[option] = value

    f.close()

    return control_options


def move_files_to_main_directory(directory):
    """Move all files from subdirectories into the main directory"""
    directory = Path(directory)
    for subdir in directory.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(directory))

def generate_gru_start_and_count(sim_size, chunk_size: int = None, num_chunks: int = None):
    '''
    Generate a list of GRU start and stop times for parallel simulation.
    Original code: https://github.com/UW-Hydro/pysumma/blob/master/pysumma/distributed.py#L104

    The GRU format is automatically determined based on sim_size.
    For example, if sim_size is 99, then "G{:02d}" formatting is used; if sim_size is 100, then
    "G{:03d}" formatting is used.
    '''
    import numpy as np

    # Only specify at most one of `chunk_size` or `num_chunks`
    assert not (chunk_size and num_chunks), \
        "Only specify at most one of `chunk_size` or `num_chunks`!"

    # Default to a chunk_size if neither is provided
    if not (chunk_size or num_chunks):
        chunk_size = 12

    # Determine the chunk boundaries based on chunk_size or num_chunks
    if chunk_size:
        sim_truncated = (chunk_size - 1) * (sim_size // (chunk_size - 1))
        starts = np.arange(1, sim_truncated + 1, chunk_size).astype(int)
        stops = np.append(starts[1:], sim_size + 1)
        chunks = np.vstack([starts, stops]).T
    elif num_chunks:
        chunk_size = np.ceil(sim_size / num_chunks).astype(int)
        starts = np.arange(1, sim_size, chunk_size)
        stops = np.append(starts[1:], sim_size + 1)
        chunks = np.vstack([starts, stops]).T

    gru_chunk_dict = [{'startGRU': start, 'countGRU': stop - start}
                      for start, stop in chunks]

    # Determine the width dynamically: number of digits for sim_size.
    # For example, if sim_size == 99 -> width is 2, if sim_size == 100 -> width is 3.
    width = len(str(sim_size))
    fmt_str = f"G{{:0{width}d}}-{{:0{width}d}}"

    gru_chunk_strings = [
        fmt_str.format(item['startGRU'], item['startGRU'] + item['countGRU'] - 1)
        for item in gru_chunk_dict
    ]

    return gru_chunk_strings

def calc_num_grus(attribute_nc):
    """
    Calculate the number of GRUs in the SUMMA model.
    """
    with xr.open_dataset(attribute_nc) as ds:
        num_grus = ds['gruId'].shape[0]
    ds.close()
    return num_grus
    
def extract_gru_int(s):
    """Extract an integer value for the startGRU from the gru_chunk using regex"""
    match = re.search(r'G(\d+)-\d+', s)  # Changed to re.search for more flexibility
    if match:
        return int(match.group(1))  # Convert the captured group to an integer
    else:
        raise ValueError(f"Invalid gru_chunk format: {s}. Ensure the filename follows the 'GXX-YY' format.")

def merge_netcdf_output(unsorted_input_file_list, output_file):
    """Merge a list of netcdf files into a single file"""
    #Sort input file list

    # Sort the list based on the extracted numbers
    input_file_list = sorted(unsorted_input_file_list, key=extract_gru_int)

    out_ds = [xr.open_dataset(input_file) for input_file in input_file_list]
    hru_vars = [] # variables that have hru dimension
    gru_vars = [] # variables that have gru dimension
    for name, var in out_ds[0].variables.items():
        if 'hru' in var.dims:
            hru_vars.append(name)
        elif 'gru' in var.dims:
            gru_vars.append(name)
    hru_ds = [ds[hru_vars] for ds in out_ds]
    gru_ds = [ds[gru_vars] for ds in out_ds]
    hru_merged = xr.concat(hru_ds, dim='hru')
    gru_merged = xr.concat(gru_ds, dim='gru')

    merged_ds = xr.merge([hru_merged, gru_merged])

    if 'time' in merged_ds.dims:  # Check if 'time' dimension exists
        merged_ds['time'].encoding = {'units': 'hours since 1900-01-01', 'calendar': 'standard'}
    else:
        print("The 'time' dimension does not exist in the dataset. Encoding not applied.")

    merged_ds.to_netcdf(output_file)

    return merged_ds

def promote_keys(d):
    """
    Promote all keys from nested dictionaries to the top level.
    
    Parameters:
    d (dict): The dictionary from which to promote keys.

    Returns:
    dict: A new dictionary with all keys promoted to the top level.
    """
    result = {}
    
    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively promote keys from the nested dictionary
            promoted = promote_keys(value)
            # Update the result dictionary, promoting nested keys to the top
            result.update(promoted)
        else:
            # If it's not a dictionary, keep the current key-value pair
            result[key] = value
            
    return result

def get_files_by_date(directory: str, filter_str: str, months: Optional[List[int]] = None) -> Dict[str, List[str]]:
    """
    Retrieves and groups files from a directory based on a date extracted from filenames.

    Args:
        directory (str): The directory to search for files.
        filter_str (str): A string to filter filenames.
        months (Optional[List[int]]): A list of months (as integers) to include. If None, all months are included.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are date strings and values are lists of file paths.
    """
    file_list = os.listdir(directory)
    filtered_files = [f for f in file_list if filter_str in f]

    # Updated regex pattern to match date-like strings (YYYYMMDDHH) in different positions
    date_pattern = re.compile(r'_(\d{8,10})(?:_base_G\d{2}-\d{2})?\.nc$')

    files_by_date = defaultdict(list)

    for filename in filtered_files:
        match = date_pattern.search(filename)  # Use search to find date in the filename
        if match:
            date_str = match.group(1)  # Extracted date string: YYYYMMDDHH or YYYYMMDD
            month = int(date_str[4:6])  # Extract month from YYYYMMDD or YYYYMMDDHH
            if months is None or month in months:
                file_path = os.path.join(directory, filename)
                files_by_date[date_str].append(file_path)
        else:
            print(f"Warning: Could not extract date from filename: {filename}")

    return files_by_date

def get_restart_files_by_date(directory: str, filter_str: str, months: Optional[List[int]] = None) -> Dict[str, List[str]]:
    """
    Retrieves and groups files from a directory based on a date extracted from filenames.

    Args:
        directory (str): The directory to search for files.
        filter_str (str): A string to filter filenames.
        months (Optional[List[int]]): A list of months (as integers) to include. If None, all months are included.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are date strings (YYYYMMDDHH) and values are lists of file paths.
    """
    file_list = os.listdir(directory)
    filtered_files = [f for f in file_list if filter_str in f]

    # Generalized regex pattern to match a date in the format YYYYMMDDHH regardless of surrounding text
    date_pattern = re.compile(r'_(\d{10})')

    files_by_date = defaultdict(list)

    for filename in filtered_files:
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1)  # Extracted date string: YYYYMMDDHH
            month = int(date_str[4:6])  # Extract month from YYYYMMDDHH
            if months is None or month in months:
                file_path = os.path.join(directory, filename)
                # Group files by date (YYYYMMDDHH)
                files_by_date[date_str].append(file_path)
        else:
            print(f"Warning: Could not extract date from filename: {filename}")

    return files_by_date

    return files_by_date

def generate_times(input_date, start_year, stop_year, forecast_length, 
                   date_format_in="%Y%m%d%H", date_format_out="%Y-%m-%d %H:%M"):
    """
    Generate start and end times for forecasts over a range of years.

    Args:
        input_date (str): The initial date and time in the format specified by `date_format_in`.
        start_year (int): The starting year of the forecast generation.
        stop_year (int): The stopping year (inclusive) of the forecast generation.
        forecast_length (int): The number of days to add to each start time for the end time.
        date_format_in (str, optional): Input date format. Defaults to "%Y%m%d%H".
        date_format_out (str, optional): Output date format. Defaults to "%Y-%m-%d %H:%M".

    Returns:
        tuple: Two lists, `start_times` and `end_times`, containing the formatted start and end times 
               for each year in the range.
    """
    start_times = []
    end_times = []
    
    # Parse input date using the input format
    input_dt = datetime.strptime(input_date, date_format_in)
    
    # Loop through the range of years
    for year in range(start_year, stop_year + 1):
        try:
            # Adjust the year of the input date
            start_dt = input_dt.replace(year=year)
        except ValueError:
            # Handle February 29 on non-leap years
            if input_dt.month == 2 and input_dt.day == 29:
                start_dt = input_dt.replace(year=year, day=28)
            else:
                raise
        
        # Calculate end time by adding forecast length in days
        end_dt = start_dt + timedelta(days=forecast_length)
        
        # Append formatted start and end times to lists
        start_times.append(start_dt.strftime(date_format_out))
        end_times.append(end_dt.strftime(date_format_out))
    
    return start_times, end_times
