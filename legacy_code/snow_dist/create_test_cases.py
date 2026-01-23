"""

Create Test Cases

This script is used to create test cases, to provide a basis for algorithm testing and comparison.
The test cases are output to NetCDF, and can be used then by the SSNOW model.

Daily is initially created on a daily timestep

Test Case #1: Snow accumulation and apart melt.
Test Case #2: Interupting Snowfall Event
Test Case #3: Multiple Snowfall Events

"""

import yaml
from metsim import MetSim
from pathlib import Path
import xarray as xr
import numpy as np
import os
import pandas as pd
import subprocess
import shutil
from collections import OrderedDict


import logging

logging.getLogger().setLevel(logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)

def read_metsim_base_config(settings):
    """Create metsim domain file from summa model attribute file"""

    metsim_base_path = Path(settings["metsim"]["metsim_settings_file"])
    with open(metsim_base_path) as file:
        metsim_base_config = OrderedDict(yaml.load(file, Loader=yaml.FullLoader))

    return metsim_base_config

def generate_synthetic_data(settings):

    # Define the start and end dates for the year
    start_date = settings["forcing"]["start_time"].strftime("%Y-%m-%d")
    end_date = settings["forcing"]["end_time"].strftime("%Y-%m-%d")

    # Create a pandas date range for the entire year
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    num_days = len(date_range)
    # Create a pandas DataFrame with the date range as the index
    time_series_df = pd.DataFrame(index=date_range)

    # Simulate precipitation data (in mm)
    time_series_df['pptrate'] = np.random.exponential(3, len(date_range))  # Use exponential distribution for positive values
    
    # Simulate temperature data with seasonal trends (in Celsius)
    days = np.arange(1, num_days + 1)
    temperature_annual_amplitude = 15  # Amplitude of temperature fluctuations (adjust as needed)
    diurnal_temperature_range = 15  # Daily temperature range (adjust as needed)
    temperature_mean = -10  # Average temperature (adjust as needed)
    temperature_variation = temperature_annual_amplitude * np.cos(2 * np.pi * (days+180) / num_days)
    time_series_df['temp_min'] = temperature_mean - temperature_variation
    time_series_df['temp_max'] = time_series_df['temp_min'] + diurnal_temperature_range

    return time_series_df

def create_xarray_dataset_from_csv():
    """From the basis csv file, create a xarray dataset.
    Note that later conversion to SI units will be done by MetSim"""

    # Read in test data from csv
    #test_case_data = pd.read_csv(test_case_csv, index_col="DateTime")
    #logging.info(f"Reading test data from {test_case_csv}")
    #timeseries_data = pd.to_datetime(test_case_data.index)

    # Generate synthetic data for one year
    test_case_data = generate_synthetic_data(settings)
    timeseries_data = pd.to_datetime(test_case_data.index)
    
    # Create DataArrays
    temp_max_xr_array = xr.DataArray(
        test_case_data["temp_max"],
        dims=["time"],
        coords={"time": timeseries_data.values},
    )
    temp_min_xr_array = xr.DataArray(
        test_case_data["temp_min"],
        dims=["time"],
        coords={"time": timeseries_data.values},
    )
    pptrate_xr_array = xr.DataArray(
        test_case_data["pptrate"],
        dims=["time"],
        coords={"time": timeseries_data.values},
    )

    # Create DataSet
    meteo_dataset = xr.Dataset(
        {
            "t_max": temp_max_xr_array,
            "t_min": temp_min_xr_array,
            "pptrate": pptrate_xr_array,
        }
    )

    # Add hru coordinate
    meteo_dataset["hru"] = xr.DataArray([1], dims=["hru"])
    meteo_dataset["t_max"] = meteo_dataset["t_max"].expand_dims("hru")
    meteo_dataset["t_min"] = meteo_dataset["t_min"].expand_dims("hru")
    meteo_dataset["pptrate"] = meteo_dataset["pptrate"].expand_dims("hru")
    meteo_dataset = meteo_dataset.assign_coords({"hru": meteo_dataset["hru"]})

    meteo_dataset = meteo_dataset.transpose("time", "hru")

    return meteo_dataset


def read_metsim_base_config(settings):
    """Create metsim domain file from summa model attribute file"""

    metsim_base_path = Path(settings["metsim"]["metsim_base_config"])
    with open(metsim_base_path) as file:
        metsim_base_config = OrderedDict(yaml.load(file, Loader=yaml.FullLoader))

    return metsim_base_config


def create_metsim_config(settings, test_case_num):
    """Create metsim configuration file, both from master configuration file and from hardwired settings"""

    # Read settings from config file
    metsim_base_config = read_metsim_base_config(settings)

    # Add in run specific settings
    metsim_base_config["start"] = settings["forcing"]["start_time"].strftime("%Y-%m-%d")
    metsim_base_config["stop"] = settings["forcing"]["end_time"].strftime("%Y-%m-%d")
    metsim_base_config["scheduler"] = "threading"
    metsim_base_config["time_step"] = settings["metsim"]["timestep_minutes"]
    metsim_base_config["out_prefix"] = f"test_case_{test_case_num}"

    # Add the location of state, forcing and domain files
    forcing_str = Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["forcing_nc"]
    )
    metsim_base_config["forcing"] = str(forcing_str)
    state_str = Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["state_nc"]
    )
    metsim_base_config["state"] = str(state_str)
    domain_str = Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["domain_nc"]
    )
    metsim_base_config["domain"] = str(domain_str)

    # Add final output directory
    out_dir_path = Path(settings["metsim"]["metsim_result_dir"])
    metsim_base_config["out_dir"] = str(out_dir_path)

    # Intialize metsim config
    ms = MetSim(metsim_base_config)

    return ms


def create_domain_nc(settings):
    """See as reference https://github.com/UW-Hydro/MetSim-tutorial/blob/master/3_creating_input_datasets.ipynb"""
    coords = {"hru": [int(settings["state"]["hrus"])]}
    domain = xr.Dataset(coords=coords)
    domain["hruId"] = xr.DataArray(
        data=np.full((1), np.nan), coords=coords, dims=("hru")
    )
    domain["elevation"] = xr.DataArray(
        data=np.full((1), np.nan), coords=coords, dims=("hru")
    )
    domain["mask"] = xr.DataArray(
        data=np.full((1), np.nan), coords=coords, dims=("hru")
    )
    domain["lat"] = xr.DataArray(data=np.full((1), np.nan), coords=coords, dims=("hru"))
    domain["lon"] = xr.DataArray(data=np.full((1), np.nan), coords=coords, dims=("hru"))
    # Add the data
    domain["hruId"][0] = int(1001)
    domain["elevation"][0] = float(settings["state"]["elevation"])
    domain["mask"][0] = 1
    domain["lat"][0] = float(settings["state"]["elevation"])
    domain["lon"][0] = float(settings["state"]["elevation"])
    domain_output_path = Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["domain_nc"]
    )
    domain.to_netcdf(domain_output_path)
    logging.info(f"Domain netcdf output to {domain_output_path}")


def create_state_nc(settings, test_case_meteo_ds):
    """Create state file based on 90 days of preceeding data. In this case, resampled from the end of water year.
    See as reference https://github.com/UW-Hydro/MetSim-tutorial/blob/master/3_creating_input_datasets.ipynb
    """

    # Read data and shift one year back
    dates = pd.to_datetime(test_case_meteo_ds["time"])
    dates = dates.shift(freq=pd.DateOffset(years=-1))

    # Create state array with 90 days data
    dims = ("time", "hru")

    coords = {"time": dates[-90:], "hru": [int(settings["state"]["hrus"])]}
    state = xr.Dataset(coords=coords)
    for varname in ["pptrate", "t_min", "t_max"]:
        state[varname] = xr.DataArray(
            data=test_case_meteo_ds[varname][-90:].values,
            coords=coords,
            dims=dims,
            name=varname,
        )

    # Output to NetCDF
    state_output_path = Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["state_nc"]
    )
    state.to_netcdf(state_output_path)
    logging.info(f"State netcdf output to {state_output_path}")


def create_forcing_nc(settings, test_case_dataset):
    """Create forcing netcdf directly from meteo inputs"""

    forcing_output_path = Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["forcing_nc"]
    )

    test_case_dataset.to_netcdf(forcing_output_path)
    logging.info(f"Forcing netcdf output to {forcing_output_path}")


def metsim_to_summa(settings, ms_output_files):
    """From the metsim output, create summa model forcing files from metsim output"""

    # Set directory paths. In this case doing the file manipulation in place
    input_dir = Path(settings["metsim"]["metsim_result_dir"])
    output_dir = Path(settings["metsim"]["metsim_result_dir"])

    # Read the domain nc
    hru_file = Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["domain_nc"]
    )

    # Set the timestep for the datastep variable to be added.
    timestep = int(settings["metsim"]["timestep_minutes"]) * 60

    # Create an interim file that contain hruids
    hru_id_nc = Path(output_dir, "hruIds.nc")
    if not hru_id_nc.exists():
        create_hruId_file_cmd = f"ncks -v hruId {hru_file} {hru_id_nc}"
        run_shell_cmd(create_hruId_file_cmd)

    for input_file_str in ms_output_files:
        input_file = Path(input_file_str)
        # Prepare output file path
        output_file = Path(output_dir, input_file.name)

        # Create interim files
        output_file_temp = Path(
            output_file.parent, f"{output_file.stem}_temp{input_file.suffix}"
        )
        input_file_backup = Path(
            input_file.parent, f"{input_file.stem}_backup{input_file.suffix}"
        )
        shutil.copy(input_file, input_file_backup)

        # Perform commands to convert to summa format
        append_hru_cmd = f"ncks -h -A {hru_id_nc} {input_file_backup}"

        drop_hru_variable_cmd = (
            f"ncks -O -C -x -v hru {input_file_backup} {output_file_temp}"
        )

        add_data_step_cmd = (
            f'ncap2 -O -s "data_step={timestep}" {output_file_temp} {output_file}'
        )
        add_data_step_units_cmd = (
            f'ncatted -O -a units,data_step,c,c,"seconds" {output_file}'
        )

        add_hruId_units_cmd = f'ncatted -O -a units,hruId,a,c,"-" {output_file}'

        run_shell_cmd(append_hru_cmd)
        run_shell_cmd(drop_hru_variable_cmd)
        run_shell_cmd(add_data_step_cmd)
        run_shell_cmd(add_data_step_units_cmd)
        run_shell_cmd(add_hruId_units_cmd)

        output_file_temp.unlink()
        input_file_backup.unlink()

    hru_id_nc.unlink()
    for tmp_file in output_dir.glob("*tmp*"):
        tmp_file.unlink()


def create_directories(settings):
    """Create needed directories"""

    Path(settings["metsim"]["metsim_input_dir"]).mkdir(parents=True, exist_ok=True)
    Path(settings["metsim"]["metsim_result_dir"]).mkdir(parents=True, exist_ok=True)


def delete_previous_files(settings):
    """Create needed directories"""

    Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["domain_nc"]
    ).unlink(missing_ok=True)
    Path(settings["metsim"]["metsim_input_dir"], settings["metsim"]["state_nc"]).unlink(
        missing_ok=True
    )
    Path(
        settings["metsim"]["metsim_input_dir"], settings["metsim"]["forcing_nc"]
    ).unlink(missing_ok=True)
    logging.info(f"Deleting previously generated domain, state and forcing files")


def run_shell_cmd(cmd):
    """Simple function to run shell command with subprocess"""
    logging.info(f"Running shell cmd: {cmd}")
    cmd_log = subprocess.run(cmd, shell=True, text=True)
    if cmd_log.stderr is not None:
        logging.error(cmd.stderr)


if __name__ == "__main__":
    
    config_file = Path(
        "/Users/drc858/GitHub/snow_dist/settings/metsim/create_test_cases_config.yaml"
    )
    with open(config_file) as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)

    create_directories(settings)
    delete_previous_files(settings)

    # met_data_empty = create_empty_dataset(settings)
    test_case_csvs = [
        "/Users/drc858/GitHub/snow_dist/example_data/test_cases/test_case_1.csv",
        "/Users/drc858/GitHub/snow_dist/example_data/test_cases/test_case_2.csv",
        "/Users/drc858/GitHub/snow_dist/example_data/test_cases/test_case_3.csv",
    ]
    test_case_ds_array = []

    for test_case_num, test_case_csv in enumerate(test_case_csvs, 1):
        if test_case_num == 1:
            test_case_meteo_ds = create_xarray_dataset_from_csv(test_case_csv)
            create_domain_nc(settings)
            create_state_nc(settings, test_case_meteo_ds)
        else:
            test_case_meteo_ds = create_xarray_dataset_from_csv(test_case_csv)

        settings["metsim"]["forcing_nc"] = f"forcing_test_{test_case_num}.nc"
        create_forcing_nc(settings, test_case_meteo_ds)

        ms = create_metsim_config(settings, test_case_num)
        ms.run()
        ms_output_files = [ms._get_output_filename(times) for times in ms._times]
        metsim_to_summa(settings, ms_output_files)
