"""

This script is used to control data input and output.

Scripted by dave.casson@usask.ca

"""

import glob, os
import xarray as xr
from pathlib import Path
import numpy as np


def read_output_file_txt_to_ds(output_file_txt_path):
    with open(output_file_txt_path) as f:
        output_files = f.read().splitlines()

    ds = xr.open_mfdataset(output_files)

    return ds

def read_forcing_data_nc(forcing_data_path, forcing_data_pattern):
    """For 5 parameter snow model.
    Read input forcing data from available netcdf files, and convert to needed format.
    """

    # List all matching files
    forcing_files = glob.glob(f"{forcing_data_path}/*{forcing_data_pattern}")
    complete_ds = xr.open_mfdataset(forcing_files, decode_coords=True)

    # Seperate air temperature and precipitation. Update units on both
    airtemp_degC = (
        complete_ds["airtemp"] - 273.15
    )  # Convert degrees Kelvin to Degrees Celsius
    precip_mm_hr = complete_ds["pptrate"] * 3600  # Convert  kg*m2/s to mm/hr

    # Resample temperature and precipitation data
    airtemp = airtemp_degC.resample(time="D").mean()
    pptrate = precip_mm_hr.resample(time="D").sum()

    # Add the day of year
    doy = pptrate.time.dt.strftime("%j")
    doy_da = xr.DataArray(doy, coords={"time": pptrate.time}, dims=["time"], name="doy")
    # Merge into dataset
    forcing_ds = xr.merge([airtemp, pptrate, doy_da])

    return forcing_ds


def get_model_hrus(settings, ds_type="forcing"):
    """Retrieve model dimensions from a set file and return as array"""

    if ds_type == "forcing":
        # List all matching files
        forcing_files = glob.glob(f"{settings['paths']['forcing_data_path']}/*.nc")
        # Use forcing dataset
        sample_ds = xr.open_dataset(forcing_files[0], decode_coords=True)
        model_hrus = sample_ds["hru"].values
        sample_ds.close
    else:
        logger.warning(
            "Unable to read number of hrus. Check data source type specified"
        )

    return model_hrus


def read_param_from_file(settings, param_type):
    """Read in model parameters based on the"""
    param_nc_path = os.path.join(
        settings["paths"]["model_settings_path"],
        settings[f"{param_type}_parameters"]["file_name"],
    )

    ds = xr.open_dataset(param_nc_path)

    return ds


def write_default_model_param_to_file(settings):
    """Write default (lumped) cv settings to an output file"""

    model_hrus = get_model_hrus(settings)
    ds_list = []

    for var, value in settings["snow_5_parameters"]["lumped_settings"].items():
        array = np.full(
            np.shape(model_hrus), settings["snow_5_parameters"]["lumped_settings"][var]
        )

        da = xr.DataArray(
            array,
            coords={
                "hru": model_hrus,
            },
            dims=["hru"],
        )

        ds = da.to_dataset(name=var)
        ds_list.append(ds)

    ds_merged = xr.merge(ds_list)
    model_param_nc_path = Path(
        settings["paths"]["model_settings_path"],
        settings["snow_5_parameters"]["file_name"],
    )
    Path.unlink(model_param_nc_path, missing_ok=True)
    ds_merged.to_netcdf(model_param_nc_path)

    return


def output_to_nc(settings, results):
    """Output model results to NetCDF"""
    output_nc_path = Path(
        f'{settings["paths"]["model_run_path"]}/{settings["filenames"]["model_results_file"]}'
    )
    Path.unlink(output_nc_path, missing_ok=True)

    results.to_netcdf(output_nc_path)

def convert_arrays_to_dataset(output_arrays,basis_ds):

    new_ds = xr.Dataset()

    # Get the timesteps from the original dataset
    timesteps = basis_ds["time"].values

    # Iterate over the output arrays and add them to the new dataset
    for label, array in output_arrays.items():
        da = xr.DataArray(
            np.vstack(array),
            coords=[timesteps, basis_ds["hru"]],
            dims=["time", "hru"],
            name=label,
        )
        new_ds[label] = da

    #Merge the new dataset with the original dataset
    #new_ds = xr.merge([basis_ds, new_ds])

    new_ds.time.encoding['units'] = "seconds since 1970-01-01 00:00:00"

    return new_ds
