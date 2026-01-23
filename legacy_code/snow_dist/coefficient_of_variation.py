"""

This script is used to prepare the Coefficient of Variation (CV) values needed for distributed snow modelling

Scripted by dave.casson@usask.ca

"""

import numpy as np
import xarray as xr
import glob
from pathlib import Path
from data_io import get_model_hrus


def write_default_cv_to_file(settings):
    """Write default (lumped) cv settings to an output file"""

    model_hrus = get_model_hrus(settings)

    array = np.full(
        np.shape(model_hrus), settings["cv_parameters"]["lumped_settings"]["cv"]
    )
    # print(array)
    da = xr.DataArray(
        array,
        coords={
            "hru": model_hrus,
        },
        dims=["hru"],
    )

    ds = da.to_dataset(name="cv")
    cv_nc_path = Path(
        f'{settings["paths"]["model_settings_path"]}/{settings["cv_parameters"]["file_name"]}'
    )
    Path.unlink(cv_nc_path, missing_ok=True)

    ds.to_netcdf(cv_nc_path)

    return


def read_param_from_file(settings, param_type):
    param_nc_path = os.path.join(
        settings["paths"]["model_settings_path"],
        settings[f"{param_type}_parameters"]["file_name"],
    )
    ds = xr.open_dataset(param_nc_path)

    return ds


def set_lumped_cv(sample_ds, lumped_cv_value):
    """Apply Coefficient of Variation values for the model"""

    cv_array = np.full(np.shape(sample_ds["hru"]), lumped_cv_value)

    return cv_array
