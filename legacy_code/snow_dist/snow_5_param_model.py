"""
This script is used to emulate a 5 parameter snow model, roughly based off SNOW17.
The basis for is code is Slater et al, 2013, where the following description is given:
This is a simple 5 parameter snow model that follows largely
from the NWS SNOW-17 model. It operates on a daily timestep,
using forcing data taken from the SNOTEL sites. Forcing data is
mean daily air temperature (Tair), total daily precipitation (P) and
day of the year (DOY). SWE is always greater than or equal to zero.

Usage:
This model is developed to run off meterological forcing data, either for a single point or distributed over a grid.
In current form, it reads forcing data developed for a semi-distributed (hru-based) model.

Forcing data:
pptrate --> daily total precipitation
airtemp --> daily mean temperature
doy --> julian day of year

State Variables:
swe -> snow water equivalent

Model Parameters:
guc -> gauge undercatch
src -> snow rain
f_max -> maximum melt factor
f_min -> minimum melt factor
t_mlt -> air temperature below which no melt occurs

Scripted by Dave Casson, dave.casson@usask.ca
Reference: Slater, A. G., A. P. Barrett, M. P. Clark, J. D. Lundquist, and M. S. Raleigh. 2013. “Uncertainty in Seasonal Snow Reconstruction: Relative Impacts of Model Forcing and Image Availability.” Advances in Water Resources, Snow–Atmosphere Interactions and Hydrological Consequences, 55 (May): 165–77. https://doi.org/10.1016/j.advwatres.2012.07.006.
"""

# Import python packages
import sys 
import math

import xarray as xr

import numpy as np

from ssnowd import ssnowd

import logging

logger = logging.getLogger("log_to_file")

sys.path.append("../snow_dist")
from fsca_models import initialize_model_arrays, initialize_output_arrays, compute_fsca_hru
import snow_5_param_simulation


def generate_snowfall(precip, airtemp, model_param):
    """Step 1: Generate snow if the air temperature is less than the snow/rain criteria"""

    snowfall = np.where(
        airtemp < model_param["src"], precip * model_param["guc"], np.zeros_like(precip)
    )

    return snowfall


def update_snowpack(swe, snowfall):
    """Step 2: Update snowpack with snowfall"""

    return swe + snowfall


def phase_shift_day_of_year(doy):
    """Step 3: Phase shift the day of year"""

    mdoy = int(doy) - (366 / 4) + 10

    return mdoy


def compute_melt_factor(mdoy, model_param):
    """Step 4: Compute the melt factor for a given day of year"""
    mean_melt_factor = (model_param["f_max"] + model_param["f_min"]) / 2

    sin_term = math.sin((2 * math.pi * mdoy) / 366)

    diff_melt_factor = (model_param["f_max"] - model_param["f_min"]) / 2

    melt_rate_factor = mean_melt_factor * sin_term * diff_melt_factor

    # Set all negative values to zero
    melt_rate_factor[melt_rate_factor < 0] = 0

    return melt_rate_factor


def compute_potential_melt(airtemp, melt_rate_factor, model_param):
    """Step 5: Compute the potential melt"""

    melt_potential = np.where(
        airtemp > model_param["t_mlt"],
        melt_rate_factor * (airtemp - model_param["t_mlt"]),
        0,
    )

    return melt_potential


def melt_swe(swe, melt_potential):
    """Step 6: Melt the SWE"""

    melt_rate = melt_potential

    melt = np.where(swe > melt_rate, melt_rate, swe)
    swe = np.where(swe > melt_rate, swe - melt_rate, 0)

    return swe, melt

def run_snow_5_param_timestep(pptrate, airtemp, doy, swe_5_param, model_param):
    """Run the 5 parameter snow model for 1 time step"""

    # Step 1: Generate snow if the air temperature is less than the snow/rain criteria
    snowfall = generate_snowfall(pptrate, airtemp, model_param)
    # Step 2: Update snowpack with snowfall
    swe_5_param = update_snowpack(swe_5_param, snowfall)
    # Step 3: Phase shift the day of year
    mdoy = phase_shift_day_of_year(doy)
    # Step 4: Compute the melt factor for a given day of year
    melt_rate_factor = compute_melt_factor(mdoy, model_param)
    # Step 5: Compute the potential melt
    melt_potential = compute_potential_melt(airtemp, melt_rate_factor, model_param)
    # Step 6: Melt the SWE
    swe_5_param, melt_rate = melt_swe(swe_5_param, melt_potential)

    return snowfall, melt_rate, melt_potential, swe_5_param


def run_snow_5_param_model(forcing_ds, model_param, cv_ds, use_model_swe=True):
    """
    Run the 5 parameter snow model on the input dataset.

    Args:
        forcing_ds (xarray.Dataset): The input dataset with model forcing variables.
        model_param (dict): Parameters for the snow model.
        cv_ds (xarray.Dataset): Dataset with cv values for each HRU.
        use_model_swe (bool, optional): Flag to indicate whether to use model-calculated SWE or input SWE. Defaults to True.

    Returns:
        xarray.Dataset: The output dataset with the model results.
    """

    # Extract necessary variables from the input dataset
    timesteps = forcing_ds["time"].values
    pptrate = forcing_ds["pptrate"].values
    airtemp = forcing_ds["airtemp"].values
    doy = forcing_ds["doy"].values
    cv_array = cv_ds.to_array().values[0]  # cv values for each HRU

    # Initialize model state, flux, and output arrays
    zeros_array = np.zeros_like(forcing_ds["hru"])

    model_arrays = initialize_model_arrays(zeros_array)
    output_arrays = initialize_output_arrays(timesteps, zeros_array)

    # Initial SWE for each timestep
    swe_start_timestep = model_arrays['swe']

    # Run the model forward in time, through each time step
    for i, timestep in enumerate(timesteps):

        # Run the 5 parameter snow model for the timestep
        (
            snowfall_timestep,
            melt_timestep,
            _,
            swe_end_timestep,
        ) = run_snow_5_param_timestep(
            pptrate[i, :], airtemp[i, :], doy[i], swe_start_timestep, model_param
        )

        # Iterate over HRUs
        for j, cv in enumerate(cv_array):

            # Current state of the model for HRU
            fsca_model_input = {
                "swe": swe_end_timestep[j],
                "snowfall": snowfall_timestep[j],
                "melt": melt_timestep[j],
                "snowfall_total": model_arrays["snowfall_total"][j],
                "melt_total": model_arrays["melt_total"][j],
                "residual_new_snow": model_arrays["residual_new_snow"][j],
                }

            # Compute SWE and other values for each HRU
            hru_values = compute_fsca_hru(fsca_model_input, cv)

            # Update model arrays and save output for the timestep
            for label, value in hru_values.items():
                if label in model_arrays:
                    model_arrays[label] = np.copy(model_arrays[label])
                    model_arrays[label][j] = value
                    output_arrays[label][i, j] = value

        # Prepare for the next timestep
        swe_start_timestep = swe_end_timestep

    # Convert output arrays to an xarray dataset

    return output_arrays



if __name__ == '__main__':
    # Define location of settings file
    settings_file = '/Users/drc858/GitHub/snow_dist/settings/config_5_param_model.yaml'
    # Create a model instance
    model_run = snow_5_param_simulation.Snow5ParamSimulation(settings_file)
    model_run.run()
