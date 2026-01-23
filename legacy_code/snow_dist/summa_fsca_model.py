"""

This script is used to run the SUMMA model, and calculate fractional Snow Cover Area.
pysumma is used to control the model simulation

Currently the summa model is run to completion, then the output is run through the SSNOWD model.

"""

# Import python packages
import xarray as xr
import numpy as np
from pathlib import Path
import pysumma as ps
from decorators import timer
from fsca_models import initialize_output_arrays, compute_fsca_hru
import data_io
import logging

logger = logging.getLogger(__name__)

def prepare_summa_fsca_input(summa_output_nc, fsca_input):
    """
    Prepare the input dataset for SUMMA's fractional Snow Covered Area (fSCA) computation.
    
    This function processes the output from a SUMMA simulation, and prepares it for the fSCA calculation. 
    The processed dataset includes snowpack input, snowpack output, and the snow water balance, among other parameters.
    
    Parameters:
    -----------
    summa_output_nc : list of str or str
        Paths to the output netCDF files from the SUMMA model run. 
        This can be a single file or a list of files.
        
    fsca_input : str
        Path where the prepared input for fSCA computation will be saved.
        
    Returns:
    --------
    None
        The function writes the prepared dataset to the provided output path and doesn't return any value.
    
    Notes:
    ------
    1. The function uses xarray to handle the netCDF dataset.
    2. Based on the presence of snow on the ground, certain variables are adjusted.
    3. Units for some variables are converted.
    4. The snowpack input and output are calculated based on various parameters from the SUMMA output.
    5. The final dataset is saved to the provided input path for subsequent fSCA computations.
    """

    logger.info(f'Preparing summa model input {summa_output_nc} for fsca simulation, output to {fsca_input}')

    # Read the netcdf files
    if len(summa_output_nc) == 1:
        summa_ds = xr.open_dataset(summa_output_nc[0])
    else:
        summa_ds = xr.open_mfdataset(summa_output_nc, combine='by_coords')

    fsca_ds = xr.Dataset(coords=summa_ds.coords)

    
    # Include variables when there is snow on the ground
    for var in ["scalarCanopyLiqDrainage_total", "scalarThroughfallRain_total"]:
        fsca_ds[var] = xr.where(summa_ds["nSnow"] != 0, summa_ds[var], 0)
    
    # Convert units
    for var in ["scalarSnowDrainage_total","scalarInfiltration_total"]:
        fsca_ds[var] = summa_ds[var] * 1000

    # Calculate snowpack input
    fsca_ds["snow_input"] = (
        summa_ds["scalarCanopySnowUnloading_total"]
        + summa_ds["scalarThroughfallSnow_total"]
        + fsca_ds["scalarCanopyLiqDrainage_total"]
        + fsca_ds["scalarThroughfallRain_total"]
        
    )

    # Calculate snowpack output
    fsca_ds["snow_output"] = (
        fsca_ds["scalarSnowDrainage_total"] 
        - summa_ds["scalarSnowSublimation_total"]
         + summa_ds["scalarSfcMeltPond"]
    )
    """
    for var in ["scalarCanopyLiqDrainage", "scalarThroughfallRain"]:
        fsca_ds[var] = xr.where(summa_ds["nSnow"] != 0, summa_ds[var], 0)
    
    # Convert units
    for var in ["scalarSnowDrainage","scalarInfiltration"]:
        fsca_ds[var] = summa_ds[var] * 1000

    # Calculate snowpack input
    fsca_ds["snow_input"] = (
        summa_ds["scalarCanopySnowUnloading"]
        + summa_ds["scalarThroughfallSnow"]
        + fsca_ds["scalarCanopyLiqDrainage"]
        + fsca_ds["scalarThroughfallRain"]
        
    )

    # Calculate snowpack output
    fsca_ds["snow_output"] = (
        fsca_ds["scalarSnowDrainage"] 
        - summa_ds["scalarSnowSublimation"]
         + summa_ds["scalarSfcMeltPond"]
    )
    """
    # Calculate snowpack balance
    fsca_ds["snow_water_balance"] = (
        fsca_ds["snow_input"].cumsum(dim='time') 
        - fsca_ds["snow_output"].cumsum(dim='time') 
        - summa_ds["scalarSWE"]
    )

    # Copy over additional variables
    for var in summa_ds.data_vars:
        if var not in fsca_ds.data_vars:
            fsca_ds[var] = summa_ds[var]

    # Resample all current variables to a daily timestep
    fsca_ds_daily = fsca_ds.resample(time='D').sum()

    # Copy additional needed variables
    fsca_ds_daily["scalarSWE"] = summa_ds["scalarSWE"].resample(time='D').mean()

    # Output to netcdf
    fsca_ds_daily.to_netcdf(fsca_input)
    logger.info(f'Prepared summa model input for fsca simulation, output to {fsca_input}')

    return


#@timer
def calc_summa_fsca(fsca_input, lumped_cv_value,output_file, fsca_model='ssnowd'):

    """
    Calculate the fractional Snow Covered Area (fSCA) for SUMMA.
    
    This function processes a given input dataset to compute the fSCA for each Hydrologic Response Unit (HRU) and timestep.
    The computed results are then saved into a netCDF output file.
    
    Parameters:
    -----------
    fsca_input : str
        Path to the input netCDF file that contains data required for fSCA computation.
        
    fsca_output : str
        Path to the output netCDF file where the computed fSCA values will be saved.
        
    lumped_cv_value : float
        Lumped CV value to be used for all HRUs.
        
    fsca_model : str, optional (default='ssnowd')
        Name of the fSCA model to be used. Default is 'ssnowd'.
        
    Returns:
    --------
    None
        The function writes the results to the output netCDF file and doesn't return any value.
    
    Notes:
    ------
    1. The function uses xarray to handle the netCDF dataset.
    2. It initializes various variables required for the fSCA computation, and iteratively processes each HRU and timestep.
    3. After processing, it prepares an output dataset and saves it to the provided output path.
    
    """

    logger.info(f'Calculating fSCA for {fsca_input} using {fsca_model}')

    summa_ds = xr.open_dataset(fsca_input)

    # Read number of hrus
    num_hrus = summa_ds["hru"].sizes["hru"]

    # Read the cv array
    cv_array = np.full(num_hrus, lumped_cv_value)

    # Read values from SUMMA run simulation
    timesteps = summa_ds["time"].values
    swe_array = summa_ds["scalarSWE"].values

    # Read model specific snowpack input values
    snowfall_array = summa_ds["snow_input"].values
    snowfall_total_array = summa_ds["snow_input"].cumsum(dim='time').values
    
    #Read model specific snowpack output values
    melt_array = summa_ds["snow_output"].values
    melt_total_array = summa_ds["snow_output"].cumsum(dim='time').values

    # Set initial values
    zeros_array = np.zeros_like(summa_ds["hru"])
    output_arrays = initialize_output_arrays(timesteps, zeros_array)

    # Run the model forward in time, through each time step
    for j, cv in enumerate(cv_array):
        # Intitialize accumulation variables

        snowfall_total = snowfall_total_array[0,j]
        melt_total = melt_total_array[0,j]
        fsca = 0 #Need to update for model re-start use case!!!
        residual_new_snow = 0 #Need to update for model re-start use case!!!

        for i, timestep in enumerate(timesteps):
        
            # Current state of the model for HRU 
            fsca_model_input = {
                "swe": swe_array[i,j],
                "snowfall": snowfall_array[i,j],
                "melt": melt_array[i,j],
                "snowfall_total": snowfall_total, 
                "melt_total": melt_total,
                'fsca': fsca,
                'residual_new_snow': residual_new_snow
                }
            
            # Run the model for the current timestep
            hru_values = compute_fsca_hru(fsca_model_input, cv,fsca_model=fsca_model)

            # Update accumulative variables
            snowfall_total = hru_values["snowfall_total"]
            melt_total = hru_values["melt_total"]
            fsca = hru_values["fsca"]
            # Used for linear inversion calculation
            residual_new_snow = hru_values["residual_new_snow"]

            for label, value in hru_values.items():
                    output_arrays[label][i, j] = value
    
    result_ds = data_io.convert_arrays_to_dataset(output_arrays, summa_ds)

    result_ds.to_netcdf(output_file)
    
    return result_ds

def read_summa_model_output(settings):
    """Find location and names of output files"""

    # Create simulation object
    sim = ps.Simulation(
        settings["summa"]["summa_exe"], settings["summa"]["file_manager"]
    )
    # Read location of output files
    output_txt_file = Path(
        sim.backup["manager"]["outputPath"].value, settings["summa"]["output_file_name"]
    )

    with open(output_txt_file) as f:
        output_files = f.read().splitlines()

    logger.info(f"Reading summa output files for fSCA simulation {output_files}")

    #output_ds = xr.open_mfdataset(output_files)

    return output_files

def save_output_file_location(sim, settings, no_snow_hru=False):
    """Save a text file with the summa model output"""

    """ Adapted from pysumma https://github.com/UW-Hydro/pysumma/blob/62664da408c046cf0d0bae83cef4626d93c70c7a/pysumma/simulation.py#L469"""

    if no_snow_hru == True:
        output_file_name = settings["summa_no_snow_hru"]["output_file_name"]
    else:
        output_file_name = settings["summa"]["output_file_name"]

    output_txt_file = Path(sim.backup["manager"]["outputPath"].value, output_file_name)

    """Find output files given the ``stdout`` generated from a run"""
    new_file_text = "Created output file:"
    out_files = []
    for l in sim.stdout.split("\n"):
        if new_file_text in l:
            out_files.append(l.split(";")[0].replace(new_file_text, "").strip())

    with open(output_txt_file, "w") as f:
        for out_file in out_files:
            f.write(f"{out_file}\n")

    logger.info(f"Text file recording output files written to to {output_txt_file}")
