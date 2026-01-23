"""

Configuration file reader

Scripting by Dave Casson, dave.casson@usask.ca
"""

# Package Import
import os
from pathlib import Path
import logging
import xarray as xr
import numpy as np
from datetime import datetime
from pprint import pformat

logger = logging.getLogger("log_to_file")

def resolve_paths(config,log_config = False):
    """Resolve paths from the configuration file"""

    # Log the settings if required
    if log_config:
        logger.info(f'Settings logged from {config}')
        config_to_log = pformat(config)
        logger.debug(f'Settings: {config_to_log}')

    promoted_config = promote_keys(config)
    config.update(promoted_config)

    config['settings_path'] = Path(config['settings_path']).resolve()
    config['forcing_path'] = Path(config['forcing_path']).resolve()
    config['output_path'] = Path(config['output_path']).resolve()
    config['fsca_path'] = Path(config['fsca_path']).resolve()

    config['file_manager'] = Path(config['settings_path'], config['file_manager'])
    config['fsca_input']   = Path(config['fsca_path'],'input',config['fsca_input_nc'])
    config['fsca_output']   = Path(config['fsca_path'],'output',config['fsca_output_nc'])

    for config_str in 'settings_path', 'forcing_path', 'output_path', 'fsca_path':
        config[config_str] = Path(config[config_str]).resolve()
        if not str(config[config_str]).endswith('/'):
            config[config_str] = str(config[config_str]) + '/'

    return config                  

def promote_keys(nested_dict):
    """Promote keys from a nested dictionary to the top level"""
    promoted_dict = {}

    for primary_key, secondary_dict in nested_dict.items():
        # Check that secondary_dict is a dictionary
        if isinstance(secondary_dict, dict):
            for secondary_key, value in secondary_dict.items():
                promoted_dict[secondary_key] = value

    return promoted_dict

def convert_arrays_to_dataset(output_arrays,forcing_ds):

    # Package all results in a xarray DataArray, and then add to the dataset
    timesteps = forcing_ds["time"].values

    for label, array in output_arrays.items():
        da = xr.DataArray(
            np.vstack(array),
            coords=[timesteps, forcing_ds["hru"]],
            dims=["time", "hru"],
            name=label,
        )
        forcing_ds[label] = da

    return forcing_ds

def get_output_files(sim):
    """Save a text file with the summa model output"""

    """ Adapted from pysumma https://github.com/UW-Hydro/pysumma/blob/62664da408c046cf0d0bae83cef4626d93c70c7a/pysumma/simulation.py#L469"""

    """Find output files given the ``stdout`` generated from a run"""
    new_file_text = "Created output file:"
    summa_output = []
    for l in sim.stdout.split("\n"):
        if new_file_text in l:
            summa_output.append(l.split(";")[0].replace(new_file_text, "").strip())

    return summa_output


def start_logging(settings):
    """This function will create a logs folder and file in the same directory as the main script
    It uses the logger_config.ini to set properties

    Parameters
    ----------
    file_name : str
        optional argument for name to be given to logfile

    Returns
    -------
    logger : logging object
        logger can be called with logger.debug("") etc to log to the console and output file

    """

    now = datetime.now()
    logging_path = Path(settings["logging"]["logging_dir"])
    log_file_name = f'{settings["logging"]["logging_file_name"]}_{now.strftime("%Y-%m-%d_%H:%M")}.log'

    logfile = Path(logging_path, log_file_name)

    logging.config.fileConfig(
        os.path.join(
            settings["logging"]["logging_config"],
        ),
        defaults={"logfilename": logfile},
        disable_existing_loggers=False,
    )
    logging.getLogger("matplotlib.font_manager").disabled = True
    logger = logging.getLogger("log_to_file")
    logger.info(f"Log File Generated: {logfile}")

