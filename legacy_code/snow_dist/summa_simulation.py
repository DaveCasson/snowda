"""

Simulation object to control model runs.

Credit to pysumma repository for motivation of code structure.
https://github.com/DaveCasson/pysumma/blob/master/pysumma/simulation.py

"""

# Import python packages

import yaml
import sys
import xarray as xr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger("log_to_file")


import summa_fsca_model
import snow_dist_util as util
import data_io

class SummaSimulation:

    """The simulation class provides a wrapper to control model runs

    Attributes
    ----------
    settings_file:
        yaml file containing model run settings
    """

    def __init__(self, settings_file):
        """Initialize a new simulation object"""

        # Read Configuration File
        self.read_settings_file(settings_file)
        logger.info(f"Starting simulation with settings file {settings_file}")

    def read_settings_file(self, settings_file):
        """Read the settings from a yaml file"""
        with open(settings_file) as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)

        self.settings['fsca_input'] = Path(self.settings['summa']['summa_output_path'], self.settings['summa']['summa_fsca_filename'])

        #if self.settings["logging"]["logging_bool"] == True:
        #    util.start_logging(self.settings)

        self.read_parameters()

    def read_parameters(self):
        """Read parameters for model run"""
        if self.settings["cv_parameters"]["read_from_file"] == True:
            self.cv_array = data_io.read_param_from_file(self.settings, "cv")
        else:
            # Read in array shape
            hrus = data_io.get_model_hrus(self.settings)

            # Set to lumped parameter
            self.cv_array = np.full(
                np.shape(hrus), self.settings["cv_parameters"]["lumped_settings"]["cv"]
            )

    def run_summa_model(self):
        """Run summa model simulation"""

        summa_fsca_model.run_summa_model(self.settings)

        if self.settings["run_options"]["run_no_snow_hru"] == True:
            summa_fsca_model.run_summa_model(self.settings, no_snow_hru=True)

    def prepare_summa_model_output(self):
        """Prepare summa model output for fsca calculations"""

        summa_output_nc = '/Users/dcasson/Data/yukon_esp/summa_output/Chena_base_timestep.nc'#summa_fsca_model.read_summa_model_output(self.settings)
        
        summa_fsca_model.prepare_summa_fsca_input(summa_output_nc, self.settings['fsca_input'])

        return

    def run_summa_fsca_model(self):
        """Run summa fsca model"""

        model_results = summa_fsca_model.calc_summa_fsca(
            self.settings['fsca_input'], self.cv_array, self.output_file
        )
        result_ds = data_io.convert_arrays_to_dataset(model_results, xr.open_dataset(self.settings['fsca_input']))
       
        fsca_output_path = Path(self.settings['paths']['model_run_path'], self.settings['summa']['summa_fsca_filename'])
        
        result_ds.to_netcdf(fsca_output_path)

if __name__ == "__main__":
    settings_file = (
        "/Users/dcasson/GitHub/snow_dist/settings/config_summa_model_chena.yaml"
    )
    model_run = SummaSimulation(settings_file)
    model_run.read_parameters()
    model_run.settings['fsca_model'] = 'ssnowd'
    model_run.output_file = '/Users/dcasson/Data/yukon_esp/summa_output/Chena_base_timestep.nc'
    model_run.prepare_summa_model_output()
    model_run.run_summa_fsca_model()

    """

    cv_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9]
    output_files = []
    for cv_value in cv_values:
        model_run.settings["cv_parameters"]["lumped_settings"]["cv"] = cv_value
        model_run.settings["summa"][
            "summa_fsca_filename"
        ] = f"summa_fsca_result_{cv_value}.nc"
        output_files.append(
            os.path.join(
                model_run.settings["summa"]["summa_output_path"],
                model_run.settings["summa"]["summa_fsca_filename"],
            )
        )
        model_run.run_summa_fsca_model()

    fig, axs = plt.subplots(nrows=4, figsize=[10, 12])
    xticks = np.arange(0, 365, 30)

    hru_id = 0

    for output_file in output_files:
        result = xr.open_dataset(output_file)

        result["fsca"].isel(hru=hru_id).plot(ax=axs[0])
        axs[0].get_xaxis().set_visible(False)
        axs[0].set_title("fSCA ")

        result["snowfall_total"].isel(hru=hru_id).plot(ax=axs[1], color="b")
        #result["swe_total_melt_flux"].isel(hru=hru_id).plot(ax=axs[1], color="r")
        axs[1].get_xaxis().set_visible(False)
        axs[1].set_title("Snowfall and Snowmelt Accumulation")

        axs[2].set_title("Snowpack SWE, SUMMA (black) and SSNOWD calculated (red)")
        result["scalarSWE"].isel(hru=hru_id).plot(ax=axs[2], color="k")
        #result["swe_calc_mean"].isel(hru=hru_id).plot(ax=axs[2])
        #result["swe_total_acc"].isel(hru=hru_id).plot(ax=axs[2], color="b")

        # result['scalarSnowfall_total'].isel(hru=hru_id).plot(ax=axs[3], color='b')
        # ax_3 = axs[3].twinx()
        # result['scalarSnowDrainage_total'].isel(hru=hru_id).plot(ax=ax_3, color='r')

        result.close()
    plt.show()
    """
