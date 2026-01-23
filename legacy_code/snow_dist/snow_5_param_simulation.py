"""

Simulation object to control model runs.

Credit to pysumma repository for motivation of code structure.
https://github.com/DaveCasson/pysumma/blob/master/pysumma/simulation.py

"""

# Import python packages
import yaml
from pathlib import Path
import logging
import argparse

import data_io
import coefficient_of_variation as cv
from snow_5_param_model import run_snow_5_param_model
import snow_dist_util as util


logger = logging.getLogger("log_to_file")


class Snow5ParamSimulation:

    """The simulation class provides a wrapper to control model runs

    Attributes
    ----------
    settings_file:
        yaml file containing model run settings
    """

    def __init__(self, settings_file, initialize=True):
        """Initialize a new simulation object"""

        # Read Configuration File
        self.read_settings_file(settings_file)

        if initialize:
            self.initialize()

    def read_settings_file(self, settings_file):
        """Read the settings from a yaml file"""
        with open(settings_file) as file:
            self.settings = yaml.safe_load(file)

        if self.settings["logging"]["logging_bool"] == True:
            util.start_logging(self.settings)

    def initialize(self):
        """Intialize run"""
        self.read_parameters()
        self.create_dirs()

    def read_parameters(self):
        """Read parameters for model run"""

        if self.settings["run_options"]["create_new_settings_files"] == True:
            cv.write_default_cv_to_file(self.settings)
            data_io.write_default_model_param_to_file(self.settings)

        self.cv_array = data_io.read_param_from_file(self.settings, "cv")
        self.model_param_array = data_io.read_param_from_file(self.settings, "snow_5")

    def create_dirs(self):
        """Create needed directories if not existing"""
        Path(self.settings["paths"]["model_run_path"]).mkdir(parents=True, exist_ok=True)
        Path(self.settings["paths"]["model_settings_path"]).mkdir(parents=True, exist_ok=True)

    def run(self):
        """Run 5 parameter model simulation"""

        # Read forcing_data
        forcing_ds = data_io.read_forcing_data_nc(
            self.settings["paths"]["forcing_data_path"],
            self.settings["filenames"]["forcing_data_pattern"],
        )
        
        logger.info(
            f'Running model, output to {self.settings["paths"]["model_run_path"]}/{self.settings["filenames"]["model_results_file"]}'
        )

        # Run simple example model
        model_results = run_snow_5_param_model(
            forcing_ds,
            self.model_param_array,
            self.cv_array,
            use_model_swe=self.settings["run_options"]["use_model_swe"],
        )

        result_ds = data_io.convert_arrays_to_dataset(model_results, forcing_ds)
        data_io.output_to_nc(self.settings, result_ds)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run snow 5 parameter model simulation')
    parser.add_argument('settings_file', type=str, nargs='?', default='../settings/config_5_param_model.yaml', help='path to the settings file')
    args = parser.parse_args()

    model_run = Snow5ParamSimulation(args.settings_file)

    model_run.run()
