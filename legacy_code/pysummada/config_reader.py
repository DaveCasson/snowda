"""

configuration file reader

This script reads configuration from external files
to set key properties for model run.

This will start by reading all required data from the local config file.

There is also the option to use the settings in the control file of
summaWorkflow_public (https://github.com/CH-Earth/summaWorkflow_public)

Scripting by Dave Casson, dave.casson@usask.ca
"""

#Package Import
import os
import datetime as dt
import shutil
import configparser
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
import logging
import argparse

logger = logging.getLogger('fileOutput')

import pysummada.utils as utils

class ConfigReader():
    """
    The config object is used to hold run time information, specified in a configuration file
    """
    def __init__(self, logging=True):

        config_file = self.read_config_ini_file()

        if config_file is None:
            raise ValueError("No configuration file is specified, this must be specified.")

        if logging:
            logger = utils.start_logging()

        # Read Configuration File
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(config_file)

        # Log the contents of the configuration file
        logger.info(f'The run configuration settings have been read from {config_file}')
        config_logger = utils.ConfigLogger(logger)
        config_logger(config)

        #Read all the configuration file elements into a dictionary
        config_interim_dict = {sect: dict(config.items(sect)) for sect in config.sections()}
        #Remove the top level dict items, leaving only the key and value pairs
        self.option = dict(ele for sub in config_interim_dict.values() for ele in sub.items())

        # Read Path
        self.repo_base = config['Paths']['repo_base']
        self.summa_exe = config['Paths']['summa_exe']
        self.settings_path = config['Paths']['settings_path']
        self.output_dir = config['Paths']['output_dir']

        # Read Input Files
        self.input_file_manager = config['InputFiles']['base_file_manager']
        self.input_swe_hru_nc = config['InputFiles']['input_swe_hru_nc']
        self.initial_state = config['InputFiles']['initial_state']

        # Read Model Run Information
        self.model_result_output_dir = os.path.join(self.output_dir,config['ModelRun']['model_run_id'])
        self.forcing_file_base = config['ModelRun']['forcing_file_base']
        self.job_list_filename = config['ModelRun']['job_list_filename']
        self.num_ens = int(config['ModelRun']['num_ens'])
        self.start_time_txt = config['ModelRun']['start_time']
        self.end_time_txt = config['ModelRun']['end_time']
        self.start_time = dt.datetime.strptime(self.start_time_txt, '%Y-%m-%d %H:%M')
        self.end_time = dt.datetime.strptime(self.end_time_txt, '%Y-%m-%d %H:%M')
        self.time_delta_txt = config['ModelRun']['time_delta']
        self.timedelta = dt.timedelta(days=int(self.time_delta_txt))
        self.summa_log_bool = config['ModelRun']['summa_log_bool']
        self.ens_label_array = [f'{(num_ens+1):03d}' for num_ens in range(self.num_ens)]

        #Read Data Assimilation Information
        self.pf_resample = config['DA']['pf_resample']
        self.n_eff_threshold = config['PF']['n_eff_threshold']

        # Read SUMMA Arguments to Dictionary
        self.summa_arg = dict(config.items('SUMMAArg'))

    def read_config_ini_file(self):

        """Read configuration as command line argument"""
        parser = argparse.ArgumentParser(description='Run SUMMA')
        parser.add_argument("-c", "--config_file", help="Configuration file used", default='run_config.ini')
        parser.add_argument("-p", "--config_file_path", help="Path to configuration file",
                            default=str(os.path.join(Path(__name__).parent.absolute(), 'control_files')))
        args = parser.parse_args()
        logger.info(f'Runtime arguments used: {args}')

        """Read configuration file"""
        config_file = os.path.join(args.config_file_path, args.config_file)
        logger.info(f'Reading configuration file from: {config_file}')

        return config_file


    def build_ens_config(self):
        """Build a dictionary that can be used to run a summa ensemble run"""
        file_manager_dict  = {'file_manager': fm for fm in self.ens_file_managers}
        config_zip = zip(self.ens_label_array, file_manager_dict)
        config_dict = dict(config_zip)

        self.ens_config = config_dict

        return