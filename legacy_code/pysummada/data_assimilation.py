

"""

Particle Filter Class

Core components of Data Assimilation Initialization

Scripting by Dave Casson, dave.casson@usask.ca

"""


import os, glob, sys
import xarray as xr
import datetime as dt
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import subprocess
import shlex
from collections import ChainMap
import logging
logger = logging.getLogger('fileOutput')

import pysumma as ps

from . import utils
from .decorators import timer
from .config_reader import ConfigReader
from .ensemble_file_writer import EnsembleFileWriter
from .data_reader import DataReader

class DataAssimilation(DataReader):
    """Class for All Data Assimilation Algorithms"""

    def __init__(self, write_ensemble_files=True):
        super().__init__()

        if write_ensemble_files:
            # Write configuration files needed for model run
            logging.info('Creating ensemble run files')
            ensemble_file_writer = EnsembleFileWriter()
            ensemble_file_writer.write_ensemble_files()
            self.ens_file_managers = ensemble_file_writer.ens_file_managers


    def initialize_da_run(self):
        '''Perform initial tasks before beginning data assimilation run'''

        logging.info('Initializing data assimilation run by reading data and creating an ensemble dictionary')
        self.set_run_period()
        self.read_swe_data()
        self.create_ens_dict()

    def set_run_period(self):
        logging.info(f'Setting run period from {self.start_time} to {self.end_time}')
        self.run_period = pd.date_range(self.start_time, self.end_time - self.timedelta, freq='d')

    def create_ens_dict(self):
        """Build a dictionary that can be used to run a summa ensemble run

        Should be in the format:
        config = {
                'run0': {'file_manager': './data/coldeport/file_manager.txt'},
                'run1': {'file_manager': './data/reynolds/file_manager.txt'},
                 }
        """

        file_manager_dict = [{'file_manager': fm} for fm in self.ens_file_managers]
        config_zip = zip(self.ens_label_array, file_manager_dict)
        config_dict = dict(config_zip)

        self.ens_config = config_dict

        return

    def update_run_time(self, start_time, end_time):
        """Build a dictionary that can be used to run a summa ensemble run

        Should be in the format:
        config = {
                '001: {'manager': {'simStartTime' : '2010-01-01 00:00',
                                    'simEndTime'  : '2011-01-02 00:00'
                '002: {'manager': {'simStartTime' : '2010-01-01 00:00',
                    '               'simEndTime'  : '2011-01-02 00:00',
                ...
                 }
        """
        self.create_ens_dict()

        config_list = [{ens:{'manager': {'simStartTime': start_time.strftime("%Y-%m-%d %H:%M"),
                                         'simEndTime': end_time.strftime("%Y-%m-%d %H:%M")}}
                        for ens in self.ens_label_array}]

        config_dict = dict(ChainMap(*config_list))

        self.ens_config = self.merge_dict(self.ens_config, config_dict)

        return

    def update_warm_state(self,end_time):
        """Build a dictionary that can be used to run a summa ensemble run

        Should be in the format:
        config = {
                '001: {'manager': {'initialConditionFile' : '2010-01-01 00:00'}
                '002: {'manager': {'initialConditionFile' : '2010-01-01 00:00'}
                ...
                 }
        """

        logging.info(f'Updating warm states for end time: {end_time}')

        #Create list of the warm states used each timestep
        warm_states = self.create_warm_state_list(end_time)

        config_list = [{ens: {'manager': {'initConditionFile': warm_state}}
                        for ens, warm_state in zip(self.ens_label_array, warm_states)
                        }]

        config_dict = dict(ChainMap(*config_list))

        self.ens_config = self.merge_dict(self.ens_config, config_dict)

        return

    def merge_dict(self, a, b, path=None):
        """Merges two dictionaries together based on a common key (i.e. ensemble label)"""

        if path is None: path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    self.merge_dict(a[key], b[key], path + [str(key)])
                elif a[key] == b[key]:
                    pass  # same leaf value
                else:
                    raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
            else:
                a[key] = b[key]

        return a

    def create_warm_state_list(self,end_time):
        """Create list of warm states with the format: 001_restart_2010010200_2010_01_02.nc"""

        run_suffix_str1 = end_time.strftime("%Y%m%d")
        run_suffix_str2 = end_time.strftime("%Y_%m_%d")

        warm_states = [''.join([run_prefix, '_restart_',run_suffix_str1,'00_',run_suffix_str2,'.nc'])
                           for run_prefix in self.ens_label_array]

        return warm_states

    @timer
    def predict(self, summa_run_config,summa_arg):

        """Prediction step of particle filter, meaning running of the model

        Parameters
        ----------
        summa_exe : str
            The location of the summa executable
        summa_arg : dict
            Dictonary of the summa arguments

        Returns
        -------
        None

        TODO: None
        """
        # Create command argument beginning with arguments
        summa_run_cmd = []
        summa_run_cmd.append(summa_run_config.summa_exe)

        #Add all arguments
        for key, value in summa_arg.items():
            summa_run_cmd.extend([key,value])

        #Run SUMMA model with command and capture output
        process = subprocess.Popen(summa_run_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        if summa_run_config.summa_log_bool==True:
            with process.stdout:
                utils.log_subprocess_output(process.stdout)

        exitcode = process.wait()  # 0 means success

        #Create string for output
        model_run_cmd_str = ' '.join(map(str, summa_run_cmd))

        #Log success or failure of run
        if exitcode == 0:
            logger.info('test')
            logger.info(f'Successful run: {model_run_cmd_str}')
        else:
            logger.warning(f'Failed run: {model_run_cmd_str}')

        return None

    @timer
    def state_estimate(self, particles, weights, end_date_current_run, hru_index):

        """Returns mean and variance of the weighted particles

        Parameters
        ----------
        particles : array
            Array of state estimates for each ensemble member
        weights : array
            Weights calculated for each ensemble member based on likelihoods

        Returns
        -------
        mean: DataFrame
            State mean
        var: DataFrame
            State Variance

        TODO: None
        """

        mean = {}
        var = {}

        for hru_id in hru_index['hruId']:
            mean[hru_id] = np.average(particles.loc[hru_id].values[0], weights=weights.values[0], axis=0)
            var[hru_id] = np.average((particles.loc[hru_id].values[0] - mean[hru_id]) ** 2, weights=weights.values[0], axis=0)

        mean_df = pd.DataFrame.from_dict(mean,orient='index')
        var_df = pd.DataFrame.from_dict(var,orient='index')

        mean_df["time"] = end_date_current_run
        mean_df.reset_index(inplace=True)
        mean_df.set_index(['index', 'time'], inplace=True)
        mean_df.index.names = ['hru', 'time']
        mean_df.columns = ['mean_snw']

        var_df["time"] = end_date_current_run
        var_df.reset_index(inplace=True)
        var_df.set_index(['index', 'time'], inplace=True)
        var_df.index.names = ['hru', 'time']
        var_df.columns = ['var_snw']

        return mean_df, var_df

