

"""

Particle Filter Class

Core components of Particle Filter Algorithm, including the prediction and update steps

Scripting by Dave Casson, dave.casson@usask.ca

"""


import os, glob, sys
import xarray as xr
import matplotlib.pyplot as plt
import datetime as dt
import multiprocessing as mp
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import subprocess
import shlex
import logging
logger = logging.getLogger('fileOutput')

import pysumma as ps

from . import utils
from .decorators import timer
from .config_reader import ConfigReader
from .ensemble_file_writer import EnsembleFileWriter
from .data_assimilation import DataAssimilation

class ParticleFilter(DataAssimilation):
    """Class for All Particle Filter Algorithms"""

    def __init__(self, write_ensemble_files=True):
        super().__init__()

    @timer
    def run_particle_filter(self):

        for start_time in self.run_period:

            end_time_current_run = start_time + self.timedelta
            logger.info(f'Starting runs for {start_time} to {end_time_current_run}. Running until {self.end_time}.')

            self.update_run_time(start_time, end_time_current_run)

            if not start_time == self.run_period[0]:
                self.update_warm_state(start_time)

            predict_ens = ps.Ensemble(self.summa_exe,self.ens_config,num_workers=5)

            predict_ens.run(run_option='local', run_suffix=end_time_current_run.strftime("%Y_%m_%d"),
                            freq_restart='d', write_config=False,update_manager=True)

            logging.info(f'Ensemble Run Summary: {predict_ens.summary()}')
            logging.info(predict_ens.simulations['001'].stdout)



    def update_particle_filter(self, particles,swe_measurements,z_var,end_date_current_run,ens_label_array,hru_index):
        """Update step of the Particle Filter

        This function updates the weights of each particle filter member based on
        the error compared to measurements
        """
        #Get values for specific date
        z_df = swe_measurements.loc[swe_measurements.index.get_level_values('time') == end_date_current_run]
        weights_dict = {}
        likelihoods = []

        #Loop through number of particles (columns)
        for i in range(len(particles.columns)):
            errors = []
            particle_by_hru = particles.iloc[:,i]

            for hru_id in hru_index['hruId']:
                meas = z_df.loc[hru_id].values
                particle = particle_by_hru.loc[hru_id].values
                #Calculate the mean of errors
                errors.append(abs(particle - meas))
            error_mean = np.nanmean(errors)
            likelihoods.append(np.exp(-0.5 * error_mean * z_var))

        #Calculate weights
        weights = likelihoods / np.sum(likelihoods)

        #Add date to weights
        weights_dict[end_date_current_run] = weights

        logger.info(f' Particle weights are calculated as {weights}')

        weights_df = pd.DataFrame.from_dict(weights_dict, orient='index')
        weights_df.columns = ens_label_array

        return weights_df

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

    def resample(self, summa_run_config,summa_run,weights,warm_states):
        """
        PF resampling, adapted from https://github.com/bertrandcz/CrocO_toolbox/blob/master/ParticleFilter.py

        TODO: Update to match SUMMA patterns
        TODO: Implement in actual filter
        """

        weights_np = weights.iloc[0].to_numpy()
        effweights = np.around(1. / (summa_run_config.num_ens * np.mean([w ** 2 for w in weights_np])))

        if effweights > float(summa_run_config.n_eff_threshold):

            resample = np.empty((summa_run_config.num_ens), dtype=int)
            zweightcumul = np.cumsum(weights_np)
            zrand = np.random.random() / summa_run_config.num_ens

            for ires in range(0, summa_run_config.num_ens):
                if zrand <= zweightcumul[0]:  #
                    resample[ires] = 1
                for rk in range(1, summa_run_config.num_ens):
                    if (zweightcumul[rk - 1] < zrand) and (zrand <= zweightcumul[rk]):
                        resample[ires] = rk + 1

                zrand += 1. / summa_run_config.num_ens

            resample_index = resample - 1

            logger.info(f"Resampling Index is {resample_index}")

            for i, index in enumerate(resample_index):
                # Update Warm Start if needed. This should not occur until the first run through of all ensembles.
                utils.update_fm_option(summa_run.fm_list[i], 'initConditionFile', '\'' + warm_states[index] + '\'')
                logger.debug(f"{summa_run.fm_list[i]} updated with warm state {warm_states[index]}")

        return None

    def efficient_weights(self,summa_run_config,weights):
        """
        PF resampling, adapted from https://github.com/bertrandcz/CrocO_toolbox/blob/master/ParticleFilter.py
        """

        weights_np = weights.iloc[0].to_numpy()
        effweights = np.around(1. / (summa_run_config.num_ens * np.mean([w ** 2 for w in weights_np])))

        return effweights

