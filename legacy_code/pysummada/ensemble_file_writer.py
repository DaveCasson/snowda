
"""

This script produces configuration files for a SUMMA ensemble run.

It intended to be called as a class using the provided information below.

See further examples at the end of the script for usage.

Scripting by Dave Casson, dave.casson@usask.ca
"""

import pandas as pd
import numpy as np
import glob
import os
import datetime as dt
import shutil
import configparser
import logging
logger = logging.getLogger('fileOutput')

from collections import OrderedDict
from pathlib import Path

from .config_reader import ConfigReader

class EnsembleFileWriter(ConfigReader):

    def write_ensemble_files(self):
        """Function to write needed files for ensemble model runs"""
        logging.info('Writing files needed for ensemble run')
        self.create_output_directories()
        self.clear_ensemble_files()
        self.clear_state_files()
        self.write_ens_file_managers()
        self.write_forcing_file_list()
        self.build_ens_config()


    def create_output_directories(self):
        """Function to write needed output directories"""

        logging.info('Creating needed output directories')

        # Set output directories
        # Currently using only flat file directory
        self.file_manager_dir = os.path.dirname(self.settings_path)
        self.file_list_dir = os.path.dirname(self.settings_path)

        # Create required directories if they do not exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_result_output_dir).mkdir(parents=True, exist_ok=True)

    def write_ens_file_managers(self):

        '''Write File Managers to Run Ensemble Files'''
        logger.info('Writing Ensemble File Managers')

        # Read Example File Manager
        input_fm_df = pd.read_csv(self.input_file_manager, delim_whitespace=True, index_col=0, header=None)
        fm_list = []

        for num_ens in np.arange(1, self.num_ens + 1):
            # Set File Output Name and Path
            output_fm_name = 'fileManagerEns' + f"{num_ens:03d}" + '.txt'
            output_fm_path = os.path.join(self.file_manager_dir, output_fm_name)

            # Initialize Output from First File
            output_fm_df = input_fm_df
            # Modify FileManager forcing file list (i.e. forcingFileListEns001)
            output_fm_df.loc['forcingListFile', 1] = ''.join(
                ['\'','forcingFileListEns', f"{num_ens:03d}",'.txt','\''])
            # Modify FileManager outfilePrefix (i.e. 001)
            output_fm_df.loc['outFilePrefix', 1] = ''.join(['\'', f"{num_ens:03d}", '\''])

            # Output File
            output_fm_df.to_csv(output_fm_path, sep="\t", header=False)
            logger.debug(f'Writing File Manager: {output_fm_path}')

            fm_path = os.path.join(self.file_manager_dir, output_fm_name)
            fm_list.append(fm_path)

        self.ens_file_managers = fm_list

        return

    def write_forcing_file_list(self):
        '''Write Forcing File Lists'''
        logging.info('Writing Ensemble File Lists')
        year_months_array = OrderedDict(((self.start_time + dt.timedelta(_)).strftime(r"%Y%m"), None) for _ in range((self.end_time - self.start_time).days)).keys()

        for num_ens in np.arange(1, self.num_ens + 1):
            forcing_file_list = []
            for year_month_str in year_months_array:
                    forcing_file_list.append(self.forcing_file_base + year_month_str + '.' + f"{num_ens:03d}" + '.nc')

            output_file_list_name = ''.join(['forcingFileListEns', f"{num_ens:03d}",'.txt'])
            output_file_list_path = os.path.join(self.file_list_dir, output_file_list_name)
            forcing_file_list.sort()

            logger.debug(f'Writing Forcing File List: {output_file_list_path}')
            with open(output_file_list_path, "w") as fl_file:
                for line in forcing_file_list:
                    fl_file.write("".join(line) + "\n")  # works with any number of elements in a line

        return None

    def clear_ensemble_files(self):

        #Delete existing files in folders
        for folder in [self.file_manager_dir,self.file_list_dir]:

            logger.info(f'Deleting files in folder: {folder}')
            for filename in glob.glob(os.path.join(folder,'*Ens*')):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

        return None

    def clear_state_files(self):

        # Read input files
        input_fm = pd.read_csv(self.input_file_manager, delim_whitespace=True, index_col=0, header=None)

        # Modify FileManager
        state_folder = input_fm.loc['statePath', 1].replace("'", "")
        cold_state = input_fm.loc['initConditionFile', 1].replace("'", "")
        logger.info(f'Deleting state files in folder: {state_folder}')

        # Delete existing files in folders
        for filename in os.listdir(state_folder):
            file_path = os.path.join(state_folder, filename)
            if filename == cold_state:
                logger.debug('Cold State File Not Deleted: %s' % (filename))
            else:
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.info('Failed to delete %s. Reason: %s' % (file_path, e))

        return None

    def clear_output_files(self):

        # Read input files
        input_fm = pd.read_csv(self.input_file_manager, delim_whitespace=True, index_col=0, header=None)

        # Modify FileManager
        output_folder = input_fm.loc['outputPath', 1].replace("'", "")
        logger.info(f'Deleting all files in folder: {output_folder}')
        # Delete existing files in folders
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
               logger.info('Failed to delete %s. Reason: %s' % (file_path, e))

        return None


