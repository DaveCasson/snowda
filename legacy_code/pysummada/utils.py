"""
Utility functions

Scripting by Dave Casson, dave.casson@usask.ca
"""

#Relevent 
import pandas as pd
import os
from pathlib import Path
import logging
import logging.config
from datetime import datetime

logger = logging.getLogger('fileOutput')

def log_subprocess_output(pipe):
    '''Function to log the output from a subprocess call'''
    for line in iter(pipe.readline, b''):  # b'\n'-separated lines
        logger.info('SUMMA Logging: %r', line)


def get_SUMMA_workflow_control_setting(workflow_control_file, setting):
    """Read  line item  from the SUMMAworkflow (https://github.com/CH-Earth/summaWorkflow_public)
       format control file.

    Parameters
    ----------
    file : SUMMAworkflow control file path
        i.e./summaWorkflow_public/0_control_files/control_active.txt

    setting : line item of SUMMA workflow control file
        i.e. catchment_shp_name

    Returns
    -------
    substring : configuration setting from control file
        i.e. bow_distributed_elevation_zone.shp

    """

    # Open 'control_active.txt' and ...
    with open(workflow_control_file) as contents:
        for line in contents:
            # ... find the line with the requested setting
            if setting in line:
                break

    # Extract the setting's value
    substring = line.split('|', 1)[1]  # Remove the setting's name (split into 2 based on '|', keep only 2nd part)
    substring = substring.split('#', 1)[0]  # Remove comments, does nothing if no '#' is found
    substring = substring.strip()  # Remove leading and trailing whitespace, tabs, newlines

    # Return this value
    return substring


def read_SUMMA_workflow_control_file(workflow_control_file):
    """Read complete control data from the SUMMAworkflow (https://github.com/CH-Earth/summaWorkflow_public)
       format control file

    Parameters
    ----------
    workflow_control_file : SUMMAworkflow control file path
        i.e./summaWorkflow_public/0_control_files/control_active.txt

    Returns
    -------
    control_options --> dict
        dictionary containing all options from the control file

    """
    comment_char = '#'
    option_char = '|'
    control_options = {}
    f = open(workflow_control_file)
    for line in f:
        # First, remove comments:
        if comment_char in line:
            # split on comment char, keep only the part before
            line, comment = line.split(comment_char, 1)
        # Second, find lines with an option=value:
        if option_char in line:
            # split on option char:
            option, value = line.split(option_char, 1)
            # strip spaces:
            option = option.strip()
            value = value.strip()
            # store in dictionary:
            control_options[option] = value
    f.close()

    return control_options


class ConfigLogger(object):
    """ Function to log the reading of the configuration file """
    def __init__(self, log):
        self.__log = log
    def __call__(self, config):
        self.__log.info("Logging of configuration file:")
        config.write(self)
    def write(self, data):
        # stripping the data makes the output nicer and avoids empty lines
        line = data.strip()
        self.__log.info(line)

def isstr(s):
    # Returns the suffix if defined, otherwise ''
    return '' if s is None else str(s)

def output_netcdf(xarray_ds, output_nc):
    """ Function to compress and output netcdf file from xarray dataset"""
    encoding_dic = {}
    for key in xarray_ds.keys():
        encoding_dic[key] = {'zlib': True, 'complevel': 3}
    xarray_ds.to_netcdf(os.path.join(output_nc),
                        engine='netcdf4',
                        encoding=encoding_dic)

    return None

def calculate_ensemble_statistics(df):
    '''Function to calculate statistics from an input ensemble
    This will take the average across rows.
    '''

    df['mean'] = df.mean(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['0.75'] = df.quantile(q=0.75, axis=1)
    df['0.25'] = df.quantile(q=0.25, axis=1)

    return df

def start_logging(file_name='log_file'):

    '''This function will create a logs folder and file in the same directory as the main script
    It uses the logger_config.ini to set properties

    Parameters
    ----------
    file_name : str
        optional argument for name to be given to logfile

    Returns
    -------
    logger : logging object
        logger can be called with logger.debug("") etc to log to the console and output file

    '''

    now = datetime.now()
    log_file_name = f'{file_name}_{now.strftime("%Y-%m-%d_%H:%M:%S")}.log'
    working_folder = os.getcwd()

    Path(working_folder, 'logs').mkdir(parents=True, exist_ok=True)
    logfile = os.path.join(working_folder, 'logs', log_file_name)

    logging.config.fileConfig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../control_files/logger_config.ini'),
                              defaults={'logfilename': logfile}, disable_existing_loggers=False)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger('fileOutput')
    logger.info(f'Log File Generated: {logfile}')

    return logger

def code_providence(config_file):

    '''Save appropriate model run files for code providence'''

    logger.info(f'Model run configuration saved to {os.path.join(self.model_result_output_dir,"model_config.ini")}')
    shutil.copy(config_file,os.path.join(self.model_result_output_dir,'model_config.ini'))
