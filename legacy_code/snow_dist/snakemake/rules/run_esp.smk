
import os
import sys
from pathlib import Path
import pandas as pd
import pysumma as ps
import logging

# Add the directory containing the helper scripts to the path
sys.path.append('../scripts/')
import ss_utils
import model_run_utils

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
my_logger = logging.getLogger(__name__)

# Promote config keys
config = ss_utils.promote_keys(config)

# Read in necessary configurations
summa_output_path = Path(config['summa_output_path'])
summa_restart_path = Path(config['summa_restart_path'])
summa_exe = config['summa_exe']
file_manager = Path(config['file_manager'])
filter_str = config['restart_file_filter']
months = config['esp_months']
start_year = int(config['esp_start_year'])
stop_year = int(config['esp_stop_year'])
forecast_length = int(config['forecast_length_days'])
case_name = config['case_name']


# Read in MizuRoute configurations
mizuroute_exe = config['mizuroute_exe']
mizuroute_output_path = Path(config['mizuroute_output_path'])
mizuroute_restart_path = Path(config['mizuroute_restart_path'])
control_file_base = Path(config['mizuroute_settings_path'], config['mizuroute_control_file'])
ancil_dir = config['mizuroute_settings_path']

# Get restart files by date
files_by_date = ss_utils.get_files_by_date(summa_restart_path, filter_str, months)
state_dates = list(files_by_date.keys())

# Generate list of start years
start_years = [str(year) for year in range(start_year, stop_year + 1)]
for state_date in state_dates:
    Path(summa_output_path, f"{case_name}/{state_date}").mkdir(parents=True, exist_ok=True)


rule run_esp:
    input:
        expand(f"{summa_output_path}/{case_name}/{{state_date}}/summa_{{start_year}}.nc",
                state_date=state_dates,
                start_year=start_years),
        expand(f"{mizuroute_output_path}/{case_name}/{{state_date}}/mizuroute_{{start_year}}.nc",
                state_date=state_dates,
                start_year=start_years)


# Helper function to calculate start_time
def calculate_start_time(wildcards):
    state_date = pd.to_datetime(wildcards.state_date, format='%Y%m%d%H')
    start_year = int(wildcards.start_year)
    # Handle potential invalid dates when replacing the year
    try:
        start_time = state_date.replace(year=start_year)
    except ValueError:
        # Adjust the day if necessary (e.g., for February 29 in non-leap years)
        max_day = pd.Timestamp(f"{start_year}-{state_date.month}-01").days_in_month
        day = min(state_date.day, max_day)
        start_time = state_date.replace(year=start_year, day=day)
    return start_time

# Helper function to calculate end_time
def calculate_end_time(wildcards):
    start_time = calculate_start_time(wildcards)
    end_time = start_time + pd.Timedelta(days=forecast_length)
    return end_time.strftime('%Y-%m-%d')

# Rule to run SUMMA ensemble
rule run_summa_esp:
    output:
        summa_initial_output = f"{summa_output_path}/{case_name}_state{{state_date}}_year{{start_year}}/{case_name}_state{{state_date}}_year{{start_year}}_timestep.nc",
        summa_output = f"{summa_output_path}/{case_name}/{{state_date}}/summa_{{start_year}}.nc"
    params:
        summa_exe=Path(summa_exe),
        file_manager=Path(file_manager),
        state_path = Path(summa_restart_path),
        state_date=lambda wildcards: pd.to_datetime(wildcards.state_date, format='%Y%m%d%H'),
        state_file=lambda wildcards: files_by_date[wildcards.state_date],
        start_year=lambda wildcards: int(wildcards.start_year),
        # Use helper function to calculate start_time
        start_time=lambda wildcards: calculate_start_time(wildcards).strftime("%Y-%m-%d"),
        # Use helper function to calculate end_time
        end_time=lambda wildcards: calculate_end_time(wildcards),
        output_path=lambda wildcards: Path(
            summa_output_path,
            f"{case_name}_state{wildcards.state_date}_year{wildcards.start_year}"
        )
    log:
        f"logs/summa_{{state_date}}_{{start_year}}.log"
    run:
        my_logger.info(f"Running SUMMA for state date: {wildcards.state_date}, year: {wildcards.start_year}")

        sim = ps.Simulation(params.summa_exe, params.file_manager)
        state = f"{case_name}_restart_{wildcards.state_date}.nc"
        my_logger.info(f"Using state file: {state}")

        sim_config = {
            'manager': {
                'simStartTime': params.start_time,
                'simEndTime': params.end_time,
                'statePath': str(params.state_path) + "/",
                'initConditionFile': state,
                'outputPath': str(params.output_path) + "/"
            }
        }

        sim.apply_config(sim_config)

        my_logger.info(f"Running SUMMA simulation with config: {sim_config}")
        sim.run(
            run_suffix=f"state{wildcards.state_date}_year{wildcards.start_year}",
            write_config=True,
            write_manager=False,
        )
        print(sim.status)

        # Shift time output, so ESP run starts at the state time.
        summa_output = f"{summa_output_path}/{case_name}_state{wildcards.state_date}_year{params.start_year}/{case_name}_state{wildcards.state_date}_year{params.start_year}_timestep.nc"
        summa_shifted_output = f"{summa_output_path}/{case_name}/{wildcards.state_date}/summa_{params.start_year}.nc"
        
        state_year_int = pd.to_datetime(wildcards.state_date, format='%Y%m%d%H').strftime('%Y')
        logger.info(f"Shifting SUMMA output time to {state_year_int}")
        
        model_run_utils.update_netcdf_time(summa_output, state_year_int, summa_shifted_output, file=True)
        

# Rule to run MizuRoute
rule run_mizuroute_esp:
    input:
        summa_output=f"{summa_output_path}/{case_name}/{{state_date}}/summa_{{start_year}}.nc"
    output:
        mizuroute_output = f"{mizuroute_output_path}/{case_name}/{{state_date}}/mizuroute_{{start_year}}.nc"
    log:
        f"logs/mizuroute_{{state_date}}_{{start_year}}.log"
    params:
        mizuroute_exe=mizuroute_exe,
        control_file_base=control_file_base,
        case_name=case_name,
        ancil_dir=ancil_dir,
        mizuroute_input_path=summa_output_path,
        mizuroute_output_path=mizuroute_output_path,
        restart_dir=mizuroute_restart_path,
        start_year=lambda wildcards: wildcards.state_date[:4],
        start_time=lambda wildcards: f"{pd.to_datetime(wildcards.state_date, format='%Y%m%d%H').strftime('%Y-%m-%d')}",
        end_time=lambda wildcards: (
            pd.to_datetime(f"{pd.to_datetime(wildcards.state_date, format='%Y%m%d%H').strftime('%Y-%m-%d')}") + pd.Timedelta(days=forecast_length)
            -pd.Timedelta(hours=1)).strftime("%Y-%m-%d"),
        output_path=lambda wildcards: Path(
            summa_output_path,
            f"{case_name}/{wildcards.state_date}"
        )
    run:
        my_logger.info(f"Running MizuRoute for state date: {wildcards.state_date}, year: {wildcards.start_year}")

        mizuroute_input_file = Path(
            f"summa_{wildcards.start_year}.nc"
        )

        mizuroute_state_file = Path(f"{case_name}.r.{params.start_time}-00000.nc")

        control_file_case = Path(
            f"{summa_output_path}/{case_name}_state{wildcards.state_date}_year{wildcards.start_year}",
            f"{case_name}_state{wildcards.state_date}_year{wildcards.start_year}.control"
        )
        start_time_pd = pd.to_datetime(params.start_time)
        end_time_pd = pd.to_datetime(params.end_time)

        my_logger.info(f"Updating MizuRoute control file: {control_file_case}")


        #Update control file
        model_run_utils.update_mizuroute_run_settings(
            input_control_file = params.control_file_base,
            output_control_file = control_file_case,
            input_summa_file = mizuroute_input_file,
            input_mizuroute_state = mizuroute_state_file,
            case_name = params.case_name,
            sim_start = start_time_pd,
            sim_end = end_time_pd,
            ancil_dir = params.ancil_dir,
            input_dir = Path(input.summa_output).parent,
            output_dir = Path(params.output_path),
            restart_dir = params.restart_dir
        )

        Path(params.output_path,f'{wildcards.start_year}').mkdir(parents=True, exist_ok=True)
        my_logger.info(f"Running MizuRoute with input: {mizuroute_input_file}")

        #Run MizuRoute
        model_run_utils.run_mizuroute(params.mizuroute_exe, control_file_case)

        # Renaming the MizuRoute output file to meet expected naming convention
        generated_file = Path(params.output_path, f"{params.case_name}.h.{params.start_time}-00000.nc")
        my_logger.info(f"Renaming file {generated_file} to {output.mizuroute_output}")
        generated_file.rename(output.mizuroute_output)

        #Path(params.output_path,f'{wildcards.start_year}').unlink()
