"""
Snakemake file to run the base SUMMA simulations.

The model simulation is chunked by GRU to allow for parallelization on a cluster.

The chunks of GRUs are defined by the user.
"""

from pathlib import Path
import sys
import pandas as pd
import pysumma as ps
import shutil

sys.path.append('../scripts/')
import ss_utils
import model_run_utils

# Promote config keys
config = ss_utils.promote_keys(config)

# Read configuration
summa_output_path = Path(config['summa_output_path'])
summa_restart_path = Path(config['summa_restart_path'])
summa_cold_state_name = config['summa_cold_state']
summa_cold_state_path = Path(config['summa_settings_path'], summa_cold_state_name)
summa_restart_path.mkdir(parents=True, exist_ok=True)

attributes_nc = Path(config['summa_settings_path'], config['attributes_nc'])
file_manager = Path(config['file_manager'])
summa_exe = Path(config['summa_exe'])
sim_start_time = config['model_start_time']
sim_end_time = config['model_end_time']
case_name = config['case_name']


# Read mizuroute configuration
mizuroute_exe = config['mizuroute_exe']
mizuroute_output_path = Path(config['mizuroute_output_path'])
control_file_base = Path(config['mizuroute_settings_path'], config['mizuroute_control_file'])
mizuroute_cold_state_name = config['mizuroute_cold_state']
mizuroute_cold_state_path = Path(config['mizuroute_settings_path'], config['mizuroute_cold_state'])
ancil_dir = config['mizuroute_settings_path']
mizuroute_restart_path = Path(config['mizuroute_restart_path'])
mizuroute_restart_path.mkdir(parents=True, exist_ok=True)

# Calculate the number of GRUs from the attributes_nc and generate chunk strings
num_grus = ss_utils.calc_num_grus(attributes_nc)
gru_chunks = ss_utils.generate_gru_start_and_count(
    num_grus, chunk_size=config['gru_chunk_size']
)

# Copy summa state file to restart directory
shutil.copy(summa_cold_state_path, Path(summa_restart_path, summa_cold_state_name))
shutil.copy(mizuroute_cold_state_path, Path(mizuroute_restart_path, mizuroute_cold_state_name))

# Run the base SUMMA simulation
rule run_pysumma_base:
    input:
        Path(summa_output_path, f"{config['case_name']}_{config['run_suffix']}_timestep.nc"),
        Path(mizuroute_output_path, f"{config['case_name']}_{config['run_suffix']}_routed.nc")

# Run SUMMA simulations for each GRU chunk
rule run_summa_in_gru_chunks:
    wildcard_constraints:
        gru_chunk='|'.join(gru_chunks)
    input:
        file_manager = file_manager
    output:
        summa_chunked_output = temp(Path(summa_output_path, f"{config['case_name']}_{config['run_suffix']}_{{gru_chunk}}_timestep.nc"))
    params:
        summa_exe = summa_exe,
        gru_start = lambda wildcards: ss_utils.extract_gru_int(wildcards.gru_chunk),
        gru_count = config['gru_chunk_size']
    log:
        log_file =  f"logs/summa_{config['case_name']}_{config['run_suffix']}_{{gru_chunk}}.log"
    run:
        sim = ps.Simulation(params.summa_exe, input.file_manager)

        sim_config = {
            'manager': {
                'simStartTime': sim_start_time,
                'simEndTime': sim_end_time,
                #'statePath': str(summa_restart_path) + '/',
                'outputPath': str(summa_output_path) + '/'
            }
        }
       
        # Set the configuration
        sim.apply_config(sim_config)

        # Set the last file name for the GRU chunking
        if params.gru_start + params.gru_count > num_grus:
            params.gru_count = num_grus - params.gru_start + 1

        sim.run(
            run_suffix=config['run_suffix'],
            startGRU=params.gru_start,
            countGRU=params.gru_count,
            write_config=False,
            write_manager=True,
            freq_restart='m'
        )

        print(f"SUMMA simulation for GRUs {params.gru_start} to {params.gru_start + params.gru_count - 1} finished with {sim.status}.")

# Merge SUMMA output files
rule merge_summa_output_files:
    input:
        chunked_output_files = expand(
            Path(summa_output_path, f"{config['case_name']}_{config['run_suffix']}_{{gru_chunk}}_timestep.nc"),
            gru_chunk=gru_chunks
        )
    output:
        merged_output_file = Path(summa_output_path, f"{config['case_name']}_{config['run_suffix']}_timestep.nc")
    run:
        print(f"Merging SUMMA output files into {output.merged_output_file}.")
        ss_utils.merge_netcdf_output(input.chunked_output_files, output.merged_output_file)
# Run MizuRoute
rule run_mizuroute:
    input:
        mizuroute_input = Path(summa_output_path, f"{config['case_name']}_{config['run_suffix']}_timestep.nc")
    output:
        mizuroute_output = Path(mizuroute_output_path, f"{config['case_name']}_{config['run_suffix']}_routed.nc")
    params:
        mizuroute_exe=mizuroute_exe,
        mizuroute_input_path=summa_output_path,
        mizuroute_output_path=mizuroute_output_path,
        control_file_base=control_file_base,
        case_name=case_name,
        ancil_dir=ancil_dir,
        sim_start_time=sim_start_time,
        sim_end_time=sim_end_time,
        output_path=mizuroute_output_path,
        restart_dir=mizuroute_restart_path
    run:
        #Set MizuRoute control file
        control_file_case = Path(params.mizuroute_output_path,f"{case_name}_mizuroute.control")
        #Set MizuRoute input file
        mizuroute_input_file = Path(input.mizuroute_input).name
        # Set timing
        start_time_pd = pd.to_datetime(params.sim_start_time)
        end_time_pd = pd.to_datetime(params.sim_end_time) - pd.Timedelta('1h')
        start_time_for_file_name=start_time_pd.strftime('%Y-%m-%d-%H')
        #Update control file
        model_run_utils.update_mizuroute_run_settings(
            input_control_file = params.control_file_base,
            output_control_file = control_file_case,
            input_summa_file = mizuroute_input_file,
            input_mizuroute_state = mizuroute_cold_state_name,
            case_name = params.case_name,
            sim_start = start_time_pd,
            sim_end = end_time_pd,
            ancil_dir = params.ancil_dir,
            input_dir = params.mizuroute_input_path,
            output_dir = params.mizuroute_output_path,
            restart_dir = params.restart_dir
        )

        #Run MizuRoute
        model_run_utils.run_mizuroute(params.mizuroute_exe, control_file_case)
        #Rename output file
        generated_file = Path(params.output_path,f"{params.case_name}.h.{start_time_for_file_name}000.nc")
        generated_file.rename(output.mizuroute_output)
