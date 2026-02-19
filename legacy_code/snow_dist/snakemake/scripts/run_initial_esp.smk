
import os, sys
# Directory containing the files
sys.path.append('../scripts/')
import ss_utils
# Promote config keys
config = ss_utils.promote_keys(config)

# Read configuration settings
summa_output_path = Path(config['summa_output_path'])
summa_restart_path = Path(config['summa_restart_path'])
filter_str = config['restart_file_filter']
months = config['esp_months']

files_by_date = ss_utils.get_files_by_date(summa_restart_path, filter_str, months)
state_dates   = list(files_by_date.keys())
state_files   = list(files_by_date.values())

for state_date in state_dates:
    start_times, end_times = generate_times(state_date, start_year, stop_year, forecast_length)

run_suffixes ={}
output_paths = {}
for state_file, state_date in zip(state_files, state_dates):
    for start_time, end_time in zip(start_times, end_times):
        # Generate the run suffix
        start_year = pd.to_datetime(start_time).strftime("%Y")
        run_suffix = f"state{state_date}_year{start_year}"
        run_suffixes.append(run_suffix)
        # Generate the output path
        output_path = Path(config['summa_output_path'], f"{config['case_name']}_{run_suffix}")
        output_paths.append(output_path)

    
rule run_initial_esp:
    input: 
        expand(Path(output_path,f"{config['case_name']}_{config['run_suffix']}_timestep.nc"),
                output_path=output_paths)

rule prepare_esp_runs:
    output:
        summa_output = Path({output_path}, f"{config['case_name']}_{config['run_suffix']}_timestep.nc")
    params:
        summa_exe = config['summa_exe'],
        file_manager = Path(config['file_manager']),
        output_dir = {output_dir}
    run:
        sim = ps.Simulation(params.summa_exe, params.file_manager)

        sim_config = {
            'manager': {
                'simStartTime': {start_time},
                'simEndTime': {end_time},
                'initConditionFile': {state_file},
                'outputPath': {output_path}
            }
        }
        
        sim.apply_config(sim_config)

        sim.run(
            run_suffix=config['run_suffix'],
            write_config=True,
            write_manager=False,
        )
