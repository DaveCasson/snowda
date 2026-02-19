import os, sys
from pathlib import Path
from collections import defaultdict

# Directory containing the files
sys.path.append('../scripts/')
import ss_utils

# Promote config keys
config = ss_utils.promote_keys(config)

# Read configuration settings
summa_output_path = Path(config['summa_output_path'])
summa_restart_path = Path(config['summa_restart_path'])
filter_str = config['restart_file_filter']

# Get the files grouped by date
files_by_date = ss_utils.get_restart_files_by_date(summa_restart_path, filter_str)

# Create a list of dates (used as wildcards)
date_list = list(files_by_date.keys())
print(f"Date List: {date_list}")

rule all:
    input:
        expand(
            Path(summa_restart_path, f"{config['case_name']}_restart_{{date}}.nc"),
            date=date_list
        )

rule merge_restart_files_by_date:
    input:
        chunked_restart_files=lambda wildcards: files_by_date[wildcards.date]
    output:
        merged_output_file=Path(summa_restart_path, f"{config['case_name']}_restart_{{date}}.nc")
    wildcard_constraints:
        date=r"\d{10}"
    run:
        print(f"Merging restart files for date {wildcards.date}")
        print(f"Input files: {input.chunked_restart_files}")
        ss_utils.merge_netcdf_output(input.chunked_restart_files, output.merged_output_file)
        for file in input.chunked_restart_files:
            os.remove(file)
