# Read the configuration and resolve paths for simpler reading
import os, sys
from pathlib import Path
import numpy as np
import xarray as xr
import geopandas as gpd
import shutil

def resolve_paths(config):
    """Resolve paths to make clearer reading"""

    # Resolve summa settings file path
    # Each resulting path is the summa settings file path + the filename
    # E.g. config['file_manager'] = path/to/summa/settings/file_manager.txt

    # Set base outout directory
    config['base_output_dir'] = Path(config['base_settings']['base_dir'], config['base_settings']['case_name'])
    
    #Set watershed_tools output files
    for key in config["watershed_tools"]['wt_output_files']:
        config[key] = Path(
            config["watershed_tools"]["results_path"],
            config["watershed_tools"]['wt_output_files'][key],
        )
    # Set summa model files and paths
    config['summa_sample_settings_dir'] = Path(config['base_output_dir'], config['summa']['summa_paths']['sample_settings'])
    config['summa_settings_dir'] = Path(config['base_output_dir'], config['summa']['summa_paths']['settings'])
    config['summa_forcing_dir']  = Path(config['base_output_dir'], config['summa']['summa_paths']['forcing'])
    config['summa_output_dir']   = Path(config['base_output_dir'], config['summa']['summa_paths']['output'])
    
    for key in config["summa"]["summa_settings_files"]:
        config[key] = Path(
            config['summa_settings_dir'], config["summa"]["summa_settings_files"][key]
        )

       # Set easymore file and paths
    if config['easymore']['easymore_path'] == 'default':
        config['easymore_base_dir'] = Path(config['base_output_dir'], 'easymore')
    else:
        config['easymore_base_dir']       = Path(config['easymore']['easymore_path'])
    
    config['easymore_temp_dir']       = Path(config['easymore_base_dir'],'temp')
    config['easymore_intersect_dir']  = Path(config['easymore_base_dir'],'intersect')
    config['easymore_output_dir']     = Path(config['easymore_base_dir'],'output')
    config['easymore_remap_csv']      = Path(config['easymore_temp_dir'], f"{config['base_settings']['case_name']}_remapping.csv")
 
    return config

def get_num_hrus_from_shp(config, input_shp):
    """ Read the number of HRUs from the catchment shapefile """
    shp_hru_attr = config['watershed_tools']["discretized_shp_attr"]["hru_id"]

    gdf = gpd.read_file(input_shp)

    hru_ids = gdf[shp_hru_attr].values.astype(int)
    num_hru = len(hru_ids)

    return num_hru, hru_ids

def get_num_hrus_from_forcing(config):
    """Read the number of HRUs from the forcing file"""

    # Read all forcing files in the forcing directory
    _, _, forcing_files = next(os.walk(config["summa_forcing_dir"]))
    first_forcing_file = Path(config["summa_forcing_dir"], forcing_files[0])

    try:
        # Read the first forcing file
        forc = xr.open_dataset(first_forcing_file)
    except:
        sys.exit("Could not read forcing file: {}".format(first_forcing_file))

    # Get the sorting order from the forcing file
    forcing_hruIds = forc["hruId"].values.astype(
        int
    )  # 'hruId' is prescribed by SUMMA so this variable must exist

    # Number of HRUs
    num_hru = len(forcing_hruIds)

    forc.close()

    return num_hru, forcing_hruIds


def read_cwarhm_control_file(workflow_control_file):
    """Read complete control data from the CWARHM (https://github.com/CH-Earth/CWARHM)
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
    comment_char = "#"
    option_char = "|"
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


def move_files_to_main_directory(directory):
    """Move all files from subdirectories into the main directory"""
    directory = Path(directory)
    for subdir in directory.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.is_file():
                    shutil.move(str(file), str(directory))


def generate_gru_start_and_count(sim_size, chunk_size: int=None, num_chunks: int=None):
        '''
        Generate a list of gru start and stop times for parallel simulation.
        Original code: https://github.com/UW-Hydro/pysumma/blob/master/pysumma/distributed.py#L104
        '''
        assert not (chunk_size and num_chunks), \
            "Only specify at most one of `chunk_size` or `num_chunks`!"
        start, stop = 0, 0
        if not (chunk_size or num_chunks):
            chunk_size = 12
        if chunk_size:
            sim_truncated = (chunk_size-1) * (sim_size // (chunk_size-1))
            starts = np.arange(1, sim_truncated+1, chunk_size).astype(int)
            stops = np.append(starts[1:], sim_size+1)
            chunks = np.vstack([starts, stops]).T
        elif num_chunks:
            chunk_size = np.ceil(sim_size / num_chunks).astype(int)
            starts = np.arange(1, sim_size, chunk_size)
            stops = np.append(starts[1:], sim_size+1)
            chunks = np.vstack([starts, stops]).T
        
        gru_chunk_dict = [{'startGRU': start, 'countGRU': stop - start}
                for start, stop in chunks]
        
        gru_chunk_strings = ["G{:02d}-{:02d}".format(item['startGRU'], item['startGRU'] + item['countGRU'] - 1) for item in gru_chunk_dict]

        return gru_chunk_strings


def calc_num_grus(attribute_nc):
    """
    Calculate the number of GRUs in the SUMMA model.
    """
    with xr.open_dataset(attribute_nc) as ds:
        num_grus = ds['gruId'].shape[0]
    ds.close()
    return num_grus
    
def extract_gru_int(s):
    """Extract an integer value for the startGRU from the gru_chunk"""
    string_value = list(s)[0]   
    return int(string_value[1:string_value.index('-')])

def merge_netcdf_output(input_file_list, output_file):
    """Merge a list of netcdf files into a single file"""
    out_ds = [xr.open_dataset(input_file) for input_file in input_file_list]
    hru_vars = [] # variables that have hru dimension
    gru_vars = [] # variables that have gru dimension
    for name, var in out_ds[0].variables.items():
        if 'hru' in var.dims:
            hru_vars.append(name)
        elif 'gru' in var.dims:
            gru_vars.append(name)
    hru_ds = [ds[hru_vars] for ds in out_ds]
    gru_ds = [ds[gru_vars] for ds in out_ds]
    hru_merged = xr.concat(hru_ds, dim='hru')
    gru_merged = xr.concat(gru_ds, dim='gru')

    merged_ds = xr.merge([hru_merged, gru_merged])
    merged_ds.to_netcdf(output_file)

    return merged_ds
