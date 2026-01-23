"""
Plotting

Scripting by Dave Casson, dave.casson@usask.ca

"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import xarray as xr

def plot_5_param_snow_model_results(result_ds):
    """Plot the model results for review"""

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))
    result_ds["swe_5_param"].plot.line(ax=ax[0], x="time", color="k", add_legend=False)
    result_ds["swe_calc"].plot.line(
        ax=ax[0], x="time", color="r", linestyle="dashed", add_legend=False
    )
    ax[0].set_ylabel("SWE")
    ax[0].set_xlabel("Time")
    ax[0].set_title("SWE calculated from 5 parameter model (black) and SSNOWD (red)")

    result_ds["swe_frac_area"].plot.line(
        ax=ax[1], x="time", color="k", linestyle="dashed", add_legend=False
    )
    ax[1].set_ylabel("Snow Fractional Area")
    ax[1].set_xlabel("Time")
    ax[1].set_title("Fractional Snow Cover Area")
    plt.tight_layout()
    plt.show()
    
    return

def plot_summa_results(result_ds, fsca= False):
    """Plot the model results for review"""

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    result_ds["airtemp"].plot.line(ax=ax[0], x="time", color="r", add_legend=False)
    ax[0].grid(True)
    ax[0].set_ylabel("Temperature (degC)", fontsize=14)
    ax[0].set_xlabel("Time", fontsize=14)
    ax[0] = ax[0].twinx()
    
    result_ds["pptrate"].plot.line(ax=ax[0], x="time", color="b", add_legend=False)
    ax[0].set_ylabel("Precipitation (mm)", fontsize=14)
    ax[0].set_xlabel("Time", fontsize=14)
    plt.gca().invert_yaxis()
    ax[0].set_title("Meteorological Forcing Data", fontsize=16)

    if fsca == True:
        plot_percentile_range(result_ds["swe_frac_area"], ax[1])
    plot_percentile_range(result_ds["scalarSWE"],ax[1])

    plt.show()

    return

def plot_summa_fsca_results_test(result_ds):
    """Plot the model results for review"""

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    plot_percentile_range(result_ds["scalarSWE"],ax[0])
    plot_percentile_range(result_ds["swe_frac_area"], ax[1])
    

    plt.show()

    return

def plot_percentile_range(dataarray, ax):
    """Plot the percentile range across the 'hru' dimension"""

    # Calculate the percentile ranges across the 'hru' dimension
    p10 = np.percentile(dataarray, 10, axis=dataarray.dims.index('hru'))
    p90 = np.percentile(dataarray, 90, axis=dataarray.dims.index('hru'))
    pmin = np.min(dataarray, axis=dataarray.dims.index('hru'))
    pmax = np.max(dataarray, axis=dataarray.dims.index('hru'))

    # Plot the percentile range
    ax.fill_between(dataarray.time.values, p10, p90, alpha=0.5, color='blue', label='10-90 percentile range')
    ax.fill_between(dataarray.time.values, pmin, pmax, alpha=0.2, color='lightblue', label='min-max range')
    ax.plot(dataarray.time.values, dataarray.mean(dim='hru'), color='darkblue', label='mean')
    ax.set_xlabel('Time')
    ax.set_ylabel(dataarray.name)
    ax.set_title(f'{dataarray.name} Percentile Range')
    ax.legend()

    return

def plot_summa_fsca_results(result_ds):
    """Plot the model results for review"""

    fig, ax = plt.subplots(nrows=3, ncols=1)
    result_ds["airtemp"].plot.line(ax=ax[0], x="time", color="r", add_legend=False)

    ax[0].set_ylabel("Temperature (degC)")
    ax[0].set_xlabel("Time")
    ax[0] = ax[0].twinx()
    result_ds["pptrate"].plot.line(ax=ax[0], x="time", color="b", add_legend=False)
    ax[0].set_ylabel("Precipitation (mm)")
    ax[0].set_xlabel("Time")
    plt.gca().invert_yaxis()
    ax[0].set_title("Meterological Forcing Data")

    result_ds["scalarSWE"].plot.line(ax=ax[1], x="time", color="k", add_legend=False)
    result_ds["swe_calc_mean"].plot.line(
        ax=ax[1], x="time", color="r", linestyle="dashed", add_legend=False
    )
    ax[1].set_ylabel("SWE")
    ax[1].set_xlabel("Time")
    ax[1].set_title("SWE calculated from summa model (black) and SSNOWD (red)")

    result_ds["swe_frac_area"].plot.line(
        ax=ax[2], x="time", color="k", linestyle="dashed", add_legend=False
    )
    ax[2].set_ylabel("Snow Fractional Area")
    ax[2].set_xlabel("Time")
    ax[2].set_title("Fractional Snow Cover Area")
    plt.show()

    return

def plot_simple_fsca_results(output_files, cv_values, plot_output_file):
    """Plot the model results for review"""
    hru_id = 0

    _, axs = plt.subplots(nrows=2, figsize=[10, 12])

    for output_file, cv_value in zip(output_files,cv_values):

        result_ds = xr.open_dataset(output_file)

        axs[0].plot(result_ds['fsca'].isel(hru=hru_id),label=f'CV:{cv_value}')
        
        axs[1].plot(result_ds['snowfall_total'].isel(hru=hru_id),label='Total Accumulated Snowfall')
        axs[1].plot(result_ds['melt_total'].isel(hru=hru_id),linestyle='dashdot',label='Total Accumulated Melt')
        
        result_ds.close()
    axs0 = axs[0].twinx()
    axs0.plot(result_ds['scalarSWE'].isel(hru=hru_id),color='k',label='SWE')
    axs0.set_ylabel('SWE')
    axs0.set_ylim(0)
    axs0.legend(loc='upper right')

    axs[0].set_ylabel('fSCA')
    axs[0].get_xaxis().set_visible(False)
    axs[1].get_xaxis().set_visible(False)
    axs[0].set_title('fSCA')
    axs[0].legend()
    axs[0].set_ylim(0, 1.1)
    axs[1].set_title('Snowmelt Accumulation (Blue), Snowmelt Accumulation (Green), Snowmelt Flux Red')
    plt.savefig(plot_output_file)

    return

def plot_snow_variables(data):
    """Plot the snow variables for review"""

    # Calculate the mean and range of fsca, snowfall, snowfall_total, and melt_total across all HRUs
    fsca_mean = data.fsca.mean(dim='hru')
    fsca_min = data.fsca.min(dim='hru')
    fsca_max = data.fsca.max(dim='hru')

    snowfall_mean = data.snowfall.mean(dim='hru')
    snowfall_min = data.snowfall.min(dim='hru')
    snowfall_max = data.snowfall.max(dim='hru')

    snowfall_total_mean = data.snowfall_total.mean(dim='hru')
    snowfall_total_min = data.snowfall_total.min(dim='hru')
    snowfall_total_max = data.snowfall_total.max(dim='hru')

    melt_total_mean = data.melt_total.mean(dim='hru')
    melt_total_min = data.melt_total.min(dim='hru')
    melt_total_max = data.melt_total.max(dim='hru')

    # Create a figure with 3 subplots
    fig, ax = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Plot fsca
    ax[0].plot(data.time, fsca_mean, color='blue', label='Mean fsca')
    ax[0].fill_between(data.time.values, fsca_min, fsca_max, color='blue', alpha=0.3, label='Range of fsca across HRUs')
    ax[0].set_title('Fractional Snow-Covered Area (fsca) over Time', fontsize=16)
    ax[0].set_ylabel('fsca', fontsize=14)
    ax[0].legend()
    ax[0].grid(True)

    # Plot snowfall
    ax[1].plot(data.time, snowfall_mean, color='green', label='Mean snowfall')
    ax[1].fill_between(data.time.values, snowfall_min, snowfall_max, color='green', alpha=0.3, label='Range of snowfall across HRUs')
    ax[1].set_title('Snowfall over Time', fontsize=16)
    ax[1].set_ylabel('Snowfall', fontsize=14)
    ax[1].legend()
    ax[1].grid(True)

    # Plot snowfall_total and melt_total
    ax[2].plot(data.time, snowfall_total_mean, color='purple', label='Mean snowfall_total')
    ax[2].fill_between(data.time.values, snowfall_total_min, snowfall_total_max, color='purple', alpha=0.3, label='Range of snowfall_total across HRUs')
    ax[2].plot(data.time, melt_total_mean, color='red', label='Mean melt_total')
    ax[2].fill_between(data.time.values, melt_total_min, melt_total_max, color='red', alpha=0.3, label='Range of melt_total across HRUs')
    ax[2].set_title('Snowfall Total and Melt Total over Time', fontsize=16)
    ax[2].set_xlabel('Time', fontsize=14)
    ax[2].set_ylabel('Snowfall Total / Melt Total', fontsize=14)
    ax[2].legend()
    ax[2].grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def calculate_cumsum(input_ds, var, multiplier=1,no_snow_flux=False):
    
    if no_snow_flux == True:
        input_ds[f'{var}_corrected'] = xr.where(input_ds['nSnow'] != 0, input_ds[var], 0)
        input_ds[f'{var}_sum'] = input_ds[f'{var}_corrected'].cumsum(dim='time') * multiplier
        print(f'{var} sum calculated with only when nSnow !0')
    else:
        input_ds[f'{var}_sum'] = input_ds[var].cumsum(dim='time') * multiplier
    
    return input_ds

def plot_snow_balance(input_nc, hru=0):

    input_ds = xr.open_dataset(input_nc)

    flux_list            = {
                            #Precipitation Flux
                        'scalarSnowfall_total',
                        'scalarRainfall_total',
                        
                            # Canopy Flux
                        'scalarThroughfallSnow_total',
                        'scalarThroughfallRain_total',
                        'scalarCanopySnowUnloading_total',
                        'scalarCanopyLiqDrainage_total',
                        'scalarCanopyEvaporation_total',
                        'scalarCanopySublimation_total',
                        
                        # Snow Flux
                        'scalarSnowSublimation_total',
                        'scalarSnowDrainage_total',
                        #'scalarSfcMeltPond',
                        
                        #Soil surface flux
                        'scalarGroundEvaporation_total',
                        'scalarInfiltration_total',

                        # Snowpack input and output
                        'snow_input',
                        'snow_output',
                        }

    for var in flux_list:
            input_ds = calculate_cumsum(input_ds, var)

    fig, ax = plt.subplots(nrows=3,ncols=1, figsize=[10,12])
    ax[0].plot(input_ds.time,input_ds['scalarSWE'].isel(hru=hru),color='black',linestyle='solid', label='SUMMA SWE')
    ax[0].set_ylabel('Snow Water Equivalent [mm]')
    
    ax0 = ax[0].twinx()
    ax0.plot(input_ds.time,input_ds['scalarGroundSnowFraction'].isel(hru=hru),color='gray',linestyle='dotted', label='SUMMA fSCA')
    ax0.set_ylabel('fSCA')
    ax[0].set_title('Snowpack')
    ax[0].legend(loc='upper left')
    ax0.legend(loc='upper right')

    ax[1].plot(input_ds.time, input_ds['scalarThroughfallSnow_total_sum'].isel(hru=hru),color='green',linestyle='--', label='Throughfall Snow')
    ax[1].plot(input_ds.time,input_ds['scalarThroughfallRain_total_sum'].isel(hru=hru),color='green',linestyle=':', label='Throughfall Rain')
    ax[1].plot(input_ds.time,input_ds['scalarCanopySnowUnloading_total_sum'].isel(hru=hru),color='green',linestyle='dotted', label='Canopy Snow Unloading')
    ax[1].plot(input_ds.time,input_ds['scalarCanopyLiqDrainage_total_sum'].isel(hru=hru),color='green',linestyle='dashdot', label='Canopy Liquid Drainage')
    ax[1].plot(input_ds.time,input_ds['snow_input_sum'].isel(hru=hru),color='darkgreen',linestyle='-', label='Total Snowpack Input')

    ax[1].plot(input_ds.time,input_ds['scalarSnowDrainage_total_sum'].isel(hru=hru),color='red',linestyle='--', label='Snow Drainage')
    ax[1].plot(input_ds.time,input_ds['scalarSnowSublimation_total_sum'].isel(hru=hru),color='red',linestyle='dotted', label='Snow Sublimation')
    ax[1].plot(input_ds.time,input_ds['scalarSfcMeltPond'].isel(hru=hru),color='red',linestyle='dashdot', label='Surface Ponded Water')
    ax[1].plot(input_ds.time,input_ds['snow_output_sum'].isel(hru=hru),color='darkred',linestyle='--', label='Total Snowpack Output')

    ax[1].set_title('Total Snowpack Inputs and Outputs')
    ax[1].set_label('kg m-2')
    ax[1].legend()

    ax[2].set_title('Snowpack Water Balance')
    ax[2].plot(input_ds.time,input_ds['snow_water_balance'].isel(hru=hru),color='black',linestyle='-', label='Snowpack Water Balance')

    plt.show()


def plot_test_case_results(summa_output_nc, validation_nc_path):
    ''' Create a plot for the scalarSWE and scalarSnowDepth variables '''

    # Read the netcdf files
    if len(summa_output_nc) == 1:
        results_ds = xr.open_dataset(summa_output_nc[0])
    else:
        results_ds = xr.open_mfdataset(summa_output_nc, combine='by_coords')

    validation_ds = xr.open_dataset(validation_nc_path)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].set_title("Snow Depth")
    ax[0].set_ylabel("SD (m)")
    ax[0].plot(results_ds.time, results_ds.scalarSnowDepth * 100, label="Modelled")
    ax[0].plot(validation_ds.time, validation_ds.zs_sheltered, label="Observed")
    ax[0].set_xlabel("Time")
    ax[0].grid(True)
    ax[0].set_xlim(results_ds.time[0], results_ds.time[-1])
    ax[0].legend()
    ax[0] = ax[0].twinx()
    ax[1].set_title("Snow Water Equivalent")
    ax[1].set_ylabel("SWE (m)") 
    ax[1].plot(results_ds.time, results_ds.scalarSWE, label="Modelled")
    ax[1].plot(validation_ds.time, validation_ds.SWE, label="Observed")
    ax[1].set_xlabel("Time")
    ax[1].grid(True)
    ax[1].set_xlim(results_ds.time[0], results_ds.time[-1])
    ax[1].legend()

    plt.tight_layout()  
    plt.show()

def plot_fsca_snow_result(fsca_nc):
    
        # Read the netcdf files
    if len(fsca_nc) == 1:
        ds = xr.open_dataset(fsca_nc[0])
    else:
        ds = xr.open_mfdataset(fsca_nc, combine='by_coords')

    # Define colors for each HRU using the 'Paired' colormap
    colors = plt.cm.Paired([0, 2, 4])  # Extracting 3 distinct colors from the colormap

    # Set up the plot with 2 main subplots
    fig, main_axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12), sharex=True)

    # Iterate through each HRU to plot the data
    for idx, hru_val in enumerate(ds['hru'].values):
        
        # Plot fsca and scalarSWE on the first subplot with twin axes
        ax1 = main_axes[0]
        ax2 = ax1.twinx()
        ds['fsca'].sel(hru=hru_val).plot(ax=ax1, label=f'fsca HRU {hru_val}', color=colors[idx])
        ds['scalarSWE'].sel(hru=hru_val).plot(ax=ax2, label=f'scalarSWE HRU {hru_val}', linestyle='--', color=colors[idx])
        
        # Plot snow_input and snow_output on the second subplot
        ax3 = main_axes[1]
        ds['snow_input'].sel(hru=hru_val).plot(ax=ax3, label=f'snow_input HRU {hru_val}', color=colors[idx])
        ds['snow_output'].sel(hru=hru_val).plot(ax=ax3, label=f'snow_output HRU {hru_val}', linestyle='--', color=colors[idx])

    # Set plot titles, legends and other details
    ax1.set_title('fsca and scalarSWE over Time')
    ax1.set_ylabel('fsca Value')
    ax2.set_ylabel('scalarSWE Value')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax3.set_title('snow_input and snow_output over Time')
    ax3.legend(loc='upper left')
    ax3.set_ylabel('Value')
    ax3.grid(True)

    plt.tight_layout()
    plt.xlabel('Time')
    plt.show()
