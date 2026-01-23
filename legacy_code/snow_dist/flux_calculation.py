"""

Flux calculation

"""

# Import python packages
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Import local packages
from data_io import read_output_file_txt_to_ds
from summa_simulation import SummaSimulation

import logging

logger = logging.getLogger("log_to_file")


def calc_flux(fsca, snow_cover_flux, snow_free_flux):
    """Return the grid average flux, calculating snow cover and snow free seperately"""

    grid_averaged_flux = fsca * snow_cover_flux + (1 - fsca) * snow_free_flux

    return grid_averaged_flux


def calc_snow_cover_flux_reverse(fsca, grid_average_flux):
    """Calculate snow cover flux, when the snow free flux is known to be zero"""

    snow_cover_flux = xr.where(
        fsca != 1, grid_average_flux * (1 - fsca), grid_average_flux
    )

    return snow_cover_flux


def read_summa_flux_data(settings):
    output_dir = settings["summa"]["summa_output_path"]
    snow_free_data_path = Path(
        output_dir, settings["summa_no_snow_hru"]["output_file_name"]
    )
    snow_cover_data_path = Path(output_dir, settings["summa"]["output_file_name"])
    fsca_data_path = Path(output_dir, settings["summa"]["summa_fsca_filename"])

    snow_free_ds = read_output_file_txt_to_ds(snow_free_data_path)
    snow_cover_ds = read_output_file_txt_to_ds(snow_cover_data_path)

    fsca_ds = xr.open_dataset(fsca_data_path)

    return snow_free_ds, snow_cover_ds, fsca_ds


def calculate_fluxes(settings, var_list):
    snow_free_ds, snow_cover_ds, fsca_ds = read_summa_flux_data(settings)
    grid_average_flux = {}
    grid_average_flux["fsca"] = fsca_ds["swe_frac_area"]

    for var in var_list:
        grid_average_flux_calc = calc_flux(
            grid_average_flux["fsca"], snow_cover_ds[var], snow_free_ds[var]
        )
        grid_average_flux[f"{var}_avg"] = grid_average_flux_calc.cumsum(dim="time")
        grid_average_flux[f"{var}_snow_free"] = snow_free_ds[var].cumsum(dim="time")
        grid_average_flux[f"{var}_snow_cover"] = snow_cover_ds[var].cumsum(dim="time")

        back_calculated_flux = calc_snow_cover_flux_reverse(
            grid_average_flux["fsca"], snow_cover_ds[var]
        )
        grid_average_flux[f"back_calculate_rate_{var}"] = back_calculated_flux.cumsum(
            dim="time"
        )

    grid_average_flux_ds = xr.Dataset(grid_average_flux)

    output_data_path = Path(
        settings["summa"]["summa_output_path"],
        settings["summa_fsca"]["grid_average_flux_file_name"],
    )
    grid_average_flux_ds.to_netcdf(output_data_path)

    return grid_average_flux_ds, snow_free_ds, snow_cover_ds, fsca_ds


def plot_fluxes(grid_average_flux_ds, var_list, hru):
    fig, ax = plt.subplots(nrows=len(var_list), ncols=1, figsize=(10, 14))

    for i, var in enumerate(var_list):
        grid_average_flux_ds[f"{var}_avg"].isel(hru=hru).plot.line(
            ax=ax[i], x="time", color="k", linestyle="dotted", label="Average Flux"
        )
        grid_average_flux_ds[f"{var}_snow_free"].isel(hru=hru).plot.line(
            ax=ax[i], x="time", color="b", linestyle="dotted", label="Snow Free"
        )
        grid_average_flux_ds[f"{var}_snow_cover"].isel(hru=hru).plot.line(
            ax=ax[i], x="time", color="r", linestyle="dotted", label="Snow Cover"
        )
        # grid_average_flux_ds[f'{var}_error'].plot.line(ax=ax[0], x="time", color="k", add_legend=False)

        ax[i].set_ylabel(var)
        ax[i].set_xlabel("Time")
        ax[i].set_title("")
        ax[i].legend(loc="upper right")

        # ax[i].set_xlim([datetime.date(2010, 11, 15), datetime.date(2011, 9, 1)])

        ax2 = ax[i].twinx()
        ax2.set_title("")
        grid_average_flux_ds["fsca"].isel(hru=hru).plot.line(
            ax=ax2, x="time", color="k", linestyle="solid", add_legend=False
        )

        # ax2.set_xlim([datetime.date(2010, 11, 15), datetime.date(2011, 9, 1)])
        plt.tight_layout()

    plt.show()


def plot_back_calculated_fluxes(grid_average_flux_ds, fsca_ds, hru):
    fig1, ax1 = plt.subplots()
    # grid_average_flux_ds['back_calculate_rate_scalarSnowDrainage_total_mm'] = grid_average_flux_ds[
    #                                                                              'back_calculate_rate_scalarSnowDrainage_total'] * 1000
    # grid_average_flux_ds['back_calculate_rate_scalarSnowDrainage_total_mm'].isel(hru=hru).plot.line(ax=ax1,x="time", color="k", linestyle='dotted',
    #                                                           label='Back calculated melt')
    grid_average_flux_ds["scalarSnowDrainage_total_mm"] = (
        grid_average_flux_ds["back_calculate_rate_scalarSnowDrainage_total"] * 1000
    )
    grid_average_flux_ds["scalarSnowDrainage_total_mm"].isel(hru=hru).plot.line(
        ax=ax1, x="time", color="k", linestyle="dotted", label="Back calculated melt"
    )
    fsca_ds["swe_total_melt_flux"].isel(hru=hru).plot.line(
        ax=ax1, x="time", color="r", linestyle="dotted", label="SSNOWD melt"
    )

    ax1.set_ylabel("Cumulative melt")
    ax1.set_xlabel("Time")
    ax1.set_title("")
    plt.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    settings_file = (
        "/Users/drc858/GitHub/snow_dist/settings/config_summa_test_cases.yaml"
    )
    model_run = SummaSimulation(settings_file)
    var_list = [
        "scalarInfiltration_total",
        "scalarSurfaceRunoff_total",
        "scalarSnowDrainage_total",
    ]

    grid_average_flux_ds, snow_free_ds, snow_cover_ds, fsca_ds = calculate_fluxes(
        model_run.settings, var_list
    )

    # plot_fluxes(grid_average_flux_ds,var_list,hru)
    for hru in np.arange(0, 10, 1):
        plot_back_calculated_fluxes(grid_average_flux_ds, fsca_ds, hru)
