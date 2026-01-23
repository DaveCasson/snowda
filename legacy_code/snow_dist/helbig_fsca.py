"""

This script is a Python implementation of the Helbig et al, 2021 seasonal algorithm of the snow-covered area algorithm.

Helbig, Nora, Michael Schirmer, Jan Magnusson, Flavia Mäder, Alec van Herwijnen, Louis Quéno, Yves Bühler, Jeff S. Deems, and Simon Gascoin. 2021. “A Seasonal Algorithm of the Snow-Covered Area Fraction for Mountainous Terrain.” The Cryosphere 15 (9): 4607–24. https://doi.org/10.5194/tc-15-4607-2021.

"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def snow_cover_fraction(
    snow_depth_n,
    swe_tmp,
    swe_hist_n,
    swe_min_n,
    swe_max_n,
    snow_depth_hist_n,
    snow_depth_min_n,
    snow_depth_max_n,
    slope_mu_n,
    xi_n,
    Ld_n,
    snow_depth_threshold,
    buffer_length
):
    """

    :param fsnow_n: Snow cover fraction fSCA
    :param snowdepth_n: Snow depth (m)
    :param swe_tmp: Temporary snow water equivalent for fSCA calculation (kg/m^2)
    :param swe_hist_n: history of snow water equivalent (SWE) during last 14 days (kg/m^2)
    :param swe_min_n: Min swe during the season (m)
    :param swe_max_n: Max swe during the season (m)
    :param snow_depth_hist_n: history of snow depth during last 14 days (m)
    :param snow_depth_min_n: Min snow depth at time step swemin(m)
    :param slope_mu_n: squared slope related parameter (cf. Helbig et al, 2021)
    :param xi_n: terrain correlation length (cf. Helbig et al, 2021)
    :param Ld_n: Coarse-grid cell size

    :return: fsnow_n: Snow cover fraction fSCA
    """
    # Initialize empty arrays
    # Buffer with 14 day history of SWE plus hourly current SWEtmp
    swe_buffer = np.empty(buffer_length)
    # Buffer with 14 day history of snow depth plus hourly current snow depth
    snow_depth_buffer = np.empty(buffer_length)
    # Difference betwen entries in swe_buffer
    diff_swe_buffer = np.empty(buffer_length-1)

    # Calculate topo terms needed for standard deviation of snow depth
    sd_snow_depth_1 = np.exp(-1 / (Ld_n / xi_n) ** 2)
    sd_snow_depth_3 = slope_mu_n ** (0.3193 * (Ld_n**0.1034))

    # Merge current SWEtmp with SWEtmp history from past 14 days
    swe_buffer[0] = swe_tmp
    swe_buffer[1:buffer_length] = swe_hist_n[:]
    snow_depth_buffer[0] = snow_depth_n
    snow_depth_buffer[1:buffer_length] = snow_depth_hist_n[:]

    # Update 14 days history
    history_update_length = buffer_length-1
    swe_hist_n[0:history_update_length] = swe_buffer[0:history_update_length]
    snow_depth_hist_n[0:history_update_length] = snow_depth_buffer[0:history_update_length]

    # Calculate snowdepthmin_buffer, snowdepthmax_buffer, snowdepthmin_recent
    # Find indices of global min and max in SWEbuffer
    i_abs_max = np.argmax(swe_buffer)
    i_abs_min = np.argmin(swe_buffer)

    # Find index of recent min in swe_buffer
    # Calculate diff vector of swe_buffer
    for i in range(1, history_update_length, 1):
        diff_swe_buffer[i] = swe_buffer[i+1] - swe_buffer[i]
        if diff_swe_buffer[i] > 0.5:
            break

    i_recent_min = np.argmin(swe_buffer[0:i])

    # Use indices to determine snow depth amounts
    snow_depth_min_buffer = snow_depth_buffer[i_abs_min]
    snow_depth_max_buffer = snow_depth_buffer[i_abs_max]
    snow_depth_min_recent = snow_depth_buffer[i_recent_min]

    # Compute storage of new snow on old snow in snowdepthbuffer
    d_snow_depth = max(snow_depth_n - snow_depth_min_buffer, 0)

    # Compute dswemax in swe buffer
    d_snow_depth_max = max(snow_depth_max_buffer - snow_depth_min_buffer, 0)

    # Don't accept dsnowdepthmax to be larger then dsnowdepth, otherwise larger fnsnow values
    d_snow_depth_max = max(d_snow_depth_max, d_snow_depth)

    # Compute storage of recent new snow on old snow in SWEbuffer
    d_snow_depth_recent = max(snow_depth_n - snow_depth_min_recent, 0)

    # Set swe_max and swe_min equal to zero if no snow, same with corresponding snow depth values
    if swe_tmp == 0:
        swe_max_n = 0
        swe_min_n = 0
        snow_depth_max_n = 0
        snow_depth_min_n = 0

    # Set swemax and swemin equal to SWEtmp if maximum, store also snowdepthmax and snowdepthmin of those time steps
    if swe_tmp >= swe_max_n:
        swe_max_n = swe_tmp
        swe_min_n = swe_tmp
        snow_depth_max_n = snow_depth_n
        snow_depth_min_n = snow_depth_n

    # Set swe_min equal swe_tmp if smaller than swe_min, same with corresponding snow depth value
    if swe_tmp < swe_max_n and swe_tmp < swe_min_n:
        swe_min_n = swe_tmp
        snow_depth_min_n = snow_depth_n

    """ Start calculating fSCA """
    # Initial guess of fSCAs
    fsnow_season = 0
    fsnow_nsnow = 0
    fsnow_nsnow_recent = 0

    """ Seasonal fSCA, using formulas of Helbig et al.,2015 and Egli and Jonas, 2009 """
    # Calculate standard deviation as in Helbig et al., 2015, 2021
    sd_snow_depth_2 = snow_depth_max_n ** (0.5330 * (Ld_n**0.0389))
    sd_snow_depth_0 = sd_snow_depth_1 * sd_snow_depth_2 * sd_snow_depth_3
    # Calculate standard deviation as in Egli and Jonas, 2009 for completely flat grid cells
    if not slope_mu_n > np.finfo("float").eps:
        sd_snow_depth_0 = snow_depth_max_n**0.84

    # Calculate fSCA
    if snow_depth_max_n > 0:
        fsnow_season = np.tanh(1.3 * snow_depth_min_n / sd_snow_depth_0)

    # Calculate CV
    if snow_depth_max_n > 0:
        coeff_vari = sd_snow_depth_0 / snow_depth_max_n
    else:
        coeff_vari = np.nan

    # New snow fSCA based on dswe of last 14 days
    # Calculate standard deviation of dhs as in Egli and Jonas, 2009
    sd_snow_depth_0_dhs = d_snow_depth_max**0.84
    # Calculate fSCA of new snow of last 14 days
    if d_snow_depth_max > 0:
        fsnow_nsnow = np.tanh(1.3 * d_snow_depth / sd_snow_depth_0_dhs)

    fsnow_nsnow_14days = fsnow_nsnow

    # Recent new snow fSCA based on dswe_recent since last minimum
    # Calculate standard deviation of dsnowdepth_recent as in Egli and Jonas, 2009
    sd_snow_depth_0_dhs_recent = d_snow_depth_recent**0.84
    # Calculate fSCA of recent new snow with recent dsnowdepth
    if d_snow_depth_recent > 0:
        fsnow_nsnow_recent = np.tanh(
            1.3 * d_snow_depth_recent / sd_snow_depth_0_dhs_recent
        )

    # Take maximum between the two new snow fSCA, similar to taking the maximum of all three fSCA at the end
    fsnow_nsnow = max(fsnow_nsnow, fsnow_nsnow_recent)

    # If snow amounts are too low, set fSCA to zero
    if snow_depth_min_n < snow_depth_threshold:
        fsnow_season = 0
    if d_snow_depth < snow_depth_threshold:
        fsnow_nsnow = 0

    # Use the largest of the two fSCA estimates
    fsnow_n = max(fsnow_season, fsnow_nsnow)

    if snow_depth_n < np.finfo("float").eps:
        fsnow_n = 0
    else:
        fsnow_n = min(fsnow_n, 1)

    return fsnow_n, snow_depth_min_n, snow_depth_max_n, snow_depth_hist_n, swe_min_n,swe_max_n,swe_hist_n, coeff_vari

def helbig_fsca(swe_input,snow_depth_input,settings):

    #Intitialize History with zero
    fsnow = 0
    swe_min = 0
    swe_max = 0
    snow_depth_min = 0
    snow_depth_max = 0
    swe_hist = np.zeros(settings['buffer_length']-1)
    snow_depth_hist = np.zeros(settings['buffer_length']-1)

    #Initialize empty arrays
    fsnow_array = []

    for snow_depth, swe_tmp in zip(snow_depth_input, swe_input):
        fsnow, snow_depth_min, snow_depth_max, snow_depth_hist, swe_min,swe_max,swe_hist, coeff_vari = snow_cover_fraction(
            snow_depth,
            swe_tmp,
            swe_hist,
            swe_min,
            swe_max,
            snow_depth_hist,
            snow_depth_min,
            snow_depth_max,
            settings['slope_mu'],
            settings['xi'],
            settings['Ld'],
            settings['snow_depth_threshold'],
            settings['buffer_length']
        )

        fsnow_array.append(fsnow)

    fsnow_array = np.array(fsnow_array)

    return fsnow_array


def plot_fsca_output(input_snow_data,fsnow_array,fsca_fortran_generated_data,fsca_original_data):

    fig, ax1 = plt.subplots(nrows=2,figsize=(8, 6))
    ax2 = ax1[0].twinx()
    ax3 = ax1[1].twinx()

    ax1[0].plot(fsca_fortran_generated_data, color='b', label="Fortran generated fSCA")
    ax1[0].plot(fsnow_array, color='r', label="Python generated fSCA", ls='--')
    ax2.plot(input_snow_data[1],color='k',ls='--')
    ax1[0].set_title('Comparison Fortran and Python Generated fSCA (Helbig et al, 2021)')
    ax1[0].set_xlabel('Day')
    ax1[0].set_ylabel('fSCA')
    ax1[0].get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.set_ylabel('SWE (mm)')
    ax1[0].legend(loc='upper right')

    ax1[1].plot(fsca_fortran_generated_data, color='b', label="Fortran generated fSCA")
    ax1[1].plot(fsca_original_data, color='g', label="Provided fSCA.txt, shifted 15 days", ls='--')
    ax3.plot(input_snow_data[0],color='k',ls='--')
    ax1[1].set_title('Comparison Fortran Generated and Provided fSCA.txt')
    ax1[1].set_xlabel('Day')
    ax1[1].set_ylabel('fSCA')
    ax3.set_ylabel('Snow Depth (mm)')
    ax1[1].legend(loc='upper right')

    plt.show()

if __name__ == '__main__':

    # Initilize settings
    settings = {}
    settings['snow_depth_threshold'] = 0.005714286
    settings['buffer_length'] = 15

    settings['Ld'] = 1000
    settings['slope_mu'] = 0.445
    settings['std_dem'] = 117.85
    #Calculate xi
    settings['xi'] = 2 * settings['std_dem'] / settings['slope_mu']

    # Open input and output files
    input_hs_swe_path = '/Users/dcasson/GitHub/snow_dist/test_cases/helbig_2021/HS_SWE.txt'
    fsca_fortran_generated_path = '/Users/dcasson/GitHub/snow_dist/test_cases/helbig_2021/fSCA.txt'
    fsca_original_path = '/Users/dcasson/GitHub/snow_dist/test_cases/helbig_2021/fSCA_original.txt'
    input_snow_data = pd.read_csv(input_hs_swe_path, header=None, delimiter="\t")
    fsca_fortran_generated_data = pd.read_csv(fsca_fortran_generated_path, header=None, delimiter="\t")
    fsca_original_data = pd.read_csv(fsca_original_path, header=None, delimiter="\t")

    # Run main script
    swe_input = input_snow_data[1]
    snow_depth_input = input_snow_data[0]
    fsnow_calculated_data = helbig_fsca(swe_input, snow_depth_input, settings)

    # Plot output
    # import shift for an array
    from numpy import roll as shift
    shifted_original_data = shift(fsca_original_data,0,0)
    plot_fsca_output(input_snow_data, fsnow_calculated_data, fsca_fortran_generated_data, fsca_original_data)

