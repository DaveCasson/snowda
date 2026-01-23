"""

"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import optimize
import logging
from decorators import timer

logger = logging.getLogger(__name__)

def swe_depth_function(D, zeta, lambda_):
    """Equation 4 from Liston et al, 2004
    f_D is a snow-water-equivalent depth distribution function"""

    f_D = (1 / (D * zeta * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((np.log(D) - lambda_) / zeta) ** 2
    )

    return f_D


def calc_lambda(zeta, mu):
    """Equation 5 from Liston et al, 2004
    Note the lambda_ is used as lambda is a python standard variable"""

    lambda_ = np.log(mu) - 0.5 * zeta**2

    return lambda_


def calc_zeta(CV):
    """Equation 6 from Liston et al, 2004
    zeta is a distribution parameter related to Coefficient of variation"""

    zeta_squared = np.log(1 + CV**2)
    zeta = np.sqrt(zeta_squared)

    return zeta


def calc_z(D, lambda_, zeta):
    """Equation 8 from Liston et al, 2004"""
    """Changing the integration variable from D to z"""
    z = (np.log(D) - lambda_) / zeta
    return z


def calc_swe_frac_area(z):
    """Equation 13 from Liston et al, 2004"""

    swe_frac_area = 0.5 * special.erfc(z / np.sqrt(2))

    return swe_frac_area


def calc_mean_swe(swe_total_acc, swe_total_melt, zeta, z):
    """Equation 18 from Liston et al, 2004"""
    fsca = calc_swe_frac_area(z)

    swe_calc_mean = (
        0.5 * swe_total_acc * special.erfc((z - zeta) / np.sqrt(2))
        - swe_total_melt * fsca
    )

    return swe_calc_mean


def calc_snowmelt_root(swe_total_melt, swe_calc_mean, swe_total_acc, zeta, lambda_):
    """Calculate the root of the formula Eq 18"""

    z = calc_z(swe_total_melt, lambda_, zeta)

    root = swe_calc_mean - calc_mean_swe(swe_total_acc, swe_total_melt, zeta, z)

    return root

def swe_calc_snowmelt(swe, swe_total_acc, CV, swe_total_melt_backup):
    ## Logic below can be removed, however avoids need lack of convergence in edge cases
    if swe_total_acc < swe:
        swe_total_acc = max(swe_total_acc, swe)
        swe_total_melt_result = 0
        return swe_total_melt_result

    zeta = calc_zeta(CV)
    lambda_ = calc_lambda(zeta, swe_total_acc)

    try:
        swe_total_melt_result = optimize.newton(calc_snowmelt_root, swe_total_melt_backup, args=(swe, swe_total_acc, zeta, lambda_), maxiter=10000)
    except:
        swe_total_melt_result = swe_total_melt_backup
        logger.debug(f"No converging solution. swe_mean = {swe}, swe_total_acc = {swe_total_acc} Setting to previous timestep value {swe_total_melt_backup}")

    return swe_total_melt_result


def calc_fsca_and_mean_swe(swe_acc, swe_total_melt, CV):
    """Calculate fractional snow cover area, see Liston 2004, Eq. 18"""

    zeta = calc_zeta(CV)

    lambda_ = calc_lambda(zeta, swe_acc)

    z = calc_z(swe_total_melt, lambda_, zeta)

    fsca = calc_swe_frac_area(z)

    swe_calc_mean = (
        0.5 * np.exp(lambda_ + 0.5 * zeta**2) * special.erfc((z - zeta) / np.sqrt(2))
        - swe_total_melt * fsca
    )

    return fsca, swe_calc_mean, z, lambda_, zeta


def ssnowd(
    snowfall,
    melt_rate,
    swe_total_acc,
    swe_total_melt,
    swe_input,
    residual_new_snow,
    CV,
    use_model_swe=True,
):
    """Main calculation of fsca and average snow"""
    # If the accumulated snow is less than or equal to zero
    if swe_total_acc <= 0:
        # And the snowfall is equal to zero
        if snowfall <= 0:
            # There is no snow
            fsca = 0
            swe_total_acc = 0
            swe_total_melt = 0
            swe_calc = 0
            z = np.nan
            lambda_ = np.nan
        # If there is snowfall
        else:
            # Accumulate both snow and melt
            swe_total_acc = snowfall
            swe_total_melt = melt_rate
            # If the accumulated snow is greater than than the melt
            if swe_total_acc > swe_total_melt:
                # If the melt is zero
                if swe_total_melt <= 0:
                    # The total calculated snow, is the total accumulation so far
                    fsca = 1
                    swe_calc = swe_total_acc
                    swe_total_melt = 0
                    z = np.nan
                    lambda_ = np.nan
                # If the melt is greater than zero
                else:
                    # There is snow and melt, so calculate the fsca and mean SWE
                    fsca, swe_calc, z, lambda_, zeta = calc_fsca_and_mean_swe(
                        swe_total_acc, swe_total_melt, CV
                    )
            # If the accumulated snow is less than or equal the melt
            else:
                # There is no snow
                fsca = 0
                swe_total_acc = 0
                swe_total_melt = 0
                swe_calc = 0
                z = np.nan
                lambda_ = np.nan
    # If the accumulated snow is greater than zero
    else:
        # Accumulate the melt
        swe_total_melt = swe_total_melt + melt_rate
        # If the melt is less than or close to zero
        if swe_total_melt <= 0:
            # There is complete snow cover, and no snow has melted
            fsca = 1
            swe_total_acc = swe_total_acc + snowfall
            swe_calc = swe_total_acc
            swe_total_melt = 0
            z = np.nan
            lambda_ = np.nan
        # If the melt is greater than zero
        else:
            # If there is snowfall this timestep
            if snowfall <= 0:
                # No additional snowfall interrupting the melt
                # Can calculate fsca and mean swe
                #if use_model_swe:
                #swe_total_melt = swe_calc_snowmelt(swe_input, swe_total_acc, CV, swe_total_melt)

                fsca, swe_calc, z, lambda_, zeta = calc_fsca_and_mean_swe(
                    swe_total_acc, swe_total_melt, CV
                )
            # If there is snowfall interupting the melt
            else:
                # If the current snowfall is greater than accumulated melt and the current melt potential
                if snowfall > swe_total_melt:
                    fsca = 1
                    swe_total_acc = swe_total_acc + snowfall - swe_total_melt
                    swe_calc = swe_input
                    swe_total_melt = 0
                    z = np.nan
                    lambda_ = np.nan
                # If the accumulated snowfall is less than the accumulated melt and current melt potential
                else:
                    swe_total_acc = swe_total_acc + snowfall - melt_rate
                    if swe_input > swe_total_acc:
                        fsca = 1
                        swe_calc = swe_input
                        swe_total_melt = 0
                        z = np.nan
                        lambda_ = np.nan
                    else:
                        # Use root finding methods to calculate the total melt, using known mean snowfall and total accum
                        swe_total_melt = swe_calc_snowmelt(
                            swe_input, swe_total_acc, CV,swe_total_melt
                        )
                        fsca, swe_calc, z, lambda_, zeta = calc_fsca_and_mean_swe(
                            swe_total_acc, swe_total_melt, CV
                        )

    return fsca, swe_calc, swe_total_acc, swe_total_melt, z, lambda_



def shift(arr, num, fill_value):
    """Used to shift array (accounting for melt depth values)"""
    # https://www.delftstack.com/howto/numpy/python-numpy-shift-array/

    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr

    return result


def plot_liston_2004_figure4():
    """Replicate the example Figure 4 from Liston et al, 2004"""
    swe_acc = 0.9
    snow_depth_increment = 0.01
    swe_depth_array = np.arange(0.01, 3, snow_depth_increment)
    CV = 0.5
    melt_depth_array = [0, 0.5, 1, 1.5]
    f_D_temp = np.empty(len(swe_depth_array))
    f_D = np.empty((len(melt_depth_array), len(swe_depth_array)))

    for i, melt_depth in enumerate(melt_depth_array):
        zeta = calc_zeta(CV)
        lambda_ = calc_lambda(zeta, swe_acc)

        for j, swe_depth in enumerate(swe_depth_array):
            f_D_temp[j] = swe_depth_function(swe_depth, zeta, lambda_)

        # The melt factor to create the shift is here done by shifting the distribution left.
        melt_depth_shift = -int(melt_depth / snow_depth_increment)
        f_D[i] = shift(f_D_temp, melt_depth_shift, 0)

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.title("Figure 4 Plot (Liston et al,2004)")
    plt.ylabel("f(D)")
    plt.xlabel("swe depth (m)")
    # Change major ticks to show every 20.
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.2)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    plt.fill_between(
        swe_depth_array,
        f_D[0, :],
        color="k",
        alpha=0.3,
        label=f"Melt Depth = {melt_depth_array[0]}",
    )
    plt.fill_between(
        swe_depth_array,
        f_D[1, :],
        color="k",
        alpha=0.5,
        label=f"Melt Depth = {melt_depth_array[1]}",
    )
    plt.fill_between(
        swe_depth_array,
        f_D[2, :],
        color="k",
        alpha=0.6,
        label=f"Melt Depth = {melt_depth_array[2]}",
    )
    plt.fill_between(
        swe_depth_array,
        f_D[3, :],
        color="k",
        label=f"Melt Depth = {melt_depth_array[3]}",
    )
    plt.legend()
    plt.show()


def plot_liston_2004_figure5():
    """Replicate the example Figure 5 from Liston et al, 2004"""
    swe_acc = 1
    swe_depth_array = np.arange(0.01, 2.5, 0.01)
    CV_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    f_D = np.empty((len(swe_depth_array), len(CV_array)))

    for i, CV in enumerate(CV_array):
        zeta = calc_zeta(CV)
        lambda_ = calc_lambda(zeta, swe_acc)

        for j, swe_depth in enumerate(swe_depth_array):
            f_D[j][i] = swe_depth_function(swe_depth, zeta, lambda_)

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.title("Figure 5 Plot (Liston et al,2004)")
    plt.ylabel("f(D)")
    plt.xlabel("swe depth (m)")
    # Change major ticks to show every 20.
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 4.1)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(which="major", color="#CCCCCC", linestyle="--")
    plt.plot(
        swe_depth_array,
        f_D[:, 0],
        color="k",
        linestyle="solid",
        label=f"CV = {CV_array[0]}",
    )
    plt.plot(
        swe_depth_array,
        f_D[:, 1],
        color="k",
        linestyle="dashed",
        label=f"CV = {CV_array[1]}",
    )
    plt.plot(
        swe_depth_array,
        f_D[:, 2],
        color="k",
        linestyle="dotted",
        label=f"CV = {CV_array[2]}",
    )
    plt.plot(
        swe_depth_array,
        f_D[:, 3],
        color="k",
        linestyle="dashdot",
        label=f"CV = {CV_array[3]}",
    )
    plt.plot(
        swe_depth_array,
        f_D[:, 4],
        color="k",
        linestyle=(0, (1, 5)),
        label=f"CV = {CV_array[4]}",
    )
    plt.legend()
    plt.show()
    print(np.sum(f_D[:,0]))
    print(np.sum(f_D[:,1]))
    print(np.sum(f_D[:,2]))
    print(np.sum(f_D[:,3]))
    print(np.sum(f_D[:,4]))

def plot_liston_2004_figure6():
    """Replicate the example Figure 6 from Liston et al, 2004"""
    swe_acc = 1
    swe_melt_depth_array = np.arange(0.01, 2.5, 0.01)

    CV_array = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    swe_frac_area = np.empty((len(swe_melt_depth_array), len(CV_array)))
    swe_ave = np.empty((len(swe_melt_depth_array), len(CV_array)))

    for i, CV in enumerate(CV_array):
        for j, swe_melt_depth in enumerate(swe_melt_depth_array):
            zeta = calc_zeta(CV)

            lambda_ = calc_lambda(zeta, swe_acc)

            z = calc_z(swe_melt_depth, lambda_, zeta)

            swe_frac_area[j][i] = calc_swe_frac_area(z)

            swe_ave[j][i] = (
                0.5
                * np.exp(lambda_ + 0.5 * zeta**2)
                * special.erfc((z - zeta) / np.sqrt(2))
                - swe_melt_depth * swe_frac_area[j][i]
            )

    fig, ax = plt.subplots(figsize=(8, 10))

    ax1 = plt.subplot(211)
    plt.title("Figure 6 Plot (Liston et al,2004)")
    plt.ylabel("snow-covered fraction")
    plt.xlabel("melt depth (m)")
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 1.05)
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.grid(which="major", color="#CCCCCC", linestyle="--")
    plt.plot(
        swe_melt_depth_array,
        swe_frac_area[:, 0],
        color="k",
        linestyle="solid",
        label=f"CV = {CV_array[0]}",
    )
    plt.plot(
        swe_melt_depth_array,
        swe_frac_area[:, 1],
        color="k",
        linestyle="dashed",
        label=f"CV = {CV_array[1]}",
    )
    plt.plot(
        swe_melt_depth_array,
        swe_frac_area[:, 2],
        color="k",
        linestyle="dotted",
        label=f"CV = {CV_array[2]}",
    )
    plt.plot(
        swe_melt_depth_array,
        swe_frac_area[:, 3],
        color="k",
        linestyle="dashdot",
        label=f"CV = {CV_array[3]}",
    )
    plt.plot(
        swe_melt_depth_array,
        swe_frac_area[:, 4],
        color="k",
        linestyle=(0, (1, 5)),
        label=f"CV = {CV_array[4]}",
    )
    plt.legend()

    ax2 = plt.subplot(212)
    plt.ylabel("cell-average swe depth (m)")
    plt.xlabel("melt depth (m)")
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))
    ax2.grid(which="major", color="#CCCCCC", linestyle="--")
    plt.plot(
        swe_melt_depth_array,
        swe_ave[:, 0],
        color="k",
        linestyle="solid",
        label=f"CV = {CV_array[0]}",
    )
    plt.plot(
        swe_melt_depth_array,
        swe_ave[:, 1],
        color="k",
        linestyle="dashed",
        label=f"CV = {CV_array[1]}",
    )
    plt.plot(
        swe_melt_depth_array,
        swe_ave[:, 2],
        color="k",
        linestyle="dotted",
        label=f"CV = {CV_array[2]}",
    )
    plt.plot(
        swe_melt_depth_array,
        swe_ave[:, 3],
        color="k",
        linestyle="dashdot",
        label=f"CV = {CV_array[3]}",
    )
    plt.plot(
        swe_melt_depth_array,
        swe_ave[:, 4],
        color="k",
        linestyle=(0, (1, 5)),
        label=f"CV = {CV_array[4]}",
    )

    plt.show()


def plot_simple_comparison_figure():
    swe_acc_array = np.array(range(50, 1000, 1))
    swe_melt_array = np.array(range(1, 1500, 3))
    num_zero_snowmelt = 5
    swe_melt_array[0:num_zero_snowmelt] = 0
    cv_array = np.array(np.arange(0.1, 0.9, 0.1))

    fsca_complete_array = []
    calc_swe_complete_array = []
    for cv in cv_array:
        fsca_array = []
        calc_swe_array = []
        for swe_acc, swe_total_melt in zip(swe_acc_array, swe_melt_array):
            fsca, calc_swe, z, lambda_, zeta = calc_fsca_and_mean_swe(
                swe_acc, swe_total_melt, cv
            )
            fsca_array.append(fsca)
            calc_swe_array.append(calc_swe)

        fsca_complete_array.append(fsca_array)
        calc_swe_complete_array.append(calc_swe_array)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(swe_acc_array, label="Accumulated Snowfall")
    ax[0].plot(swe_melt_array, label="Accumulated Melt")
    ax[0].legend()
    ax[0].set_title("Accumulated Snowfall and Melt")
    for i, cv in enumerate(cv_array):
        ax[1].plot(calc_swe_complete_array[i], label=f"CV: {cv}")
        ax[1].set_title("Calculated SWE")
        ax[1].legend()
        ax[2].plot(fsca_complete_array[i], label=f"CV: {cv}")
        ax[2].legend()
        ax[2].set_title("fSCA")

    plt.show(block=True)


def plot_z_efrc_figure():
    z_array = np.arange(-3, 3, 0.01)
    fsca_array = []
    for z in z_array:
        fsca = calc_swe_frac_area(z)
        fsca_array.append(fsca)

    plt.plot(z_array, fsca_array)
    plt.xlabel("z")
    plt.ylabel("fsca")
    plt.show()


def plot_basic_fsca_calculation():
    swe_acc_array = np.full(100, 100)
    cv_array = np.array(np.arange(0.1, 0.9, 0.1))

    fsca_complete_array = []
    calc_swe_complete_array = []
    for cv in cv_array:
        fsca_array = []
        calc_swe_array = []
        for swe_acc, swe_total_melt in zip(swe_acc_array, swe_melt_array):
            fsca, calc_swe, z, lambda_, zeta = run_calc_fsca(
                swe_acc, swe_total_melt, cv
            )
            fsca_array.append(fsca)
            calc_swe_array.append(calc_swe)

        fsca_complete_array.append(fsca_array)
        calc_swe_complete_array.append(calc_swe_array)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(swe_acc_array, label="Accumulated Snowfall")
    ax[0].plot(swe_melt_array, label="Accumulated Melt")
    ax[0].legend()
    ax[0].set_title("Accumulated Snowfall and Melt")
    for i, cv in enumerate(cv_array):
        ax[1].plot(calc_swe_complete_array[i], label=f"CV: {cv}")
        ax[1].set_title("Calculated SWE")
        ax[1].legend()
        ax[2].plot(fsca_complete_array[i], label=f"CV: {cv}")
        ax[2].legend()
        ax[2].set_title("fSCA")

    plt.show()


def plot_demonstration_figures():
    """PLot demonstration figures from Liston 2004"""
    plot_liston_2004_figure4()
    plot_liston_2004_figure5()
    plot_liston_2004_figure6()


if __name__ == "__main__":
    # If this script is run directly, plot the demonstration figures
    plot_demonstration_figures()
    #plot_simple_comparison_figure()
    # plot_z_efrc_figure()
    # plot_basic_fsca_calculation()
