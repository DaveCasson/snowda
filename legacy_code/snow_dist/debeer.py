"""

Implementations of fSCA algorithms from Chris De Beer's PhD thesis and publications.

References:

DeBeer, Christopher M., and John W. Pomeroy. 2009. “Modelling Snow Melt and Snowcover Depletion in a Small Alpine Cirque, Canadian Rocky Mountains.” Hydrological Processes 23 (18): 2584–99. https://doi.org/10.1002/hyp.7346.
DeBeer, Chris. 2012. “Simulating Areal Snowcover Depletion and Snowmelt Runoff in Alpine Terrain,” April. https://harvest.usask.ca/handle/10388/ETD-2012-04-432.
Moore, R. J., V. A. Bell, R. M. Austin, and R. J. Harding. 1999. “Methods for Snowmelt Forecasting in Upland Britain.” Hydrology and Earth System Sciences 3 (2): 233–46. https://doi.org/10.5194/hess-3-233-1999.


"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt


def calc_mean_y(mean_swe, CV):
    """ Calculate the mean of the lognormal distribution """

    mean_y = np.log(mean_swe) - 0.5 * (np.log(1 +CV**2))

    return mean_y


def calc_std_y(CV):
    """ Calculate the standard deviation of the lognormal distribution """

    std_y = np.sqrt(np.log(1 + CV**2))

    return std_y


def calc_mean_swe(std, mean):
    """ Calculate the mean of a normal distribution"""

    dist_mean = np.log(mean) - 0.5 * std**2

    return dist_mean


def calc_std_swe(CV):
    """ Calculate the standard deviation of a normal distribution"""

    variance = np.log(1 + CV**2)
    std = np.sqrt(variance)

    return std

def calc_normal_prob_dist(y, mean_y, std_y):
    """Frequency distribution of y, where y = ln(SWE) and is normally distributed"""

    f_y = (1 / (std_y * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y - mean_y)**2 / std_y**2))

    return f_y

def calc_log_normal_prob_dist(y, mean_y, std_y):
    """Frequency distribution of y, where y = SWE and is normally distributed"""

    f_y = (1 / (y * std_y * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(y) - mean_y)**2 / std_y**2))

    return f_y

def calc_swe(mean_swe, K, CV):
    """Equation 2a from De Beer and Pomeroy, 2009

    mean_swe is the mean swe for the HRU
    K is the frequency factor
    CV is the coefficient of variation

    """

    swe = mean_swe * (1 + K * CV)

    return swe


def calc_freq_factor(std_y, K, CV):
    """Equation 3 from De Beer and Pomeroy, 2009
    K_y is the frequency factor for the lognormal distribution
    """

    K_y = np.log(1 + K*CV)/ std_y

    return K_y

def calc_K_y(y, mean_y, std_y):
    """Equation 3 from De Beer and Pomeroy, 2009
    K_y is the frequency factor for the lognormal distribution
    """

    K_y = (y - mean_y) / std_y

    return K_y

def calc_K(K_y, CV, std_y):
    #De Beer PhD Thesis eq. 3.5
    # K is the well-known z-statistic of the normal distribution

    K = (np.exp(K_y * std_y - 0.5 * std_y**2) - 1) / CV

    return K
    
def calc_exceedance_prop(K_y):
    # Solution to De Beer PhD Thesis eq. 3.7
    P_K_y = 0.5 * special.erfc(K_y / np.sqrt(2))

    return P_K_y

def calculate_K_min(initial_swe, current_swe, CV):
    # Back calculate the K_min (x-axis intercept)

    swe_ratio  = current_swe / initial_swe
    K_min = - swe_ratio / CV

    return K_min

def back_calculate_K_y(K,CV,std_y):
    """DeBeer PhD Thesis eq. 3.9"""

    K_y = np.log(1+K*CV)/std_y + std_y / 2

    return K_y


def run_debeer_simple(premelt_swe, swe_melt, cv):
    """ Run algorithm as described on pg 55 of Chris Debeers PhD Thesis"""

    std_y = calc_std_y(cv)

    current_swe = premelt_swe - swe_melt    

    K_min = calculate_K_min(premelt_swe, current_swe, cv)
    K_y = back_calculate_K_y(K_min,cv,std_y)
    P = calc_exceedance_prop(K_y)

    return K_min, K_y, P

def debeer(snowfall, melt, snowfall_total, melt_total, fsca, residual_new_snow, cv,alpha=0.25, new_snowfall_threshold=2):
    
    """ Run algorithm  inclusive of linear reversion"""
        
    snowfall_total = snowfall_total + snowfall 
    melt_total = melt_total + melt
        
    # TO DO: Need to add in partial cover swe
    #snowfall = snowfall+residual_new_snow

     # If the accumulated snow is less than or equal to zero
    if snowfall_total <= 0:
        # And the snowfall is equal to zero
        if snowfall <= 0:
            # There is no snow
            fsca = 0
            residual_new_snow = 0
        # If there is snowfall
        else:
            # If the accumulated snow is greater than than the melt
            if snowfall > melt:
                fsca = 1
                residual_new_snow = 0
                # If the melt is greater than snowfall
            else:
                fsca = 0
                residual_new_snow = 0        
    # If the accumulated snow is greater than zero
    else:
        # If the melt is less than or close to zero
        if melt_total > 0:
            #Calculate the fsca
            _, _, fsca = run_debeer_simple(snowfall_total, melt_total, cv)
            #If there is new snowfall interupting the melt
            if snowfall > new_snowfall_threshold:
                new_fsca, residual_new_snow = linear_reversion(snowfall,melt,fsca,snowfall_total,alpha)
                fsca = max(new_fsca,fsca)
        else:
            fsca = 1
            residual_new_snow = 0
    #print(f'residual_new_snow: {residual_new_snow}')
    return fsca, snowfall_total, melt_total, residual_new_snow 

def linear_reversion(new_snowfall,melt,original_fsca,snowfall_total,alpha): 
    """ Refers to Partial Cover Curve, Moore et al 1999 """
    
    # Calculate the melt threshold
    melt_threshold = (1-alpha) * new_snowfall

    if np.greater_equal(melt, new_snowfall):
        new_fsca = original_fsca
        residual_new_snow = 0
    #If the melt is less than the threshold, fsca = 1
    elif np.less_equal(melt, melt_threshold):
        new_fsca = 1
        residual_new_snow = new_snowfall - melt
    # If the melt is greater than the threshold, but less than the new snowfall, use the linear reversion equation
    else:
        residual_melt = melt - melt_threshold
        #Calculate the slope (m) of the line using the formula:
        #m = (y2 - y1) / (x2 - x1)
        slope = (1-original_fsca) / (alpha * new_snowfall)
        #Calculate the y interchept (b) of the line using the formula:
        # b = y1 - m*x1
        y_intercept = original_fsca - slope * snowfall_total
        #Calculate the new fsca using the formula:
        # y = m*x + b
        new_fsca = slope * (snowfall_total - residual_melt) + y_intercept
        residual_new_snow = new_snowfall - melt

    return new_fsca, residual_new_snow

def plot_debeer_example_fig3_2():

    mean_swe = 1
    swe_array = np.arange(0.01, 3, 0.01)

    swe_sum = np.sum(swe_array)
    normalized_swe = swe_array / mean_swe

    y_array = np.log(swe_array)
    y_sum = np.sum(y_array)

    CV_array = np.array([0.2, 0.3, 0.4, 0.6, 0.8, 1.0])

    f_D = np.empty((len(swe_array), len(CV_array)))
    f_D_swe = np.empty((len(swe_array), len(CV_array)))
    f_D_scipy = np.empty((len(swe_array), len(CV_array)))

    for i, CV in enumerate(CV_array):

        std_swe = calc_std_swe(CV)
        mean_swe_dist = calc_mean_swe(std_swe, mean_swe)

        std_y = calc_std_y(CV)
        mean_y_dist = calc_mean_y(mean_swe, CV)

        for j, y in enumerate(y_array):

           f_D[j][i] = calc_normal_prob_dist(y, mean_y_dist, std_y) / y_sum
           f_D_scipy[j][i] = norm.pdf(y, mean_y_dist, std_y) / y_sum
           
        for k, swe in enumerate(swe_array):

            f_D_swe[k][i] = calc_log_normal_prob_dist(swe, mean_swe_dist, std_swe) / swe_sum


    fig, ax = plt.subplots(ncols=2,nrows=1,figsize=(14,6))
    for i, CV in enumerate(CV_array):
        ax[0].plot(normalized_swe, f_D[:, i],label = f"CV = {CV_array[i]}")
        ax[1].plot(normalized_swe, f_D_swe[:, i],label = f"CV = {CV_array[i]}")

    ax[0].set_title("Normal Probability Plot")
    ax[0].set_xlabel("SWE / mean SWE")
    ax[0].set_ylabel("ln(SWE)")
    ax[0].legend()
    
    ax[1].set_title("Lognormal Probability Plot")
    ax[1].set_xlabel("SWE / mean SWE")
    ax[1].set_ylabel("SWE")
    ax[1].legend()

    plt.show()

def plot_debeer_example_fig3_3():

    mean_swe = 100
    swe_array = np.arange(0, 200, 0.001)

    CV_array = np.array([0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
    K_array = np.empty((len(swe_array), len(CV_array)))
    K_plot = np.empty((len(swe_array), len(CV_array)))
    P_K_y_plot = np.empty((len(swe_array), len(CV_array)))

    for i, CV in enumerate(CV_array):
        std_y = calc_std_y(CV)
        mean_y = calc_mean_y(mean_swe, CV)

        for j, y in enumerate(swe_array):

            K_y = calc_K_y(y, mean_y, std_y)
            K = calc_K(K_y, CV, std_y)

            K_array[j,i] = K
            K_plot[j,i] = 1 + K*CV
            P_K_y_plot[j,i] = calc_exceedance_prop(K_y)

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.title("De Beer PhD Thesis, Fig 3.3")
    plt.ylabel("SWE / mean SWE")
    plt.xlabel("K")

    for i, CV in enumerate(CV_array):
        plt.plot(K_array[:,i], K_plot[:, i],label = f"CV = {CV_array[i]}")

    plt.axhline(y=1, color='k')
    plt.axvline(x=0, color='k')
    plt.xlim(-3,6)
    plt.ylim(0,4)
    plt.legend()
    plt.show()

    plt.title("De Beer PhD Thesis, Fig 3.3")
    plt.ylabel("SWE / mean SWE")
    plt.xlabel("P")

    plt.plot(P_K_y_plot[:,2], K_plot[:, 2],label = f"CV = {CV_array[2]}")
    plt.legend()
    plt.show()

def plot_debeer_example_fig3_4():

    premelt_swe_array = [200,200,200,200,200]
    CV = 0.4
    days_after_melt_onset = [0,2,5,10,15]
    swe_melt_array = [0,60,150,300,450]
    K_min_array = []
    K_y_array = []
    P_array = []
    swe_line_array = []
    swe_line_2D_array = []

    # Create arrays for line plots
    K_line_range = np.arange(-4, 6, 0.1)

    for premelt_swe, swe_melt in zip(premelt_swe_array, swe_melt_array):

        K_min, K_y, P = run_debeer_simple(premelt_swe,swe_melt, CV)
        K_min_array.append(K_min)
        K_y_array.append(K_y)
        P_array.append(P)
        swe_line_array = []

        for K in K_line_range:
            swe_line = premelt_swe * (1 + K * CV) - swe_melt 
            swe_line_array.append(swe_line)

        print(f'K_min: {K_min}, K_y: {K_y}, P: {P}')

        swe_line_2D_array.append(swe_line_array)

    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(14,6))

    for i in range(len(days_after_melt_onset)):
        ax[0].plot(K_line_range, swe_line_2D_array[i], color='k', linestyle='--', label=f"Days = {days_after_melt_onset[i]}")

    #Set limits on x and y axis
    ax[0].set_xlim(-4, 6)
    ax[0].set_ylim(0,800)
    ax[0].set_xlabel('K')
    ax[0].set_ylabel('SWE')
    ax[0].spines['left'].set_position('zero')
    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')

    ax[1].plot(days_after_melt_onset, P_array,color='k', marker='o')
    ax[1].set_xlabel('Days')
    ax[1].set_ylabel('SCA')
    ax[0].spines['right'].set_color('none')
    ax[0].spines['top'].set_color('none')


    plt.show()

    return

if __name__ == "__main__":
    # If this script is run directly, plot the demonstration figures
    #plot_debeer_example_fig3_2()
    #plot_debeer_example_fig3_3()
    plot_debeer_example_fig3_4()

