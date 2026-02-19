# Read the configuration and resolve paths for simpler reading
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

def calculate_annual_climatology(df, date_col='datetime', flow_col='discharge_cms'):
    """
    Calculate annual streamflow climatology and include the actual streamflow for each year.
    
    Args:
        df (pd.DataFrame): DataFrame containing the streamflow data.
        date_col (str): Column name for datetime.
        flow_col (str): Column name for streamflow values.
    
    Returns:
        pd.DataFrame: A DataFrame with climatology statistics and columns for each year's actual streamflow.
    """
    # Ensure the datetime column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Extract the year and day of the year for each timestamp
    df['year'] = df[date_col].dt.year
    df['day_of_year'] = df[date_col].dt.dayofyear
    
    # Pivot the data so each year has its own column for streamflow values
    df_pivot = df.pivot_table(index='day_of_year', columns='year', values=flow_col)
    
    # Calculate climatological statistics (mean, std, min, max, percentiles)
    climatology = df.groupby('day_of_year')[flow_col].agg(
        mean_flow='mean',
        std_dev='std',
        min_flow='min',
        max_flow='max',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        q10=lambda x: x.quantile(0.10),
        q90=lambda x: x.quantile(0.90)
    )
    
    # Merge the climatology statistics with the pivoted streamflow data (each year's actual values)
    climatology = pd.concat([climatology, df_pivot], axis=1)
    
    return climatology


def plot_climatology(climatology, title1='Streamflow Records', title2='Climatology with Percentiles'):
    """
    Plot the spaghetti plot of all years' streamflow and the climatology with shaded percentiles in two subplots.
    
    Args:
        climatology (pd.DataFrame): DataFrame containing the streamflow climatology and actual values for each year.
        title1 (str): Title for the spaghetti plot.
        title2 (str): Title for the climatology plot.
    
    Returns:
        None: Displays the plots.
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
    
    # First subplot: Spaghetti plot of all years' streamflow
    axes[0].set_title(title1)
    years = climatology.columns[5:]  # Assuming first 5 columns are stats, rest are years
    for year in years:
        axes[0].plot(climatology.index, climatology[year], label=f'{year}')
    
    # Customize first plot (spaghetti plot)
    axes[0].set_xlabel('Day of Year')
    axes[0].set_ylabel('Streamflow (cms)')
    axes[0].grid(True)
    #axes[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Keep legend outside plot for clarity

    # Second subplot: Climatology with shaded areas for 25th-75th percentile
    axes[1].set_title(title2)
    
    # Plot the mean streamflow
    axes[1].plot(climatology.index, climatology['mean_flow'], label='Mean Flow', color='blue')
    
    # Add shaded areas for the 25th to 75th percentile
    axes[1].fill_between(climatology.index, climatology['q25'], climatology['q75'], color='blue', alpha=0.2, label='25th-75th Percentile')
    axes[1].fill_between(climatology.index, climatology['q10'], climatology['q90'], color='blue', alpha=0.1, label='10th-90th Percentile')
    axes[1].fill_between(climatology.index, climatology['min_flow'], climatology['max_flow'], color='blue', alpha=0.05, label='Min-Max Range')
    # Customize second plot (climatology)
    axes[1].set_xlabel('Day of Year')
    axes[1].set_ylabel('Streamflow (cms)')
    axes[1].grid(True)
    axes[1].legend()

    # Adjust layout to ensure plots don't overlap
    plt.tight_layout()
    
    # Display the plots
    plt.show()
