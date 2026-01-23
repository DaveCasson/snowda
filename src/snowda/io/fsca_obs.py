from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr


def read_fsca_observations(
    obs_nc: str | Path,
    *,
    variable: str = "fsca",
    start_time: Optional[pd.Timestamp] = None,
    end_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    with xr.open_dataset(obs_nc) as ds:
        if variable not in ds:
            raise ValueError(f"Variable '{variable}' not found in {obs_nc}")
        obs = ds[variable]
        if start_time or end_time:
            obs = obs.sel(time=slice(start_time, end_time))
        obs_df = obs.to_dataframe()
    obs_df.index.names = ["hru", "time"]
    obs_df.columns = ["fsca_obs"]
    return obs_df
