from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from snowda.utils.logging import get_logger
from snowda.utils.timing import timer

log = get_logger(__name__)


@dataclass
class ParticleFilterCore:
    """Core particle filter utilities (no I/O)."""

    @timer
    def update_particle_weights(
        self,
        particles: pd.DataFrame,
        swe_measurements: pd.DataFrame,
        z_var: float,
        end_date_current_run,
        hru_index: pd.DataFrame,
        ens_label_array: list[str],
        obs_column: str,
    ) -> pd.DataFrame:
        z_df = swe_measurements.loc[
            swe_measurements.index.get_level_values("time") == end_date_current_run
        ]

        likelihoods = []
        for i in range(len(particles.columns)):
            errors = []
            particle_by_hru = particles.iloc[:, i]
            for hru_id in hru_index["hruId"]:
                meas = float(z_df[z_df["hruId"] == hru_id][obs_column].values[0])
                particle = particle_by_hru.loc[hru_id].values
                errors.append(abs(particle - meas))
            error_mean = np.nanmean(errors)
            likelihoods.append(np.exp(-0.5 * error_mean * z_var))

        likelihoods = np.array(likelihoods, dtype=float)
        total = np.sum(likelihoods)
        if total == 0:
            weights = np.full_like(likelihoods, 1.0 / len(likelihoods))
        else:
            weights = likelihoods / total

        weights_dict = {end_date_current_run: weights}
        weights_df = pd.DataFrame.from_dict(weights_dict, orient="index")
        weights_df.columns = ens_label_array
        log.info("Particle weights calculated for %s", end_date_current_run)
        return weights_df

    @timer
    def update_particle_weights_fsca(
        self,
        particles: pd.DataFrame,
        fsca_measurements: pd.DataFrame,
        z_var: float,
        end_date_current_run,
        hru_index: pd.DataFrame,
        ens_label_array: list[str],
        obs_column: str,
    ) -> pd.DataFrame:
        z_df = fsca_measurements.loc[
            fsca_measurements.index.get_level_values("time") == end_date_current_run
        ]

        likelihoods = []
        for i in range(len(particles.columns)):
            errors = []
            particle_by_hru = particles.iloc[:, i]
            for hru_id in hru_index["hruId"]:
                meas = float(z_df.loc[hru_id, obs_column])
                particle = particle_by_hru.loc[hru_id].values
                errors.append(abs(particle - meas))
            error_mean = np.nanmean(errors)
            likelihoods.append(np.exp(-0.5 * error_mean * z_var))

        likelihoods = np.array(likelihoods, dtype=float)
        total = np.sum(likelihoods)
        if total == 0:
            weights = np.full_like(likelihoods, 1.0 / len(likelihoods))
        else:
            weights = likelihoods / total

        weights_dict = {end_date_current_run: weights}
        weights_df = pd.DataFrame.from_dict(weights_dict, orient="index")
        weights_df.columns = ens_label_array
        log.info("fSCA particle weights calculated for %s", end_date_current_run)
        return weights_df

    @timer
    def state_estimate(
        self,
        particles: pd.DataFrame,
        weights: pd.DataFrame,
        end_date_current_run,
        hru_index: pd.DataFrame,
        mean_name: str = "mean_state",
        var_name: str = "var_state",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mean = {}
        var = {}
        for hru_id in hru_index["hruId"]:
            mean[hru_id] = np.average(
                particles.loc[hru_id].values[0], weights=weights.values[0], axis=0
            )
            var[hru_id] = np.average(
                (particles.loc[hru_id].values[0] - mean[hru_id]) ** 2,
                weights=weights.values[0],
                axis=0,
            )

        mean_df = pd.DataFrame.from_dict(mean, orient="index")
        var_df = pd.DataFrame.from_dict(var, orient="index")

        mean_df["time"] = end_date_current_run
        mean_df.reset_index(inplace=True)
        mean_df.set_index(["index", "time"], inplace=True)
        mean_df.index.names = ["hru", "time"]
        mean_df.columns = [mean_name]

        var_df["time"] = end_date_current_run
        var_df.reset_index(inplace=True)
        var_df.set_index(["index", "time"], inplace=True)
        var_df.index.names = ["hru", "time"]
        var_df.columns = [var_name]

        return mean_df, var_df

    def efficient_weights(self, weights: pd.DataFrame, num_ens: int) -> float:
        weights_np = weights.iloc[0].to_numpy()
        return np.around(1.0 / (num_ens * np.mean([w**2 for w in weights_np])))

    def resample_indices(
        self, weights: pd.DataFrame, num_ens: int, n_eff_threshold: float
    ) -> Optional[np.ndarray]:
        weights_np = weights.iloc[0].to_numpy()
        effweights = self.efficient_weights(weights, num_ens)
        if effweights <= float(n_eff_threshold):
            return None

        resample = np.empty((num_ens), dtype=int)
        zweightcumul = np.cumsum(weights_np)
        zrand = np.random.random() / num_ens

        for ires in range(0, num_ens):
            if zrand <= zweightcumul[0]:
                resample[ires] = 1
            for rk in range(1, num_ens):
                if (zweightcumul[rk - 1] < zrand) and (zrand <= zweightcumul[rk]):
                    resample[ires] = rk + 1
            zrand += 1.0 / num_ens

        return resample - 1
