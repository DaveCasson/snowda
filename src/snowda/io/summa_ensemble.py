from __future__ import annotations

import glob
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from snowda.adapters.pysumma import get_file_manager_value
from snowda.config.pysummada import PysummadaConfig
from snowda.utils.logging import get_logger
from snowda.utils.pysummada import calculate_ensemble_statistics, output_netcdf
from snowda.utils.timing import timer

log = get_logger(__name__)


def update_file_manager_option(file_manager_path: str | Path, key: str, value: str) -> None:
    """Update a key/value entry in a SUMMA file manager file."""
    path = Path(file_manager_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split(None, 1)
        if parts[0] == key:
            new_lines.append(f"{key}\t{value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}\t{value}")
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@dataclass
class EnsembleFileWriter:
    config: PysummadaConfig
    ens_file_managers: list[str] | None = None

    def write_ensemble_files(self) -> None:
        self.create_output_directories()
        self.clear_ensemble_files()
        self.clear_state_files()
        self.ens_file_managers = self.write_ens_file_managers()
        self.write_forcing_file_list()

    def create_output_directories(self) -> None:
        log.info("Creating output directories")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.model_result_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.file_manager_dir.mkdir(parents=True, exist_ok=True)
        self.config.file_list_dir.mkdir(parents=True, exist_ok=True)

    def write_ens_file_managers(self) -> list[str]:
        log.info("Writing ensemble file managers")
        input_fm_df = pd.read_csv(
            self.config.base_file_manager,
            delim_whitespace=True,
            index_col=0,
            header=None,
        )
        fm_list: list[str] = []

        for num_ens in range(1, self.config.num_ens + 1):
            output_fm_name = f"fileManagerEns{num_ens:03d}.txt"
            output_fm_path = self.config.file_manager_dir / output_fm_name

            output_fm_df = input_fm_df.copy()
            output_fm_df.loc["forcingListFile", 1] = (
                f"'forcingFileListEns{num_ens:03d}.txt'"
            )
            output_fm_df.loc["outFilePrefix", 1] = f"'{num_ens:03d}'"

            output_fm_df.to_csv(output_fm_path, sep="\t", header=False)
            fm_list.append(str(output_fm_path))

        return fm_list

    def write_forcing_file_list(self) -> None:
        log.info("Writing forcing file lists")
        day_count = (self.config.end_time - self.config.start_time).days
        year_months = OrderedDict(
            ((self.config.start_time + pd.Timedelta(days=i)).strftime("%Y%m"), None)
            for i in range(day_count)
        ).keys()

        for num_ens in range(1, self.config.num_ens + 1):
            forcing_file_list = [
                f"{self.config.forcing_file_base}{year_month}.{num_ens:03d}.nc"
                for year_month in year_months
            ]
            forcing_file_list.sort()

            output_file_list_name = f"forcingFileListEns{num_ens:03d}.txt"
            output_file_list_path = self.config.file_list_dir / output_file_list_name
            with output_file_list_path.open("w", encoding="utf-8") as fl_file:
                for line in forcing_file_list:
                    fl_file.write(f"{line}\n")

    def clear_ensemble_files(self) -> None:
        for folder in [self.config.file_manager_dir, self.config.file_list_dir]:
            if not folder.exists():
                continue
            log.info("Deleting ensemble files in %s", folder)
            for filename in folder.glob("*Ens*"):
                try:
                    if filename.is_file() or filename.is_symlink():
                        filename.unlink()
                    elif filename.is_dir():
                        shutil.rmtree(filename)
                except Exception as exc:
                    log.warning("Failed to delete %s: %s", filename, exc)

    def clear_state_files(self) -> None:
        state_folder = Path(
            get_file_manager_value(str(self.config.base_file_manager), "statePath").replace("'", "")
        )
        cold_state = get_file_manager_value(
            str(self.config.base_file_manager), "initConditionFile"
        ).replace("'", "")
        if not state_folder.exists():
            return
        log.info("Deleting state files in %s", state_folder)
        for filename in state_folder.iterdir():
            if filename.name == cold_state:
                continue
            try:
                if filename.is_file() or filename.is_symlink():
                    filename.unlink()
                elif filename.is_dir():
                    shutil.rmtree(filename)
            except Exception as exc:
                log.warning("Failed to delete %s: %s", filename, exc)

    def clear_output_files(self) -> None:
        output_folder = Path(
            get_file_manager_value(str(self.config.base_file_manager), "outputPath").replace("'", "")
        )
        if not output_folder.exists():
            return
        log.info("Deleting output files in %s", output_folder)
        for filename in output_folder.iterdir():
            try:
                if filename.is_file() or filename.is_symlink():
                    filename.unlink()
            except Exception as exc:
                log.warning("Failed to delete %s: %s", filename, exc)


@dataclass
class SummaEnsembleReader:
    config: PysummadaConfig

    @timer
    def read_swe_data(self, *, variable: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.input_swe_hru_nc:
            raise ValueError("Config must define input_swe_hru_nc to read SWE data.")

        with xr.open_dataset(self.config.input_swe_hru_nc) as ds:
            start_slice = self.config.start_time.strftime("%Y-%m-%d")
            end_slice = self.config.end_time.strftime("%Y-%m-%d")
            swe_ds = ds.sel(time=slice(start_slice, end_slice)).load()

        if not swe_ds.data_vars:
            raise ValueError("SWE dataset has no data variables.")
        data_var = variable or list(swe_ds.data_vars)[0]
        swe_ds = swe_ds[[data_var]]
        swe_ds = swe_ds.dropna("hru", how="all")

        swe_df = swe_ds.to_dataframe()
        swe_df.index.names = ["hru", "time"]
        swe_df.columns = ["meas_snw"]

        hru_list = swe_df.index.unique(level="hru").values
        hru_index = self.get_hru_index(hru_list)
        return swe_df, hru_index

    @timer
    def read_particles(
        self,
        particle_files: Iterable[str | Path],
        particle_variable: str,
        hru_index: pd.DataFrame,
        end_date_current_run: datetime,
        ens_label_array: list[str],
    ) -> pd.DataFrame:
        particles_dict: dict[int, np.ndarray] = {}
        for hru_index_val, hru_id in zip(hru_index["hru_index"], hru_index["hruId"]):
            particles = []
            for particle_file in particle_files:
                with xr.open_dataset(particle_file) as particle_ds:
                    particles.append(particle_ds[particle_variable].sel(hru=hru_index_val).values)
            particles_dict[hru_id] = np.hstack(particles)

        particles_df = pd.DataFrame.from_dict(particles_dict, orient="index")
        particles_df["time"] = end_date_current_run
        particles_df.reset_index(inplace=True)
        particles_df.set_index(["index", "time"], inplace=True)
        particles_df.index.names = ["hru", "time"]
        particles_df.columns = ens_label_array
        return particles_df

    def read_summa_ens_output(self, variable: str, file_suffix: str) -> pd.DataFrame:
        ens_list = self.config.ens_label_array
        model_files_ens_list = [
            glob.glob(f"{self.config.output_dir}/{ens_label}*{file_suffix}.nc")
            for ens_label in ens_list
        ]

        model_ens_df = []
        for model_files, ens_label in zip(model_files_ens_list, ens_list):
            if not model_files:
                raise FileNotFoundError(f"No model files found for {ens_label} with suffix {file_suffix}")
            with xr.open_mfdataset(model_files, combine="by_coords") as model_ds:
                model_var_df = model_ds[variable].to_dataframe()
            model_var_df.columns = [ens_label]
            model_ens_df.append(model_var_df)

        ens_result_df = pd.concat(model_ens_df, axis=1)
        ens_result_df = ens_result_df.swaplevel().sort_values("hru")
        return ens_result_df

    def read_summa_state_ens_output(self, variable: str) -> pd.DataFrame:
        start_slice = self.config.start_time.strftime("%Y-%m-%d")
        end_slice = self.config.end_time.strftime("%Y-%m-%d")
        ens_list = self.config.ens_label_array
        model_files_ens_list = [
            glob.glob(f"{self.config.output_dir}/{ens_label}*.nc") for ens_label in ens_list
        ]

        model_ens_df = []
        for model_files, ens_label in zip(model_files_ens_list, ens_list):
            if not model_files:
                raise FileNotFoundError(f"No model files found for {ens_label}")
            with xr.open_mfdataset(model_files, combine="by_coords") as model_ds:
                model_var_df = model_ds[variable].sel(time=slice(start_slice, end_slice)).to_dataframe()
            model_var_df.columns = [ens_label]
            model_ens_df.append(model_var_df)

        ens_result_df = pd.concat(model_ens_df, axis=1)
        ens_result_df = ens_result_df.swaplevel().sort_values("hru")
        return ens_result_df

    def get_hru_index(self, hru_list: Iterable[int]) -> pd.DataFrame:
        if not self.config.initial_state:
            raise ValueError("Config must define initial_state to read HRU index.")

        state_path = get_file_manager_value(str(self.config.base_file_manager), "statePath")
        initial_state_path = Path(state_path) / self.config.initial_state
        log.info("Reading HRU index from %s", initial_state_path)

        with xr.open_dataset(initial_state_path) as cold_state_ds:
            hru_df = cold_state_ds.to_dataframe()

        hru_df.index = hru_df.index.get_level_values("hru")
        hru_dict = dict(zip(hru_df.index.unique().values, hru_df["hruId"].unique()))
        hru_index = [[hru_dict.get(hru), hru] for hru in hru_list]
        return pd.DataFrame(hru_index, columns=["hru_index", "hruId"])

    def postprocess_pf_result(
        self,
        swe_measurements: pd.DataFrame,
        particle_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        state_mean_df: pd.DataFrame,
    ) -> dict[str, Path]:
        particle_df_stats = calculate_ensemble_statistics(particle_df)
        result_df = swe_measurements.merge(state_mean_df, left_index=True, right_index=True)
        result_df = result_df.merge(particle_df_stats, left_index=True, right_index=True)

        outputs: dict[str, Path] = {}
        results_ds = result_df.to_xarray()
        outputs["results"] = output_netcdf(
            results_ds, self.config.model_result_output_dir / "summa_run.nc"
        )

        particles_ds = particle_df_stats.to_xarray()
        outputs["particles"] = output_netcdf(
            particles_ds, self.config.model_result_output_dir / "particles.nc"
        )

        weights_ds = weights_df.to_xarray()
        outputs["weights"] = output_netcdf(
            weights_ds, self.config.model_result_output_dir / "weights.nc"
        )

        return outputs
