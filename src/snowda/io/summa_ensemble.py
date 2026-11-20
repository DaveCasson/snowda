from __future__ import annotations

import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr

from snowda.adapters.pysumma import get_file_manager_value
from snowda.config.pysummada import PysummadaConfig
from snowda.utils.logging import get_logger

log = get_logger(__name__)


def _read_file_manager(path: Path) -> pd.DataFrame:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            parts = stripped.split(None, 1)
            if len(parts) < 2:
                continue
            key, value = parts
            records.append((key, value.strip()))
    if not records:
        raise ValueError(f"No valid entries found in file manager: {path}")
    return pd.DataFrame(records, columns=["key", "value"]).set_index("key")


SUMMA_TIME_FORMAT = "%Y-%m-%d %H:%M"


def forcing_file_list_name(case_name: str, label: str) -> str:
    """Bare forcing-list filename SUMMA opens under ``settingsPath``.

    The case name keeps concurrent runs that share a settings directory from
    deleting or overwriting each other's lists.
    """
    return f"forcingFileList_{case_name}_Ens{label}.txt"


# SUMMA opens these as directories and appends a bare filename. Relative
# values are resolved against the base file manager's directory so they stay
# valid after summa.exe is launched from the run directory.
_SUMMA_DIRECTORY_KEYS = ("settingsPath", "forcingPath", "outputPath", "statePath")


def _absolute_summa_directory(value: str, anchor: Path) -> str:
    """Return a quoted absolute directory with the trailing slash SUMMA concatenates."""
    raw = value.strip().strip("'").strip('"')
    if not raw:
        return value
    path = Path(raw)
    if not path.is_absolute():
        path = (anchor / path).resolve()
    text = path.as_posix().rstrip("/") + "/"
    return f"'{text}'"


def _absolutize_summa_directories(frame: pd.DataFrame, anchor: Path) -> None:
    for key in _SUMMA_DIRECTORY_KEYS:
        if key not in frame.index:
            continue
        frame.loc[key, "value"] = _absolute_summa_directory(str(frame.loc[key, "value"]), anchor)


def read_file_manager_option(file_manager_path: str | Path, key: str) -> str:
    """Return a file-manager value with surrounding quotes removed."""
    frame = _read_file_manager(Path(file_manager_path))
    if key not in frame.index:
        raise KeyError(f"{key} not found in {file_manager_path}")
    return str(frame.loc[key, "value"]).strip().strip("'").strip('"')


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


def write_simulation_times(
    file_managers: Iterable[str | Path],
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Persist simStartTime/simEndTime so SUMMA reads the YAML run period from disk."""
    start = start_time.strftime(SUMMA_TIME_FORMAT)
    end = end_time.strftime(SUMMA_TIME_FORMAT)
    for file_manager in file_managers:
        update_file_manager_option(file_manager, "simStartTime", f"'{start}'")
        update_file_manager_option(file_manager, "simEndTime", f"'{end}'")


@dataclass
class EnsembleFileWriter:
    config: PysummadaConfig
    run_dir: Path | None = None
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
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def _historic_member_netcdf(self, label: str) -> Path:
        if self.config.ensemble_forcing_dir is None:
            raise ValueError(
                "`ensemble_forcing_dir` is required when `ensemble_forcing_mode` is historic."
            )
        member_dir = self.config.ensemble_forcing_dir / label
        if not member_dir.is_dir():
            raise FileNotFoundError(f"Ensemble member directory not found: {member_dir}")
        nc_files = sorted(member_dir.glob("*.nc"))
        if len(nc_files) != 1:
            raise ValueError(
                f"Expected one forcing netCDF in {member_dir}, found {len(nc_files)}."
            )
        return nc_files[0]

    def write_ens_file_managers(self) -> list[str]:
        if self.run_dir is None:
            raise ValueError("EnsembleFileWriter.run_dir is required to write file managers.")
        log.info("Writing ensemble file managers to %s", self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        input_fm_df = _read_file_manager(self.config.base_file_manager)
        path_anchor = self.config.base_file_manager.parent
        fm_list: list[str] = []
        historic = self.config.ensemble_forcing_mode == "historic"
        use_base_forcing = self.config.num_ens == 1 and not historic

        for label in self.config.ens_label_array:
            output_fm_name = f"fileManagerEns{label}.txt"
            output_fm_path = self.run_dir / output_fm_name

            output_fm_df = input_fm_df.copy()
            _absolutize_summa_directories(output_fm_df, path_anchor)
            if historic:
                member_file = self._historic_member_netcdf(label)
                member_dir = member_file.parent.as_posix().rstrip("/") + "/"
                output_fm_df.loc["forcingPath", "value"] = f"'{member_dir}'"
                output_fm_df.loc["forcingListFile", "value"] = (
                    f"'{forcing_file_list_name(self.config.case_name, label)}'"
                )
            elif not use_base_forcing:
                output_fm_df.loc["forcingListFile", "value"] = (
                    f"'{forcing_file_list_name(self.config.case_name, label)}'"
                )
            output_fm_df.loc["outFilePrefix", "value"] = f"'{label}'"
            output_fm_df.loc["simStartTime", "value"] = (
                f"'{self.config.start_time.strftime(SUMMA_TIME_FORMAT)}'"
            )
            output_fm_df.loc["simEndTime", "value"] = (
                f"'{self.config.end_time.strftime(SUMMA_TIME_FORMAT)}'"
            )

            output_fm_df.to_csv(output_fm_path, sep="\t", header=False)
            fm_list.append(str(output_fm_path))

        write_simulation_times(
            [self.config.base_file_manager],
            self.config.start_time,
            self.config.end_time,
        )
        return fm_list

    def write_forcing_file_list(self) -> None:
        if self.config.ensemble_forcing_mode == "historic":
            log.info("Writing historic ensemble forcing lists")
            for label in self.config.ens_label_array:
                member_file = self._historic_member_netcdf(label)
                output_file_list_path = self.config.file_list_dir / forcing_file_list_name(
                    self.config.case_name, label
                )
                output_file_list_path.write_text(f"{member_file.name}\n", encoding="utf-8")
            return
        if self.config.num_ens == 1:
            log.info("Single-member run: using forcingListFile from base file manager.")
            return
        log.info("Writing forcing file lists")
        day_count = (self.config.end_time - self.config.start_time).days
        year_months = OrderedDict(
            ((self.config.start_time + pd.Timedelta(days=i)).strftime("%Y%m"), None)
            for i in range(day_count)
        ).keys()

        for label in self.config.ens_label_array:
            forcing_file_list = [
                f"{self.config.forcing_file_base}{year_month}.{label}.nc"
                for year_month in year_months
            ]
            forcing_file_list.sort()

            output_file_list_path = self.config.file_list_dir / forcing_file_list_name(
                self.config.case_name, label
            )
            with output_file_list_path.open("w", encoding="utf-8") as fl_file:
                for line in forcing_file_list:
                    fl_file.write(f"{line}\n")

    def clear_ensemble_files(self) -> None:
        folder = self.config.file_list_dir
        if not folder.exists():
            return
        pattern = f"forcingFileList_{self.config.case_name}_Ens*.txt"
        log.info("Deleting forcing lists matching %s in %s", pattern, folder)
        for filename in folder.glob(pattern):
            try:
                if filename.is_file() or filename.is_symlink():
                    filename.unlink()
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

@dataclass
class SummaEnsembleReader:
    config: PysummadaConfig

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
