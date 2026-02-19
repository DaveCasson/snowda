from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import pandas as pd

from snowda.config.pysummada import PysummadaConfig
from snowda.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DataAssimilationCore:
    config: PysummadaConfig
    ens_config: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)

    def set_run_period(self) -> pd.DatetimeIndex:
        log.info(
            "Setting run period from %s to %s", self.config.start_time, self.config.end_time
        )
        self.run_period = pd.date_range(
            self.config.start_time,
            self.config.end_time - self.config.time_delta,
            freq="d",
        )
        return self.run_period

    def create_ens_config(self, ens_file_managers: list[str]) -> Dict[str, Dict[str, str]]:
        file_manager_dicts = [{"file_manager": fm} for fm in ens_file_managers]
        config_zip = zip(self.config.ens_label_array, file_manager_dicts)
        config_dict = dict(config_zip)
        self.ens_config = config_dict
        return config_dict

    def update_run_time(self, start_time: datetime, end_time: datetime) -> Dict[str, Dict[str, str]]:
        for ens in self.config.ens_label_array:
            entry = self.ens_config.setdefault(ens, {})
            manager = entry.setdefault("manager", {})
            manager["simStartTime"] = start_time.strftime("%Y-%m-%d %H:%M")
            manager["simEndTime"] = end_time.strftime("%Y-%m-%d %H:%M")
        return self.ens_config

    def update_warm_state(self, end_time: datetime) -> Dict[str, Dict[str, str]]:
        warm_states = self.create_warm_state_list(end_time)
        config_list = [
            {ens: {"manager": {"initConditionFile": warm_state}}}
            for ens, warm_state in zip(self.config.ens_label_array, warm_states)
        ]
        config_dict = self._merge_dicts(config_list)
        self.ens_config = self.merge_dict(self.ens_config, config_dict)
        return self.ens_config

    @staticmethod
    def merge_dict(a: Dict, b: Dict, path: list[str] | None = None) -> Dict:
        if path is None:
            path = []
        for key in b:
            if key in a:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    DataAssimilationCore.merge_dict(a[key], b[key], path + [str(key)])
                elif a[key] != b[key]:
                    raise ValueError(f"Conflict at {'.'.join(path + [str(key)])}")
            else:
                a[key] = b[key]
        return a

    def create_warm_state_list(self, end_time: datetime) -> list[str]:
        run_suffix_str1 = end_time.strftime("%Y%m%d")
        run_suffix_str2 = end_time.strftime("%Y_%m_%d")
        return [
            f"{run_prefix}_restart_{run_suffix_str1}00_{run_suffix_str2}.nc"
            for run_prefix in self.config.ens_label_array
        ]

    @staticmethod
    def _merge_dicts(dicts: list[Dict]) -> Dict:
        merged: Dict = {}
        for item in dicts:
            merged = DataAssimilationCore.merge_dict(merged, item)
        return merged
