import pathlib
from typing import Protocol

import pandas as pd

import config
from src import configurations, datalib, nsatlib
from src.utils import pathing as upath


class Config(Protocol):
    dirname: str
    volume: list[datalib.volumes.Volume]

    def get_nsat_filename(self) -> str: ...


def get_nsat_path(configuration: Config, cond_label: str) -> pathlib.Path:
    """
    Create and/or get the path to main Nsat directory where results are
    stored.

    """
    if isinstance(configuration, configurations.BPConfig):
        assert isinstance(nsatlib.config.CONFIG, configurations.BPConfig)
        return upath.get_path(
            config.RESULTS_DIR,
            configuration.dirname,
            nsatlib.config.CONFIG.scatter_str,
            config.NSAT_DIRNAME,
            cond_label,
        )

    return upath.get_path(
        config.RESULTS_DIR, configuration.dirname, config.NSAT_DIRNAME, cond_label
    )


def get_nsat_results_path(configuration: Config, cond_label: str) -> pathlib.Path:
    """
    Create and/or get the path to Nsat results.

    """
    return upath.get_results_path_from_path(get_nsat_path(configuration, cond_label))


def save_nsat(df: pd.DataFrame, configuration: Config, cond_label: str) -> None:
    """
    Write Nsat results to a file.

    """
    file = nsatlib.names.get_nsat_filename(configuration)
    df.to_parquet(
        get_nsat_results_path(configuration, cond_label).joinpath(file),
        index=False,
    )


def load_nsat(configuration: Config, cond_label: str) -> pd.DataFrame:
    """
    Load Nsat results.

    """
    file = nsatlib.names.get_nsat_filename(configuration)
    return pd.read_parquet(
        get_nsat_results_path(configuration, cond_label).joinpath(file)
    )
