import pathlib
from typing import Protocol

import pandas as pd

import config
from src import galhalo
from src.utils import pathing as upath


class Config(Protocol):
    dirname: str
    scatter: float
    cond_label: str

    @property
    def scatter_str(self) -> str: ...


def get_galhalo_sample_path(configuration: Config) -> pathlib.Path:
    """
    Create and/or get the path to a galhalo sample directory.

    """
    return upath.get_results_path(
        config.RESULTS_DIR,
        configuration.dirname,
        f'{configuration.scatter:.3f}',
        config.SAMPLE_DIRNAME,
        configuration.cond_label,
    )


def save_galhalo_sample(
    results_path: pathlib.Path, obs_label: str, sim_label: str, df: pd.DataFrame
) -> None:
    """
    Write galhalo sample results to a file.

    """
    file = galhalo.names.get_galhalo_sample_filename(obs_label, sim_label)
    df.to_parquet(results_path.joinpath(file), index=False)


def load_galhalo_sample(
    results_path: pathlib.Path, obs_label: str, sim_label: str
) -> pd.DataFrame:
    """
    Load galhalo sample results.

    """
    file = galhalo.names.get_galhalo_sample_filename(obs_label, sim_label)
    return pd.read_parquet(results_path.joinpath(file))


def get_galhalo_path(configuration: Config) -> pathlib.Path:
    """
    Create and/or get the path to the main galhalo directory where
    cfgrid results are stored.

    """
    return upath.get_path(
        config.RESULTS_DIR,
        configuration.dirname,
        configuration.scatter_str,
        config.CFGRID_DIRNAME,
        configuration.cond_label,
    )


def get_galhalo_tpcf_path(configuration: Config) -> pathlib.Path:
    """
    Create and/or get the path to galhalo correlation function results.

    """
    return upath.get_results_path_from_path(get_galhalo_path(configuration))
