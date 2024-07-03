import pathlib

import config
from src.utils import pathing as upath


def get_density_path(dirname: str) -> pathlib.Path:
    """
    Create and/or get the path to main density directory where results
    are stored.

    """
    return upath.get_path(config.RESULTS_DIR, dirname, config.DENSITY_DIRNAME)


def get_density_results_path(dirname: str) -> pathlib.Path:
    """
    Create and/or get the path to cfgrid correlation function results.


    """
    return upath.get_results_path_from_path(get_density_path(dirname))
