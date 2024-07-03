import pathlib

import config
from src.utils import pathing as upath


def get_cfgrid_path(dirname: str, cond_label: str) -> pathlib.Path:
    """
    Create and/or get the path to main cfgrid directory where results
    are stored.

    """
    return upath.get_path(
        config.RESULTS_DIR, dirname, config.CFGRID_DIRNAME, cond_label
    )


def get_cfgrid_tpcf_path(dirname: str, cond_label: str) -> pathlib.Path:
    """
    Create and/or get the path to cfgrid correlation function results.


    """
    return upath.get_results_path_from_path(get_cfgrid_path(dirname, cond_label))
