import pathlib
from typing import Any

import numpy as np


# =========================================================
# DIRECTORIES / FILES
# =========================================================
ROOT_DIR = pathlib.Path(__file__).parent
DATA_DIR = ROOT_DIR.joinpath('data')
CATALOG_DIR = ROOT_DIR.joinpath('catalogs')
BP_CATALOG_DIR = CATALOG_DIR.joinpath('BP_snapshots')
EMPIRE_CATALOG_DIR = CATALOG_DIR.joinpath('Empire')
SIM_DIR = DATA_DIR.joinpath('simulation')
BASE_MOCK_DIR = SIM_DIR.joinpath('base')
MSTAR_MOCK_DIR = SIM_DIR.joinpath('mstar')
BP_DATA_DIR = SIM_DIR.joinpath('mocks')

# Where results / figures will be stored
RESULTS_DIR = ROOT_DIR.joinpath('results')
CFGRID_DIRNAME = 'cfgrid'
GALHALO_DIRNAME = 'galhalo'
SAMPLE_DIRNAME = 'sample'
NSAT_DIRNAME = 'nsat'
SHMR_DIRNAME = 'shmr'
DENSITY_DIRNAME = 'density'
MISC_DIRNAME = 'misc'

# Random catalog used for SDSS
RANDOM = 'lss_random-0.dr72.dat'

# =========================================================
# DATA CONFIGURATION
# =========================================================
MASS_BINS = [10.0, 10.375, 10.75, 11.0, 11.25, 11.5]
MPA_COMPLETENESS = 1  # volume 1


def get_main_seq_fit_params(mpa_completeness: int) -> tuple[float, float, float]:
    """
    Get the main sequence fitting parameters based on a completeness
    condition used for MPAJHU.

    """
    ms_fit_params = {
        0: (0.829295, 10.9140, -0.896783),
        1: (0.829295, 10.9140, -0.896783),
        2: (0.775939, 10.7655, -0.998087),
    }
    if mpa_completeness not in ms_fit_params:
        raise ValueError(
            "No main sequence fit found for given MPAJHU completeness "
            f"(expected one of {list(ms_fit_params.keys())}, got "
            f"{mpa_completeness})."
        )
    return ms_fit_params[mpa_completeness]


MS_PSI0, MS_M0, MS_GAMMA = get_main_seq_fit_params(MPA_COMPLETENESS)

tpcf_settings: dict[str, Any] = dict(
    bootstraps=200, nthreads=10, pimax=20, bins=np.geomspace(0.1, 20, 11)
)
