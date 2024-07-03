from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from src import nsatlib
from src.protocols import binning, properties


def calc_nsat(
    cen: pd.DataFrame,
    sat: pd.DataFrame,
    group_id_str: str,
    mass_bins: npt.NDArray | Sequence[float],
    ms_bins: Sequence[float],
    window_size: float = 0.2,
) -> pd.DataFrame:
    """
    Calculate the number of satellites per central as a function of
    central stellar mass and delta MS.

    """
    delta_ms = properties.DeltaMS(cen).value
    cen = cen.assign(delta_ms_bin=np.digitize(delta_ms, bins=ms_bins))
    nsat = nsatlib.calc.nsat(cen, sat, group_id_str, mass_bins, window_size)
    return nsat


def main() -> None:
    configuration = nsatlib.config.CONFIG
    data = configuration.load_nsat()
    volume = configuration.volume[0]
    dm = nsatlib.config.CENTRAL_MASS_BIN_SIZE

    mmin, mmax = volume.mass_lims
    zmin, zmax = volume.redshift_lims
    mass_bins = np.arange(mmin, mmax + dm, dm)
    ms_bins = binning.DeltaMSBins.bins[6]

    for cen_cond, sat_cond in zip(configuration.centrals, configuration.satellites):
        cen = data[
            cen_cond(data).value & (data['z_obs'] >= zmin) & (data['z_obs'] <= zmax)
        ]
        sat = data[sat_cond(data).value]
        nsat = calc_nsat(
            cen,
            sat,
            cen_cond.group_id_label,
            mass_bins,
            ms_bins,
            window_size=nsatlib.config.WINDOW_SIZE,
        )
        nsatlib.pathing.save_nsat(nsat, configuration, cen_cond.label)
