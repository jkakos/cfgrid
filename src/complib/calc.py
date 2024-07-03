from typing import Sequence

import numpy as np
import pandas as pd

from src import datalib
from src.protocols import conditionals, properties


def vol1_type_completeness(
    mass_bins: Sequence[float],
) -> list[datalib.volumes.Volume]:
    """
    Calculate a completeness using the volume1 method applied to
    different subsamples.

    """
    dz = 0.005
    zbins = np.arange(0, 0.3, dz)
    zbins = 0.5 * (zbins[:-1] + zbins[1:])
    volumes = []

    mpa = datalib.MPAJHU(data_cuts=dict(M_star=(mass_bins[0], mass_bins[-1])))
    mpa_data = mpa.load(completeness=0)

    delta_ms = properties.DeltaMS(mpa_data).value
    data = [
        mpa_data[conditionals.AllGalaxies(mpa_data).value],
        mpa_data[conditionals.Centrals(mpa_data).value],
        mpa_data[conditionals.Satellites(mpa_data).value],
        mpa_data[delta_ms < -1],
        mpa_data[(delta_ms > -1) & (delta_ms < -0.45)],
        mpa_data[delta_ms > -0.45],
    ]
    data_complete = [
        datalib.dataprocess.make_complete_vol1(d, mass_bins=mass_bins) for d in data
    ]

    for mmin, mmax in zip(mass_bins[:-1], mass_bins[1:]):
        zmins = []
        zmaxs = []
        for d in data_complete:
            d_mbin = d.query("M_star > @mmin and M_star < @mmax")
            zmins.append(min(d_mbin['z_obs']))
            zmaxs.append(max(d_mbin['z_obs']))

        zlim = zbins[(zbins >= max(zmins)) & (zbins <= min(zmaxs))]
        zmin = min(zlim)
        zmax = max(zlim)
        volumes.append(
            datalib.volumes.Volume((mmin, mmax), (zmin - dz / 2, zmax + dz / 2))
        )

    return volumes
