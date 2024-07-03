import numpy as np
import numpy.typing as npt

import config
from src.cosmo import constants, distances


def age_of_universe(
    z: float | npt.NDArray[np.float64],
    H0: float = constants.H0,
    Om: float = constants.Om,
    Ol: float = constants.Ol,
) -> float | npt.NDArray[np.float64]:
    """
    Calculate the age of the universe at redshift z using Hubble
    parameter H0, matter density Om, and dark energy density Ol.
    H0 should be given in units of km/s/Mpc.

    """
    H0_inv_years = H0 * (1 / 3.086e19) * 3.154e7  # Convert to units of 1/yrs
    age = (
        1
        / H0_inv_years
        * 2
        / (3 * np.sqrt(Ol))
        * np.log(
            (np.sqrt(Ol / (1 + z) ** 3) + np.sqrt(Ol / (1 + z) ** 3 + Om)) / np.sqrt(Om)
        )
    )

    return age


def sdss_volume(z_min: float, z_max: float, cut_sdss_rects: bool = True) -> float:
    """
    Calculate the volume of the SDSS from z_min to z_max. This assumes
    that SDSS has been cut along the edges with smooth functions (see
    Varela et al. 2012, Cebrian & Trujillo 2014, Dragomir et al. 2018).

    """
    solid_angle_mgs_cutout = 7748.0  # in sq deg
    d_near = distances.comoving_distance(z_min, H0=100)[0]
    d_far = distances.comoving_distance(z_max, H0=100)[0]
    volume = (solid_angle_mgs_cutout * (np.pi / 180) ** 2) * (
        (d_far**3) / 3 - (d_near**3) / 3
    )
    if cut_sdss_rects:
        import pandas as pd

        rects = pd.read_csv(config.DATA_DIR.joinpath('sdss_cuts.dat'))
        rects['solid_angle'] = (
            (rects['ra_high'] - rects['ra_low'])
            * (rects['dec_high'] - rects['dec_low'])
        ) * (np.pi / 180) ** 2
        rects_vol = rects['solid_angle'].sum() * ((d_far**3) / 3 - (d_near**3) / 3)
        volume -= rects_vol

    return volume
