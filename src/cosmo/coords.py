"""
Note that these functions will assume the x axis is the long axis
similar to the redshift axis.

"""

import numpy as np
import numpy.typing as npt

from src.cosmo import distances


def convert_spherical_to_cartesian(
    ra: npt.ArrayLike, dec: npt.ArrayLike, z: npt.ArrayLike, H0: float = 100
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Convert (ra, dec, z) to (x, y, z).

    Parameters
    ----------
    ra: array
        Right ascension coordinates of galaxies.

    dec : array
        Declination coordinates of galaxies.

    z : array
        Redshifts of galaxies.

    Returns
    -------
    X, Y, Z : array
        Three-dimensional simulation coordinates of the galaxies.

    """
    # Ensure using arrays
    ra = np.array(ra)
    dec = np.array(dec)
    z = np.array(z)

    r = distances.comoving_distance(z, H0=H0)
    ra = (ra - 180) * np.pi / 180
    dec = dec * np.pi / 180

    X = r * np.cos(dec) * np.cos(ra)
    Y = r * np.cos(dec) * np.sin(ra)
    Z = r * np.sin(dec)

    return X, Y, Z


def convert_cartesian_to_spherical(
    x: npt.NDArray, y: npt.NDArray, z: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Finds the (r, ra, dec) coordinates of a satellite on the sky
    given its (x, y, z) coordinates in the light cone.

    """
    r = np.round(np.sqrt(x**2 + y**2 + z**2), 6)
    ra = 180 - np.round(np.arctan2(-y, x) * 180 / np.pi, 6)
    dec = np.round(np.arcsin(z / r) * 180 / np.pi, 6)

    return r, ra, dec
