from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate
from scipy.interpolate import CubicSpline

from src.cosmo import constants


def comoving_distance(
    z: float | npt.NDArray,
    z0: Optional[float | npt.NDArray] = None,
    H0: float = constants.H0,
    Om: float = constants.Om,
    Ol: float = constants.Ol,
    c: float = constants.c,
) -> npt.NDArray:
    """
    Calculate the comoving distance from z0 to z. If z0 is not None,
    it must be a single number or an array with the same shape as z.

    """
    if isinstance(z, int) or isinstance(z, float):
        z = np.array([z])

    if isinstance(z0, int) or isinstance(z0, float):
        z0 = np.full(z.shape, z0)
    elif z0 is None:
        z0 = np.zeros(len(z))

    if len(z) != len(z0):
        raise ValueError('z and z0 must have the same shape if z0 is given')

    d_H = c / H0
    Ez_inv = lambda z: 1 / np.sqrt(Om * (1 + z) ** 3 + Ol)
    results = []

    for z_, z0_ in zip(z, z0):
        integral = integrate.quad(Ez_inv, z0_, z_)
        result = d_H * integral[0]
        results.append(result)

    return np.array(results)


def luminosity_distance(
    z: float | npt.NDArray,
    H0: float = constants.H0,
    Om: float = constants.Om,
    Ol: float = constants.Ol,
) -> float | npt.NDArray:
    """
    Calculate the luminosity distance at redshift z.

    """
    return (1 + z) * comoving_distance(z, H0=H0, Om=Om, Ol=Ol)


def redshift_interp(
    z: float,
    z0: float = 0.0,
    H0: float = constants.H0,
    Om: float = constants.Om,
    Ol: float = constants.Ol,
    c: float = constants.c,
) -> Callable:
    """
    Calculate the redshift corresponding to a given comoving distance.

    """
    redshift = np.arange(z0, z + 0.01, 0.01)
    cdist = comoving_distance(redshift, z0=z0, H0=H0, Om=Om, Ol=Ol, c=c)
    interp_z_from_cdist = CubicSpline(cdist, redshift, extrapolate=False)

    return interp_z_from_cdist
