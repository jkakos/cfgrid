from typing import TypedDict

import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate

from src.cosmo import constants, quantities


PHI_STAR1 = 10 ** (-3.041)  # h^3 / Mpc^3
PHI_STAR2 = 10 ** (-1.885)  # h^3 / Mpc^3
M_STAR = 10.199  # Msun / h^2
ALPHA = -1.664
BETA = 0.708


class SchechterArgs(TypedDict):
    phi_star: float
    m_star: float
    alpha: float
    beta: float


PARAMS_FUNC1: SchechterArgs = {
    'phi_star': PHI_STAR1,
    'm_star': M_STAR,
    'alpha': ALPHA,
    'beta': 1,
}
PARAMS_FUNC2: SchechterArgs = {
    'phi_star': PHI_STAR2,
    'm_star': M_STAR,
    'alpha': 1 + ALPHA,
    'beta': BETA,
}


def schechter(
    logM: npt.NDArray,
    phi_star: float,
    m_star: float,
    alpha: float,
    beta: float,
    h: float = constants.H0 / 100,
) -> npt.NDArray:
    """
    Evaluate a Schechter function. The 'h' kwarg represents the
    reduced Hubble parameter. It will only be applied to 'm_star',
    leaving the result in units of h^3 / Mpc^3.

    """
    return (
        phi_star
        * np.log(10)
        * (10**logM / (10**m_star / h**2)) ** (1 + alpha)
        * np.exp(-((10**logM / (10**m_star / h**2)) ** beta))
    )


def gsmf(
    logM: npt.NDArray,
    func1_args: SchechterArgs | None = None,
    func2_args: SchechterArgs | None = None,
    h: float = constants.H0 / 100,
) -> npt.NDArray:
    """
    Compute the Galaxy Stellar Mass Function (GSMF) assuming a double
    Schechter function form. The 'h' parameter will only be applied to
    the masses (see 'schechter()' documentation).

    """
    if func1_args is None:
        func1_args = PARAMS_FUNC1
    if func2_args is None:
        func2_args = PARAMS_FUNC2

    return schechter(logM, **func1_args, h=h) + schechter(logM, **func2_args, h=h)


def _gsmf(
    logM: npt.NDArray,
    func1_args: SchechterArgs | None = None,
    func2_args: SchechterArgs | None = None,
    h: float = constants.H0 / 100,
) -> npt.NDArray:
    """
    Helper function to properly calculate the mean density when
    integrating the galaxy stellar mass function. Because the input
    is logM, to get units of mass/volume, a unit transform is required.

    Transform M -> logM:
        - d[\log(M)] = dM / M
        - \int{\phi(M) * dM}
        - \int{\phi(M) * M * (dM/M)}
        - \int{\phi(\log(M)) * 10^{\log(M)} * d\log(M)}

    Thus, we need to multiply by a factor of 10**logM before
    integrating the function.

    """
    return 10**logM * gsmf(logM, func1_args, func2_args, h)


def mean_density(
    logM_low: float,
    logM_high: float,
    func1_args: SchechterArgs | None = None,
    func2_args: SchechterArgs | None = None,
    h: float = constants.H0 / 100,
) -> float:
    """
    Compute the mean stellar mass density in a mass interval defined by
    'm_low' and 'm_high' by integrating the galaxy stellar mass
    function over the given mass range.

    """
    rho_mean = integrate.quad(
        _gsmf, logM_low, logM_high, args=(func1_args, func2_args, h)
    )
    return rho_mean[0]


def number_density(
    logM_low: float,
    logM_high: float,
    func1_args: SchechterArgs | None = None,
    func2_args: SchechterArgs | None = None,
    h: float = constants.H0 / 100,
) -> float:
    number_dens = integrate.quad(
        gsmf, logM_low, logM_high, args=(func1_args, func2_args, h)
    )
    return number_dens[0]


def residual(params, m, y):
    """
    Finds the difference between the data and the double Schechter
    function approximation. This function will be minimized to
    determine phi_star, m_star, alpha, phi_star2, and beta.

    """
    return (
        schechter(m, params[0], params[1], params[2], 1)
        + schechter(m, params[3], params[1], 1 + params[2], params[4])
        - y
    )


def calc_gsmf(logM, z_min, z_max, bin_width=0.1) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Calculates the GSMF for SDSS.

    """
    sdss_volume = quantities.sdss_volume(z_min, z_max)

    # Check if the mass bin width is a multiple of bin_width. If it is, add another bin.
    logM_min = round(min(logM), 3) * 1000
    logM_max = round(max(logM), 3) * 1000

    if not (logM_max - logM_min) % (1000 * bin_width):
        upper_lim = max(logM) + bin_width
    else:
        upper_lim = max(logM)

    n_gal_m, mass_bins = np.histogram(
        logM, bins=np.arange(min(logM), upper_lim, bin_width)
    )
    bins_plot = 0.5 * (mass_bins[1:] + mass_bins[:-1])
    mass_func = n_gal_m / (sdss_volume * bin_width)

    return mass_func, bins_plot
