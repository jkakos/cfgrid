import warnings
from typing import Mapping, Protocol, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate

import config
import src.cosmo.constants as consts
from src.cosmo.distances import luminosity_distance
from src.cosmo.quantities import sdss_volume


class Volume(Protocol):
    mass_lims: tuple[float, float]
    redshift_lims: tuple[float, float]


def check_log(data: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that mass and SFR are stored as logs.

    """
    try:
        mass = data['M_star']
    except KeyError:
        mass = None
    try:
        sfr = data['SFR']
    except KeyError:
        sfr = None

    if mass is not None and max(mass) > 100:
        data['M_star'] = np.log10(mass)

    if sfr is not None and max(sfr) > 100:
        data['SFR'] = np.log10(sfr)

    return data


def get_mass_bins(
    data_cuts: dict[str, tuple[float, float]] | None, mass_bins: Sequence[float] | None
) -> list[float] | None:
    """
    Combine any lower and upper mass limits defined by data_cuts
    with any given mass bins defined by mass_bins to get a full set
    of mass bins.

    """
    if mass_bins is None and data_cuts is None:
        return list(config.MASS_BINS)

    if mass_bins is not None:
        mass_bins = list(mass_bins)

    if data_cuts is None:
        return mass_bins

    mass_lims_ = data_cuts.get('M_star')
    if mass_lims_ is None:
        return mass_bins

    mass_lims = list(mass_lims_)

    if mass_bins is None or not len(mass_bins):
        return mass_lims

    ret = mass_bins
    if mass_lims[0] != ret[0]:
        if mass_lims[0] > ret[0]:
            raise ValueError(
                "The lower limit of `data_cuts['M_star']` must be <="
                " the lower limit of `mass_bins`"
                f" (got data_cuts['M_star'][0]={mass_lims[0]}"
                f" and mass_bins[0]={mass_bins[0]})."
            )
        ret.insert(0, mass_lims[0])

    if mass_lims[-1] != ret[-1]:
        if mass_lims[-1] < ret[-1]:
            raise ValueError(
                "The upper limit of `data_cuts['M_star']` must be >="
                " the upper limit of `mass_bins`"
                f" (got data_cuts['M_star'][-1]={mass_lims[-1]}"
                f" and mass_bins[-1]={mass_bins[-1]})."
            )
        ret.append(mass_lims[-1])

    return ret


def adjust_cosmology(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust the cosmology using cosmo.constants.H0 and assuming MPA
    has H0 = 70 km/s/Mpc.

    """
    hubble_correction = 2 * np.log10(0.7 / (consts.H0 / 100))
    data['M_star'] = data['M_star'] + hubble_correction

    return data


def apply_cuts(
    data: pd.DataFrame, cuts: Mapping[str, tuple[float, float]] | None
) -> pd.DataFrame:
    """
    Cut the data according to a given parameter and corresponding
    lower and upper limits.

    """
    if cuts is None:
        return data

    data_cuts = []
    for k, v in cuts.items():
        data_cuts.append((data[k] > v[0]) & (data[k] < v[-1]))

    return data[np.logical_and.reduce(data_cuts)]


def cut_sdss_mgs(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cut out the smoothed main sample of the SDSS.

    """
    southern_limit = (data['DEC']) > 0
    western_limit = (data['DEC']) > -2.555556 * (data['RA'] - 131)
    eastern_limit = (data['DEC']) > 1.70909 * (data['RA'] - 235)
    northern_limit = (data['DEC']) < (
        180
        / np.pi
        * np.arcsin(
            (0.93232 * np.sin(np.pi / 180 * (data['RA'] - 95.9)))
            / np.sqrt(1 - (0.93232 * np.cos(np.pi / 180 * (data['RA'] - 95.9))) ** 2)
        )
    )

    within_limits = southern_limit & western_limit & eastern_limit & northern_limit

    return data[within_limits]


def cut_sdss_rects(data: pd.DataFrame) -> pd.DataFrame:
    cuts = pd.read_csv(config.DATA_DIR.joinpath('sdss_cuts.dat'))
    for ra1, ra2, dec1, dec2 in zip(
        cuts['ra_low'], cuts['ra_high'], cuts['dec_low'], cuts['dec_high']
    ):
        cut = (
            (data['RA'] >= ra1)
            & (data['RA'] <= ra2)
            & (data['DEC'] >= dec1)
            & (data['DEC'] <= dec2)
        )

        data = data[~cut]

    return data


def cut_to_volume(
    logM: npt.NDArray[np.float64], z: npt.NDArray[np.float64], volumes: Sequence[Volume]
) -> npt.NDArray[np.bool_]:
    """
    Select galaxies that are bounded by any volume in 'volumes'.

    """
    completeness_cuts = []
    for v in volumes:
        completeness_cuts.append(
            (logM > v.mass_lims[0])
            & (logM < v.mass_lims[-1])
            & (z > v.redshift_lims[0])
            & (z < v.redshift_lims[-1])
        )
    return np.logical_or.reduce(np.array(completeness_cuts))


def make_complete_vol1(
    data: pd.DataFrame,
    mass_bins: Sequence[float] | None = None,
) -> pd.DataFrame:
    """
    Apply completeness condition from Yang et al. 2012
    (volume 1, figure 19).

    """
    from scipy.ndimage import gaussian_filter as gf

    Z_MIN = 0.02
    Z_MAX = 0.2
    COMPLETENESS = 0.5

    z = data['z_obs']
    logM = data['M_star']
    completeness_cond_list = []

    if mass_bins is None:
        mass_bins = [min(logM), max(logM)]

    zbins = np.arange(0, 0.3, 0.005)  # ensure max here > Z_MAX for gaussian filter
    zbin_centers = 0.5 * (zbins[:-1] + zbins[1:])
    volumes = np.array([sdss_volume(z1, z2) for (z1, z2) in zip(zbins[:-1], zbins[1:])])

    for m_low, m_high in zip(mass_bins[:-1], mass_bins[1:]):
        mbin_cond = (logM > m_low) & (logM < m_high)
        z_mbin = z[mbin_cond]
        z_digitize = np.digitize(z_mbin, bins=zbins[1:-1])
        counts, _ = np.histogram(z_digitize, bins=np.arange(0, len(zbins), 1))
        num_density = gf(counts / volumes, 1)
        nmax = max(num_density[(zbin_centers >= Z_MIN) & (zbin_centers <= Z_MAX)])
        nmax_idx = np.argmax(num_density)

        z_min = Z_MIN
        z_max = Z_MAX
        for i in range(nmax_idx, 0, -1):
            if num_density[i] < COMPLETENESS * nmax:
                z_min = zbins[i + 1]
                break

        for i in range(nmax_idx, len(zbins) - 1):
            if num_density[i] < COMPLETENESS * nmax:
                z_max = zbins[i - 1]
                break

        completeness_cond_list.append(
            (z >= z_min) & (z <= z_max) & (logM > m_low) & (logM <= m_high)
        )

    completeness_cond = np.logical_or.reduce(np.array(completeness_cond_list))
    return data[completeness_cond & (data['z_obs'] >= Z_MIN) & (data['z_obs'] <= Z_MAX)]


def make_complete_vol2(
    data: pd.DataFrame,
    mass_bins: Sequence[float] | None = None,
    min_redshifts: Sequence[float] | None = None,
) -> pd.DataFrame:
    """
    Apply completeness condition from van den Bosch et al. 2008.

    """
    z = data['z_obs']
    logM = data['M_star']
    completeness_cond_list = []

    if mass_bins is None:
        mass_bins = [min(logM), max(logM)]

    if min_redshifts is None:
        min_redshifts = [0 for _ in range(len(mass_bins) - 1)]

    if len(min_redshifts) != len(mass_bins) - 1:
        warnings.warn(
            "The number of minimum redshifts does not match the number "
            "of mass bins! The resulting behavior of the completeness "
            "cut(s) may be unexpected."
        )

    for m_low, m_high, min_redshift in zip(
        mass_bins[:-1], mass_bins[1:], min_redshifts
    ):
        max_redshift = interp_mass_to_redshift(
            m_low, H0=consts.H0, Om=consts.Om, Ol=consts.Ol
        )  # type: ignore

        if len(max_redshift) != 1:
            raise ValueError(
                "Something went wrong finding the redshift at a given "
                "mass. More than one value was found. Check data "
                "and/or 'interp_mass_to_redshift' in 'split_utils.py.'"
            )

        completeness_cond_list.append(
            (z > min_redshift)
            & (z < max_redshift[0])
            & (logM > m_low)
            & (logM <= m_high)
        )

    completeness_cond = np.logical_or.reduce(np.array(completeness_cond_list))
    return data[completeness_cond]


def vdB_completeness(
    z: npt.NDArray, H0: float = consts.H0, Om: float = consts.Om, Ol: float = consts.Ol
) -> np.typing.NDArray:
    """
    Completeness limit from Eq A8 in van den Bosch+2008

    """
    lum_dist = luminosity_distance(z, H0=H0, Om=Om, Ol=Ol)
    logM = (
        4.852 + 2.246 * np.log10(lum_dist) + 1.123 * np.log10(1 + z) - 1.186 * z
    ) / (1 - 0.067 * z) + 2 * np.log10(H0 / 100)

    return logM


def interp_mass_to_redshift(
    mass: float, H0: float = consts.H0, Om: float = consts.Om, Ol: float = consts.Ol
) -> list[float]:
    """
    Create an interpolation function to determine redshift limits
    given mass limits for volume completeness.

    """
    z = np.arange(0.01, 0.5, 0.001)

    # Completeness limit from Eq A8 in van den Bosch+2008
    logM = vdB_completeness(z, H0=H0, Om=Om, Ol=Ol)

    # Find the redshift where the difference between the completeness
    # limit and the given mass is zero.
    y = logM - mass
    f = interpolate.InterpolatedUnivariateSpline(z, y)

    return f.roots()
