from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from src.protocols import binning


def percentile_bins(
    x: npt.NDArray,
    n: int,
) -> npt.NDArray:
    """
    Determine the bin edges that divide x into n equal percentile bins.

    """
    base_percentile = 100 / n
    pbins = base_percentile * np.arange(1, n)

    return np.percentile(x, pbins)


def get_ybins(
    x: npt.NDArray,
    y: npt.NDArray,
    xbins: Sequence[float],
    ybin_strategy: type[binning.BinningStrategy] | Sequence[float],
    num_ybins: int,
) -> dict[int, list[float]]:
    """
    Evaluate 'ybin_strategy' as a function of 'x'. If only a single
    xbin is desired, pass 'xbins' as an empty list. The return type
    will be a dictionary where each key is an integer corresponding to
    the xbin and each value will be a list of values where 'y' is
    divided into bins.

    """
    ybins: dict[int, list[float]] = {}
    num_xbins = len(xbins) + 1

    if isinstance(ybin_strategy, Sequence):
        if not all(isinstance(ybin, (int, float, np.number)) for ybin in ybin_strategy):
            raise ValueError(
                "If 'ybin_strategy' is given as a sequence, it must "
                "contain all ints or floats."
            )
        elif len(ybin_strategy) + 1 != num_ybins:
            raise ValueError(
                "The inferred number of bins in 'ybin_strategy' does not match"
                f" 'num_ybins' (got {len(ybin_strategy)+1=}) and {num_ybins=}).'"
            )
        ybins = {i: list(ybin_strategy) for i in range(num_xbins)}

    elif isinstance(ybin_strategy, binning.BinningStrategy):
        xcoords = np.digitize(x, xbins)
        ybins = {
            i: ybin_strategy(y[xcoords == i]).get_bins(num_ybins)
            for i in range(num_xbins)
        }

    else:
        raise ValueError(f"Invalid input for 'ybin_strategy' ({ybin_strategy} given).")

    return ybins


def get_grid_coords(
    x: npt.NDArray,
    y: npt.NDArray,
    xbins: Sequence[float] | npt.NDArray[np.float64],
    ybins: Mapping[int, Sequence[float]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Create a 2D grid in x and y using xbins and ybins. The endpoints of
    x and y should not be included in xbins and ybins. The return values
    are an array that contains an (xbin, ybin) coordinate pair for each
    point (x, y) and a dictionary that holds the ybins within each xbin.

    """
    xcoords = np.digitize(x, xbins)

    if all(ybin == ybins[0] for ybin in ybins.values()):
        ycoords = np.digitize(y, ybins[0])
        return xcoords, ycoords

    ycoords = np.full(len(y), -1, dtype=int)
    num_xbins = len(xbins) + 1

    for i in range(num_xbins):
        selection = xcoords == i
        ycoords[selection] = np.digitize(y[selection], bins=ybins[i])

    if any(ycoords == -1):
        raise ValueError('A y-coordinate has not been properly set.')

    return xcoords, ycoords


def fit_main_seq(
    mass: npt.NDArray,
    ssfr: npt.NDArray,
    slope0: float = -0.1,
    intercept0: float = -1.0,
    shift: float = 0.5,
    min_ssfr: float | None = None,
):
    """
    Recursively find the main sequence split by applying the
    following steps:
        1. Fit a straight line to mass and ssfr.
        2. Lower the intercept of the line by <shift> dex.
        3. Define points above the line as the new data set.
        4. Fit a straight line to the new data set.
        5. Repeat steps 2-4 until the slope and intercept of the
           line is stable within 1%.

    Parameters
    ----------

    mass, ssfr : array
        Input data to fit given as logs.

    slope0, intercept0 : float
        Prior values of the line parameters that the new line
        parameters will be tested against for stability. These
        are only used once the recursion has begun, so they do
        not need to be provided initially.

    shift : float
        How much to shift the line by vertically when running
        the algorithm above.

    min_ssfr : float
        If given, the data will be cut initially such that
        ssfr >= min_ssfr.

    Returns
    -------

    slope, intercept : int or float
        The final, stable results of the fit line parameters.

    """
    if min_ssfr is not None:
        ssfr_cond = ssfr >= min_ssfr
        mass = mass[ssfr_cond]
        ssfr = ssfr[ssfr_cond]

    # Fit the points and enforce the slope to be <= 0
    fit = lambda x, m, b: -1 * abs(m) * x + b
    (slope, intercept), *_ = curve_fit(fit, mass, ssfr, p0=[slope0, intercept0 + shift])

    slope = -1 * abs(slope)
    intercept -= shift

    if abs(slope) < 1e-3:
        slope = 0

    # If the slope is 0, only check for changes in the intercept
    if slope == 0:
        delta_slope = 0
    else:
        delta_slope = 1 - slope / slope0

    delta_intercept = 1 - intercept / intercept0

    # Exit if the line parameters are within 1%
    if abs(delta_slope) < 0.01 and abs(delta_intercept) < 0.01:
        return slope, intercept

    line = lambda x: slope * x + intercept

    new_points = ssfr > line(mass)
    new_mass = mass[new_points]
    new_ssfr = ssfr[new_points]

    new_slope, new_intercept = fit_main_seq(
        new_mass, new_ssfr, slope0=slope, intercept0=intercept, min_ssfr=None
    )

    return new_slope, new_intercept
