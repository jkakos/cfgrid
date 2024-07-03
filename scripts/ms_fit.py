from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import config
from src import configurations, datalib
from src.figlib import save
from src.protocols import properties
from src.utils import split


def plot_bin_medians(
    ax: plt.Axes,
    x: npt.NDArray,
    y: npt.NDArray,
    lower_xlim: float,
    upper_xlim: float,
    num_medians: int,
    logscale: bool = False,
    kwargs: dict[str, Any] | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Plot the median y value after binning x. Pass in kwargs to
    customize errorbars.

    Parameters
    ----------
    ax : matplotlib axes object
        Axes object to plot medians on.

    x, y : (N,) array
        Data used to calculate bin medians.

    lower_xlim, upper_xlim : float
        Determines x axis binning range.

    num_medians : int
        Number of medians to calculate.

    logscale : bool, optional
        Whether the x axis is a log scale. Used to create bins.

    kwargs : dict, optional
        Kwargs for ax.errorbar() (or for ax.plot() if elinewidth is
        not given).

    """
    import warnings

    if logscale:
        xbins = np.logspace(lower_xlim, upper_xlim, num_medians)
    else:
        xbins = np.linspace(lower_xlim, upper_xlim, num_medians)

    median_list = []
    error_list = []
    for i in range(len(xbins) - 1):
        cond = (x >= xbins[i]) & (x < xbins[i + 1])
        aux = y[cond]

        # Hide runtime warnings that appear in bins with no data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            median_list += [np.median(aux)]
            error_list += [np.std(aux)]

    median = np.array(median_list)
    error = np.array(error_list)
    x_mid = (xbins[:-1] + xbins[1:]) / 2

    if kwargs is not None:
        if 'elinewidth' not in kwargs:
            ax.plot(x_mid, median, **kwargs)
        else:
            ax.errorbar(x_mid, median, yerr=error, **kwargs)
    else:
        ax.errorbar(
            x_mid,
            median,
            yerr=error,
            lw=0,
            elinewidth=1,
            color='r',
            marker='o',
            ms=5,
            mfc='r',
            mec='k',
            mew=1,
            zorder=2,
        )

    return x_mid, median


def main() -> None:
    """
    Plot sSFR vs stellar mass and apply an iterative fitting process to
    define a main sequence. Then find the median sSFR as a function of
    stellar mass for a more refined fit.

    """
    configuration = configurations.MPAConfigVolume1()
    data = configuration.load()
    mass_bins = datalib.volumes.mass_bins_from_volume(configuration.volume)

    mass = properties.Mstar(data).value
    ssfr = properties.SSFR(data).value

    slope, intercept = split.fit_main_seq(mass, ssfr, shift=0.45)
    ms_line = lambda m: slope * m + intercept

    sf_gals = ssfr >= ms_line(mass)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.scatter(mass, ssfr, s=0.1, edgecolor='none', rasterized=True)
    ax.plot(mass, ms_line(mass), color='k', ls='--')

    ms_fit = (
        lambda Ms: config.MS_PSI0
        - np.log10(1 + 10 ** (config.MS_GAMMA * (Ms - config.MS_M0)))
        - Ms
    )
    mx = np.linspace(mass_bins[0], mass_bins[-1], 30)
    ax.plot(mx, ms_fit(mx), color='k')
    ax.plot(mx, ms_fit(mx) - 0.45, color='green')
    ax.plot(mx, ms_fit(mx) - 1, color='red')

    mass_mid, ssfr_median = plot_bin_medians(
        ax, mass[sf_gals], ssfr[sf_gals], min(mass[sf_gals]), max(mass[sf_gals]), 10
    )
    medians = pd.DataFrame.from_dict({'mass': mass_mid, 'ssfr': ssfr_median})
    print(medians)

    ax.set(
        xlabel=r'$\log(M_*/M_\odot)$',
        ylabel=r'$\log({\rm sSFR/yr}^{-1})$',
        xlim=(10, 11.5),
        ylim=(-12.6, -9.2),
    )
    save.savefig(
        fig,
        filename='MS',
        path=config.RESULTS_DIR.joinpath(configuration.dirname, config.MISC_DIRNAME),
        ext='pdf',
        dpi=300,
    )
