from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import configurations, cosmo, datalib
from src.figlib import colors as fcolors
from src.figlib import save


def plot_gsmf(data: pd.DataFrame, mass_bins: Sequence[float]) -> plt.Figure:
    """
    Plot the GSMF fit from Dragomir et al. 2018 and the calculated GSMF
    from 'data' within 'mass_bins'.

    """
    fig, ax = plt.subplots(constrained_layout=True)
    func_bins = np.arange(mass_bins[0] - 0.2, mass_bins[-1] + 0.2, 0.1)
    ax.plot(
        func_bins,
        cosmo.gsmf.schechter(func_bins, **cosmo.gsmf.PARAMS_FUNC1)
        + cosmo.gsmf.schechter(func_bins, **cosmo.gsmf.PARAMS_FUNC2),
        color='k',
        label='Dragomir et al. 2018 best fit',
        zorder=2,
    )
    colors = fcolors.get_qualitative_colors(len(mass_bins) - 1)

    for mlow, mhigh, c in zip(mass_bins[:-1], mass_bins[1:], colors):
        data_mbin = data.query("M_star > @mlow and M_star < @mhigh")
        mass_func, mass = cosmo.gsmf.calc_gsmf(
            data_mbin['M_star'], min(data_mbin['z_obs']), max(data_mbin['z_obs']), 0.05
        )
        ax.scatter(
            mass,
            mass_func,
            s=15,
            color=c,
            edgecolor='k',
            lw=0.25,
            zorder=3,
            label=fr'$\log(M_*/M_\odot)={mlow}-{mhigh}$',
        )

    ax.set(
        yscale='log',
        ylabel=r'$\phi(M_*)~[h^3{\rm Mpc}^{-3}{\rm dex}^{-1}]$',
        xlabel=r'$\log(M_*/M_{\odot})$',
        xticks=[10, 10.5, 11, 11.5],
    )
    ax.legend(loc='lower left', handlelength=1)
    return fig


def vol(configuration: configurations.MPAConfig, filename: str) -> None:
    data = configuration.load()
    mass_bins = datalib.volumes.mass_bins_from_volume(configuration.volume)
    fig = plot_gsmf(data, mass_bins)
    save.savefig(
        fig,
        filename=filename,
        path=config.RESULTS_DIR.joinpath(configuration.dirname, 'completeness'),
        ext='pdf',
    )


def vol1() -> None:
    """
    Plot the GSMF for volume 1.

    """
    vol(configurations.MPAConfigVolume1(), 'vol1_gsmf')


def vol2() -> None:
    """
    Plot the GSMF for volume 2.

    """
    vol(configurations.MPAConfigVolume2(), 'vol2_gsmf')


def vol3() -> None:
    """
    Plot the GSMF for volume 3.

    """
    vol(configurations.MPAConfigVolume3(), 'vol3_gsmf')
