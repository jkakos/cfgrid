import pathlib
from typing import Callable, Sequence
import warnings

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import numpy.typing as npt
import pandas as pd

import config
from src import complib, configurations, datalib
from src.figlib import colors as fcolors
from src.figlib import FIGWIDTH, LEGEND_FONTSIZE, save
from src.protocols import conditionals, properties
from src.utils import output


def completeness_dir(dataset: str) -> pathlib.Path:
    return config.RESULTS_DIR.joinpath(dataset, 'completeness')


def plot_vol(
    configuration: configurations.MPAConfig,
    completeness_func: Callable[
        [npt.NDArray, npt.NDArray, Sequence[float], Sequence[datalib.volumes.Volume]],
        plt.Figure,
    ],
    filename: str,
) -> None:
    """
    Make a plot of log(M) vs redshift that shows the volumes used for
    completeness.

    """
    mpa = configuration.dataset
    data = mpa().load(completeness=0)
    mass_bins = datalib.volumes.mass_bins_from_volume(configuration.volume)

    mass = data['M_star'].to_numpy()
    z = data['z_obs'].to_numpy()

    fig = completeness_func(z, mass, mass_bins, configuration.volume)
    save.savefig(
        fig,
        filename=filename,
        path=completeness_dir(configuration.dirname),
        ext='pdf',
        dpi=300,
    )


def vol1() -> None:
    """
    Make a plot of log(M) vs redshift that shows the volumes used after
    applying the Yang et al. 2012 volume 1 completeness condition.

    """
    plot_vol(
        configurations.MPAConfigVolume1(),
        complib.plot.plot_vol_completeness,
        'vol1_completeness',
    )


def vol2() -> None:
    """
    Make a plot of log(M) vs redshift that shows the completeness limit
    and volumes used after applying the Yang et al. 2012 volume 2
    completeness condition which uses the van den Bosch et al. 2008
    completeness limit.

    """
    plot_vol(
        configurations.MPAConfigVolume2(),
        complib.plot.plot_vol2_completeness,
        'vol2_completeness',
    )


def vol3() -> None:
    """
    Make a plot of log(M) vs redshift that shows the volumes used after
    applying the Yang et al. 2012 volume 1 completeness condition for a
    single volume.

    """
    plot_vol(
        configurations.MPAConfigVolume3(),
        complib.plot.plot_vol_completeness,
        'vol3_completeness',
    )


def plot_vol_with_hist(configuration: configurations.MPAConfig, filename: str) -> None:
    """
    Make a plot of log(M) vs redshift that shows the volumes used after
    applying the Yang et al. 2012 volume 1 completeness condition. This
    figure includes the redshift histograms used for determining
    redshift limits for the volumes.

    """
    mpa = configuration.dataset
    data = mpa().load(completeness=0)
    mass_bins = datalib.volumes.mass_bins_from_volume(configuration.volume)

    mass = data['M_star'].to_numpy()
    z = data['z_obs'].to_numpy()

    fig = complib.plot.plot_vol_completeness_with_hist(
        z, mass, mass_bins, configuration.volume
    )
    save.savefig(
        fig,
        filename=filename,
        path=completeness_dir(configuration.dirname),
        ext='pdf',
        dpi=300,
    )


def vol1_with_hist() -> None:
    plot_vol_with_hist(configurations.MPAConfigVolume1(), 'vol1_completeness_hist')


def vol2_with_hist() -> None:
    plot_vol_with_hist(configurations.MPAConfigVolume2(), 'vol2_completeness_hist')


def vol3_with_hist() -> None:
    plot_vol_with_hist(configurations.MPAConfigVolume3(), 'vol3_completeness_hist')


def _plot_vol_completeness_density(
    min_mass: float, max_mass: float, mass_bins: Sequence[float]
) -> plt.Figure:
    """
    Make a plot of number density vs redshift for all, central,
    satellite, star-forming, green valley, and quiescent galaxies with
    the ranges of completeness shaded.

    """
    if len(mass_bins) != 6:
        warnings.warn(
            "vol_completeness_density is currently only set up to handle 5 mass bins"
            f" (got {len(mass_bins)-1}). Other mass bin numbers may not create a proper"
            " figure."
        )

    dz = 0.005
    zbins = np.arange(0, 0.3, dz)

    mpa = datalib.MPAJHU(data_cuts=dict(M_star=(min_mass, max_mass)))
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

    labels = ['All', 'Centrals', 'Satellites', 'Quiescent', 'GV', 'SFMS']
    colors = [
        'k',
        *plt.cm.inferno([0.35, 0.75]),
        *fcolors.get_colors(3, delta_MS=True),
    ]
    nrows = 2
    ncols = 3
    if nrows * ncols < len(mass_bins) - 1:
        raise ValueError("Not enough axes to pair with mass bins.")

    fig, axes = plt.subplots(
        figsize=(2 * FIGWIDTH, 1.25 * FIGWIDTH),
        nrows=nrows,
        ncols=ncols,
        constrained_layout=True,
    )
    axes = axes.flatten()
    axes, redshift_lims = complib.plot.plot_vol_completeness_density(
        data,
        data_complete,
        axes,
        mass_bins,
        zbins,
        labels,
        colors,
    )

    # Find sub-volume redshift limits based on individual completenesses
    for mbin_idx, zlims in redshift_lims.items():
        zmin = max(z[0] for z in zlims)
        zmax = min(z[-1] for z in zlims)
        axes[mbin_idx].axvline(x=zmin, color='k', lw=0.5)
        axes[mbin_idx].axvline(x=zmax, color='k', lw=0.5)

    for ax in [axes[0], axes[3]]:
        ax.set(
            ylabel=r'$n_{\rm vol}$ [arb. units]',
        )
    for ax in [axes[3], axes[4]]:
        ax.set(xlabel=r'$z$', xticks=[0.05, 0.10, 0.15, 0.20])
    for ax in axes:
        ax.set(
            xlim=(0.01, 0.24),
            ylim=(10**0.1, 10**3.7),
            yscale='log',
        )

    handles = [
        mlines.Line2D([], [], color=c, label=l) for (c, l) in zip(colors, labels)
    ]
    axes[-1].legend(
        loc='upper left',
        handles=handles,
        fontsize=LEGEND_FONTSIZE + 1,
        handlelength=1.2,
    )

    for ax in [axes[1], axes[2], axes[4], axes[5]]:
        ax.set(yticklabels=[])
    for ax in [axes[0], axes[1], axes[2]]:
        ax.set(xticklabels=[])
    for ax in axes[:3]:
        ax.set(xticks=axes[3].get_xticks())

    # Hide unused subplot
    axes[-1].set(xticks=[], yticks=[])
    axes[-1].minorticks_off()
    for side in ['top', 'bottom', 'left', 'right']:
        axes[-1].spines[side].set_visible(False)

    return fig


def plot_vol_completeness_density(
    configuration: configurations.MPAConfig, filename: str
) -> None:
    mass_bins = datalib.volumes.mass_bins_from_volume(configuration.volume)
    min_mass = min(mass_bins)
    max_mass = max(mass_bins)
    fig = _plot_vol_completeness_density(min_mass, max_mass, mass_bins)
    save.savefig(
        fig,
        filename=filename,
        path=completeness_dir(configuration.dirname),
        ext='pdf',
        dpi=300,
    )


def vol1_completeness_density() -> None:
    plot_vol_completeness_density(
        configurations.MPAConfigVolume1(), 'vol1_completeness_num_dens'
    )


def print_volume_stats(configuration: configurations.MPAConfig) -> None:
    """
    Print out the mass and redshift boundaries of each volume and the
    number of galaxies in each volume.

    """
    data = configuration.load()
    mass_bins = datalib.volumes.mass_bins_from_volume(configuration.volume)
    output.print_volume_table(data, mass_bins)
    print()


def print_volume1_stats() -> None:
    print_volume_stats(configurations.MPAConfigVolume1())


def print_volume2_stats() -> None:
    print_volume_stats(configurations.MPAConfigVolume2())


def print_volume3_stats() -> None:
    print_volume_stats(configurations.MPAConfigVolume3())


def sample_vol2_completeness_points() -> None:
    """
    Sample points from the completeness limit and save them to a table
    that can be used to recreate the completeness limit by fitting.

    """
    z_complete = np.linspace(0.001, 0.3, 100)
    logM_complete = datalib.dataprocess.vdB_completeness(z_complete)
    df = pd.DataFrame.from_dict({'z': z_complete, 'logM': logM_complete})
    df.round(decimals=6).to_csv(
        config.DATA_DIR.joinpath('completeness.dat'), index=False
    )
