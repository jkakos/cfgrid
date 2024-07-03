from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import gaussian_filter as gf

from src import datalib
from src.cosmo.quantities import sdss_volume
from src.figlib import FIGWIDTH, LEGEND_FONTSIZE
from src.figlib import colors as fcolors


RED = 'tab:red'
BLUE = 'steelblue'
GREY = 'silver'


def plot_volume(
    ax: plt.Axes,
    subvol_data: pd.DataFrame,
    zlims: tuple[float, float],
    mlims: tuple[float, float],
    last_vol: bool = False,
) -> None:
    """
    Plot a box with colored points inside to denote selected region of
    a volume.

    """
    zmin, zmax = zlims
    mmin, mmax = mlims

    ax.scatter(
        subvol_data['z_obs'],
        subvol_data['M_star'],
        s=0.25,
        color=BLUE,
        rasterized=True,
    )
    if last_vol:
        ax.add_patch(
            patches.Rectangle(
                (zmin, mmin),
                zmax - zmin,
                mmax - mmin,
                color='k',
                ls='-',
                lw=1.25,
                fill=None,
                label='Sub-volumes',
                zorder=2,
            )
        )
    else:
        ax.add_patch(
            patches.Rectangle(
                (zmin, mmin),
                zmax - zmin,
                mmax - mmin,
                color='k',
                ls='-',
                lw=1.25,
                fill=None,
                label=None,
                zorder=2,
            )
        )


def plot_vol_completeness(
    z: npt.NDArray,
    logM: npt.NDArray,
    mass_bins: Sequence[float],
    volume: Sequence[datalib.volumes.Volume],
) -> plt.Figure:
    """
    Plot the completeness limit for volume 1 in Yang et al. 2012
    figure 19.

    """
    fig, ax = plt.subplots(constrained_layout=True)
    data = pd.DataFrame.from_dict({'z_obs': z, 'M_star': logM})
    complete_data = data[datalib.dataprocess.cut_to_volume(logM, z, volume)]

    ax.scatter(z, logM, s=0.25, color='silver', rasterized=True)

    mass_shift = 0.005
    redshift_shift = 0.0004
    for mmin, mmax in zip(mass_bins[:-1], mass_bins[1:]):
        subvol_mass_cut_data = complete_data.query(
            "M_star > (@mmin + @mass_shift) and M_star < (@mmax - @mass_shift)"
        )
        zmin = min(subvol_mass_cut_data['z_obs'])
        zmax = max(subvol_mass_cut_data['z_obs'])
        subvol_data = subvol_mass_cut_data.query(
            "z_obs > (@zmin + @redshift_shift) and z_obs < (@zmax - @redshift_shift)"
        )
        last_vol = mmax == mass_bins[-1]
        plot_volume(ax, subvol_data, (zmin, zmax), (mmin, mmax), last_vol=last_vol)

    ax.set(
        xlabel=r'$z$',
        ylabel=r'$\log(M_*/M_\odot)$',
        xlim=(0.01, 0.24),
        ylim=(9.75, 11.75),
        xticks=[0.05, 0.10, 0.15, 0.20],
        yticks=[10.0, 10.5, 11.0, 11.5],
    )
    ax.legend(loc='lower right')

    return fig


def plot_vol2_completeness(
    z: npt.NDArray,
    logM: npt.NDArray,
    mass_bins: Sequence[float],
    volume: Sequence[datalib.volumes.Volume],
) -> plt.Figure:
    """
    Plot the completeness limit using van den Bosch et al. 2008.

    """
    fig, ax = plt.subplots(constrained_layout=True)
    zmin = 0.02

    z_complete = np.arange(0.001, 0.3, 0.001)
    logM_complete = datalib.dataprocess.vdB_completeness(z_complete)

    mass_redshifts = []
    for mbin in mass_bins[:-1]:
        mass_redshifts.append(datalib.dataprocess.interp_mass_to_redshift(mbin)[0])

    data = pd.DataFrame.from_dict({'z_obs': z, 'M_star': logM})
    complete_data = data[datalib.dataprocess.cut_to_volume(logM, z, volume)]

    ax.scatter(z, logM, s=0.25, color='silver', rasterized=True)
    ax.plot(
        z_complete,
        logM_complete,
        color=RED,
        lw=2,
        label='Completeness Limit',
        zorder=1,
    )

    mass_shift = 0.005
    redshift_shift = 0.0004
    for mmin, mmax, zmax in zip(mass_bins[:-1], mass_bins[1:], mass_redshifts):
        subvol_data = complete_data.query(
            "M_star > (@mmin + @mass_shift) and "
            "M_star < (@mmax - @mass_shift) and "
            "z_obs > (@zmin + @redshift_shift) and "
            "z_obs < (@zmax - @redshift_shift)"
        )
        last_vol = mmax == mass_bins[-1]
        plot_volume(ax, subvol_data, (zmin, zmax), (mmin, mmax), last_vol=last_vol)

    ax.set(
        xlabel=r'$z$',
        ylabel=r'$\log(M_*/M_\odot)$',
        xlim=(0.01, 0.24),
        ylim=(9.75, 11.75),
        xticks=[0.05, 0.10, 0.15, 0.20],
        yticks=[10.0, 10.5, 11.0, 11.5],
    )
    ax.legend(loc='lower right')

    return fig


def plot_vol_completeness_with_hist(
    z: npt.NDArray,
    logM: npt.NDArray,
    mass_bins: Sequence[float],
    volume: Sequence[datalib.volumes.Volume],
) -> plt.Figure:
    """
    Plot the completeness limit for volume 1 in Yang et al. 2012
    figure 19. This figure includes the number densities as a function
    of redshift used for determining redshift limits for the volumes.

    """
    fig, axes = plt.subplots(
        figsize=(2 * FIGWIDTH, 0.8 * FIGWIDTH), ncols=2, constrained_layout=True
    )
    colors = fcolors.get_qualitative_colors(len(mass_bins) - 1)
    ls = '-'
    data = pd.DataFrame.from_dict({'z_obs': z, 'M_star': logM})
    data_complete = data[datalib.dataprocess.cut_to_volume(logM, z, volume)]

    zbins = np.arange(0, 0.3, 0.005)
    zplot = 0.5 * (zbins[:-1] + zbins[1:])
    vols = np.array([sdss_volume(z1, z2) for (z1, z2) in zip(zbins[:-1], zbins[1:])])

    axes[0].scatter(z, logM, s=0.1, color='silver', edgecolor='none', rasterized=True)

    # Shift limits so points don't show outside the edges of the volume box
    mass_shift = 0.005
    redshift_shift = 0.0004
    hist_scales = [10**i for i in range(len(mass_bins) - 1)][::-1]

    for i, (mmin, mmax) in enumerate(zip(mass_bins[:-1], mass_bins[1:])):
        subvol_mass_cut_data = data.query(
            "M_star > (@mmin + @mass_shift) and M_star < (@mmax - @mass_shift)"
        )
        subvol_mass_cut_data_complete = data_complete.query(
            "M_star > (@mmin + @mass_shift) and M_star < (@mmax - @mass_shift)"
        )
        zmin = min(subvol_mass_cut_data_complete['z_obs'])
        zmax = max(subvol_mass_cut_data_complete['z_obs'])
        subvol_data = subvol_mass_cut_data_complete.query(
            "z_obs > (@zmin + @redshift_shift) and z_obs < (@zmax - @redshift_shift)"
        )

        axes[0].scatter(
            subvol_data['z_obs'],
            subvol_data['M_star'],
            s=0.1,
            color=colors[i],
            edgecolor='none',
            rasterized=True,
        )
        label = 'Sub-volumes' if mmax == mass_bins[-1] else None
        axes[0].add_patch(
            patches.Rectangle(
                (zmin, mmin),
                zmax - zmin,
                mmax - mmin,
                color='k',
                ls=ls,
                lw=1.25,
                fill=None,
                label=label,
                zorder=2,
            )
        )

        z_digitize = np.digitize(subvol_mass_cut_data['z_obs'], bins=zbins[1:-1])
        counts, _ = np.histogram(z_digitize, bins=np.arange(0, len(zbins), 1))
        num_density = gf(counts / vols, 1)
        y = num_density * hist_scales[i]

        axes[1].plot(
            zplot, y, color=fcolors.lighten(colors[i], -0.1), zorder=-10 + i + 1
        )
        z_complete_cond = (zplot >= zmin) & (zplot <= zmax)
        axes[1].fill_between(
            zplot[z_complete_cond],
            0,
            y[z_complete_cond],
            edgecolor=colors[i],
            facecolor='none',
            hatch='/' * 15,
            zorder=-10 + i,
        )

    axes[0].set(
        xlabel=r'$z$',
        ylabel=r'$\log(M_*/M_\odot)$',
        xlim=(0.01, 0.24),
        ylim=(9.75, 11.75),
        xticks=[0.05, 0.10, 0.15, 0.20],
        yticks=[10.0, 10.5, 11.0, 11.5],
    )
    axes[0].legend(loc='lower right')
    axes[1].set(
        xlabel=r'$z$',
        ylabel=r'$n_{\rm vol}(z)~[h^3{\rm Mpc}^{-3}]$',
        xlim=(0.01, 0.24),
        xticks=[0.05, 0.10, 0.15, 0.20],
        yscale='log',
    )
    return fig


def plot_vol_completeness_density(
    data: Sequence[pd.DataFrame],
    data_complete: Sequence[pd.DataFrame],
    axes: Sequence[plt.Axes],
    mass_bins: Sequence[float],
    zbins: npt.NDArray[np.float64],
    labels: Sequence[str],
    colors: Sequence[str | npt.NDArray],
) -> tuple[plt.Figure, dict[int, list[tuple[float, float]]]]:
    """
    Make a plot of number density vs redshift that shows the volumes
    used after applying the Yang et al. 2012 volume 1 completeness
    condition.

    """
    hist_scales = [10 ** (0.5 * (i + 1)) for i in range(len(data))][::-1]
    zplot = 0.5 * (zbins[:-1] + zbins[1:])
    vols = np.array([sdss_volume(z1, z2) for (z1, z2) in zip(zbins[:-1], zbins[1:])])
    redshift_lims: dict[int, list[tuple[float, float]]] = {
        i: [] for i in range(len(mass_bins) - 1)
    }

    for i, d in enumerate(data):
        for j, (mlow, mhigh, ax) in enumerate(zip(mass_bins[:-1], mass_bins[1:], axes)):
            d_mbin_comp = data_complete[i].query("M_star > @mlow and M_star < @mhigh")
            d_mbin = d.query("M_star > @mlow and M_star < @mhigh")
            zmin = min(d_mbin_comp['z_obs'])
            zmax = max(d_mbin_comp['z_obs'])
            z_digitize = np.digitize(d_mbin['z_obs'], bins=zbins[1:-1])
            counts, _ = np.histogram(z_digitize, bins=np.arange(0, len(zbins), 1))
            num_density = gf(counts / vols, 1)

            if i == 0:
                ax.text(
                    0.95,
                    0.95,
                    fr'${mlow}<\log(M_*/M_\odot)<{mhigh}$',
                    fontsize=LEGEND_FONTSIZE,
                    ha='right',
                    va='top',
                    transform=ax.transAxes,
                    bbox=dict(edgecolor='k', alpha=0.95, facecolor='white'),
                )

            z_limit = (zplot >= zmin) & (zplot <= zmax)
            redshift_lims[j].append((min(zplot[z_limit]), max(zplot[z_limit])))
            y = num_density * hist_scales[i] / (max(num_density[z_limit]))
            ax.plot(zplot, y, color=colors[i], label=labels[i])
            ax.fill_between(
                zplot[z_limit],
                0,
                y[z_limit],
                edgecolor=colors[i],
                facecolor='none',
                hatch='/' * 15,
            )

    return axes, redshift_lims
