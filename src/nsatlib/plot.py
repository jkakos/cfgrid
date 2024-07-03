from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import numpy.typing as npt
import pandas as pd

from src import configurations, nsatlib
from src.figlib import colors as fcolors
from src.figlib import FIGWIDTH, LEGEND_FONTSIZE, save
from src.protocols import properties


GALHALO_LABELS: dict[type[properties.Property], str] = {
    properties.AccretionRate: r'$\dot{M}_{\rm h}$ Model',
    properties.Concentration: r'$C_{\rm vir}$ Model',
    properties.Mvir: r'$M_{\rm vir}$ Model',
    properties.Vmax: r'$V_{\rm max}$ Model',
    properties.Vpeak: r'$V_{\rm peak}$ Model',
    properties.SpecificAccretionRate: r's$\dot{M}_{\rm h}$ Model',
}


def get_line_values(
    m_star: npt.NDArray[np.float64],
    data: pd.DataFrame,
    df_label: str,
    min_cts: int = 0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Extract Nsat and Nsat error as a function of stellar mass from
    'data'. Remove any results that correspond to bins with fewer than
    'min_cts' data points.

    """
    y = data[f'{nsatlib.constants.RESULTS_LABELS["Ns"]}_{df_label}'].to_numpy()
    yerr = data[f'{nsatlib.constants.RESULTS_LABELS["err_Ns"]}_{df_label}'].to_numpy()

    cond = data[f'{nsatlib.constants.RESULTS_LABELS["n_gals"]}_{df_label}'] > min_cts
    m = m_star[cond]
    y = y[cond]
    yerr = yerr[cond]

    return m, y, yerr


def plot_line(
    ax: plt.Axes,
    m_star: npt.NDArray[np.float64],
    data: pd.DataFrame,
    df_label: str,
    color: str | npt.NDArray[np.float64],
    min_cts: int = 0,
    log: bool = True,
    no_err: bool = False,
    mass_range: tuple[float, float] | None = None,
    **kwargs,
) -> plt.Line2D:
    """
    Plot a line with a shaded region to denote errors. Filter out
    points that correspond to statistics calculated with fewer than
    'min_cts' number of data points.

    """
    m, y, yerr = get_line_values(m_star, data, df_label, min_cts)

    if mass_range is not None:
        mass_cond = ((m >= mass_range[0]) | np.isclose(m, mass_range[0])) & (
            (m <= mass_range[-1]) | np.isclose(m, mass_range[-1])
        )
        m = m[mass_cond]
        y = y[mass_cond]
        yerr = yerr[mass_cond]

    if log:
        if not no_err:
            ax.fill_between(
                m, np.log10(y - yerr), np.log10(y + yerr), color=color, lw=0, alpha=0.2
            )

        (line,) = ax.plot(m, np.log10(y), color=color, **kwargs)
    else:
        if not no_err:
            ax.fill_between(m, (y - yerr), (y + yerr), color=color, lw=0, alpha=0.2)

        (line,) = ax.plot(m, y, color=color, **kwargs)

    return line


def plot_nsat(min_cts: int = 50) -> None:
    """
    Plot Nsat as a function of delta MS.

    """
    configuration = nsatlib.config.CONFIG
    colors = fcolors.get_colors(6, delta_MS=True)
    lw = 1.5

    for cen_cond in configuration.centrals:
        load_filename = nsatlib.names.get_nsat_filename(configuration)
        data = nsatlib.pathing.load_nsat(configuration, cen_cond.label)
        m_star = data['logMs'].to_numpy()

        fig, ax = plt.subplots(constrained_layout=True)
        plot_line(ax, m_star, data, 'all', 'k', min_cts, lw=1, ls='--', label='All')
        plot_line(ax, m_star, data, 'Q', colors[0], min_cts, lw=lw, label='Q')
        plot_line(ax, m_star, data, 'GV', colors[1], min_cts, lw=1, ls='-', label='GV')
        plot_line(ax, m_star, data, 'SFMS', colors[4], min_cts, lw=lw, label='LMS+UMS')

        ax.set(
            xlabel=r'$\log(M_*/M_\odot)$',
            ylabel=r'$\log(\langle N_{\rm sat}\rangle)$',
            xticks=[10.0, 10.5, 11.0, 11.5],
        )
        ax.legend(labelspacing=0.3)
        save.savefig(
            fig,
            filename=load_filename.rsplit('.', maxsplit=1)[0],
            path=nsatlib.pathing.get_nsat_path(configuration, cen_cond.label),
            ext='pdf',
            dpi=300,
        )


def plot_nsat_two_panel(min_cts: int = 50) -> None:
    """
    Plot Nsat in a left panel and Nsat for all the main sequence
    sub-samples in a right panel.

    """
    configuration = nsatlib.config.CONFIG
    colors = fcolors.get_colors(6, delta_MS=True)
    lw = 1.5

    for cen_cond in configuration.centrals:
        load_filename = nsatlib.names.get_nsat_filename(configuration)
        data = nsatlib.pathing.load_nsat(configuration, cen_cond.label)
        m_star = data['logMs'].to_numpy()

        fig, (ax, ax2) = plt.subplots(
            figsize=(2 * FIGWIDTH, 0.8 * FIGWIDTH),
            ncols=2,
            constrained_layout=True,
        )
        plot_line(ax, m_star, data, 'all', 'k', min_cts, lw=1, ls='--', label='All')
        plot_line(ax, m_star, data, 'Q', colors[0], min_cts, lw=lw, label='Q')
        plot_line(ax, m_star, data, 'GV', colors[1], min_cts, lw=1, ls='-', label='GV')
        plot_line(ax, m_star, data, 'SFMS', colors[4], min_cts, lw=lw, label='LMS+UMS')

        plot_line(ax2, m_star, data, 'all', 'grey', min_cts, lw=1, ls='-', label='All')
        plot_line(ax2, m_star, data, 'BMS', colors[2], min_cts, lw=1, label='BMS')
        plot_line(
            ax2, m_star, data, 'LMS', colors[3], min_cts, ls='-.', lw=1, label='LMS'
        )
        plot_line(
            ax2, m_star, data, 'UMS', colors[4], min_cts, ls='--', lw=1, label='UMS'
        )
        plot_line(ax2, m_star, data, 'HSF', colors[5], min_cts, lw=1, label='HSF')

        ax.set(
            xlabel=r'$\log(M_*/M_\odot)$',
            ylabel=r'$\log(\langle N_{\rm sat}\rangle)$',
            xticks=[10.0, 10.5, 11.0, 11.5],
        )
        ax.legend(labelspacing=0.3)
        ax2.set(
            xlabel=r'$\log(M_*/M_\odot)$',
            ylabel=r'$\log(\langle N_{\rm sat}\rangle)$',
            xlim=(9.9, 11.4),
            ylim=(-2.7, 0.5),
            xticks=[10.0, 10.5, 11.0],
        )
        ax2.legend(labelspacing=0.3)
        save.savefig(
            fig,
            filename=f"{load_filename.rsplit('.', maxsplit=1)[0]}_two_panel",
            path=nsatlib.pathing.get_nsat_path(configuration, cen_cond.label),
            ext='pdf',
            dpi=300,
        )


def plot_nsat_group_cats(min_cts: int = 50) -> None:
    """
    Plot Nsat for each of the different MPA group catalogs.

    """
    configuration = configurations.MPAConfigVolume3()
    centrals = configuration.centrals
    filename = nsatlib.names.get_nsat_filename(configuration)
    data = [nsatlib.pathing.load_nsat(configuration, c.label) for c in centrals]
    m_star = data[0]['logMs'].to_numpy()
    colors = fcolors.get_colors(6, delta_MS=True)
    legend_labels = [c.legend_label for c in centrals]
    linestyles = ['-', '--', '-.']

    fig, ax = plt.subplots(constrained_layout=True)
    for d, label, ls in zip(data, legend_labels, linestyles):
        plot_line(ax, m_star, d, 'Q', colors[0], min_cts, lw=1, ls=ls, label=label)
        plot_line(ax, m_star, d, 'GV', colors[1], min_cts, lw=1, ls=ls)
        plot_line(ax, m_star, d, 'SFMS', colors[4], min_cts, lw=1, ls=ls)

    ax.set(xlabel=r'$\log(M_*/M_\odot)$', ylabel=r'$\log(N_{\rm sat})$', ylim=(-2, 2))
    ax.legend(labelspacing=0.3)
    save.savefig(
        fig,
        filename=f"{filename.rsplit('.', maxsplit=1)[0]}_groups",
        path=nsatlib.pathing.get_nsat_path(configuration, centrals[0].label),
        ext='pdf',
        dpi=300,
    )


def plot_nsat_comparison(
    halo_props: Sequence[type[properties.Property]], min_cts: int = 50
) -> None:
    """
    Plot Nsat for MPA and different halo property models in BP.

    """
    mpa_config = configurations.MPAConfigVolume3()
    centrals = mpa_config.centrals[0]
    mpa_data = nsatlib.pathing.load_nsat(mpa_config, centrals.label)

    m_star = mpa_data['logMs'].to_numpy()
    colors = fcolors.get_colors(6, delta_MS=True)
    linestyles = ['-', '-', '-']

    fig, axes = plt.subplots(
        figsize=(0.8 * FIGWIDTH, 1.25 * FIGWIDTH),
        nrows=len(halo_props),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    handles = []
    model_handles = []

    obs_kwargs = dict(
        lw=0,
        elinewidth=0.5,
        marker='o',
        markersize=3,
        markeredgewidth=0.5,
        capsize=1,
        capthick=0.5,
        color='k',
    )
    for ax in axes:
        m, y, yerr = get_line_values(m_star, mpa_data, 'all', min_cts=min_cts)
        ax.errorbar(m, y, yerr=yerr, markerfacecolor='grey', **obs_kwargs)

        m, y, yerr = get_line_values(m_star, mpa_data, 'Q', min_cts=min_cts)
        ax.errorbar(m, y, yerr=yerr, markerfacecolor=colors[0], **obs_kwargs)

        m, y, yerr = get_line_values(m_star, mpa_data, 'SF', min_cts=min_cts)
        ax.errorbar(m, y, yerr=yerr, markerfacecolor=colors[4], **obs_kwargs)

    handles.append(mlines.Line2D([], [], color='k', lw=1, label='All'))
    handles.append(mlines.Line2D([], [], color=colors[0], lw=1, label='Quiescent'))
    handles.append(mlines.Line2D([], [], color=colors[4], lw=1, label='LMS+UMS'))
    model_handles.append(
        mlines.Line2D(
            [],
            [],
            marker='o',
            markersize=3,
            markerfacecolor='grey',
            markeredgecolor='k',
            markeredgewidth=0.5,
            lw=0,
            label='SDSS',
        )
    )

    for ax, halo, ls in zip(axes, halo_props, linestyles):
        bp_config = configurations.BPConfigVolume3()
        bp_config.set_galhalo_props(properties.SSFR.file_label, halo.file_label)
        bp_data = nsatlib.pathing.load_nsat(bp_config, bp_config.centrals[0].label)

        plot_line(ax, m_star, bp_data, 'all', 'k', min_cts, log=False, lw=1, ls=ls)
        plot_line(ax, m_star, bp_data, 'Q', colors[0], min_cts, log=False, lw=1, ls=ls)
        plot_line(
            ax, m_star, bp_data, 'SFMS', colors[4], min_cts, log=False, lw=1, ls=ls
        )
        ax.text(
            0.5,
            0.97,
            GALHALO_LABELS[halo],
            ha='center',
            va='top',
            fontsize=LEGEND_FONTSIZE,
            transform=ax.transAxes,
        )

    for ax in axes:
        ax.set(
            yscale='log',
            xlim=(9.9, 11.9),
            ylim=(10 ** (-2.95), 10**1.6),
            ylabel=r'$\langle N_{\rm sat}\rangle$',
        )

    axes[-1].set(xlabel=r'$\log(M_*/{\rm M}_\odot)$')
    axes[2].legend(
        handles=handles,
        loc='lower right',
        labelspacing=0.3,
        columnspacing=1,
        ncols=1,
        handlelength=1.25,
        handletextpad=0.4,
        framealpha=0.95,
    )
    axes[1].legend(
        handles=model_handles,
        loc='lower right',
        labelspacing=0.3,
        columnspacing=1,
        ncols=1,
        handlelength=1.25,
        handletextpad=0.4,
        framealpha=0.95,
    )
    save.savefig(
        fig,
        filename=f'nsat_bp_mpa_comparison',
        path=nsatlib.pathing.get_nsat_path(bp_config, centrals.label),
        ext='pdf',
    )


def plot_nsat_empire(mpa_min_cts: int = 100, empire_min_cts: int = 25) -> None:
    """
    Plot Nsat for MPA and Empire.

    """
    mpa_config = configurations.MPAConfigVolume3()
    centrals = mpa_config.centrals[0]
    mpa_data = nsatlib.pathing.load_nsat(mpa_config, centrals.label)

    empire_config = configurations.EmpireConfigVolume3()
    empire_data = nsatlib.pathing.load_nsat(empire_config, centrals.label)

    m_star = mpa_data['logMs'].to_numpy()
    colors = fcolors.get_colors(6, delta_MS=True)
    lw = 1

    fig, ax = plt.subplots(constrained_layout=True)

    lookup_labels = ['all', 'Q', 'GV', 'SFMS', 'HSF']
    legend_labels = ['All', 'Q', 'GV', 'LMS+UMS', 'HSF']
    colors_list = ['grey', colors[0], colors[1], colors[4], colors[5]]
    lws = [1, lw, lw, lw, 1]
    lss = ['--', '-', '-', '-', '-']
    obs_kwargs = dict(
        lw=0,
        elinewidth=0.5,
        marker='o',
        # markerfacecolor='none',
        markersize=3,
        markeredgewidth=0.5,
        capsize=1,
        capthick=0.5,
        color='k',
    )

    for lookup_label, color in zip(lookup_labels, colors_list):
        m, y, yerr = get_line_values(
            m_star, mpa_data, lookup_label, min_cts=mpa_min_cts
        )
        ax.errorbar(m, y, yerr=yerr, markerfacecolor=color, **obs_kwargs)

    colors_list[0] = 'k'
    for lookup_label, legend_label, color, lw_, ls in zip(
        lookup_labels, legend_labels, colors_list, lws, lss
    ):
        plot_line(
            ax,
            m_star,
            empire_data,
            lookup_label,
            color,
            empire_min_cts,
            lw=lw_,
            ls=ls,
            label=legend_label,
            log=False,
        )

    ax.set(
        xlabel=r'$\log(M_*/M_\odot)$',
        ylabel=r'$\langle N_{\rm sat}\rangle$',
        yscale='log',
    )
    ax.legend(labelspacing=0.3)

    save.savefig(
        fig,
        filename=f'nsat_empire_mpa_comparison',
        path=nsatlib.pathing.get_nsat_path(empire_config, centrals.label),
        ext='pdf',
        dpi=300,
    )
