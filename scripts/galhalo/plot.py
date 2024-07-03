from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import cfgrid, galhalo
from src.figlib import colors as fcolors
from src.figlib import save
from src.protocols import binning, properties


GAL = properties.SSFR
HALO = properties.Vpeak
N_SAMPLE = 20_000


def load_data(
    gal: properties.Property | type[properties.Property],
    halo: properties.Property | type[properties.Property],
    n_sample: int | None = None,
) -> pd.DataFrame:
    """
    Load galhalo data and merge in 'gal' results based on 'halo' model.
    Return only 'n_sample' random rows from the data if given.

    """
    halo_config = galhalo.config.CONFIG_TYPE()
    halo_config.set_galhalo_props(gal.file_label, halo.file_label)
    data = halo_config.load()
    data = data.dropna(subset=[gal.label])
    data = data[halo_config.centrals[0](data).value]

    if n_sample is not None:
        data = data.sample(n=n_sample, replace=False)

    return data


def scatter_ssfr() -> None:
    halo_config = galhalo.config.CONFIG
    mass_bins = halo_config.mass_bins
    cond_type = halo_config.centrals[0]
    num_ybins = 2
    num_xbins = int(np.ceil((len(mass_bins) - 1) / num_ybins))
    colors = np.array(fcolors.get_colors(6, delta_MS=True))
    dms_bins = binning.DeltaMSBins.bins[6]
    sample_results_path = galhalo.pathing.get_galhalo_sample_path(halo_config)

    for (gal, halo), _ in list(galhalo.config.PAIRINGS.items()):
        print(gal.file_label, halo.file_label)
        halo_config = galhalo.config.CONFIG_TYPE()
        halo_config.set_galhalo_props(gal.file_label, halo.file_label)
        data = halo_config.load()
        cond_obj = cond_type(data)
        filtered_data = data[cond_obj.value]

        mstar = properties.Mstar(filtered_data)
        x_property = halo(filtered_data)
        y_property = gal(filtered_data)
        y_property.value += mstar.value

        # Get good x, y
        [x, y, m], all_good = properties.standardize(x_property, y_property, mstar)
        data_good = filtered_data.loc[all_good]
        delta_ms = properties.DeltaMS(data_good)
        dms_bin = np.digitize(delta_ms.value, bins=dms_bins)

        if not isinstance(x, properties.Mvir) or isinstance(x, properties.Mpeak):
            x.value = np.log10(x.value)

        fig, axes = cfgrid.plot.get_grid(
            num_xbins, num_ybins, ylabel=gal.full_label, height=4, left=0.6
        )
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i == 5:
                break

            xcond = (m.value > mass_bins[i]) & (m.value < mass_bins[i + 1])
            ax.scatter(
                x.value[xcond],
                y.value[xcond],
                s=1,
                color=colors[dms_bin[xcond]],
                edgecolor='none',
                rasterized=True,
            )
            ax.text(
                0.97,
                0.97,
                fr'$\log(M_*/M_\odot)={np.round(np.median(mass_bins[i : i+2]), 2)}$',
                ha='right',
                va='top',
                transform=ax.transAxes,
                fontsize=7,
            )

        axes[-1].scatter(
            x.value,
            y.value,
            s=1,
            color=colors[dms_bin],
            edgecolor='none',
            rasterized=True,
        )

        for ax in axes[3:]:
            ax.set(xlabel=x.full_label)

        save.savefig(
            fig,
            filename=f'{gal.file_label}_{halo.file_label}',
            path=sample_results_path.parent.joinpath('scatter'),
            ext='pdf',
            dpi=300,
        )
        plt.close('all')


def plot_halo_halo(
    x: properties.Property,
    y: properties.Property,
    c: properties.Property,
) -> None:
    gridspec = {'width_ratios': [1, 0.05]}
    fig, (ax, cax) = plt.subplots(
        ncols=2, gridspec_kw=gridspec, constrained_layout=True
    )

    sc = ax.scatter(
        x.value,
        y.value,
        s=1,
        c=c.value,
        edgecolor='none',
        cmap='turbo_r',
        rasterized=True,
    )
    xlabel = x.file_label
    ylabel = y.file_label
    clabel = c.file_label
    ax.set(xlabel=xlabel, ylabel=ylabel)
    cbar = fig.colorbar(sc, cax=cax, use_gridspec=True)
    cbar.set_label(clabel)

    sample_results_path = galhalo.pathing.get_galhalo_sample_path(galhalo.config.CONFIG)
    save.savefig(
        fig,
        f'{xlabel}_{ylabel}_{clabel}',
        sample_results_path.parent.joinpath('scatter_combos'),
    )
    plt.close('all')


def plot_halo_halo_quarters(
    x: properties.Property,
    y: properties.Property,
    c: properties.Property,
) -> None:
    """
    Plot scatter plots broken into 4 percentile bins of the color
    property.

    """
    fig = plt.figure(constrained_layout=True)
    gridspec = fig.add_gridspec(
        nrows=2, ncols=3, width_ratios=[1, 1, 0.075], height_ratios=[0.5, 0.5]
    )
    ax1 = fig.add_subplot(gridspec[0, 0])
    ax2 = fig.add_subplot(gridspec[0, 1])  # , sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gridspec[1, 0])  # , sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gridspec[1, 1])  # , sharex=ax1, sharey=ax1)
    cax = fig.add_subplot(gridspec[:, 2])
    axes = [ax1, ax2, ax3, ax4]
    percentiles = [0, 25, 50, 75, 100]

    vmin = min(c.value)
    vmax = max(c.value)

    if any(
        [
            isinstance(x, properties.SSFR),
            isinstance(y, properties.SSFR),
            isinstance(c, properties.SSFR),
        ]
    ):
        pass
    else:
        return

    for ax, p1, p2 in zip(axes, percentiles[:-1], percentiles[1:]):
        pcond = (c.value > np.percentile(c.value, p1)) & (
            c.value <= np.percentile(c.value, p2)
        )
        ax.scatter(
            x.value[~pcond],
            y.value[~pcond],
            s=1,
            c='silver',
            edgecolor='none',
            rasterized=True,
        )
        sc = ax.scatter(
            x.value[pcond],
            y.value[pcond],
            s=1,
            c=c.value[pcond],
            edgecolor='none',
            cmap='turbo_r',
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

    xlabel = x.file_label
    ylabel = y.file_label
    clabel = c.file_label
    ax1.set(xticklabels=[], ylabel=ylabel)
    ax2.set(xticklabels=[], yticklabels=[])
    ax3.set(xlabel=xlabel, ylabel=ylabel)
    ax4.set(xlabel=xlabel, yticklabels=[])
    ax1.yaxis.label.set_fontsize(8)
    ax3.xaxis.label.set_fontsize(8)
    ax3.yaxis.label.set_fontsize(8)
    ax4.xaxis.label.set_fontsize(8)
    cbar = fig.colorbar(sc, cax=cax, use_gridspec=True)
    cbar.set_label(clabel, fontsize=8)

    sample_results_path = galhalo.pathing.get_galhalo_sample_path(galhalo.config.CONFIG)
    save.savefig(
        fig,
        f'{xlabel}_{ylabel}_{clabel}',
        sample_results_path.parent.joinpath('scatter_combo_quarters'),
    )
    plt.close('all')


def _plot_loop(func: Callable, n_sample: int | None = None) -> None:
    """
    Loop through all x, y, z combinations of 'all_properties' and make
    scatter plots.

    """
    data = load_data(GAL, HALO, n_sample)
    all_properties = [
        properties.Mstar,
        properties.SSFR,
        properties.Vpeak,
        # properties.Vmax,
        properties.Mvir,
        properties.Concentration,
        properties.AccretionRate,
        # properties.Mpeak,
        properties.DeltaMS,
    ]
    for p1 in all_properties:
        for p2 in all_properties:
            if p2 == p1:
                continue

            for p3 in all_properties:
                if p3 == p2 or p3 == p1:
                    continue

                [x, y, c], _ = properties.standardize(
                    p1(data),
                    p2(data),
                    p3(data),
                )
                func(x, y, c=c)


def plot_all() -> None:
    _plot_loop(plot_halo_halo, N_SAMPLE)


def plot_quarters() -> None:
    _plot_loop(plot_halo_halo_quarters)
