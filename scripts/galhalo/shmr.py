from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import numpy.typing as npt
import pandas as pd

import config
from src import configurations, galhalo
from src.cosmo import constants as consts
from src.figlib import FIGWIDTH, save
from src.figlib import colors as fcolors
from src.protocols import binning, properties


SHMR_DIR = config.RESULTS_DIR.joinpath(
    galhalo.config.CONFIG.dirname,
    f'{galhalo.config.CONFIG.scatter:.3f}',
    'shmr',
    galhalo.config.CONFIG.centrals[0].label,
)
GAL = properties.SSFR
COLORS_ = fcolors.get_colors(6, delta_MS=True)
COLORS = {
    'qu': COLORS_[0],
    'qgv': COLORS_[0],
    'gv': COLORS_[1],
    'bms': COLORS_[2],
    'lms': COLORS_[3],
    'ums': COLORS_[4],
    'sf': COLORS_[4],
    'hsf': COLORS_[5],
    'sfms': COLORS_[4],
}
DELTA_MS_LABELS = {
    'qu': 'Quiescent',
    'qgv': 'Q+GV',
    'gv': 'GV',
    'bms': 'BMS',
    'lms': 'LMS',
    'ums': 'UMS',
    'sf': 'LMS+UMS',
    'hsf': 'HSF',
    'sfms': 'SFMS',
}
COMP_LABELS = [
    'More et al. 2011',
    'Rodriguez-Puebla et al. 2015',
    'Mandelbaum et al. 2016',
]
COMP_MARKERS = ['^', 's', 'o']
COMP_COLORS = [
    fcolors.lighten('r', 0.1),
    fcolors.lighten('b', 0.25),
    fcolors.lighten('r', 0.1),
    fcolors.lighten('b', 0.25),
    fcolors.lighten('r', 0.1),
    fcolors.lighten('b', 0.25),
]


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


def get_delta_ms_bin_conds(
    delta_ms: npt.NDArray, delta_ms_bins: Sequence[float]
) -> dict[str, npt.NDArray]:
    bin_nums: npt.NDArray = np.digitize(delta_ms, bins=delta_ms_bins)
    d: dict[str, npt.NDArray] = {
        'qu': bin_nums == 0,
        'qgv': np.isin(bin_nums, [0, 1]),
        'gv': bin_nums == 1,
        'bms': bin_nums == 2,
        'lms': bin_nums == 3,
        'ums': bin_nums == 4,
        'sf': np.isin(bin_nums, [3, 4]),
        'hsf': bin_nums == 5,
        'sfms': np.isin(bin_nums, [2, 3, 4, 5]),
    }
    return d


def more_2011_shmr(mh: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    shmr = lambda mh, M0, M1, gamma1, gamma2: M0 * (
        (mh / M1) ** gamma1 / (1 + (mh / M1)) ** (gamma1 - gamma2)
    )
    red = np.log10(shmr(10**mh, 10**10.84, 10**12.18, 3.34, 0.22)) - 2 * np.log10(
        consts.H0 / 100
    )
    blue = np.log10(shmr(10**mh, 10**9.38, 10**11.32, 2.41, 1.12)) - 2 * np.log10(
        consts.H0 / 100
    )

    return red, blue


def rpa_2015_shmr() -> (
    tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
):
    red = pd.read_csv(config.DATA_DIR.joinpath('RPA15_data', 'mhmsR.dat'), sep='\s+')
    blue = pd.read_csv(config.DATA_DIR.joinpath('RPA15_data', 'mhmsB.dat'), sep='\s+')

    h = consts.H0 / 100
    red['logMs'] += 2 * np.log10(0.7 / h)
    blue['logMs'] += 2 * np.log10(0.7 / h)
    red['MeanlogMh'] -= np.log10(h)
    blue['MeanlogMh'] -= np.log10(h)
    red['SD'] -= np.log10(h)
    blue['SD'] -= np.log10(h)

    return (
        red['logMs'].to_numpy(),
        red['MeanlogMh'].to_numpy(),
        red['SD'].to_numpy(),
        blue['logMs'].to_numpy(),
        blue['MeanlogMh'].to_numpy(),
        blue['SD'].to_numpy(),
    )


def mandelbaum_2016_shmr() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    h = consts.H0 / 100
    ms_red = np.array([10.39, 10.70, 10.97, 11.20, 11.38, 11.56]) + 2 * np.log10(
        0.7 / h
    )
    mh_red = np.array([12.17, 12.14, 12.50, 12.89, 13.63, 14.05]) - np.log10(h)
    ms_blue = np.array([10.29, 10.63, 10.94, 11.18, 11.35, 11.54]) + 2 * np.log10(
        0.7 / h
    )
    mh_blue = np.array([11.80, 11.73, 12.15, 12.61, 12.69, 12.79]) - np.log10(h)

    return ms_red, mh_red, ms_blue, mh_blue


def plot_more_2011_shmr(
    axes: np.ndarray[plt.Axes, Any] | Sequence[plt.Axes],
    mh: npt.NDArray,
    mstar_min: float,
    mstar_max: float,
) -> None:
    red, blue = more_2011_shmr(mh)
    for ax in axes:
        ax.scatter(
            red[(red > mstar_min) & (red < mstar_max)],
            mh[(red > mstar_min) & (red < mstar_max)],
            marker=COMP_MARKERS[0],
            s=5,
            facecolor=COMP_COLORS[0],
            edgecolor='k',
            lw=0.5,
        )
        ax.scatter(
            blue[(blue > mstar_min) & (blue < mstar_max)],
            mh[(blue > mstar_min) & (blue < mstar_max)],
            marker=COMP_MARKERS[0],
            s=5,
            facecolor=COMP_COLORS[1],
            edgecolor='k',
            lw=0.5,
        )


def plot_rpa_2015_shmr(
    axes: np.ndarray[plt.Axes, Any] | Sequence[plt.Axes],
    mstar_min: float,
    mstar_max: float,
) -> None:
    ms_red, mh_red, sd_red, ms_blue, mh_blue, sd_blue = rpa_2015_shmr()
    for ax in axes:
        ax.scatter(
            ms_red[(ms_red > mstar_min) & (ms_red < mstar_max)],
            mh_red[(ms_red > mstar_min) & (ms_red < mstar_max)],
            marker=COMP_MARKERS[1],
            s=5,
            facecolor=COMP_COLORS[2],
            edgecolor='k',
            lw=0.5,
        )
        ax.scatter(
            ms_blue[(ms_blue > mstar_min) & (ms_blue < mstar_max)],
            mh_blue[(ms_blue > mstar_min) & (ms_blue < mstar_max)],
            marker=COMP_MARKERS[1],
            s=5,
            facecolor=COMP_COLORS[3],
            edgecolor='k',
            lw=0.5,
        )


def plot_mandelbaum_2016_shmr(
    axes: np.ndarray[plt.Axes, Any] | Sequence[plt.Axes],
    mstar_min: float,
    mstar_max: float,
) -> None:
    ms_red, mh_red, ms_blue, mh_blue = mandelbaum_2016_shmr()
    for ax in axes:
        ax.scatter(
            ms_red[(ms_red > mstar_min) & (ms_red < mstar_max)],
            mh_red[(ms_red > mstar_min) & (ms_red < mstar_max)],
            marker=COMP_MARKERS[2],
            s=5,
            facecolor=COMP_COLORS[4],
            edgecolor='k',
            lw=0.5,
        )
        ax.scatter(
            ms_blue[(ms_blue > mstar_min) & (ms_blue < mstar_max)],
            mh_blue[(ms_blue > mstar_min) & (ms_blue < mstar_max)],
            marker=COMP_MARKERS[2],
            s=5,
            facecolor=COMP_COLORS[5],
            edgecolor='k',
            lw=0.5,
        )


def plot_shmr(
    axes: np.ndarray[plt.Axes, Any],
    mstar: properties.Property,
    mhalo: properties.Property,
    delta_ms1: properties.Property,
    delta_ms2: properties.Property,
    delta_ms3: properties.Property,
    delta_ms_bin_names: Sequence[str],
    ls: tuple[str, str, str],
    min_cts: int = 50,
) -> plt.Axes:
    """
    Plot the inverted SHMR as a function of sSFR for different sSFR models.

    """
    mhalo_avgs: list[dict[str, list[float]]] = [
        {b: [] for b in delta_ms_bin_names} for _ in range(3)
    ]
    mhalo_errs: list[dict[str, list[float]]] = [
        {b: [] for b in delta_ms_bin_names} for _ in range(3)
    ]
    mstar_mid = []
    mstar_bins = np.arange(
        round(min(mstar.value), 2),
        np.nextafter(round(max(mstar.value), 2), np.inf),
        0.1,
    )

    # Identify the sub-samples based on the delta_ms_bin_names
    delta_ms_bins = binning.DeltaMSBins.bins[6]
    bin_conds = [
        get_delta_ms_bin_conds(dms.value, delta_ms_bins)
        for dms in [delta_ms1, delta_ms2, delta_ms3]
    ]

    # Calculate avg Mhalo(sSFR) in Mstar bins
    for mlow, mhigh in zip(mstar_bins[:-1], mstar_bins[1:]):
        mbin_cond = (mstar.value > mlow) & (mstar.value < mhigh)
        mstar_mid.append(np.median([mlow, mhigh]))

        for mhalo_avg, mhalo_err, delta_ms_bin_cond in zip(
            mhalo_avgs, mhalo_errs, bin_conds
        ):
            for delta_ms_bin in delta_ms_bin_names:
                mhalos = mhalo.value[mbin_cond & delta_ms_bin_cond[delta_ms_bin]]
                if len(mhalos) < min_cts:
                    mhalo_avg[delta_ms_bin].append(-99.0)
                    mhalo_err[delta_ms_bin].append(-99.0)
                else:
                    mhalo_avg[delta_ms_bin].append(np.mean(mhalos))
                    mhalo_err[delta_ms_bin].append(np.std(mhalos))

    # Plot results for all the different models
    for ax, mhalo_avg, mhalo_err, ls_ in zip(axes, mhalo_avgs, mhalo_errs, ls):
        for ms_bin in delta_ms_bin_names:
            x = np.array(mstar_mid)
            y = np.array(mhalo_avg[ms_bin])
            yerr = np.array(mhalo_err[ms_bin])
            cond = y != -99.0
            ax.plot(x[cond], y[cond], lw=1, ls=ls_, color=COLORS[ms_bin], zorder=0)
            ax.fill_between(
                x[cond],
                (y - yerr)[cond],
                (y + yerr)[cond],
                lw=0,
                color=COLORS[ms_bin],
                alpha=0.2,
                zorder=0,
            )

    return axes


def shmr() -> None:
    """
    Plot the stellar-to-halo mass relation as Mstar vs Mhalo for
    star-forming and quiescent galaxies.

    """
    delta_ms_bin_names = ['qgv', 'sfms']
    data_accrate = load_data(GAL, properties.AccretionRate, None)
    data_conc = load_data(GAL, properties.Concentration, None)
    data_vpeak = load_data(GAL, properties.Vpeak, None)

    [mstar, mhalo, ssfr_vpeak, ssfr_accrate, ssfr_conc], _ = properties.standardize(
        properties.Mstar(data_vpeak),
        properties.Mvir(data_vpeak),
        properties.DeltaMS(data_vpeak),
        properties.DeltaMS(data_accrate),
        properties.DeltaMS(data_conc),
    )

    fig, axes = plt.subplots(
        figsize=(FIGWIDTH, 1.25 * FIGWIDTH),
        nrows=3,
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': 0},
        constrained_layout=True,
    )

    model_labels = [r'$V_{\rm peak}$', r'$C_{\rm vir}$', r'$\dot{M}_{\rm h}$']
    ls = ('-', '--', '-.')
    axes = plot_shmr(
        axes, mhalo, mstar, ssfr_vpeak, ssfr_conc, ssfr_accrate, delta_ms_bin_names, ls
    )

    # Legend
    color_handles = [
        mlines.Line2D([], [], color=COLORS[b], lw=1, label=DELTA_MS_LABELS[b])
        for b in delta_ms_bin_names
    ]
    style_handles = [
        mlines.Line2D([], [], color='k', ls=ls_, lw=1, label=l)
        for (l, ls_) in zip(model_labels, ls)
    ]
    handles = [color_handles, style_handles]
    for ax, handle in zip(axes, handles):
        ax.legend(
            handles=handle,
            loc='upper left',
            labelspacing=0.3,
            columnspacing=1,
            ncols=1,
            handlelength=1.5,
            handletextpad=0.4,
            framealpha=0.95,
        )

    mstar_label = r'$\langle\log(M_*/M_\odot)\rangle$'
    mhalo_label = r'$\log(M_{\rm vir}/M_\odot)$'
    for ax in axes:
        ax.set(ylabel=mstar_label)
    axes[-1].set_xlabel(mhalo_label)

    save.savefig(fig, 'shmr', SHMR_DIR)
    plt.close('all')


def inverted_shmr() -> None:
    """
    Plot the stellar-to-halo mass relation as Mshalo vs Mstar for
    star-forming and quiescent galaxies.

    """
    delta_ms_bin_names = ['qgv', 'sfms']
    data_accrate = load_data(GAL, properties.AccretionRate, None)
    data_conc = load_data(GAL, properties.Concentration, None)
    data_vpeak = load_data(GAL, properties.Vpeak, None)

    [mstar, mhalo, ssfr_vpeak, ssfr_accrate, ssfr_conc], _ = properties.standardize(
        properties.Mstar(data_vpeak),
        properties.Mvir(data_vpeak),
        properties.DeltaMS(data_vpeak),
        properties.DeltaMS(data_accrate),
        properties.DeltaMS(data_conc),
    )

    fig, axes = plt.subplots(
        figsize=(FIGWIDTH, 1.25 * FIGWIDTH),
        nrows=3,
        sharex=True,
        sharey=True,
        gridspec_kw={'hspace': 0},
        constrained_layout=True,
    )

    model_labels = [r'$V_{\rm peak}$', r'$C_{\rm vir}$', r'$\dot{M}_{\rm h}$']
    ls = ('-', '--', '-.')
    axes = plot_shmr(
        axes, mstar, mhalo, ssfr_vpeak, ssfr_conc, ssfr_accrate, delta_ms_bin_names, ls
    )
    mstar_min = 10  # min(mstar.value)
    mstar_max = 11.5  # max(mstar.value)
    plot_more_2011_shmr(axes, np.linspace(11, 14, 25), mstar_min, mstar_max)
    plot_rpa_2015_shmr(axes, mstar_min, mstar_max)
    plot_mandelbaum_2016_shmr(axes, mstar_min, mstar_max)

    # Legend
    color_handles = [
        mlines.Line2D([], [], color=COLORS[b], lw=1, label=DELTA_MS_LABELS[b])
        for b in delta_ms_bin_names
    ]
    style_handles = [
        mlines.Line2D([], [], color='k', ls=ls_, lw=1, label=l)
        for (l, ls_) in zip(model_labels, ls)
    ]
    comp_handles = [
        mlines.Line2D([], [], color='k', marker=m, ms=2, lw=0, label=l)
        for (l, m) in zip(COMP_LABELS, COMP_MARKERS)
    ]
    handles = [color_handles, style_handles, comp_handles]
    for ax, handle in zip(axes, handles):
        ax.legend(
            handles=handle,
            loc='upper left',
            fancybox=False,
            edgecolor='k',
            fontsize=7,
            labelspacing=0.3,
            columnspacing=1,
            ncols=1,
            handlelength=1.5,
            handletextpad=0.4,
            framealpha=0.95,
        )

    mstar_label = r'$\log(M_*/M_\odot)$'
    mhalo_label = r'$\langle\log(M_{\rm vir}/M_\odot)\rangle$'
    for ax in axes:
        ax.set(ylabel=mhalo_label)
    axes[-1].set_xlabel(mstar_label)

    save.savefig(fig, 'shmr_inverted', SHMR_DIR)
    plt.close('all')


def both() -> None:
    """
    Plot the stellar-to-halo mass relation (left panels) and the
    inverted stellar-to-halo mass relation (right panels) for
    star-forming and quiescent galaxies.

    """
    delta_ms_bin_names = ['qgv', 'sfms']
    data_accrate = load_data(GAL, properties.AccretionRate, None)
    data_conc = load_data(GAL, properties.Concentration, None)
    data_vpeak = load_data(GAL, properties.Vpeak, None)

    [mstar, mhalo, ssfr_vpeak, ssfr_accrate, ssfr_conc], _ = properties.standardize(
        properties.Mstar(data_vpeak),
        properties.Mvir(data_vpeak),
        properties.DeltaMS(data_vpeak),
        properties.DeltaMS(data_accrate),
        properties.DeltaMS(data_conc),
    )

    fig = plt.figure(
        figsize=(1.8 * FIGWIDTH, 1.25 * FIGWIDTH),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(nrows=3, ncols=2)
    axes = np.empty(3, dtype=plt.Axes)
    axes_inv = np.empty(3, dtype=plt.Axes)

    axes[0] = fig.add_subplot(gs[0])
    axes[1] = fig.add_subplot(gs[2], sharex=axes[0], sharey=axes[0])
    axes[2] = fig.add_subplot(gs[4], sharex=axes[0], sharey=axes[0])
    axes_inv[0] = fig.add_subplot(gs[1])
    axes_inv[1] = fig.add_subplot(gs[3], sharex=axes_inv[0], sharey=axes_inv[0])
    axes_inv[2] = fig.add_subplot(gs[5], sharex=axes_inv[0], sharey=axes_inv[0])

    model_labels = [
        r'$V_{\rm peak}$ Model',
        r'$C_{\rm vir}$ Model',
        r'$\dot{M}_{\rm h}$ Model',
    ]
    ls = ('-', '-', '-')  # ('-', '--', '-.')

    axes_inv = plot_shmr(
        axes_inv,
        mstar,
        mhalo,
        ssfr_vpeak,
        ssfr_conc,
        ssfr_accrate,
        delta_ms_bin_names,
        ls,
    )
    lower_mhalo_lim = mhalo.value >= 11.9
    mstar.value = mstar.value[lower_mhalo_lim]
    mhalo.value = mhalo.value[lower_mhalo_lim]
    ssfr_vpeak.value = ssfr_vpeak.value[lower_mhalo_lim]
    ssfr_conc.value = ssfr_conc.value[lower_mhalo_lim]
    ssfr_accrate.value = ssfr_accrate.value[lower_mhalo_lim]
    axes = plot_shmr(
        axes,
        mhalo,
        mstar,
        ssfr_vpeak,
        ssfr_conc,
        ssfr_accrate,
        delta_ms_bin_names,
        ls,
    )

    mstar_min = 10  # min(mstar.value)
    mstar_max = 11.5  # max(mstar.value)
    plot_more_2011_shmr(axes_inv, np.linspace(11, 14, 25), mstar_min, mstar_max)
    plot_rpa_2015_shmr(axes_inv, mstar_min, mstar_max)
    plot_mandelbaum_2016_shmr(axes_inv, mstar_min, mstar_max)

    # Legend
    color_handles = [
        mlines.Line2D([], [], color=COLORS[b], lw=1, label=DELTA_MS_LABELS[b])
        for b in delta_ms_bin_names
    ]
    # style_handles = [
    #     mlines.Line2D([], [], color='k', ls=ls_, lw=1, label=l)
    #     for (l, ls_) in zip(model_labels, ls)
    # ]
    comp_handles = [
        mlines.Line2D([], [], color='k', marker=m, ms=2, lw=0, label=l)
        for (l, m) in zip(COMP_LABELS, COMP_MARKERS)
    ]
    fontsizes = [7, 7]
    handles = [color_handles, comp_handles]
    legend_axes = [axes[0], axes_inv[0]]
    locs = ['lower right', 'upper left']
    for ax, handle, fs, loc in zip(legend_axes, handles, fontsizes, locs):
        ax.legend(
            handles=handle,
            loc=loc,
            fancybox=False,
            edgecolor='k',
            fontsize=fs,
            labelspacing=0.3,
            columnspacing=1,
            ncols=1,
            handlelength=1.5,
            handletextpad=0.4,
            framealpha=0.95,
        )
    for ax, label in zip(axes, model_labels):
        ax.text(
            0.5, 0.97, label, ha='center', va='top', fontsize=7, transform=ax.transAxes
        )

    mstar_label_noninv = r'$\langle\log(M_*/M_\odot)\rangle$'
    mhalo_label_noninv = r'$\log(M_{\rm vir}/M_\odot)$'
    mstar_label_inv = r'$\log(M_*/M_\odot)$'
    mhalo_label_inv = r'$\langle\log(M_{\rm vir}/M_\odot)\rangle$'
    for ax in axes_inv:
        ax.set(ylabel=mhalo_label_inv)
    axes_inv[-1].set_xlabel(mstar_label_inv)

    for ax in axes:
        ax.set(ylim=(9.9, 11.7), ylabel=mstar_label_noninv)
    axes[-1].set_xlabel(mhalo_label_noninv)

    for i in range(2):
        axes[i].tick_params(axis='x', labelbottom=False)
        axes_inv[i].tick_params(axis='x', labelbottom=False)

    save.savefig(fig, f'shmr_and_inverted', SHMR_DIR)
    plt.close('all')


def empire() -> None:
    """
    Plot the stellar-to-halo mass relation for Empire for star-forming
    and quiescent galaxies.

    """
    delta_ms_bin_names = ['qgv', 'sfms']

    empire_config = configurations.EmpireConfigVolume1()
    centrals = empire_config.centrals[0]
    data = empire_config.load()
    data = data[centrals(data).value]
    empire_Mstar = properties.Mstar(data)
    empire_Mstar.value = np.log10(data['Ms_obs'])

    [mstar, mhalo, delta_ms], _ = properties.standardize(
        empire_Mstar,
        properties.Mvir(data),
        properties.DeltaMS(data),
    )
    print('Removing h correction with Mvir for Empire')
    mhalo.value += np.log10(0.678)  # undo h adjustment from properties.Mvir

    fig, ax = plt.subplots(constrained_layout=True)

    mstar_mid = []
    mstar_bins = np.arange(
        round(min(mstar.value), 2),
        np.nextafter(round(max(mstar.value), 2), np.inf),
        0.1,
    )
    mhalo_avg: dict[str, list[float]] = {b: [] for b in delta_ms_bin_names}
    mhalo_err: dict[str, list[float]] = {b: [] for b in delta_ms_bin_names}

    # Identify the sub-samples based on the delta_ms_bin_names
    delta_ms_bins = binning.DeltaMSBins.bins[6]
    delta_ms_bin_cond = get_delta_ms_bin_conds(delta_ms.value, delta_ms_bins)

    # Calculate avg Mhalo(sSFR) in Mstar bins
    for mlow, mhigh in zip(mstar_bins[:-1], mstar_bins[1:]):
        mbin_cond = (mstar.value > mlow) & (mstar.value < mhigh)
        mstar_mid.append(np.median([mlow, mhigh]))

        for delta_ms_bin in delta_ms_bin_names:
            mhalos = mhalo.value[mbin_cond & delta_ms_bin_cond[delta_ms_bin]]
            if len(mhalos) < 50:
                mhalo_avg[delta_ms_bin].append(-99.0)
                mhalo_err[delta_ms_bin].append(-99.0)
            else:
                mhalo_avg[delta_ms_bin].append(np.mean(mhalos))
                mhalo_err[delta_ms_bin].append(np.std(mhalos))

    # Plot results for all the different models
    for ms_bin in delta_ms_bin_names:
        x = np.array(mstar_mid)
        y = np.array(mhalo_avg[ms_bin])
        yerr = np.array(mhalo_err[ms_bin])
        cond = y != -99.0
        ax.plot(x[cond], y[cond], color=COLORS[ms_bin], zorder=0)
        ax.fill_between(
            x[cond],
            (y - yerr)[cond],
            (y + yerr)[cond],
            lw=0,
            color=COLORS[ms_bin],
            alpha=0.2,
            zorder=0,
        )

    mstar_min = 10  # min(mstar.value)
    mstar_max = 11.5  # max(mstar.value)
    plot_more_2011_shmr([ax], np.linspace(11, 14, 25), mstar_min, mstar_max)
    plot_rpa_2015_shmr([ax], mstar_min, mstar_max)
    plot_mandelbaum_2016_shmr([ax], mstar_min, mstar_max)

    # Legend
    color_handles = [
        mlines.Line2D([], [], color=COLORS[b], lw=1, label=DELTA_MS_LABELS[b])
        for b in delta_ms_bin_names
    ]
    comp_handles = [
        mlines.Line2D([], [], color='k', marker=m, ms=2, lw=0, label=l)
        for (l, m) in zip(COMP_LABELS, COMP_MARKERS)
    ]
    handles = color_handles + comp_handles
    ax.legend(
        handles=handles,
        loc='upper left',
        fancybox=False,
        edgecolor='k',
        fontsize=7,
        labelspacing=0.3,
        columnspacing=1,
        ncols=1,
        handlelength=1.5,
        handletextpad=0.4,
        framealpha=0.95,
    )

    mstar_label = r'$\log(M_*/M_\odot)$'
    mhalo_label = r'$\langle\log(M_{\rm vir}/M_\odot)\rangle$'
    ax.set(ylabel=mhalo_label, xlabel=mstar_label)

    path = config.RESULTS_DIR.joinpath(empire_config.dirname, 'shmr', centrals.label)
    save.savefig(fig, f'shmr_inverted', path)
    plt.close('all')


def formation_redshift() -> None:
    """
    Plot the z_form-mass relation with both Mstar and Mhalo for
    star-forming and quiescent galaxies.

    """
    delta_ms_bin_names = ['qgv', 'sfms']
    data_accrate = load_data(GAL, properties.AccretionRate, None)
    data_conc = load_data(GAL, properties.Concentration, None)
    data_vpeak = load_data(GAL, properties.Vpeak, None)

    [
        mstar,
        mhalo,
        zform,
        ssfr_vpeak,
        ssfr_accrate,
        ssfr_conc,
    ], _ = properties.standardize(
        properties.Mstar(data_vpeak),
        properties.Mvir(data_vpeak),
        properties.FormationRedshift(data_vpeak),
        properties.DeltaMS(data_vpeak),
        properties.DeltaMS(data_accrate),
        properties.DeltaMS(data_conc),
    )

    fig = plt.figure(
        figsize=(1.5 * FIGWIDTH, 1.25 * FIGWIDTH),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(nrows=3, ncols=2)
    axes = np.empty(3, dtype=plt.Axes)
    axes_inv = np.empty(3, dtype=plt.Axes)

    axes[0] = fig.add_subplot(gs[0])
    axes[1] = fig.add_subplot(gs[2], sharex=axes[0], sharey=axes[0])
    axes[2] = fig.add_subplot(gs[4], sharex=axes[0], sharey=axes[0])
    axes_inv[0] = fig.add_subplot(gs[1])
    axes_inv[1] = fig.add_subplot(gs[3], sharex=axes_inv[0], sharey=axes_inv[0])
    axes_inv[2] = fig.add_subplot(gs[5], sharex=axes_inv[0], sharey=axes_inv[0])

    model_labels = [
        r'$V_{\rm peak}$ Model',
        r'$C_{\rm vir}$ Model',
        r'$\dot{M}_{\rm h}$ Model',
    ]
    ls = ('-', '-', '-')

    axes_inv = plot_shmr(
        axes_inv,
        mstar,
        zform,
        ssfr_vpeak,
        ssfr_conc,
        ssfr_accrate,
        delta_ms_bin_names,
        ls,
    )
    lower_mhalo_lim = mhalo.value >= 11.9
    mstar.value = mstar.value[lower_mhalo_lim]
    mhalo.value = mhalo.value[lower_mhalo_lim]
    ssfr_vpeak.value = ssfr_vpeak.value[lower_mhalo_lim]
    ssfr_conc.value = ssfr_conc.value[lower_mhalo_lim]
    ssfr_accrate.value = ssfr_accrate.value[lower_mhalo_lim]
    zform.value = zform.value[lower_mhalo_lim]
    axes = plot_shmr(
        axes,
        mhalo,
        zform,
        ssfr_vpeak,
        ssfr_conc,
        ssfr_accrate,
        delta_ms_bin_names,
        ls,
    )

    # Legend
    color_handles = [
        mlines.Line2D([], [], color=COLORS[b], lw=1, label=DELTA_MS_LABELS[b])
        for b in delta_ms_bin_names
    ]
    fontsizes = [7, 7]
    handles = [color_handles]
    legend_axes = [axes[0]]
    locs = ['lower right']
    for ax, handle, fs, loc in zip(legend_axes, handles, fontsizes, locs):
        ax.legend(
            handles=handle,
            loc=loc,
            fancybox=False,
            edgecolor='k',
            fontsize=fs,
            labelspacing=0.3,
            columnspacing=1,
            ncols=1,
            handlelength=1.5,
            handletextpad=0.4,
            framealpha=0.95,
        )
    for ax, label in zip(axes, model_labels):
        ax.text(
            0.5, 0.97, label, ha='center', va='top', fontsize=7, transform=ax.transAxes
        )

    mhalo_label = r'$\log(M_{\rm vir}/M_\odot)$'
    mstar_label = r'$\log(M_*/M_\odot)$'
    zform_label = r'$\langle\log(z_{\rm form})\rangle$'
    for ax in axes_inv:
        ax.set(ylim=(-0.59, 0.44), yticklabels=[])
    axes_inv[-1].set_xlabel(mstar_label)

    for ax in axes:
        ax.set(ylim=(-0.59, 0.44), ylabel=zform_label)
    axes[-1].set_xlabel(mhalo_label)

    for i in range(2):
        axes[i].tick_params(axis='x', labelbottom=False)
        axes_inv[i].tick_params(axis='x', labelbottom=False)

    save.savefig(fig, f'formation_redshift', SHMR_DIR)
    plt.close('all')
