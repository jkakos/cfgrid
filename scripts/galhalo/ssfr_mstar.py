import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from src import galhalo
from src.figlib import save
from src.protocols import properties


GAL = properties.SSFR
HALO = properties.Vpeak
N_SAMPLE = 10_000


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


def plot(
    x: npt.NDArray,
    y: npt.NDArray,
    c: npt.NDArray,
    cbar_label: str,
    save_suffix: str,
    vlim1: tuple[float, float] | None = None,
    vlim2: tuple[float, float] | None = None,
) -> None:
    """
    Make scatter plots of the sSFR-Mstar plane of a galhalo sample with
    points below the median halo property (left panel) and above the
    median halo property (right panel). The color shows the value of
    the halo property and the histograms show its distribution.

    """
    gridspec = {'width_ratios': [1, 0.05, 0.1]}
    fig = plt.figure(figsize=(16, 8))
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=0.06)
    ax1, cax1, hax1 = subfigs[0].subplots(ncols=3, gridspec_kw=gridspec)
    ax2, cax2, hax2 = subfigs[1].subplots(ncols=3, gridspec_kw=gridspec)
    subfigs[0].subplots_adjust(wspace=0)
    subfigs[1].subplots_adjust(wspace=0)
    cmap = 'turbo'

    median = np.median(c)
    below_median = c < median
    above_median = c >= median

    if vlim1 is not None:
        vmin1, vmax1 = vlim1
    else:
        vmin1 = np.percentile(c[below_median], 10)
        vmax1 = np.percentile(c[below_median], 90)

    if vlim2 is not None:
        vmin2, vmax2 = vlim2
    else:
        vmin2 = np.percentile(c[above_median], 10)
        vmax2 = np.percentile(c[above_median], 90)

    sc1 = ax1.scatter(
        x[below_median],
        y[below_median],
        s=5,
        c=c[below_median],
        vmin=vmin1,
        vmax=vmax1,
        edgecolor='none',
        cmap=cmap,
        rasterized=True,
    )
    sc2 = ax2.scatter(
        x[above_median],
        y[above_median],
        s=5,
        c=c[above_median],
        vmin=vmin2,
        vmax=vmax2,
        edgecolor='none',
        cmap=cmap,
        rasterized=True,
    )
    ax1.set(xlabel='Mstar', ylabel='sSFR', xlim=(9.9, 11.6), ylim=(-12.8, -8.7))
    ax2.set(xlabel='Mstar', ylabel='sSFR', xlim=(9.9, 11.6), ylim=(-12.8, -8.7))
    ax1.text(0.05, 0.05, 'Below median', transform=ax1.transAxes)
    ax2.text(0.05, 0.05, 'Above median', transform=ax2.transAxes)

    cbar1 = fig.colorbar(sc1, cax=cax1, use_gridspec=True)
    cbar2 = fig.colorbar(sc2, cax=cax2, use_gridspec=True)
    cbar1.ax.yaxis.set_ticks_position('left')
    cbar2.ax.yaxis.set_ticks_position('left')
    cax1.set(xticklabels=[], yticklabels=[])
    cax2.set(xticklabels=[], yticklabels=[])

    hax1.hist(
        c[below_median],
        bins=np.linspace(vmin1, vmax1, 50),
        orientation='horizontal',
        color='silver',
    )
    hax2.hist(
        c[above_median],
        bins=np.linspace(vmin2, vmax2, 50),
        orientation='horizontal',
        color='silver',
    )
    xmin = min(hax1.get_xlim()[0], hax2.get_xlim()[0])
    xmax = max(hax1.get_xlim()[-1], hax2.get_xlim()[-1])
    hax1.set(
        xlim=(xmin, xmax),
        ylim=(vmin1, vmax1),
        xticklabels=[],
    )
    hax2.set(
        xlim=(xmin, xmax),
        ylim=(vmin2, vmax2),
        xticklabels=[],
    )
    hax1.set_ylabel(cbar_label, rotation=270, labelpad=15)
    hax2.set_ylabel(cbar_label, rotation=270, labelpad=15)
    hax1.yaxis.set_ticks_position('right')
    hax2.yaxis.set_ticks_position('right')
    hax1.yaxis.set_label_position('right')
    hax2.yaxis.set_label_position('right')

    galhalo_path = galhalo.pathing.get_galhalo_sample_path(galhalo.config.CONFIG)
    save.savefig(
        fig,
        f'ssfr_mstar_{save_suffix}',
        path=galhalo_path.parent.joinpath('ssfr_mstar'),
        ext='pdf',
    )


def plot_concentration() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    concentration = properties.Concentration(data).value
    median = np.median(concentration)
    plot(
        properties.Mstar(data).value,
        GAL(data).value,
        concentration,
        r'$\log$(Cvir)',
        'concentration',
        vlim1=(0.85, median),
        vlim2=(median, 1.25),
    )


def plot_delta_v() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    delta_v = np.log10(
        1 + (10 ** properties.Vpeak(data).value - 10 ** properties.Vmax(data).value)
    )
    median = np.median(delta_v)
    plot(
        properties.Mstar(data).value,
        GAL(data).value,
        delta_v,
        r'$\log(1+\Delta$V)',
        'delta_v',
        vlim1=(0.9, median),
        vlim2=(median, 1.8),
    )


def plot_delta_v_over_v() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    delta_v = np.log10(
        2 - (10 ** (properties.Vmax(data).value - properties.Vpeak(data).value))
    )
    plot(
        properties.Mstar(data).value,
        GAL(data).value,
        delta_v,
        r'$\log(1+\Delta$V/Vpeak)',
        'delta_v_over_v',
        vlim1=(0.02, 0.04),
        vlim2=(0.045, 0.075),
    )


def plot_halfmass_scale() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    halfmass_scale = data['Halfmass_Scale'].to_numpy()
    median = np.median(halfmass_scale)
    plot(
        properties.Mstar(data).value,
        GAL(data).value,
        halfmass_scale,
        'Half-mass Scale',
        'halfmass_scale',
        vlim1=(0.34, median),
        vlim2=(median, 0.65),
    )


def plot_scale_of_last_mm() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    scale_last_mm = data['scale_of_last_MM'].to_numpy()
    plot(
        properties.Mstar(data).value,
        GAL(data).value,
        scale_last_mm,
        'Scale of last MM',
        'scale_last_mm',
    )


def plot_specific_accretion_rate() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    spec_acc_rate = properties.SpecificAccretionRate(data).value
    pos_vals = spec_acc_rate > 0
    median = np.median(spec_acc_rate[pos_vals])
    plot(
        properties.Mstar(data[pos_vals]).value,
        GAL(data[pos_vals]).value,
        spec_acc_rate[pos_vals],
        r'$\log$(Spec Acc Rate)',
        'spec_acc_rate',
        vlim2=(median, 1e-10),
    )


def plot_spin() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    spin = properties.Spin(data).value
    median = np.median(spin)
    plot(
        properties.Mstar(data).value,
        GAL(data).value,
        spin,
        r'$\log$(spin)',
        'spin',
        vlim1=(-1.9, median),
    )


def plot_tidal_force() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    tidal_force = data['Tidal_Force'].to_numpy()
    median = np.median(tidal_force)
    plot(
        properties.Mstar(data).value,
        GAL(data).value,
        tidal_force,
        'Tidal Force',
        'tidal_force',
        vlim2=(median, 1.2),
    )


def plot_t_over_u() -> None:
    data = load_data(GAL, HALO, N_SAMPLE)
    t_over_u = np.log10(data['T/|U|'].to_numpy())
    median = np.median(t_over_u)
    plot(
        properties.Mstar(data).value,
        GAL(data).value,
        t_over_u,
        r'$\log$(T/U)',
        't_over_u',
        vlim2=(median, -0.18),
    )
