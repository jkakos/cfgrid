from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy.typing as npt

from src.cfgrid.constants import COMPARISON_LABELS, COMPARISON_OPTIONS
from src.protocols import properties


DELTA_MS_LABELS = {
    3: ['Q', 'GV', 'SFMS'],
    4: ['Q', 'GV', 'BMS+LMS', 'UMS+HSF'],
    5: ['Q', 'GV', 'BMS', 'LMS+UMS', 'HSF'],
    6: ['Q', 'GV', 'BMS', 'LMS', 'UMS', 'HSF'],
}
GALHALO_LABELS = {
    properties.AccretionRate: r'$\dot{M}_{\rm h}$ Model',
    properties.Concentration: r'$C_{\rm vir}$ Model',
    properties.Mvir: r'$M_{\rm vir}$ Model',
    properties.Spin: 'Spin Model',
    properties.Vmax: r'$V_{\rm max}$ Model',
    properties.Vpeak: r'$V_{\rm peak}$ Model',
    properties.SpecificAccretionRate: r's$\dot{M}_{\rm h}$ Model',
}


def add_legend(
    fig: plt.Figure,
    num_ybins: int,
    colors: npt.NDArray | Sequence[str],
    comparison: COMPARISON_OPTIONS | None = None,
    autocorr: bool = True,
    loc: str | None = None,
) -> plt.Figure:
    """
    Add a legend to 'fig' with delta MS labels based on the number of
    ybins.

    """
    if loc is None:
        loc = 'upper left' if autocorr else 'lower right'

    ncols = num_ybins
    handles = [
        mlines.Line2D([], [], color=c, lw=1.5, label=l)
        for (c, l) in zip(colors, DELTA_MS_LABELS[num_ybins])
    ]

    if comparison is not None:
        ncols += 1
        handles.append(
            mlines.Line2D(
                [],
                [],
                color='k',
                lw=1,
                ls='--',
                label=COMPARISON_LABELS[comparison],
            )
        )

    ax_legend = fig.add_subplot(frameon=False)
    ax_legend.set(xticks=[], yticks=[])
    ax_legend.legend(
        handles=handles,
        loc=loc,
        fancybox=False,
        edgecolor='k',
        labelspacing=0.3,
        columnspacing=1,
        ncols=ncols,
        handlelength=1.2,
        handletextpad=0.4,
        framealpha=0.95,
    )
    return fig


def add_compare_legend(
    fig: plt.Figure,
    loc: str,
    labels: Sequence[str],
    plot_kwargs: Sequence[dict[str, float | str]],
    comparison1: bool = True,
    comparison2: bool = True,
) -> plt.Figure:
    """
    Create a legend for 2-4 different sets of plotted data. All of
    'colors', 'labels', and 'plot_kwargs' should be the same length in
    order of data1, comparison1, data2, comparison2.

    """
    if not len(labels) == len(plot_kwargs):
        raise ValueError("Number of labels and number of kwargs do not match.")

    # 'fmt' is used for ax.errorbar() which should be changed 'marker' for Line2D()
    for i in range(len(plot_kwargs)):
        if plot_kwargs[i].get('fmt') is not None:
            plot_kwargs[i]['marker'] = plot_kwargs[i].pop('fmt')
            if plot_kwargs[i].get('lw') is not None:
                plot_kwargs[i]['lw'] = 0

    num_lines = 2
    handles = []
    handles.append(mlines.Line2D([], [], label=labels[0], **plot_kwargs[0]))

    if comparison1:
        num_lines += 1
        handles.append(
            mlines.Line2D(
                [], [], label=labels[num_lines - 2], **plot_kwargs[num_lines - 2]
            )
        )

    handles.append(
        mlines.Line2D([], [], label=labels[num_lines - 1], **plot_kwargs[num_lines - 1])
    )

    if comparison2:
        num_lines += 1
        handles.append(
            mlines.Line2D(
                [], [], label=labels[num_lines - 1], **plot_kwargs[num_lines - 1]
            )
        )

    for i in range(num_lines, len(plot_kwargs)):
        num_lines += 1
        handles.append(mlines.Line2D([], [], label=labels[i], **plot_kwargs[i]))

    ax_legend = fig.add_subplot(frameon=False)
    ax_legend.set(xticks=[], yticks=[])
    ax_legend.legend(
        handles=handles,
        loc=loc,
        fancybox=False,
        edgecolor='k',
        labelspacing=0.3,
        columnspacing=1,
        ncols=num_lines,
        handlelength=1.25,
        handletextpad=0.4,
        framealpha=0.95,
    )
    return fig


def add_galhalo_legend(
    fig: plt.Figure,
    halo_prop: Sequence[type[properties.Property]],
    plot_kwargs: Sequence[dict[str, float | str]],
    comparison1: COMPARISON_OPTIONS | None = None,
    comparison2: COMPARISON_OPTIONS | None = None,
) -> plt.Figure:
    """
    Add a legend to 'fig' for a galhalo grid that will label different
    halo properties that models are based on, as well as SDSS results
    for comparison.

    """
    labels = ['SDSS']
    if comparison1 is not None:
        labels.append(f'SDSS {COMPARISON_LABELS[comparison1]}')

    for hp in halo_prop:
        labels.append(f'{GALHALO_LABELS[hp]}')

    if comparison2 is not None:
        labels.append(
            f'{GALHALO_LABELS[halo_prop[0]]}' f' {COMPARISON_LABELS[comparison2]}'
        )

    add_compare_legend(
        fig,
        'upper left',
        labels,
        plot_kwargs,
        comparison1 is not None,
        comparison2 is not None,
    )
    return fig


def add_empire_legend(
    fig: plt.Figure,
    plot_kwargs: Sequence[dict[str, float | str]],
    comparison1: COMPARISON_OPTIONS | None = None,
    comparison2: COMPARISON_OPTIONS | None = None,
    # autocorr: bool = True,
) -> plt.Figure:
    """
    Add a legend to 'fig' that labels SDSS and Empire results.

    """
    labels = ['SDSS']
    if comparison1 is not None:
        labels.append(f'SDSS {COMPARISON_LABELS[comparison1]}')

    labels.append('Empire')

    if comparison2 is not None:
        labels.append(f'Empire {COMPARISON_LABELS[comparison2]}')

    add_compare_legend(
        fig,
        'upper left',
        labels,
        plot_kwargs,
        comparison1 is not None,
        comparison2 is not None,
    )
    return fig
