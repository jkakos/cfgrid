import copy
import pathlib
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import numpy.typing as npt
import pandas as pd

from src import cfgrid
from src.figlib import FIGWIDTH, grid, legend, LEGEND_FONTSIZE, mask, save
from src.figlib import colors as fcolors
from src.protocols import binning, properties
from src.tpcf.utils import calc_ratio_errors
from src.utils import pathing, split


PLANE_YLIMS = {
    properties.SSFR: (-12.6, -9.2),
    properties.Size: (-0.4, 1.8),
    properties.TForm: (9.3, 10.15),
}
COUNTS_LABEL_OFFSET = {
    properties.SSFR: 0.05,
    properties.Size: 0.02,
    properties.TForm: 0.01,
}


def get_grid_compare_plot_kwargs() -> tuple[dict[str, float | str], ...]:
    colors = fcolors.get_colors(4, delta_MS=True)
    points_kwargs: dict[str, float | str] = dict(
        color='grey',
        fmt='o',
        markeredgecolor='k',
        markeredgewidth=0.25,
        markersize=3,
        lw=0.5,
        zorder=2,
    )
    points_comp_kwargs: dict[str, float | str] = dict(
        color=colors[-2],
        fmt='o',
        markeredgecolor='k',
        markeredgewidth=0.25,
        markersize=3,
        lw=0.5,
        zorder=2,
    )
    line_kwargs: dict[str, float | str] = dict(color='k', ls='-', lw=1.0, zorder=1)
    line_comp_kwargs: dict[str, float | str] = dict(
        color=colors[-2], ls='--', lw=1, zorder=1
    )

    return points_kwargs, points_comp_kwargs, line_kwargs, line_comp_kwargs


def get_grid(
    num_xbins: int,
    num_ybins: int,
    ylabel: str | None = None,
    prop_label: str | None = None,
    width: float | None = None,
    height: float | None = None,
    left: float | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Set up grid figure and size specifications.

    """
    fig_width, fig_height = grid.get_grid_size(num_xbins, num_ybins)
    if width is not None:
        fig_width = width
    if height is not None:
        fig_height = height

    cf_grid = grid.Grid(
        figsize=(fig_width, fig_height),
        nrows=num_ybins,
        ncols=num_xbins,
        xlabel=cfgrid.constants.RP_LABEL,
        ylabel=None,
        constrained_layout=False,
    )
    fig, axes = cf_grid.fig, cf_grid.ax

    if prop_label is not None:
        fig_left, fig_right, fig_top, fig_bottom = grid.get_grid_adjustments(
            fig_width, fig_height, right_label=True
        )
    else:
        fig_left, fig_right, fig_top, fig_bottom = grid.get_grid_adjustments(
            fig_width, fig_height, right_label=False
        )

    if left is not None:
        fig_left = left / fig_width

    fig.subplots_adjust(
        left=fig_left,
        right=fig_right,
        bottom=fig_bottom,
        top=fig_top,
        wspace=0,
        hspace=0,
    )

    if ylabel is not None:
        grid.add_spanning_ylabel(
            fig, fig_height * fig_top, fig_height * fig_bottom, ylabel
        )

    if prop_label is not None:
        arrow_height = (
            (0.95 - 0.05 * num_xbins) * fig_height * (1 - (fig_bottom + (1 - fig_top)))
        )
        grid.add_right_vertical_arrow(
            fig, fig_height * fig_top, fig_height * fig_bottom, prop_label, arrow_height
        )

    return fig, axes


def get_wp_label(rp_scale: bool = False, autocorr: bool = True) -> str:
    if rp_scale:
        if autocorr:
            label = r'$r_{\rm p}w_{\rm a}(r_{\rm p})~[h^{-2}{\rm Mpc}^{2}]$'
        else:
            label = r'$r_{\rm p}w_{\rm c}(r_{\rm p})~[h^{-2}{\rm Mpc}^{2}]$'
    else:
        if autocorr:
            label = r'$w_{\rm a}(r_{\rm p})~[h^{-1}{\rm Mpc}]$'
        else:
            label = r'$w_{\rm c}(r_{\rm p})~[h^{-1}{\rm Mpc}]$'

    return label


def add_column_label(
    fig: plt.Figure, ax: plt.Axes, col_label: str, col: int, col_bins: Sequence[float]
) -> None:
    """
    Add a text label to the top of a column in the form of
    'col_label = median(col_bins)' for that column.

    """
    fig_height = fig.get_size_inches()[1]
    ax_height = fig_height * ax.get_position().height
    ax.text(
        0.5,
        (ax_height + 0.02) / ax_height,
        fr'{col_label}$={np.round(np.median(col_bins[col : col+2]), 2)}$',
        ha='center',
        va='bottom',
        transform=ax.transAxes,
        fontsize=7,
    )


def get_y_logticks(ax: plt.Axes) -> npt.NDArray[np.int64]:
    """
    Set the yticks for a log scale.

    """
    ylow, yhigh = ax.get_ylim()
    ylowtick = np.ceil(np.log10(ylow))
    yhightick = np.floor(np.log10(yhigh))
    yticks = 10 ** np.arange(ylowtick, yhightick + 1, 1)

    return yticks


def mask_cond(wp: npt.NDArray, wp_err: npt.NDArray, weight: float) -> npt.NDArray:
    return (weight * np.abs(wp) < wp_err) | (wp <= 0)


def plot_grid(
    path: pathlib.Path,
    results_filename: str,
    prop: type[properties.Property],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    rp_scale: bool = False,
    autocorr: bool = True,
    comp_filename: str | None = None,
) -> None:
    """
    Plot a grid of correlation functions.

    """
    # Get results path
    results_path = pathing.get_results_path_from_path(path)

    # Load results
    tpcf_results = cfgrid.calc.load_results(results_path, results_filename)

    # Load any comparison results
    if comp_filename is not None:
        comparison = cfgrid.names.get_comp_from_filename(comp_filename)
        comp_results = cfgrid.calc.load_results(results_path, comp_filename)

    # Need to handle delta_MS when getting colors
    delta_MS = prop == properties.DeltaMS
    colors = fcolors.get_colors(num_ybins, delta_MS=delta_MS)
    ylabel = get_wp_label(rp_scale=rp_scale, autocorr=autocorr)

    # Set up figure and size specifications
    fig, axes = get_grid(num_xbins, num_ybins, ylabel=ylabel, prop_label=prop.label)

    rp = tpcf_results[cfgrid.constants.CF_BIN_LABELS['rp']].to_numpy()
    cf_bins = cfgrid.calc.get_tpcf_bins(centers=False)
    mask_weight = 0.95
    min_ys = []
    max_ys = []

    if rp_scale:
        scale = rp
    else:
        scale = np.ones(len(rp))

    for col in range(num_xbins):
        for row in range(num_ybins):
            row_to_ax_idx = (num_ybins - 1) - row
            if num_ybins == 1:
                suffix = cfgrid.names.get_results_suffix(col)
            else:
                suffix = cfgrid.names.get_results_suffix(col, row)

            wp, wp_err = cfgrid.calc.get_stats(
                tpcf_results, suffix, mean=True, err=True
            )
            rp_ = mask.mask_data(rp, mask_cond(wp, wp_err, mask_weight))

            # Plot wp
            axes[row_to_ax_idx, col].fill_between(
                rp_,
                (wp - wp_err) * scale,
                (wp + wp_err) * scale,
                color=colors[row],
                ls='-',
                lw=0,
                alpha=0.2,
            )
            axes[row_to_ax_idx, col].plot(
                rp_,
                wp * scale,
                color=colors[row],
                ls='-',
                lw=1.5,
                label=prop.label,
            )

            if comp_filename is not None:
                comp_suffix = cfgrid.names.get_results_suffix(col)
                wp_comp, wp_comp_err = cfgrid.calc.get_stats(
                    comp_results, comp_suffix, mean=True, err=True
                )
                rp_comp = mask.mask_data(
                    rp, mask_cond(wp_comp, wp_comp_err, mask_weight)
                )
                axes[row_to_ax_idx, col].errorbar(
                    rp_comp,
                    wp_comp * scale,
                    yerr=wp_comp_err * scale,
                    color='k',
                    ls='--',
                    lw=1,
                    label=cfgrid.constants.COMPARISON_LABELS[comparison],
                )

            axes[row_to_ax_idx, col].set_yscale('log')
            wps = wp * scale
            if col == 0:
                min_ys.append(min(wps[wps > 0]) / 2)

            try:
                max_ys.append(1.2 * max(wps[wps > 0]))
            except ValueError:  # empty array wps
                pass

        # Add column label
        add_column_label(fig, axes[0, col], col_label, col, xbins)

    # Add legend
    fig = legend.add_legend(fig, num_ybins, colors, comparison, autocorr)

    for ax in axes.flatten():
        # Set axes ylims
        ax.set(
            xscale='log',
            yscale='log',
            xlim=(cf_bins[0], cf_bins[-1]),
            ylim=(4, 700),
        )

        # Set axes ticks
        yticks = get_y_logticks(ax)
        ax.set_xticks(ticks=[1, 10])
        ax.set_yticks(ticks=yticks)

    # Set the figure file name
    fig_filename = results_filename.split('.')[0]
    if comp_filename is not None:
        fig_filename = f'{fig_filename}_{comparison}'
    if not rp_scale:
        fig_filename = f'{fig_filename}_norp'

    save.savefig(fig, filename=fig_filename, path=path)
    plt.close('all')


def _grid_compare(
    path: Sequence[pathlib.Path],
    results_filename: Sequence[str],
    prop: Sequence[type[properties.Property]],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    rp_scale: bool = False,
    autocorr: bool = True,
    comp_filename: tuple[str | None, str | None] = (None, None),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a grid comparing multiple correlation functions.

    """
    # Get results path
    results_path = [pathing.get_results_path_from_path(p) for p in path]

    # Load results
    tpcf_results = [
        cfgrid.calc.load_results(rpath, rfile)
        for (rpath, rfile) in zip(results_path, results_filename)
    ]

    # Load any comparison results
    if comp_filename is not None:
        comp_results = [
            cfgrid.calc.load_results(rpath, cfile) if cfile is not None else None
            for (rpath, cfile) in zip(results_path, comp_filename)
        ]

    (
        points_kwargs,
        points_comp_kwargs,
        line_kwargs_,
        line_comp_kwargs,
    ) = get_grid_compare_plot_kwargs()
    ylabel = get_wp_label(rp_scale=rp_scale, autocorr=autocorr)

    # Add additional lines
    if len(path) == 2:
        line_kwargs = [line_kwargs_]
    elif len(path) > 2:
        line_kwargs = []
        colors = [
            'k',
            fcolors.LIGHT_BLUE,
            fcolors.MAGENTA,
            fcolors.YELLOW,
            fcolors.GREEN,
            fcolors.DARK_BLUE,
        ]
        linestyles = ['-.', '--', '-']
        for i in range(len(path) - 1):
            _line_kwargs = copy.deepcopy(line_kwargs_)
            _line_kwargs['color'] = colors[i]
            _line_kwargs['ls'] = linestyles[i]
            line_kwargs.append(_line_kwargs)

    # Set up figure and size specifications
    fig, axes = get_grid(num_xbins, num_ybins, ylabel=ylabel, prop_label=prop[0].label)

    rp = tpcf_results[0][cfgrid.constants.CF_BIN_LABELS['rp']].to_numpy()
    cf_bins = cfgrid.calc.get_tpcf_bins(centers=False)
    mask_weight = 1

    if rp_scale:
        scale = rp
    else:
        scale = np.ones(len(rp))

    for col in range(num_xbins):
        for row in range(num_ybins):
            row_to_ax_idx = (num_ybins - 1) - row
            if num_ybins == 1:
                suffix = cfgrid.names.get_results_suffix(col)
            else:
                suffix = cfgrid.names.get_results_suffix(col, row)

            for i in range(len(path)):
                wp, wp_err = cfgrid.calc.get_stats(
                    tpcf_results[i], suffix, mean=True, err=True
                )
                if i == 0:
                    rp_ = mask.mask_data(
                        rp, mask_cond(wp, wp_err, mask_weight), mask_iso=False
                    )
                elif i >= 1:
                    rp_ = mask.mask_data(rp, mask_cond(wp, wp_err, mask_weight))

                all_nan = np.isnan(rp).all()

                # Plot wp
                if i == 0 and not all_nan:
                    axes[row_to_ax_idx, col].errorbar(
                        rp_, wp * scale, yerr=wp_err * scale, **points_kwargs
                    )
                elif i >= 1 and not all_nan:
                    axes[row_to_ax_idx, col].fill_between(
                        rp_,
                        (wp - wp_err) * scale,
                        (wp + wp_err) * scale,
                        color=line_kwargs[i - 1]['color'],
                        ls='-',
                        lw=0,
                        alpha=0.2,
                    )
                    axes[row_to_ax_idx, col].plot(rp_, wp * scale, **line_kwargs[i - 1])

                    kwargs = copy.deepcopy(line_kwargs[i - 1])
                    kwargs['ls'] = ':'
                    rp_2 = mask.mask_data(rp, mask_cond(wp, wp_err, 2 * mask_weight))
                    axes[row_to_ax_idx, col].plot(rp_2, wp * scale, **kwargs)

                if i > 1:
                    continue

                if comp_filename[i] is not None:
                    comp_suffix = cfgrid.names.get_results_suffix(col)
                    comp_result = comp_results[i]
                    assert comp_result is not None
                    wp_comp, wp_comp_err = cfgrid.calc.get_stats(
                        comp_result, comp_suffix, mean=True, err=True
                    )
                    if i == 0:
                        rp_comp = mask.mask_data(
                            rp,
                            mask_cond(wp_comp, wp_comp_err, mask_weight),
                            mask_iso=False,
                        )
                    elif i == 1:
                        rp_comp = mask.mask_data(
                            rp, mask_cond(wp_comp, wp_comp_err, mask_weight)
                        )
                    all_nan_comp = np.isnan(rp_comp).all()

                    if i == 0 and not all_nan_comp:
                        axes[row_to_ax_idx, col].errorbar(
                            rp_comp,
                            wp_comp * scale,
                            yerr=wp_comp_err * scale,
                            **points_comp_kwargs,
                        )
                    if i == 1 and not all_nan_comp:
                        axes[row_to_ax_idx, col].errorbar(
                            rp_comp,
                            wp_comp * scale,
                            yerr=wp_comp_err * scale,
                            **line_comp_kwargs,
                        )

        # Add column label
        add_column_label(fig, axes[0, col], col_label, col, xbins)

    for ax in axes.flatten():
        if autocorr:
            ylim = (1.001, 1900)
        else:
            ylim = (3, 3000)

        ax.set(xscale='log', yscale='log', xlim=(cf_bins[0], cf_bins[-1]), ylim=ylim)

        # Set axes ticks
        yticks = get_y_logticks(ax)
        ax.set_xticks(ticks=[1, 10])
        ax.set_yticks(ticks=yticks)

    return fig, axes


def grid_galhalo(
    path: tuple[pathlib.Path, pathlib.Path],
    results_filename: tuple[str, str],
    prop: tuple[type[properties.Property], type[properties.Property]],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    rp_scale: bool = False,
    autocorr: bool = True,
    comp_filename: tuple[str | None, str | None] = (None, None),
    fig_filename: str | None = None,
) -> None:
    """
    Plot a grid comparing multiple correlation functions.

    """
    fig, axes = _grid_compare(
        path,
        results_filename,
        prop,
        num_xbins,
        num_ybins,
        xbins,
        col_label,
        rp_scale,
        autocorr,
        comp_filename,
    )

    # Get any comparisons
    if comp_filename is not None:
        comparison: list[cfgrid.constants.COMPARISON_OPTIONS | None] = [
            cfgrid.names.get_comp_from_filename(cf) if cf is not None else None
            for cf in comp_filename
        ]

    (
        points_kwargs,
        points_comp_kwargs,
        line_kwargs,
        line_comp_kwargs,
    ) = get_grid_compare_plot_kwargs()

    plot_kwargs = [points_kwargs]
    if comparison[0] is not None:
        plot_kwargs.append(points_comp_kwargs)

    plot_kwargs.append(line_kwargs)
    if comparison[1] is not None:
        plot_kwargs.append(line_comp_kwargs)

    # Add legend
    fig = legend.add_galhalo_legend(
        fig,
        prop[1:],
        plot_kwargs,
        comparison[0],
        comparison[1],
    )

    if autocorr:
        ylim = (1.001, 1900)
    else:
        ylim = (3, 9000)

    axes[0, 0].set_ylim(ylim)

    # Set the figure file name
    if fig_filename is not None:
        fig_filename = f"{fig_filename}_{results_filename[-1].split('.')[0]}"
    else:
        fig_filename = results_filename[-1].split('.')[0]

    if comp_filename is not None:
        fig_filename = f'{fig_filename}_{comparison[0]}'
    if not rp_scale:
        fig_filename = f'{fig_filename}_norp'

    save.savefig(fig, filename=fig_filename, path=path[1])
    plt.close('all')


def grid_empire(
    path: tuple[pathlib.Path, pathlib.Path],
    results_filename: tuple[str, str],
    prop: tuple[type[properties.Property], type[properties.Property]],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    rp_scale: bool = False,
    autocorr: bool = True,
    comp_filename: tuple[str | None, str | None] = (None, None),
    fig_filename: str | None = None,
) -> None:
    """
    Plot a grid comparing multiple correlation functions.

    """
    fig, axes = _grid_compare(
        path,
        results_filename,
        prop,
        num_xbins,
        num_ybins,
        xbins,
        col_label,
        rp_scale,
        autocorr,
        comp_filename,
    )

    # Load any comparison results
    if comp_filename is not None:
        comparison: list[cfgrid.constants.COMPARISON_OPTIONS | None] = [
            cfgrid.names.get_comp_from_filename(cf) if cf is not None else None
            for cf in comp_filename
        ]

    (
        points_kwargs,
        points_comp_kwargs,
        line_kwargs,
        line_comp_kwargs,
    ) = get_grid_compare_plot_kwargs()

    plot_kwargs = [points_kwargs]
    if comparison[0] is not None:
        plot_kwargs.append(points_comp_kwargs)

    plot_kwargs.append(line_kwargs)
    if comparison[1] is not None:
        plot_kwargs.append(line_comp_kwargs)

    # Add legend
    fig = legend.add_empire_legend(
        fig,
        plot_kwargs,
        comparison[0],
        comparison[1],
    )

    if autocorr:
        ylim = (1.001, 1900)
    else:
        ylim = (3, 9000)

    axes[0, 0].set_ylim(ylim)

    # Set the figure file name
    if fig_filename is not None:
        fig_filename = f"{fig_filename}_{results_filename[-1].split('.')[0]}"
    else:
        fig_filename = results_filename[-1].split('.')[0]

    if comp_filename is not None:
        fig_filename = f'{fig_filename}_{comparison[0]}'
    if not rp_scale:
        fig_filename = f'{fig_filename}_norp'

    save.savefig(fig, filename=fig_filename, path=path[1])
    plt.close('all')


def condensed_grid(
    path: pathlib.Path,
    results_filename: str,
    prop: type[properties.Property],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    rp_scale: bool = False,
    autocorr: bool = True,
    comp_filename: str | None = None,
) -> None:
    """
    Plot a grid of correlation functions.

    """
    # Get results path
    results_path = pathing.get_results_path_from_path(path)

    # Load results
    tpcf_results = cfgrid.calc.load_results(results_path, results_filename)

    # Load any comparison results
    if comp_filename is not None:
        comparison = cfgrid.names.get_comp_from_filename(comp_filename)
        comp_results = cfgrid.calc.load_results(results_path, comp_filename)

    # Need to handle delta_MS when getting colors
    delta_MS = prop == properties.DeltaMS
    colors = fcolors.get_colors(num_ybins, delta_MS=delta_MS)
    ylabel = get_wp_label(rp_scale=rp_scale, autocorr=autocorr)

    # Set up figure and size specifications
    fig, axes = get_grid(num_xbins, 1, ylabel=ylabel)
    try:
        axes = axes.flatten()
    except AttributeError:
        axes = np.array([axes])

    rp = tpcf_results[cfgrid.constants.CF_BIN_LABELS['rp']].to_numpy()
    cf_bins = cfgrid.calc.get_tpcf_bins(centers=False)
    mask_weight = 0.95

    if rp_scale:
        scale = rp
    else:
        scale = np.ones(len(rp))

    for col in range(num_xbins):
        for row in range(num_ybins):
            if num_ybins == 1:
                suffix = cfgrid.names.get_results_suffix(col)
            else:
                suffix = cfgrid.names.get_results_suffix(col, row)

            wp, wp_err = cfgrid.calc.get_stats(
                tpcf_results, suffix, mean=True, err=True
            )
            rp_ = mask.mask_data(rp, mask_cond(wp, wp_err, mask_weight))

            legend_label = None if row > 0 else prop.label

            # Plot wp
            axes[col].fill_between(
                rp_,
                (wp - wp_err) * scale,
                (wp + wp_err) * scale,
                color=colors[row],
                ls='-',
                lw=0,
                alpha=0.2,
            )
            axes[col].plot(
                rp_,
                wp * scale,
                color=colors[row],
                ls='-',
                lw=1.5,
                label=legend_label,
            )

        if comp_filename is not None:
            if not (comparison == 'ms' and num_ybins == 3):
                comp_suffix = cfgrid.names.get_results_suffix(col)
                wp_comp, wp_comp_err = cfgrid.calc.get_stats(
                    comp_results, comp_suffix, mean=True, err=True
                )
                rp_comp = mask.mask_data(
                    rp, mask_cond(wp_comp, wp_comp_err, mask_weight)
                )
                axes[col].errorbar(
                    rp_comp,
                    wp_comp * scale,
                    yerr=wp_comp_err * scale,
                    color='k',
                    ls='--',
                    lw=1,
                    label=cfgrid.constants.COMPARISON_LABELS[comparison],
                )

        # Add column label
        add_column_label(fig, axes[col], col_label, col, xbins)

    # Add legend
    # loc = 'lower right' if rp_scale else 'lower left'
    # axes[num_xbins - 1].legend(loc=loc)

    if comparison == 'ms' and num_ybins == 3:
        fig = legend.add_legend(fig, num_ybins, colors, None, autocorr)
    else:
        fig = legend.add_legend(fig, num_ybins, colors, comparison, autocorr)

    for ax in axes:
        # Set axes ylims
        ylim = (4, 800)
        ax.set(xscale='log', yscale='log', xlim=(cf_bins[0], cf_bins[-1]), ylim=ylim)

        # Set axes ticks
        yticks = get_y_logticks(ax)
        ax.set_xticks(ticks=[1, 10])
        ax.set_yticks(ticks=yticks)

    # Set the figure file name
    fig_filename = f"condensed_{results_filename.split('.')[0]}"
    if comp_filename is not None:
        fig_filename = f'{fig_filename}_{comparison}'
    if not rp_scale:
        fig_filename = f'{fig_filename}_norp'

    save.savefig(fig, filename=fig_filename, path=path)
    plt.close('all')


def condensed_grid_multi(
    path: Sequence[pathlib.Path],
    results_filename: str,
    prop: type[properties.Property],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    row_labels: Sequence[str] | None = None,
    rp_scale: bool = False,
    autocorr: bool = True,
    comp_filename: str | None = None,
) -> None:
    """
    Plot a grid of correlation functions.

    """
    # Get results path
    results_path = [pathing.get_results_path_from_path(p) for p in path]

    # Load results
    tpcf_results = [
        cfgrid.calc.load_results(resp, results_filename) for resp in results_path
    ]

    # Load any comparison results
    if comp_filename is not None:
        comparison = cfgrid.names.get_comp_from_filename(comp_filename)
        comp_results = [
            cfgrid.calc.load_results(resp, comp_filename) for resp in results_path
        ]

    # Need to handle delta_MS when getting colors
    delta_MS = prop == properties.DeltaMS
    colors = fcolors.get_colors(num_ybins, delta_MS=delta_MS)
    ylabel = get_wp_label(rp_scale=rp_scale, autocorr=autocorr)

    # Set up figure and size specifications
    num_rows = len(path)
    fig, axes = get_grid(num_xbins, num_rows, ylabel=ylabel)
    fig.set_figheight(1.5 * fig.get_figheight())
    fig_left, fig_right, fig_top, fig_bottom = grid.get_grid_adjustments(
        fig.get_figwidth(), fig.get_figheight()
    )
    fig.subplots_adjust(top=fig_top, bottom=fig_bottom)

    rp = tpcf_results[0][cfgrid.constants.CF_BIN_LABELS['rp']].to_numpy()
    cf_bins = cfgrid.calc.get_tpcf_bins(centers=False)
    mask_weight = 0.95

    if rp_scale:
        scale = rp
    else:
        scale = np.ones(len(rp))

    for col in range(num_xbins):
        for row in range(num_ybins)[::-1]:
            if num_ybins == 1:
                suffix = cfgrid.names.get_results_suffix(col)
            else:
                suffix = cfgrid.names.get_results_suffix(col, row)

            for ax_row in range(num_rows):
                ax_row_to_ax_idx = (num_rows - 1) - ax_row
                wp, wp_err = cfgrid.calc.get_stats(
                    tpcf_results[ax_row_to_ax_idx], suffix, mean=True, err=True
                )
                rp_ = mask.mask_data(rp, mask_cond(wp, wp_err, mask_weight))

                legend_label = None if row > 0 else prop.label

                # Plot wp
                axes[ax_row_to_ax_idx, col].fill_between(
                    rp_,
                    (wp - wp_err) * scale,
                    (wp + wp_err) * scale,
                    color=colors[row],
                    ls='-',
                    lw=0,
                    alpha=0.2,
                )
                axes[ax_row_to_ax_idx, col].plot(
                    rp_,
                    wp * scale,
                    color=colors[row],
                    ls='-',
                    lw=1.5,
                    label=legend_label,
                )

                if comp_filename is not None and row == 0:
                    if not (comparison == 'ms' and num_ybins == 3):
                        comp_suffix = cfgrid.names.get_results_suffix(col)
                        wp_comp, wp_comp_err = cfgrid.calc.get_stats(
                            comp_results[ax_row_to_ax_idx],
                            comp_suffix,
                            mean=True,
                            err=True,
                        )
                        rp_comp = mask.mask_data(
                            rp, mask_cond(wp_comp, wp_comp_err, mask_weight)
                        )
                        axes[ax_row_to_ax_idx, col].errorbar(
                            rp_comp,
                            wp_comp * scale,
                            yerr=wp_comp_err * scale,
                            color='k',
                            ls='--',
                            lw=1,
                            label=cfgrid.constants.COMPARISON_LABELS[comparison],
                        )

                if row == 0 and col == 0 and row_labels is not None:
                    axes[ax_row_to_ax_idx, col].text(
                        0.05,
                        0.95,
                        row_labels[ax_row_to_ax_idx],
                        ha='left',
                        va='top',
                        transform=axes[ax_row_to_ax_idx, col].transAxes,
                        fontsize=LEGEND_FONTSIZE,
                    )

        # Add column label
        add_column_label(fig, axes[0, col], col_label, col, xbins)

    # Add legend
    # loc = 'lower right' if rp_scale else 'lower left'
    # axes[num_rows - 1, num_xbins - 1].legend(loc=loc)

    if comparison == 'ms' and num_ybins == 3:
        fig = legend.add_legend(fig, num_ybins, colors, None, autocorr)
    else:
        fig = legend.add_legend(fig, num_ybins, colors, comparison, autocorr)

    for ax in axes.flatten():
        # Set axes ylims
        ylim = (4, 800)
        ax.set(xscale='log', yscale='log', xlim=(cf_bins[0], cf_bins[-1]), ylim=ylim)

        # Set axes ticks
        yticks = get_y_logticks(ax)
        ax.set_xticks(ticks=[1, 10])
        ax.set_yticks(ticks=yticks)

    # Set the figure file name
    fig_filename = f"multi_condensed_{results_filename.split('.')[0]}"
    if comp_filename is not None:
        fig_filename = f'{fig_filename}_{comparison}'
    if not rp_scale:
        fig_filename = f'{fig_filename}_norp'

    save.savefig(fig, filename=fig_filename, path=path[0])
    plt.close('all')


def grid_all_corr(
    path: tuple[pathlib.Path, pathlib.Path, pathlib.Path],
    results_filename: str,
    counts_filename: str,
    prop: type[properties.Property],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    rp_scale: bool = False,
    comp_filename: str | None = None,
) -> None:
    """
    Plot a grid of correlation functions.

    """
    # Get results path
    results_path = [pathing.get_results_path_from_path(p) for p in path]

    # Load results
    all_results = cfgrid.calc.load_results(results_path[0], results_filename)
    centrals_results = cfgrid.calc.load_results(results_path[1], results_filename)
    satellites_results = cfgrid.calc.load_results(results_path[2], results_filename)
    all_counts = cfgrid.calc.load_results(results_path[0], counts_filename)
    centrals_counts = cfgrid.calc.load_results(results_path[1], counts_filename)
    satellites_counts = cfgrid.calc.load_results(results_path[2], counts_filename)

    # Load any comparison results
    if comp_filename is not None:
        comparison = cfgrid.names.get_comp_from_filename(comp_filename)
        comp_cent_results = cfgrid.calc.load_results(results_path[1], comp_filename)

    colors = ['k', fcolors.LIGHT_BLUE, fcolors.MAGENTA]
    labels = ['All', 'Central', 'Satellite']
    ls = ['-', '-.', '-']
    ylabel = get_wp_label(rp_scale=rp_scale, autocorr=True)
    fill_alpha = 0.2
    mask_weight = 0.95

    # Set up figure and size specifications
    fig, axes = get_grid(num_xbins, num_ybins, ylabel=ylabel, prop_label=prop.label)

    rp = all_results[cfgrid.constants.CF_BIN_LABELS['rp']].to_numpy()
    cf_bins = cfgrid.calc.get_tpcf_bins(centers=False)
    min_ys = []
    max_ys = []

    if rp_scale:
        scale = rp
    else:
        scale = np.ones(len(rp))

    for col in range(num_xbins):
        for row in range(num_ybins):
            row_to_ax_idx = (num_ybins - 1) - row
            if num_ybins == 1:
                suffix = cfgrid.names.get_results_suffix(col)
            else:
                suffix = cfgrid.names.get_results_suffix(col, row)

            wp_all, wp_all_err = cfgrid.calc.get_stats(
                all_results, suffix, mean=True, err=True
            )
            wp_cent, wp_cent_err = cfgrid.calc.get_stats(
                centrals_results, suffix, mean=True, err=True
            )
            wp_sat, wp_sat_err = cfgrid.calc.get_stats(
                satellites_results, suffix, mean=True, err=True
            )
            n_all = all_counts.at[0, f'counts_{col}_{row}']
            n_cent = centrals_counts.at[0, f'counts_{col}_{row}']
            n_sat = satellites_counts.at[0, f'counts_{col}_{row}']

            rp_all = mask.mask_data(rp, mask_cond(wp_all, wp_all_err, mask_weight))
            rp_cent = mask.mask_data(rp, mask_cond(wp_cent, wp_cent_err, mask_weight))
            rp_sat = mask.mask_data(rp, mask_cond(wp_sat, wp_sat_err, mask_weight))

            if comp_filename is not None:
                comp_suffix = cfgrid.names.get_results_suffix(col)
                wp_cent_comp, wp_cent_comp_err = cfgrid.calc.get_stats(
                    comp_cent_results, comp_suffix, mean=True, err=True
                )
                rp_cent_comp = mask.mask_data(
                    rp, mask_cond(wp_cent_comp, wp_cent_comp_err, mask_weight)
                )
                axes[row_to_ax_idx, col].errorbar(
                    rp_cent_comp,
                    wp_cent_comp * scale * (n_cent / n_all) ** 2,
                    yerr=wp_cent_comp_err * scale * (n_cent / n_all) ** 2,
                    color=colors[1],
                    # ls='--',
                    # lw=1,
                    alpha=1,
                    # label=f'Cent {comparison}',
                    fmt='o',
                    markeredgecolor='k',
                    markeredgewidth=0.25,
                    markersize=3,
                    lw=0.5,
                )

            # Plot wp
            # All
            axes[row_to_ax_idx, col].fill_between(
                rp_all,
                (wp_all - wp_all_err) * scale,
                (wp_all + wp_all_err) * scale,
                color=colors[0],
                ls='-',
                lw=0,
                alpha=fill_alpha,
            )
            axes[row_to_ax_idx, col].plot(
                rp_all,
                wp_all * scale,
                color=colors[0],
                ls=ls[0],
                lw=1,
                alpha=1,
            )

            # Centrals
            axes[row_to_ax_idx, col].fill_between(
                rp_cent,
                (wp_cent - wp_cent_err) * scale * (n_cent / n_all) ** 2,
                (wp_cent + wp_cent_err) * scale * (n_cent / n_all) ** 2,
                color=colors[1],
                ls='-',
                lw=0,
                alpha=fill_alpha,
            )
            axes[row_to_ax_idx, col].plot(
                rp_cent,
                wp_cent * scale * (n_cent / n_all) ** 2,
                color=colors[1],
                ls=ls[1],
                lw=1,
                alpha=1,
            )

            # Satellites
            axes[row_to_ax_idx, col].fill_between(
                rp_sat,
                (wp_sat - wp_sat_err) * scale * (n_sat / n_all) ** 2,
                (wp_sat + wp_sat_err) * scale * (n_sat / n_all) ** 2,
                color=colors[2],
                ls='-',
                lw=0,
                alpha=fill_alpha,
            )
            axes[row_to_ax_idx, col].plot(
                rp_sat,
                wp_sat * scale * (n_sat / n_all) ** 2,
                color=colors[2],
                ls=ls[2],
                lw=1,
                alpha=1,
            )

            axes[row_to_ax_idx, col].set_yscale('log')
            wp_cent_ = wp_cent * scale * (n_cent / n_all) ** 2
            wp_all_ = wp_all * scale
            if col == 0:
                min_ys.append(min(wp_cent_[wp_cent_ > 0]) / 2)

            max_ys.append(2 * max(wp_all_[wp_all_ > 0]))

        # Add column label
        add_column_label(fig, axes[0, col], col_label, col, xbins)

    # Add legend
    handles = [
        mlines.Line2D([], [], color=c, ls=s, lw=1, label=l)
        for (c, s, l) in zip(colors, ls, labels)
    ]
    if comp_filename is not None:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color='k',
                ls='--',
                label=f'{cfgrid.constants.COMPARISON_LABELS[comparison]} (Central)',
            )
        )
    ax_legend = fig.add_subplot(frameon=False)
    ax_legend.set(xticks=[], yticks=[])
    ax_legend.legend(
        loc='upper left',
        handles=handles,
        ncols=len(handles),
        labelspacing=0.3,
        columnspacing=1,
        handlelength=1.25,
        framealpha=0.95,
    )

    for ax in axes.flatten():
        ax.set(
            xscale='log',
            yscale='log',
            xlim=(cf_bins[0], cf_bins[-1]),
            ylim=(min(min_ys), max(max_ys)),
        )

        # Set axes ticks
        yticks = get_y_logticks(ax)
        ax.set_xticks(ticks=[1, 10])
        ax.set_yticks(ticks=yticks)

    # Set the figure file name
    fig_filename = f"all_corr_{results_filename.split('.')[0]}"

    if not rp_scale:
        fig_filename = f'{fig_filename}_norp'

    save.savefig(fig, filename=fig_filename, path=path[0])
    plt.close('all')


def _plane(
    x_property: properties.Property,
    y_property: properties.Property,
    xcoord: npt.NDArray[np.int64],
    ycoord: npt.NDArray[np.int64],
    num_xbins: int,
    num_ybins: int,
    colors: npt.NDArray[np.float64] | Sequence[str],
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a scatter plot of y_property vs x_property coloring by the
    different regions of the grid.

    """
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 0.9 * FIGWIDTH), constrained_layout=True)
    mean_x = []
    median_x = []
    median_y = []

    for col in range(num_xbins):
        xcond = xcoord == col

        for row in range(num_ybins):
            selection = (xcond) & (ycoord == row)
            n_obs = sum(selection)

            x = x_property.value[selection]
            y = y_property.value[selection]

            if not len(x) or not len(y):
                continue

            mean_x.append(np.mean(x))
            median_x.append(np.median(x))
            median_y.append(np.median(y))

            delta = 0.05
            xbins = np.arange(min(x), max(x) + delta, delta)
            ybins = np.arange(min(y), max(y) + delta, delta)

            # If the range is too narrow to form at least 2 bins, create
            # linspace bins to ensure there are enough.
            if len(xbins) <= 2:
                xbins = np.linspace(min(x), max(x), 3)
            if len(ybins) <= 2:
                ybins = np.linspace(min(y), max(y), 3)

            cts, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
            nmax = np.max(cts)
            levels = [0.25 * nmax, 0.5 * nmax, 0.75 * nmax]
            ax.scatter(
                x,
                y,
                s=0.25,
                color=fcolors.lighten(colors[row], amount=0.5),
                edgecolor='none',
                alpha=0.5,
                rasterized=True,
            )
            xbin_centers = 0.5 * (xbins[:-1] + xbins[1:])
            ybin_centers = 0.5 * (ybins[:-1] + ybins[1:])
            ax.contour(
                xbin_centers,
                ybin_centers,
                cts.T,
                levels,
                colors=fcolors.lighten(colors[row], amount=0),
                linewidths=0.5,
            )
            ax.text(
                np.median(x),
                np.median(y) + COUNTS_LABEL_OFFSET[type(y_property)],
                f'{n_obs}',
                ha='center',
                va='bottom',
                fontsize=LEGEND_FONTSIZE - 1,
                bbox={
                    'boxstyle': 'square',
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'alpha': 0.75,
                    'pad': 0.1,
                },
            )

    ax.scatter(
        median_x,
        median_y,
        marker='o',
        s=8,
        color='grey',
        edgecolor='k',
        lw=0.5,
        zorder=2,
        label=r'Median $M_*$',
    )
    ax.scatter(
        mean_x,
        median_y,
        marker='d',
        s=8,
        color='lightgrey',
        edgecolor='k',
        lw=0.5,
        zorder=2,
        label=r'Mean $M_*$',
    )
    ax.set(xlabel=x_property.full_label, ylabel=y_property.full_label)
    ax.legend(labelspacing=0.3, handlelength=1, framealpha=1)

    return fig, ax


def plane(
    x_property: properties.Property,
    y_property: properties.Property,
    xbins: Sequence[float],
    ybins: Mapping[int, Sequence[float]],
    path: pathlib.Path,
) -> None:
    """
    Plot a scatter plot of y_property vs x_property coloring by the
    different regions of the grid.

    """
    num_ybins = len(ybins[0]) + 1
    colors = fcolors.get_colors(num_ybins, delta_MS=False)

    # Get grid coordinates
    xcoord, ycoord = split.get_grid_coords(
        x_property.value, y_property.value, xbins[1:-1], ybins
    )

    fig, ax = _plane(
        x_property=x_property,
        y_property=y_property,
        xcoord=xcoord,
        ycoord=ycoord,
        num_xbins=len(xbins) - 1,
        num_ybins=num_ybins,
        colors=colors,
    )
    for xb in xbins[1:-1]:
        ax.axvline(x=xb, color='k', lw=0.5)

    for i, (mb1, mb2) in enumerate(zip(xbins[:-1], xbins[1:])):
        for ybin in ybins[i]:
            ax.hlines(y=ybin, xmin=mb1, xmax=mb2, color='k', lw=0.5)

    ax.set(xlim=(xbins[0] - 0.05, xbins[-1] + 0.05), ylim=PLANE_YLIMS[type(y_property)])

    # Get figure file name
    fig_filename = cfgrid.names.get_plane_filename(
        x_property.file_label, y_property.file_label, num_ybins=num_ybins
    )
    save.savefig(fig, filename=fig_filename, path=path, ext='pdf')
    plt.close('all')


def plane_delta_ms(
    mass: properties.Mstar,
    ssfr: properties.SSFR,
    delta_ms: properties.DeltaMS,
    xbins: Sequence[float],
    ybins: Mapping[int, Sequence[float]],
    path: pathlib.Path,
) -> None:
    """
    Plot a scatter plot of y_property vs x_property coloring by the
    different regions of the grid.

    """
    num_ybins = len(ybins[0]) + 1
    colors = fcolors.get_colors(num_ybins, delta_MS=True)

    # Get grid coordinates
    xcoord, ycoord = split.get_grid_coords(
        mass.value, delta_ms.value, xbins[1:-1], ybins
    )

    fig, ax = _plane(
        x_property=mass,
        y_property=ssfr,
        xcoord=xcoord,
        ycoord=ycoord,
        num_xbins=len(xbins) - 1,
        num_ybins=num_ybins,
        colors=colors,
    )
    for xb in xbins[1:-1]:
        ax.axvline(x=xb, color='k', lw=0.5)

    limits = (xbins[0], xbins[-1])
    logM = np.linspace(limits[0], limits[1], 100)
    ms_bins = binning.DeltaMSBins().get_bins(num_ybins)
    for msbin in ms_bins:
        ax.plot(logM, delta_ms.ms_fit(logM) + msbin, color='k', lw=0.5)

    ax.set(
        xlim=(limits[0] - 0.05, limits[-1] + 0.05),
        ylim=PLANE_YLIMS[properties.SSFR],
        # xticks=[10.0, 10.5, 11.0, 11.5],
        # xticklabels=['10.0', '10.5', '11.0', '11.5'],
    )

    # Get figure file name
    fig_filename = cfgrid.names.get_plane_filename(
        mass.file_label, delta_ms.file_label, num_ybins=num_ybins
    )
    save.savefig(fig, filename=fig_filename, path=path, ext='pdf')
    plt.close('all')


def grid_bias(
    path: pathlib.Path,
    results_filename: str,
    prop: type[properties.Property],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    autocorr: bool = True,
    rp_scale: bool = False,
    comp_filename: str | None = None,
) -> None:
    """
    Plot a grid of correlation functions.

    """
    # Get results path
    results_path = pathing.get_results_path_from_path(path)

    # Load results
    tpcf_results = cfgrid.calc.load_results(results_path, results_filename)

    # Load any comparison results
    if comp_filename is not None:
        comp_results = cfgrid.calc.load_results(results_path, comp_filename)

    # Need to handle delta_MS when getting colors
    delta_MS = prop == properties.DeltaMS
    colors = fcolors.get_colors(num_ybins, delta_MS=delta_MS)
    ylabel = r'$b\equiv \sqrt{w/w_{\rm bin}}$'

    # Set up figure and size specifications
    fig, axes = get_grid(num_xbins, 1, ylabel=ylabel, prop_label=None)

    rp = tpcf_results[cfgrid.constants.CF_BIN_LABELS['rp']].to_numpy()
    cf_bins = cfgrid.calc.get_tpcf_bins(centers=False)
    mask_weight = 0.95
    min_ys = []
    max_ys = []

    bias_df_list = []

    for col in range(num_xbins):
        comp_suffix = cfgrid.names.get_results_suffix(col)
        wp_comp, wp_comp_err = cfgrid.calc.get_stats(
            comp_results, comp_suffix, mean=True, err=True
        )
        rp_comp = mask.mask_data(rp, mask_cond(wp_comp, wp_comp_err, mask_weight))

        for row in range(num_ybins):
            row_to_ax_idx = 0  # num_ybins - 1  # (num_ybins - 1) - row
            if num_ybins == 1:
                suffix = cfgrid.names.get_results_suffix(col)
            else:
                suffix = cfgrid.names.get_results_suffix(col, row)

            wp, wp_err = cfgrid.calc.get_stats(
                tpcf_results, suffix, mean=True, err=True
            )

            # ignore sqrt invalids from low-statistics results
            with np.errstate(invalid='ignore'):
                bias = np.sqrt(wp / wp_comp)
            bias_err = 0.5 * calc_ratio_errors(wp, wp_comp, wp_err, wp_comp_err)
            rp_ = mask.mask_data(rp_comp, mask_cond(wp, wp_err, mask_weight))
            rp_ = mask.mask_data(rp_, mask_cond(bias, bias_err, mask_weight))
            nonnan = ~np.isnan(rp_)

            one_halo_term = (rp >= 0.1) & (rp < 1) & nonnan
            two_halo_term = (rp >= 1) & (rp < 10) & nonnan
            bias_err_sq = bias_err**2
            one_halo_bias = np.nanmean(bias[one_halo_term])
            one_halo_bias_err = np.sqrt(np.nansum(bias_err_sq[one_halo_term])) / sum(
                one_halo_term
            )
            two_halo_bias = np.nanmean(bias[two_halo_term])
            two_halo_bias_err = np.sqrt(np.nansum(bias_err_sq[two_halo_term])) / sum(
                two_halo_term
            )

            bias_df_list.append(
                {
                    '1h': one_halo_bias,
                    '1h_err': one_halo_bias_err,
                    '2h': two_halo_bias,
                    '2h_err': two_halo_bias_err,
                }
            )

            # Plot wp
            axes[row_to_ax_idx, col].fill_between(
                rp_,
                (bias - bias_err),
                (bias + bias_err),
                color=colors[row],
                ls='-',
                lw=0,
                alpha=0.2,
            )
            axes[row_to_ax_idx, col].plot(
                rp_,
                bias,
                color=colors[row],
                ls='-',
                lw=1.5,
                label=prop.label,
            )
            axes[row_to_ax_idx, col].errorbar(
                0.266 + row * 0.05,
                one_halo_bias,
                yerr=one_halo_bias_err,
                marker='o',
                markersize=2,
                elinewidth=1,
                linewidth=1,
                color=fcolors.lighten(colors[row], -0.2),
                capsize=2,
                zorder=3,
            )
            axes[row_to_ax_idx, col].errorbar(
                2.66 + row * 0.5,
                two_halo_bias,
                yerr=two_halo_bias_err,
                marker='o',
                markersize=2,
                elinewidth=1,
                linewidth=1,
                color=fcolors.lighten(colors[row], -0.2),
                capsize=2,
                zorder=3,
            )

            axes[row_to_ax_idx, col].set_yscale('log')
            if col == 0:
                min_ys.append(min(wp[wp > 0]) / 2)

            try:
                max_ys.append(1.2 * max(wp[wp > 0]))
            except ValueError:  # empty array wps
                pass

        # Add column label
        add_column_label(fig, axes[0, col], col_label, col, xbins)

    # Add legend
    fig = legend.add_legend(fig, num_ybins, colors, autocorr=autocorr)

    for ax in axes.flatten():
        # Set axes ylims
        ylim = (0.4, 3)
        ax.set(
            xscale='log',
            yscale='log',
            xlim=(cf_bins[0], cf_bins[-1]),
            # ylim=(min(min_ys), max(max_ys)),
            ylim=ylim,
        )

        # Set axes ticks
        yticks = get_y_logticks(ax)
        ax.set_xticks(ticks=[1, 10])
        ax.set_yticks(ticks=yticks)

    # bias_df = pd.DataFrame(bias_df_list)
    # print(bias_df.round(2))

    # Set the figure file name
    fig_filename = results_filename.split('.')[0]
    save.savefig(fig, filename=f'bias_{fig_filename}', path=path)
    plt.close('all')


def grid_bias_multi(
    path: Sequence[pathlib.Path],
    results_filename: str,
    prop: type[properties.Property],
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    cond_labels: Sequence[str] | None = None,
    comp_filename: str | None = None,
) -> None:
    """
    Plot a grid of correlation functions.

    """
    # Get results path
    results_path = [pathing.get_results_path_from_path(p) for p in path]

    # Load results
    tpcf_results = [
        cfgrid.calc.load_results(rp, results_filename) for rp in results_path
    ]

    # Load any comparison results
    if comp_filename is not None:
        comp_results = [
            cfgrid.calc.load_results(rp, comp_filename) for rp in results_path
        ]

    # Need to handle delta_MS when getting colors
    delta_MS = prop == properties.DeltaMS
    colors = fcolors.get_colors(num_ybins, delta_MS=delta_MS)
    ylabel = r'$b_{\rm rel} \equiv \sqrt{w_{\rm a}/w_{\rm a,bin}}$'

    # Set up figure and size specifications
    fig, axes = get_grid(num_xbins, len(path), ylabel=ylabel, height=1.2 * FIGWIDTH)
    for ax in axes.flatten():
        ax.axhline(y=1, color='silver', lw=0.5, zorder=-1)

    rp = tpcf_results[0][cfgrid.constants.CF_BIN_LABELS['rp']].to_numpy()
    cf_bins = cfgrid.calc.get_tpcf_bins(centers=False)
    mask_weight = 0.95
    bias_df_list = []

    for i in range(len(path)):
        for col in range(num_xbins):
            comp_suffix = cfgrid.names.get_results_suffix(col)
            wp_comp, wp_comp_err = cfgrid.calc.get_stats(
                comp_results[i], comp_suffix, mean=True, err=True
            )
            rp_comp = mask.mask_data(rp, mask_cond(wp_comp, wp_comp_err, mask_weight))

            for row in range(num_ybins):
                row_to_ax_idx = i  # num_ybins - 1  # (num_ybins - 1) - row
                if num_ybins == 1:
                    suffix = cfgrid.names.get_results_suffix(col)
                else:
                    suffix = cfgrid.names.get_results_suffix(col, row)

                wp, wp_err = cfgrid.calc.get_stats(
                    tpcf_results[i], suffix, mean=True, err=True
                )

                # ignore sqrt invalids from low-statistics results
                with np.errstate(invalid='ignore'):
                    bias = np.sqrt(wp / wp_comp)

                bias_err = 0.5 * calc_ratio_errors(wp, wp_comp, wp_err, wp_comp_err)
                rp_ = mask.mask_data(rp_comp, mask_cond(wp, wp_err, mask_weight))
                rp_ = mask.mask_data(rp_, mask_cond(bias, bias_err, mask_weight))
                nonnan = ~np.isnan(rp_)

                one_halo_term = (rp >= 0.1) & (rp < 1) & nonnan
                two_halo_term = (rp >= 1) & (rp < 10) & nonnan
                bias_err_sq = bias_err**2
                one_halo_bias = np.nanmean(bias[one_halo_term])
                one_halo_bias_err = np.sqrt(
                    np.nansum(bias_err_sq[one_halo_term])
                ) / sum(one_halo_term)
                two_halo_bias = np.nanmean(bias[two_halo_term])
                two_halo_bias_err = np.sqrt(
                    np.nansum(bias_err_sq[two_halo_term])
                ) / sum(two_halo_term)

                bias_df_list.append(
                    {
                        '1h': one_halo_bias,
                        '1h_err': one_halo_bias_err,
                        '2h': two_halo_bias,
                        '2h_err': two_halo_bias_err,
                    }
                )

                # Plot wp
                axes[row_to_ax_idx, col].fill_between(
                    rp_,
                    (bias - bias_err),
                    (bias + bias_err),
                    color=colors[row],
                    ls='-',
                    lw=0,
                    alpha=0.2,
                )
                axes[row_to_ax_idx, col].plot(
                    rp_,
                    bias,
                    color=colors[row],
                    ls='-',
                    lw=1.5,
                    label=prop.label,
                )
                axes[row_to_ax_idx, col].errorbar(
                    0.266 + row * 0.05,
                    one_halo_bias,
                    yerr=one_halo_bias_err,
                    marker='o',
                    markersize=2,
                    elinewidth=1,
                    linewidth=1,
                    color=fcolors.lighten(colors[row], -0.2),
                    capsize=2,
                    zorder=3,
                )
                axes[row_to_ax_idx, col].errorbar(
                    2.66 + row * 0.5,
                    two_halo_bias,
                    yerr=two_halo_bias_err,
                    marker='o',
                    markersize=2,
                    elinewidth=1,
                    linewidth=1,
                    color=fcolors.lighten(colors[row], -0.2),
                    capsize=2,
                    zorder=3,
                )
                axes[row_to_ax_idx, col].set_yscale('log')

            # Add column label
            add_column_label(fig, axes[0, col], col_label, col, xbins)

    # Add legend
    fig = legend.add_legend(fig, num_ybins, colors, loc='upper right')

    # Label the conditional
    if cond_labels is not None:
        for i, label in enumerate(cond_labels):
            axes[i, 0].text(
                0.95,
                0.95,
                label,
                ha='right',
                va='top',
                transform=axes[i, 0].transAxes,
                fontsize=LEGEND_FONTSIZE,
            )

    for ax in axes.flatten():
        # Set axes ylims
        ylim = (0.4, 3)
        ax.set(
            xscale='log',
            yscale='log',
            xlim=(cf_bins[0], cf_bins[-1]),
            ylim=ylim,
        )

        # Set axes ticks
        yticks = get_y_logticks(ax)
        ax.set_xticks(ticks=[1, 10])
        ax.set_yticks(ticks=yticks)

    # bias_df = pd.DataFrame(bias_df_list)
    # print(bias_df.round(2))

    # Set the figure file name
    fig_filename = results_filename.split('.')[0]
    save.savefig(fig, filename=f'bias_multi_{fig_filename}', path=path[0])
    plt.close('all')


def grid_bias_compare(
    path: Sequence[pathlib.Path],
    results_filename: str,
    num_xbins: int,
    num_ybins: int,
    xbins: Sequence[float],
    col_label: str,
    cond_labels: Sequence[str] | None = None,
    comp_filename: str | None = None,
    bias_sq: bool | None = False,
) -> None:
    """
    Plot the one-halo and two-halo bias as a function of delta MS.

    """
    # Get results path
    results_path = [pathing.get_results_path_from_path(p) for p in path]

    # Load results
    tpcf_results = [
        cfgrid.calc.load_results(rp, results_filename) for rp in results_path
    ]

    # Load any comparison results
    if comp_filename is not None:
        comp_results = [
            cfgrid.calc.load_results(rp, comp_filename) for rp in results_path
        ]

    # Set up figure and size specifications
    # colors = ['k'] + fcolors.get_qualitative_colors(len(path) - 1)
    # colors = ['k'] + [fcolors.MAGENTA, fcolors.LIGHT_BLUE, fcolors.GREEN]
    colors = [fcolors.LIGHT_BLUE, 'k']
    if not bias_sq:
        ylabel = r'$b_{\rm rel} \equiv \sqrt{w_{\rm a}/w_{\rm a,bin}}$'
    else:
        ylabel = r'$b^2_{\rm rel} \equiv w_{\rm a}/w_{\rm a,bin}$'
    fig, axes = get_grid(num_xbins, 2, ylabel=ylabel, height=0.8 * FIGWIDTH)

    rp = tpcf_results[0][cfgrid.constants.CF_BIN_LABELS['rp']].to_numpy()
    mask_weight = 0.95

    if num_ybins == 3:
        delta_ms = np.array([-1.5, -0.75, 0])
    elif num_ybins == 4:
        delta_ms = np.array([-1.5, -0.75, -0.25, 0.25])
    elif num_ybins == 6:
        delta_ms = np.array([-1.5, -0.75, -0.375, -0.125, 0.125, 0.375])

    for ax in axes.flatten():
        ax.axhline(y=1, color='silver', lw=0.5, zorder=-1)

    for i in range(len(path)):
        for col in range(num_xbins):
            axes[-1, col].set_xlabel(r'$\Delta {\rm MS}$')
            comp_suffix = cfgrid.names.get_results_suffix(col)
            wp_comp, wp_comp_err = cfgrid.calc.get_stats(
                comp_results[i], comp_suffix, mean=True, err=True
            )
            rp_comp = mask.mask_data(rp, mask_cond(wp_comp, wp_comp_err, mask_weight))
            one_halo_bias = [-1.0] * num_ybins
            two_halo_bias = [-1.0] * num_ybins
            one_halo_bias_err = [-1.0] * num_ybins
            two_halo_bias_err = [-1.0] * num_ybins

            for row in range(num_ybins):
                if num_ybins == 1:
                    suffix = cfgrid.names.get_results_suffix(col)
                else:
                    suffix = cfgrid.names.get_results_suffix(col, row)

                wp, wp_err = cfgrid.calc.get_stats(
                    tpcf_results[i], suffix, mean=True, err=True
                )

                # ignore sqrt invalids from low-statistics results
                with np.errstate(invalid='ignore'):
                    if not bias_sq:
                        bias = np.sqrt(wp / wp_comp)
                        bias_err = 0.5 * calc_ratio_errors(
                            wp, wp_comp, wp_err, wp_comp_err
                        )
                    else:
                        bias = wp / wp_comp
                        bias_err = calc_ratio_errors(wp, wp_comp, wp_err, wp_comp_err)

                rp_ = mask.mask_data(rp_comp, mask_cond(wp, wp_err, mask_weight))
                rp_ = mask.mask_data(rp_, mask_cond(bias, bias_err, mask_weight))
                nonnan = ~np.isnan(rp_)

                one_halo_term = (rp >= 0.1) & (rp < 1) & nonnan
                two_halo_term = (rp >= 1) & (rp < 10) & nonnan
                bias_err_sq = bias_err**2
                one_halo_bias[row] = np.nanmedian(bias[one_halo_term])
                two_halo_bias[row] = np.nanmedian(bias[two_halo_term])

                with np.errstate(invalid='ignore'):
                    one_halo_bias_err[row] = np.sqrt(
                        np.nansum(bias_err_sq[one_halo_term])
                    ) / sum(one_halo_term)

                    two_halo_bias_err[row] = np.sqrt(
                        np.nansum(bias_err_sq[two_halo_term])
                    ) / sum(two_halo_term)

                # Plot bias
                # axes[0, col].scatter(
                #     [delta_ms[row] + i * 0.05] * sum(one_halo_term),
                #     bias[one_halo_term],
                #     marker='s',
                #     edgecolor=fcolors.lighten(colors[i], 0.5),
                #     facecolor='none',
                #     s=5,
                # )
                # axes[1, col].scatter(
                #     [delta_ms[row] + i * 0.05] * sum(two_halo_term),
                #     bias[two_halo_term],
                #     marker='s',
                #     edgecolor=fcolors.lighten(colors[i], 0.5),
                #     facecolor='none',
                #     s=5,
                # )

            # if cond_labels is not None:
            #     print(cond_labels[i], col)
            # print(
            #     *[
            #         f'{b:.2f}\pm{berr:.2f}'
            #         for b, berr in zip(
            #             two_halo_bias[::-1],
            #             two_halo_bias_err[::-1],
            #         )
            #     ],
            #     sep='\n',
            # )

            # Plot mean bias
            axes[0, col].errorbar(
                delta_ms - i * 0.05,
                one_halo_bias,
                yerr=one_halo_bias_err,
                marker='o',
                markersize=2,
                elinewidth=1,
                linewidth=1,
                color=colors[i],
                capsize=2,
            )
            axes[1, col].errorbar(
                delta_ms - i * 0.05,
                two_halo_bias,
                yerr=two_halo_bias_err,
                marker='o',
                markersize=2,
                elinewidth=1,
                linewidth=1,
                color=colors[i],
                capsize=2,
            )

            # Add column label
            if i == 0:
                add_column_label(fig, axes[0, col], col_label, col, xbins)

    # Add legend
    if cond_labels is not None:
        ncols = len(path)
        handles = [
            mlines.Line2D([], [], marker='o', markersize=2, color=c, lw=0, label=l)
            for (c, l) in zip(colors, cond_labels)
        ]
        ax_legend = fig.add_subplot(frameon=False)
        ax_legend.set(xticks=[], yticks=[])
        ax_legend.legend(
            handles=handles,
            loc='upper right',
            labelspacing=0.3,
            columnspacing=1,
            ncols=ncols,
            handlelength=1.2,
            handletextpad=0.4,
            framealpha=0.95,
        )
    # for i, label in enumerate(['One-halo', 'Two-halo']):
    ax_text_top = fig.add_subplot(211, frameon=False)
    ax_text_bot = fig.add_subplot(212, frameon=False)
    for i, (label, ax) in enumerate(
        zip(
            [r'$0.1<r_{\rm p}/h^{-1}{\rm Mpc}<1$', r'$1<r_{\rm p}/h^{-1}{\rm Mpc}<10$'],
            # [
            #     r'$r_{\rm p} = 0.1$' + r'$-$' + r'$1~h^{-1}$Mpc',
            #     r'$r_{\rm p} = 1$' + r'$-$' + r'$10~h^{-1}$Mpc',
            # ],
            # [
            #     r'$0.1 < \frac{r_{\rm p}}{h^{-1}{\rm Mpc}} < 1$',
            #     r'$1 < \frac{r_{\rm p}}{h^{-1}{\rm Mpc}} < 10$',
            # ],
            [ax_text_top, ax_text_bot],
        )
    ):
        ax.set(xticks=[], yticks=[])
        ax.text(
            # 0.195,
            # 0.975,
            # 0.19,
            # 0.96,
            0.22,
            0.958,
            label,
            ha='right',
            va='top',
            transform=ax.transAxes,
            fontsize=LEGEND_FONTSIZE,
            # bbox=dict(facecolor='w', edgecolor='none', alpha=0.95, pad=0),
            bbox=dict(facecolor='w', edgecolor='k', lw=0.5, alpha=0.95, pad=2),
        )

    for ax in axes.flatten():
        # Set axes ylims
        if not bias_sq:
            ylim = (0.49, 2.1)
        else:
            ylim = (0.21, 2.1**2)

        ax.set(yscale='log', ylim=ylim)

        # Set axes ticks
        yticks = get_y_logticks(ax)
        ax.set_yticks(ticks=yticks)

    # Set the figure file name
    fig_filename = results_filename.split('.')[0]
    save.savefig(fig, filename=f'bias_compare_{fig_filename}', path=path[0])
    plt.close('all')
