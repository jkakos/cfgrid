import pathlib
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

import config
from src.cfgrid import constants as consts, names
from src.protocols import properties
from src.tpcf import tpcf as cf
from src.utils import split


def get_tpcf_bins(centers: bool = False) -> npt.NDArray:
    """
    Get the tpcf binning scheme, or if 'centers' is True, return the
    bin centers.

    """
    bins = config.tpcf_settings['bins']

    if centers:
        bin_centers = 10 ** (0.5 * (np.log10(bins[:-1]) + np.log10(bins[1:])))
        return bin_centers

    return bins


def get_num_ybins(ybins: Mapping[int, Sequence[float]]) -> int:
    return len(ybins[0]) + 1


def store_stats(
    df: pd.DataFrame,
    wp: npt.NDArray,
    err: npt.NDArray,
    suffix: str,
    mean: npt.NDArray | None = None,
    median: npt.NDArray | None = None,
    counts: npt.NDArray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Store various statistics that will be written to a file.

    """
    wp_label = consts.CF_RESULTS_LABELS['wp']
    df[f'{wp_label}_{suffix}'] = wp
    df[f"{wp_label}_{consts.CF_RESULTS_LABELS['err']}_{suffix}"] = err

    if mean is not None:
        df[f"{wp_label}_{consts.CF_RESULTS_LABELS['mean']}_{suffix}"] = mean
    if median is not None:
        df[f"{wp_label}_{consts.CF_RESULTS_LABELS['median']}_{suffix}"] = median
    if counts is not None:
        counts_df = pd.DataFrame(
            [counts],
            columns=[f"{consts.CF_RESULTS_LABELS['counts']}_{suffix}"],
        )
    else:
        counts_df = None

    return df, counts_df


def get_stats(
    df: pd.DataFrame,
    suffix: str,
    wp: bool | None = False,
    mean: bool | None = False,
    median: bool | None = False,
    err: bool | None = False,
) -> tuple[npt.NDArray, ...]:
    """
    Get various statistics from tpcf results. The return order will be
    the same as the order of the function parameters.

    """
    ret = []
    wp_label = consts.CF_RESULTS_LABELS['wp']

    if wp:
        ret.append(df[f'{wp_label}_{suffix}'].to_numpy())
    if mean:
        ret.append(
            df[f"{wp_label}_{consts.CF_RESULTS_LABELS['mean']}_{suffix}"].to_numpy()
        )
    if median:
        ret.append(
            df[f"{wp_label}_{consts.CF_RESULTS_LABELS['median']}_{suffix}"].to_numpy()
        )
    if err:
        ret.append(
            df[f"{wp_label}_{consts.CF_RESULTS_LABELS['err']}_{suffix}"].to_numpy()
        )

    return tuple(ret)


def _save_results(
    results: pd.DataFrame,
    filepath: pathlib.Path,
    round_decimals: int = 0,
    merge_cols: Sequence[str] = [],
) -> None:
    """
    Save the latest version of results.

    """
    if round_decimals:
        results = results.round(decimals=round_decimals)

    # Merge new results with previous ones if they exist
    try:
        prev_data = pd.read_csv(filepath)

        # Keep the latest results if any columns repeat
        drop_cols = [
            x for x in results.columns if x in prev_data.columns and x not in merge_cols
        ]
        prev_data = prev_data.drop(drop_cols, axis=1)

        if merge_cols:
            new_data = prev_data.merge(results, how='outer', on=merge_cols)
        else:
            new_data = prev_data.join(results)

    except FileNotFoundError:
        new_data = results

    new_data.to_csv(filepath, index=False)


def save_results(results: pd.DataFrame, filepath: pathlib.Path) -> None:
    """
    Save the latest version of 2pcf results.

    """
    _save_results(
        results,
        filepath,
        round_decimals=6,
        merge_cols=[
            consts.CF_BIN_LABELS['rp'],
            consts.CF_BIN_LABELS['rp_min'],
            consts.CF_BIN_LABELS['rp_max'],
        ],
    )


def save_counts(results: pd.DataFrame, filepath: pathlib.Path) -> None:
    """
    Save the counts for each bin in the grid.

    """
    _save_results(results, filepath)


def load_results(path: pathlib.Path, filename: str) -> pd.DataFrame:
    """
    Load saved 2pcf results.

    """
    filepath = path.joinpath(filename)
    return pd.read_csv(filepath)


def get_base_tpcf_df(rp, rp_min, rp_max):
    """
    Creates a new dataframe with the projected radial distances used in
    tpcf calculations.

    """
    return pd.DataFrame.from_dict(
        {
            consts.CF_BIN_LABELS['rp']: rp,
            consts.CF_BIN_LABELS['rp_min']: rp_min,
            consts.CF_BIN_LABELS['rp_max']: rp_max,
        }
    )


def auto_corr(
    x_property: properties.Property,
    y_property: properties.Property,
    cf_runner: cf.TpcfRunner,
    xbins: Sequence[float],
    ybins: Mapping[int, Sequence[float]],
    path_tpcf: pathlib.Path,
    path_counts: pathlib.Path,
) -> None:
    """
    Calculate the two-point auto-correlation function in a grid where
    data is binned by some galaxy/halo property going up the grid and
    some other property (generally mass) going across the grid.

    """
    # Get projected radial bins
    rp_bins = get_tpcf_bins(centers=False)
    rp = get_tpcf_bins(centers=True)
    rp_min = rp_bins[:-1]
    rp_max = rp_bins[1:]
    num_ybins = get_num_ybins(ybins)

    # Get grid coordinates
    xcoord, ycoord = split.get_grid_coords(
        x_property.value, y_property.value, xbins[1:-1], ybins
    )

    for col in range(len(xbins) - 1):
        xcond = xcoord == col

        for row in range(num_ybins):
            tpcf_results = get_base_tpcf_df(rp, rp_min, rp_max)
            selection = (xcond) & (ycoord == row)
            n_obs = sum(selection)

            if num_ybins == 1:
                suffix = names.get_results_suffix(col)
            else:
                suffix = names.get_results_suffix(col, row)

            wp, wp_err, wp_mean, wp_median = cf_runner.run(
                selection, return_mean=True, return_median=True
            )
            tpcf_results, counts = store_stats(
                df=tpcf_results,
                wp=wp,
                err=wp_err,
                suffix=suffix,
                mean=wp_mean,
                median=wp_median,
                counts=n_obs,
            )
            assert counts is not None
            save_results(results=tpcf_results, filepath=path_tpcf)
            save_counts(results=counts, filepath=path_counts)


def cross_corr(
    x_property: tuple[properties.Property, properties.Property],
    y_property: tuple[properties.Property, properties.Property],
    cf_runner: cf.CrossTpcfRunner,
    xbins: Sequence[float],
    ybins: Mapping[int, Sequence[float]],
    results_path: pathlib.Path,
) -> None:
    """
    Calculate the two-point cross-correlation function in a grid where
    data is binned by some galaxy/halo property going up the grid and
    some other property (generally mass) going across the grid.

    """
    # Get path and file names
    tpcf_filename = names.get_tpcf_filename(
        consts.CROSS_BASE,
        y_property[0].file_label,
        num_ybins=get_num_ybins(ybins),
    )

    # Get projected radial bins
    rp_bins = get_tpcf_bins(centers=False)
    rp = get_tpcf_bins(centers=True)
    rp_min = rp_bins[:-1]
    rp_max = rp_bins[1:]
    num_ybins = get_num_ybins(ybins)

    # Get grid coordinates
    xcoords = []
    ycoords = []
    for x, y in zip(x_property, y_property):
        xcoord, ycoord = split.get_grid_coords(x.value, y.value, xbins[1:-1], ybins)
        xcoords.append(xcoord)
        ycoords.append(ycoord)

    for col in range(len(xbins) - 1):
        xconds = [xcoord == col for xcoord in xcoords]

        for row in range(num_ybins):
            tpcf_results = get_base_tpcf_df(rp, rp_min, rp_max)
            yconds = [ycoord == row for ycoord in ycoords]
            selections = [xcond & ycond for (xcond, ycond) in zip(xconds, yconds)]

            if num_ybins == 1:
                suffix = names.get_results_suffix(col)
            else:
                suffix = names.get_results_suffix(col, row)

            wp, wp_err, wp_mean, wp_median = cf_runner.run(
                selections[0], selections[1], return_mean=True, return_median=True
            )
            tpcf_results, counts = store_stats(
                df=tpcf_results,
                wp=wp,
                err=wp_err,
                suffix=suffix,
                mean=wp_mean,
                median=wp_median,
            )
            save_results(
                results=tpcf_results, filepath=results_path.joinpath(tpcf_filename)
            )


def _get_cross_corr_all_sat_coords(
    x_property: tuple[properties.Property, properties.Property],
    y_property: tuple[properties.Property, properties.Property],
    xbins: Sequence[float],
    ybins: Mapping[int, Sequence[float]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Get the grid coords for each set of (x_property, y_property). If
    using the full xbins (num_ybins=1), then process is simplified and
    'y_property' is not needed and can be a placeholder.

    """
    num_ybins = get_num_ybins(ybins)

    if num_ybins == 1:
        xcoord1 = np.digitize(x_property[0].value, xbins[1:-1])
        xcoord2 = np.digitize(x_property[1].value, xbins[1:-1])
        ycoord1 = np.zeros(len(x_property[0].value), dtype=np.int64)

        return xcoord1, ycoord1, xcoord2

    xcoord1, ycoord1 = split.get_grid_coords(
        x_property[0].value, y_property[0].value, xbins[1:-1], ybins
    )
    xcoord2 = np.digitize(x_property[1].value, xbins[1:-1])

    return xcoord1, ycoord1, xcoord2


def cross_corr_all_satellites(
    x_property: tuple[properties.Property, properties.Property],
    y_property: tuple[properties.Property, properties.Property],
    cf_runner: cf.CrossTpcfRunner,
    xbins: Sequence[float],
    ybins: Mapping[int, Sequence[float]],
    path_tpcf: pathlib.Path,
    below: bool = False,
) -> None:
    """
    Calculate the cross-correlations of centrals in a grid in with all
    satellites in the same xbin. If 'below' is True, then cross-
    correlate with satellites in the same or lower xbin.

    """
    # Get projected radial bins
    rp_bins = get_tpcf_bins(centers=False)
    rp = get_tpcf_bins(centers=True)
    rp_min = rp_bins[:-1]
    rp_max = rp_bins[1:]
    num_ybins = get_num_ybins(ybins)

    # Get grid coordinates
    xcoord1, ycoord1, xcoord2 = _get_cross_corr_all_sat_coords(
        x_property=x_property,
        y_property=y_property,
        xbins=xbins,
        ybins=ybins,
    )

    for col in range(len(xbins) - 1):
        xcond1 = xcoord1 == col
        xcond2 = xcoord2 == col if not below else xcoord2 <= col

        for row in range(num_ybins):
            tpcf_results = get_base_tpcf_df(rp, rp_min, rp_max)
            selection = xcond1 & (ycoord1 == row)

            if num_ybins == 1:
                suffix = names.get_results_suffix(col)
            else:
                suffix = names.get_results_suffix(col, row)

            wp, wp_err, wp_mean, wp_median = cf_runner.run(
                selection,
                xcond2,
                return_mean=True,
                return_median=True,
                return_counts=False,
            )
            tpcf_results, counts = store_stats(
                df=tpcf_results,
                wp=wp,
                err=wp_err,
                suffix=suffix,
                mean=wp_mean,
                median=wp_median,
            )
            save_results(results=tpcf_results, filepath=path_tpcf)
