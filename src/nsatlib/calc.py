from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from src import nsatlib


def calc_stats(
    centrals: pd.DataFrame, satellites: pd.DataFrame, group_id_str: str, mhigh: float
) -> tuple[float, float]:
    """
    Calculate the mean and standard deviation of the number of
    satellites below a stellar mass 'mhigh' per central. The ddof
    argument of np.std() applies a correction to produce a less biased
    estimate of the standard deviation.

    """
    group_ids = {group_id: 0 for group_id in centrals[group_id_str]}
    num_centrals = len(group_ids)

    if not num_centrals:
        return 0, 0

    sats = satellites.query(
        "M_star < @mhigh and @satellites[@group_id_str] in @group_ids"
    )
    sat_group_counts = sats.groupby(group_id_str).size()

    for gid, count in sat_group_counts.items():
        assert isinstance(count, int)
        group_ids[gid] += count

    sat_counts = list(group_ids.values())
    mean = float(np.mean(sat_counts))

    if num_centrals == 1:
        err = 0
    else:
        err = np.std(sat_counts, ddof=(1.5 - 1 / (8 * (num_centrals - 1)))) / np.sqrt(
            num_centrals
        )

    return mean, err


def nsat(
    cen: pd.DataFrame,
    sat: pd.DataFrame,
    group_id_str: str,
    mass_bins: npt.NDArray | Sequence[float],
    window_size: float,
) -> pd.DataFrame:
    """
    Calculate Nsat in various subsamples.

    """
    subsamples = nsatlib.constants.SUBSAMPLES
    results_labels = nsatlib.constants.RESULTS_LABELS
    means = {subsample: np.empty(len(mass_bins)) for subsample in subsamples.keys()}
    errs = {subsample: np.empty(len(mass_bins)) for subsample in subsamples.keys()}
    ngals = {subsample: np.empty(len(mass_bins)) for subsample in subsamples.keys()}

    for i, mcenter in enumerate(tqdm(mass_bins, total=len(mass_bins))):
        mlow = mcenter - window_size
        mhigh = mcenter + window_size
        cen_all = cen.query(subsamples['all'])
        means['all'][i], errs['all'][i] = calc_stats(cen_all, sat, group_id_str, mhigh)
        ngals['all'][i] = len(cen_all)

        for j, (subsample, query) in enumerate(subsamples.items()):
            # Other subsamples build off of cen_all
            if j == 0:
                continue

            c = cen_all.query(query)
            means[subsample][i], errs[subsample][i] = calc_stats(
                c, sat, group_id_str, mhigh
            )
            ngals[subsample][i] = len(c)

    nsat_dict = {'logMs': mass_bins}
    for subsample in subsamples.keys():
        nsat_dict[f'{results_labels["Ns"]}_{subsample}'] = means[subsample]
        nsat_dict[f'{results_labels["err_Ns"]}_{subsample}'] = errs[subsample]
        nsat_dict[f'{results_labels["n_gals"]}_{subsample}'] = ngals[subsample]

    return pd.DataFrame.from_dict(nsat_dict)
