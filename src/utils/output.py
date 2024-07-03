from typing import Sequence

import pandas as pd


def print_settings(header: str = "SETTINGS", length: int = 35, **kwargs) -> None:
    """
    Print out the given settings in a neat display, one each on a new
    line.

    """
    equal_separator = "=" * 35
    hyphen_separator = "-" * 35
    output = [f"{k}: {v:>{length-(len(k)+2)}}" for (k, v) in kwargs.items()]
    print(equal_separator, header, hyphen_separator, *output, equal_separator, sep="\n")


def print_volume_table(
    data: pd.DataFrame,
    mass_bins: Sequence[float],
    mass: str = 'M_star',
    redshift: str = 'z_obs',
    spacing: int = 9,
) -> None:
    """
    Print a table of 'M_min', 'M_max', 'z_min', 'z_max', 'N_gal' broken
    into mass bins. 'mass' and 'redshift' are strings that are used to
    access those quantities from 'data'.

    """
    left = spacing - len('M_min')
    border_len = spacing * 5 + 4 + left

    print(
        f"{'M_min':>{spacing}} {'M_max':>{spacing}}"
        f" {'z_min':>{spacing}} {'z_max':>{spacing}}"
        f" {'N_gal':>{spacing}}"
    )
    print('=' * border_len)

    for m_min, m_max in zip(mass_bins[:-1], mass_bins[1:]):
        data_mbin = data.query(f"{mass} > @m_min and {mass} < @m_max")
        z_min = min(data_mbin[redshift])
        z_max = max(data_mbin[redshift])
        n_gal = len(data_mbin)
        print(
            f"{m_min:>{spacing}.3f} {m_max:>{spacing}.3f}"
            f" {z_min:>{spacing}.3f} {z_max:>{spacing}.3f}"
            f" {n_gal:>{spacing},}"
        )
