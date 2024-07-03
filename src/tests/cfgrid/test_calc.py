import numpy as np
import pandas as pd

from src import cfgrid
from src.protocols import properties


def test_get_cross_corr_all_sat_coords(data_mass_sfr: pd.DataFrame) -> None:
    x_property = (properties.Mstar(data_mass_sfr), properties.Mstar(data_mass_sfr))
    y_property = (properties.DeltaMS(data_mass_sfr), properties.DeltaMS(data_mass_sfr))
    xbins = (10.375, 10.75, 11.0, 11.25)
    ybins = {
        0: [-1, -0.45, -0.25, 0.0, 0.25],
        1: [-1, -0.45, -0.25, 0.0, 0.25],
        2: [-1, -0.45, -0.25, 0.0, 0.25],
        3: [-1, -0.45, -0.25, 0.0, 0.25],
        4: [-1, -0.45, -0.25, 0.0, 0.25],
    }
    got = cfgrid.calc._get_cross_corr_all_sat_coords(
        x_property, y_property, xbins, ybins
    )
    # fmt: off
    expected = (
        np.array([
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4,
        ]),
        np.array([
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
            0, 1, 2, 3, 4, 5,
        ]),
        np.array([
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4,
        ]),
    )
    # fmt: on
    assert len(got) == len(expected)
    assert all(got[0] == expected[0])
    assert all(got[1] == expected[1])
    assert all(got[2] == expected[2])


def test_get_cross_corr_all_sat_coords_one_ybin(data_mass_sfr: pd.DataFrame) -> None:
    x_property = (properties.Mstar(data_mass_sfr), properties.Mstar(data_mass_sfr))
    y_property = (properties.DeltaMS(data_mass_sfr), properties.DeltaMS(data_mass_sfr))
    xbins = (10.375, 10.75, 11.0, 11.25)
    ybins = {0: [], 1: [], 2: [], 3: [], 4: []}  # type: ignore
    got = cfgrid.calc._get_cross_corr_all_sat_coords(
        x_property, y_property, xbins, ybins
    )
    # fmt: off
    expected = (
        np.array([
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4,
        ]),
        np.array([
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
        ]),
        np.array([
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4,
        ]),
    )
    # fmt: on
    assert len(got) == len(expected)
    assert all(got[0] == expected[0])
    assert all(got[1] == expected[1])
    assert all(got[2] == expected[2])
