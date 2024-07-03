import numpy as np
import numpy.typing as npt

from src.protocols import binning, properties
from src.utils import split


def test_percentile_bins(arr: npt.NDArray) -> None:
    num_bins = 4
    pbins = split.percentile_bins(arr, num_bins)
    assert np.allclose(pbins, np.array([2.5, 5.0, 7.5]))


def test_get_ybins_delta_ms(mass: npt.NDArray, ssfr: npt.NDArray) -> None:
    """
    Test 'get_ybins' when 'DeltaMSBins()' is given for 'ybin_strategy'.
    In this case, the same bins should be returned for each xbin.

    """
    xbins = (10.375, 10.75, 11, 11.25)
    num_ybins = 6
    got = split.get_ybins(mass, ssfr, xbins, binning.DeltaMSBins, num_ybins)
    expected = {
        0: [-1.0, -0.45, -0.25, 0.0, 0.25],
        1: [-1.0, -0.45, -0.25, 0.0, 0.25],
        2: [-1.0, -0.45, -0.25, 0.0, 0.25],
        3: [-1.0, -0.45, -0.25, 0.0, 0.25],
        4: [-1.0, -0.45, -0.25, 0.0, 0.25],
    }
    assert got.keys() == expected.keys()
    for key in expected.keys():
        assert np.allclose(got[key], expected[key])


def test_get_ybins_precentile_bins(mass: npt.NDArray, ssfr: npt.NDArray) -> None:
    """
    Test 'get_ybins' when 'PercentileBins()' is given for
    'ybin_strategy'. In this case, each xbin should be broken into
    ybins that would return an equal number of points in each bin.

    """
    xbins = (10.375, 10.75, 11, 11.25)
    num_ybins = 4
    got = split.get_ybins(mass, ssfr, xbins, binning.PercentileBins, num_ybins)
    expected = {
        0: list(split.percentile_bins(ssfr[0:6], num_ybins)),
        1: list(split.percentile_bins(ssfr[6:12], num_ybins)),
        2: list(split.percentile_bins(ssfr[12:18], num_ybins)),
        3: list(split.percentile_bins(ssfr[18:24], num_ybins)),
        4: list(split.percentile_bins(ssfr[24:30], num_ybins)),
    }
    assert got.keys() == expected.keys()
    for key in expected.keys():
        assert np.allclose(got[key], expected[key])


def test_get_ybins_sequence(mass: npt.NDArray, ssfr: npt.NDArray) -> None:
    """
    Test 'get_ybins' when a sequence is given for 'ybin_strategy'. In
    this case, the sequence should be given back for each xbin.

    """
    xbins = (10.375, 10.75, 11, 11.25)
    num_ybins = 4
    got = split.get_ybins(mass, ssfr, xbins, [-11.0, -10.5, -10.0], num_ybins)
    expected = {
        0: [-11.0, -10.5, -10.0],
        1: [-11.0, -10.5, -10.0],
        2: [-11.0, -10.5, -10.0],
        3: [-11.0, -10.5, -10.0],
        4: [-11.0, -10.5, -10.0],
    }
    assert got.keys() == expected.keys()
    for key in expected.keys():
        assert np.allclose(got[key], expected[key])


def test_get_grid_coords(mass: npt.NDArray, ssfr: npt.NDArray):
    """
    Test the main gridding scheme.

    """
    xbins = (10.375, 10.75, 11, 11.25)
    ybins = {
        0: [-11.0, -10.5, -10.0],
        1: [-11.0, -10.5, -10.0],
        2: [-11.0, -10.5, -10.0],
        3: [-11.0, -10.5, -10.0],
        4: [-11.0, -10.5, -10.0],
    }
    got = split.get_grid_coords(mass, ssfr, xbins, ybins)
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
            0, 1, 2, 2, 3, 3,
            0, 1, 2, 2, 3, 3,
            0, 1, 1, 2, 2, 3,
            0, 0, 1, 1, 2, 3,
            0, 0, 1, 1, 1, 2,
        ]),
    )
    # fmt: on
    assert all(got[0] == expected[0])
    assert all(got[1] == expected[1])


def test_get_grid_coords_delta_ms(mass: npt.NDArray, ssfr: npt.NDArray):
    """
    Test the main gridding scheme using delta MS.

    """
    xbins = (10.375, 10.75, 11, 11.25)
    ybins = {
        0: [-1.0, -0.45, -0.25, 0.0, 0.25],
        1: [-1.0, -0.45, -0.25, 0.0, 0.25],
        2: [-1.0, -0.45, -0.25, 0.0, 0.25],
        3: [-1.0, -0.45, -0.25, 0.0, 0.25],
        4: [-1.0, -0.45, -0.25, 0.0, 0.25],
    }
    delta_ms = ssfr - properties.DeltaMS.ms_fit(mass)
    got = split.get_grid_coords(mass, delta_ms, xbins, ybins)
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
    )
    # fmt: on
    assert all(got[0] == expected[0])
    assert all(got[1] == expected[1])
