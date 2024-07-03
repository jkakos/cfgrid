import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

# fmt: off
MASS = np.array(
    [
        10.25, 10.25, 10.25, 10.25, 10.25, 10.25,
        10.50, 10.50, 10.50, 10.50, 10.50, 10.50,
        10.90, 10.90, 10.90, 10.90, 10.90, 10.90,
        11.10, 11.10, 11.10, 11.10, 11.10, 11.10,
        11.40, 11.40, 11.40, 11.40, 11.40, 11.40,
    ]
)
SSFR = np.array(
    [
        -11.50, -10.75, -10.40, -10.20, -9.90, -9.70,
        -11.50, -10.75, -10.50, -10.30, -10.00, -9.80,
        -11.50, -11.00, -10.70, -10.50, -10.20, -9.90,
        -11.90, -11.20, -10.80, -10.60, -10.30, -10.00,
        -11.90, -11.50, -11.00, -10.80, -10.60, -10.20,
    ]
)
# fmt: on


@pytest.fixture
def arr() -> npt.NDArray:
    return np.arange(11)


@pytest.fixture
def mass() -> npt.NDArray:
    return MASS


@pytest.fixture
def ssfr() -> npt.NDArray:
    return SSFR


@pytest.fixture
def data_mass_sfr() -> pd.DataFrame:
    return pd.DataFrame.from_dict({'M_star': MASS, 'SFR': SSFR + MASS})
