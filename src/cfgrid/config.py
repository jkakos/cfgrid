from typing import Protocol, Sequence

import pandas as pd

import config
from src import configurations
from src import datalib
from src.protocols import conditionals, properties


class CFGridConfig(Protocol):
    dirname: str
    dataset: datalib.DataSet
    volume: list[datalib.volumes.Volume]
    mass_bins: list[float]
    ybin_schemes: list[configurations.YbinScheme]
    centrals: list[type[conditionals.Conditional]]
    satellites: list[type[conditionals.Conditional]]

    def load(self) -> pd.DataFrame: ...

    def set_mass_bins(self, mass_bins: Sequence[float]) -> None: ...


# =========================================================
# DATA SETUP
# =========================================================
CONFIG_TYPE: type[CFGridConfig] = configurations.MPAConfigVolume3
CONFIG: CFGridConfig = CONFIG_TYPE()
MASS_BINS = config.MASS_BINS
CONFIG.set_mass_bins(MASS_BINS)

# =========================================================
# BINNING
# =========================================================
X_PROPERTY = properties.Mstar
XBINS = MASS_BINS

# =========================================================
# CONDITIONALS
# =========================================================
ALL_GALAXIES: list[type[conditionals.Conditional]] = [conditionals.AllGalaxies]

AUTO_CONDITIONALS = [
    *CONFIG.centrals,
    *ALL_GALAXIES,
    *CONFIG.satellites,
]
PLANE_CONDITIONALS = [
    *CONFIG.centrals,
    *ALL_GALAXIES,
    *CONFIG.satellites,
]
