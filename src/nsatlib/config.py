from typing import Protocol

import pandas as pd
from src import configurations, datalib
from src.protocols import conditionals, properties


CENTRAL_MASS_BIN_SIZE = 0.1
WINDOW_SIZE = 2 * CENTRAL_MASS_BIN_SIZE
GAL_PROPERTY = properties.SSFR
HALO_PROPERTY = properties.Vpeak


class NsatConfig(Protocol):
    dirname: str
    dataset: datalib.DataSet
    volume: list[datalib.volumes.Volume]
    centrals: list[type[conditionals.Conditional]]
    satellites: list[type[conditionals.Conditional]]

    def get_nsat_filename(self) -> str: ...

    def load_nsat(self) -> pd.DataFrame: ...


CONFIG_TYPE: type[NsatConfig] = configurations.MPAConfigVolume3
CONFIG = CONFIG_TYPE()

if isinstance(CONFIG, configurations.BPConfig):
    CONFIG.set_galhalo_props(GAL_PROPERTY.file_label, HALO_PROPERTY.file_label)
