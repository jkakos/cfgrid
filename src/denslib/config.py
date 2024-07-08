from typing import Protocol

import pandas as pd

from src import configurations, datalib


class DensityConfig(Protocol):
    dirname: str
    dataset: datalib.DataSet

    def load(self) -> pd.DataFrame: ...


RADIUS = [4, 8]
RP_MAX = [1, 2]
RPI_MAX = [2, 3, 4, 5, 6, 7, 8]
DDP_MASS_BINS = (10.0, 11.0)

CONFIG_TYPE = configurations.MPAConfigVolume3
CONFIG = CONFIG_TYPE()
