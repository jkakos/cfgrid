from typing import Protocol, runtime_checkable

import pandas as pd
import pathlib

from src.protocols import coords
from src.tpcf import tpcf


class DataSet(Protocol):
    data_dir: pathlib.Path
    filename: str
    data: pd.DataFrame
    merge_col: list[str]
    data_cuts: dict[str, tuple[float, float]] | None

    def __init__(self) -> None: ...

    @property
    def filename_str(self) -> str: ...

    def load(self): ...


# For tpcf calculations
@runtime_checkable
class SimSnapshotDataSet(DataSet, Protocol):
    cf_runner: type[tpcf.SimSnapshotTPCF]
    ...


@runtime_checkable
class SimSnapshotZSpaceDataSet(DataSet, Protocol):
    cf_runner: type[tpcf.SimSnapshotZSpaceTPCF]
    ...


@runtime_checkable
class ObsLightConeDataSet(DataSet, Protocol):
    cf_runner: type[tpcf.ObsLightConeTPCF]
    random_file: str
    ...


# For density calculations
class DensityDataSet(DataSet, Protocol):
    coord_strat: type[coords.CoordinateStrategy]
    ...
