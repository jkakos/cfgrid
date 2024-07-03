from typing import Sequence

import config
from src import complib, datalib
from src.utils import output


def vol1_type_completeness(mass_bins: Sequence[float]) -> None:
    """
    Find the redshift limits that correspond to the different mass bins
    using the volume1 completeness condition.

    """
    volumes = complib.calc.vol1_type_completeness(mass_bins)
    mpa = datalib.MPAJHU(volume=volumes)
    data = mpa.load()
    output.print_volume_table(data, mass_bins)


def completeness_vol1() -> None:
    vol1_type_completeness(config.MASS_BINS)


def completeness_vol3() -> None:
    vol1_type_completeness([10.0, 12.0])
