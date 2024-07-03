from typing import Protocol
from src import datalib


class Config(Protocol):
    volume: list[datalib.volumes.Volume]

    def get_nsat_filename(self) -> str: ...


def get_nsat_filename(configuration: Config) -> str:
    filename = configuration.get_nsat_filename()
    zmin, zmax = configuration.volume[0].redshift_lims
    mmin, mmax = configuration.volume[0].mass_lims
    return f'nsat_{filename}_z_{zmin:.3f}_{zmax:.3f}_mass_{mmin:.3f}_{mmax:.3f}.dat'
