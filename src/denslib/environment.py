import enum
from typing import Protocol, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.denslib import calculator
from src.denslib.densityhandler import DensityHandler
from src.protocols import properties


class EnvironmentMeasure(Protocol):
    label: str

    def measure(self, selection: npt.NDArray) -> float: ...


class TotalGalaxies:
    label = 'num_dens'

    def measure(self, selection: npt.NDArray) -> float:
        return len(selection)


class TotalMass:
    label = 'mass_dens'

    def __init__(self, data: pd.DataFrame) -> None:
        self.mass = properties.Mstar(data, log=False).value

    def measure(self, selection: npt.NDArray) -> float:
        return sum(self.mass[selection])


class TotalSFR:
    label = 'sfr_dens'

    def __init__(self, data: pd.DataFrame) -> None:
        self.sfr = properties.SFR(data, log=False).value

    def measure(self, selection: npt.NDArray) -> float:
        return sum(self.sfr[selection])


class TotalSSFR:
    label = 'ssfr_dens'

    def __init__(self, data: pd.DataFrame) -> None:
        self.ssfr = properties.SSFR(data, log=False).value

    def measure(self, selection: npt.NDArray) -> float:
        return sum(self.ssfr[selection])


# ---------------------------------------------------------------------
# Density Types
# ---------------------------------------------------------------------
class DensityTypeDict(TypedDict):
    name: str
    lookup: str
    label: str
    file_label: str
    full_label: str


class DensityType(enum.Enum):
    NUMBER = enum.auto()
    STELLARMASS = enum.auto()
    SFR = enum.auto()
    SSFR = enum.auto()


DENSITY_TYPE: dict[DensityType, DensityTypeDict] = {
    DensityType.NUMBER: dict(
        name='Number Density',
        lookup=TotalGalaxies.label,
        label='Number Density',
        file_label='numdens',
        full_label=r'$\log(\rho/h^{3}{\rm Mpc}^{-3})$',
    ),
    DensityType.STELLARMASS: dict(
        name='Stellar Mass Density',
        lookup=TotalMass.label,
        label='Mass Density',
        file_label='massdens',
        full_label=r'$\log(\rho/M_*h^{3}{\rm Mpc}^{-3})$',
    ),
    DensityType.SFR: dict(
        name='Star Formation Rate Density',
        lookup=TotalSFR.label,
        label='SFR Density',
        file_label='sfrdens',
        full_label=r'$\log(\rho/M_*{\rm yr}^{-1}h^{3}{\rm Mpc}^{-3})$',
    ),
    DensityType.SSFR: dict(
        name='Specific Star Formation Rate Density',
        lookup=TotalSSFR.label,
        label='sSFR Density',
        file_label='ssfrdens',
        full_label=r'$\log(\rho/{\rm yr}^{-1}h^{3}{\rm Mpc}^{-3})$',
    ),
}


class Density:
    name: str
    lookup: str
    label: str
    file_label: str
    full_label: str
    radius: float | None
    rp_max: float | None
    rpi_max: float | None
    value: npt.NDArray

    def __init__(
        self,
        dens_type: DensityType,
        radius: float | None = None,
        rp_max: float | None = None,
        rpi_max: float | None = None,
    ) -> None:
        for k, v in DENSITY_TYPE[dens_type].items():
            setattr(self, k, v)

        self.radius = radius
        self.rp_max = rp_max
        self.rpi_max = rpi_max

    def __format__(self, format_spec):
        return f'{str(self):{format_spec}}'

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def get_value(
        self, data: pd.DataFrame, remove_null: bool = False, log: bool = False
    ) -> npt.NDArray:
        dens_suffix = DensityHandler.get_label(self.radius, self.rp_max, self.rpi_max)
        dens = data[f'{self.lookup}_{dens_suffix}'].to_numpy(copy=True)

        if remove_null:
            dens = dens[self.get_good_cond()]

        if log:
            dens = np.log10(dens)

        return dens

    def get_good_cond(self) -> npt.NDArray:
        return self.value != calculator.NO_DENS_NUM
