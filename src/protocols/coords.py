from abc import ABC, abstractmethod

import numpy.typing as npt
import pandas as pd

from src import cosmo


class CoordinateStrategy(ABC):
    def __init__(self, data: pd.DataFrame, ddp_mass_bins: tuple[float, float]) -> None:
        self.data = data
        self.ddp_mass_bins = ddp_mass_bins

        self._set_ddp_condition(self.data['M_star'].to_numpy(), self.ddp_mass_bins)

    @abstractmethod
    def get_data_coords(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Get the appropriate data coordinates for the data set.

        """

    def _set_ddp_condition(
        self, mass: npt.NDArray, ddp_mass_bins: tuple[float, float]
    ) -> None:
        """
        Find all galaxies within the limits of ddp_mass_bins.

        """
        self.ddp_mass_cond = (mass >= ddp_mass_bins[0]) & (mass <= ddp_mass_bins[-1])

    def get_ddp_coords(
        self, x: npt.NDArray, y: npt.NDArray, z: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        x_ddp = x[self.ddp_mass_cond]
        y_ddp = y[self.ddp_mass_cond]
        z_ddp = z[self.ddp_mass_cond]

        return x_ddp, y_ddp, z_ddp


class Cartesian(CoordinateStrategy):
    def get_data_coords(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        x = self.data['X'].to_numpy()
        y = self.data['Y'].to_numpy()
        z = self.data['Z'].to_numpy()

        return x, y, z


class LightCone(CoordinateStrategy):
    def get_data_coords(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        x, y, z = cosmo.coords.convert_spherical_to_cartesian(
            self.data['RA'], self.data['DEC'], self.data['z_obs'], H0=100
        )

        return x, y, z
