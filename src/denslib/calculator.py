from __future__ import annotations
from typing import TYPE_CHECKING, Sequence

import numpy as np
import numpy.typing as npt

from src.denslib import proj_neighbors, sphere
from src.protocols import coords

if TYPE_CHECKING:
    from src.denslib import DensityHandler, EnvironmentMeasure


NO_DENS_NUM = 0


class DensityCalculator:
    """
    Handles different methods for calculating densities.

    """

    def __init__(self, strategy: coords.CoordinateStrategy) -> None:
        self.strategy = strategy

        x, y, z = strategy.get_data_coords()
        x_ddp, y_ddp, z_ddp = strategy.get_ddp_coords(x, y, z)
        self.pos = np.array([x, y, z]).T
        self.pos_ddp = np.array([x_ddp, y_ddp, z_ddp]).T

    def _return_results(
        self, env_in_vol: dict[int, npt.NDArray], volume: float
    ) -> dict[int, npt.NDArray]:
        env_results = {}

        # Keep track of no-density results
        nodens_cond = env_in_vol[0] == 0

        for j, e_in_vol in env_in_vol.items():
            env_results[j] = e_in_vol / volume

        for result in env_results.values():
            result[nodens_cond] = NO_DENS_NUM

        return env_results

    def run(
        self,
        denshandler: DensityHandler,
        env_measure: Sequence[EnvironmentMeasure],
        radius: float | None = None,
        rp_max: float | None = None,
        rpi_max: float | None = None,
        save: bool = True,
    ) -> dict[int, npt.NDArray]:
        """
        Run the density calculation and save the results.

        """
        denshandler._arg_validation(radius, rp_max, rpi_max)

        if radius is not None:
            env_in_vol = self.calc_sphere_density(env_measure, radius, chunksize=1000)
            if save:
                denshandler.save(env_measure, env_in_vol, radius=radius)

        elif rp_max is not None and rpi_max is not None:
            env_in_vol = self.calc_proj_density(
                env_measure, rp_max, rpi_max, chunksize=1000
            )
            if save:
                denshandler.save(
                    env_measure, env_in_vol, rp_max=rp_max, rpi_max=rpi_max
                )

        return env_in_vol

    def calc_proj_density(
        self,
        env_measure: Sequence[EnvironmentMeasure],
        rp_max: float,
        rpi_max: float,
        chunksize: int = 1000,
    ) -> dict[int, npt.NDArray]:
        """
        Calculate the projected density within a cylinder.

        """
        env_in_vol = proj_neighbors.find_all_neighbors(
            env_measure, rp_max, rpi_max, self.pos, self.pos_ddp, chunksize=chunksize
        )
        volume = np.pi * rp_max**2 * 2 * rpi_max

        return self._return_results(env_in_vol, volume)

    def calc_sphere_density(
        self,
        env_measure: Sequence[EnvironmentMeasure],
        radius: float,
        chunksize: int = 1000,
    ) -> dict[int, npt.NDArray]:
        """
        Calculate the density within a sphere.

        """
        env_in_vol = sphere.query_ball(
            env_measure, self.pos, self.pos_ddp, radius, chunksize=chunksize
        )
        volume = 4 / 3 * np.pi * radius**3

        return self._return_results(env_in_vol, volume)
