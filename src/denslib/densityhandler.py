from __future__ import annotations
import pathlib
import warnings
from typing import Protocol, Sequence

import numpy.typing as npt
import pandas as pd

import config
from src import denslib


class Density(Protocol):
    lookup: str
    radius: float | None
    rp_max: float | None
    rpi_max: float | None
    value: npt.NDArray

    def get_value(
        self, data: pd.DataFrame, remove_null: bool = False, log: bool = False
    ) -> npt.NDArray: ...


class EnvironmentMeasure(Protocol):
    label: str


class DensityHandler:
    """
    Handle the density file saving and loading.

    """

    def __init__(self, configuration: denslib.config.DensityConfig) -> None:
        self.configuration = configuration
        self.data = self.configuration.load()
        self.merge_col = self.configuration.dataset.merge_col
        self._path = denslib.pathing.get_density_results_path(
            self.configuration.dirname
        )
        self._dataname = self._path.name
        self._make_filepath()

    def _make_filepath(self) -> None:
        """
        Create a Path object for the location of the density.

        """
        ddp_lim = denslib.config.DDP_MASS_BINS
        filename = (
            f'mstar{config.MASS_BINS[0]}-{config.MASS_BINS[-1]}'
            f'_ddp{ddp_lim[0]}-{ddp_lim[1]}.parquet'
        )
        self._filepath = pathlib.Path.joinpath(self._path, filename)

    @staticmethod
    def _arg_validation(
        radius: float | None = None,
        rp_max: float | None = None,
        rpi_max: float | None = None,
    ) -> None:
        """
        Ensure a proper combination of args is given.

        """
        rad_given = radius is not None
        rp_given = rp_max is not None
        rpi_given = rpi_max is not None

        kwarg_error = (
            "A 'radius' or pair of 'rp_max' and 'rpi_max' must"
            f" be given (received: {radius=}, {rp_max=}, {rpi_max=})."
        )

        if (rad_given and (rp_given or rpi_given)) or (
            not rad_given and (not rp_given or not rpi_given)
        ):
            raise ValueError(kwarg_error)

    @property
    def filepath(self) -> pathlib.Path:
        return self._filepath

    @property
    def dataname(self) -> str:
        return self._dataname

    @staticmethod
    def get_label(
        radius: float | None = None,
        rp_max: float | None = None,
        rpi_max: float | None = None,
    ) -> str:
        """
        Return column label(s) for density.

        """
        DensityHandler._arg_validation(radius, rp_max, rpi_max)

        if radius is not None:
            return f'radius{radius:.1f}'
        else:
            return f'rp{rp_max:.1f}_rpi{rpi_max:.1f}'

    def load(self, density: Sequence[Density]) -> tuple[pd.DataFrame, list[Density]]:
        """
        Load density data and merge into main table. Additionally, set
        Density.value for each in density.

        """
        dens_data = pd.read_parquet(self._filepath)

        if len(self.data) != len(dens_data):
            warnings.warn(
                message=(
                    "The number of densities is not equal to the length of the data"
                    " set. You may need to recalculate the densities with your current"
                    " settings."
                ),
                category=UserWarning,
            )

        merged_data = self.data
        for d in density:
            self._arg_validation(d.radius, d.rp_max, d.rpi_max)

            if d.radius is not None:
                suffix = self.get_label(radius=d.radius)
            else:
                suffix = self.get_label(rp_max=d.rp_max, rpi_max=d.rpi_max)

            merged_data = merged_data.merge(
                dens_data[self.merge_col + [f'{d.lookup}_{suffix}']],
                how='left',
                on=self.merge_col,
            )
            setattr(d, 'value', d.get_value(merged_data))

        return merged_data, list(density)

    def save(
        self,
        env_measure: Sequence[EnvironmentMeasure],
        env_in_vol: dict[int, npt.NDArray],
        radius: float | None = None,
        rp_max: float | None = None,
        rpi_max: float | None = None,
    ) -> None:
        """
        Save density data to a file.

        """
        self._arg_validation(radius, rp_max, rpi_max)

        if radius is not None:
            suffix = self.get_label(radius=radius)
        else:
            suffix = self.get_label(rp_max=rp_max, rpi_max=rpi_max)

        data_to_write = self.data[self.merge_col].copy()
        labels = [f'{env_meas.label}_{suffix}' for env_meas in env_measure]

        for elabel, e_in_vol in zip(labels, env_in_vol.values()):
            data_to_write[f'{elabel}'] = e_in_vol

        # Merge new results with previous ones if they exist
        try:
            prev_data = pd.read_parquet(self._filepath)

            # Keep the latest results if any columns repeat
            for l in labels:
                try:
                    prev_data = prev_data.drop(l, axis=1)
                except KeyError:
                    pass

            # If there are duplicates in the data set used, this will
            # prevent the dataframe from expanding upon merging
            data_to_write = data_to_write.drop_duplicates()

            new_data = prev_data.merge(data_to_write, how='left', on=self.merge_col)

        except FileNotFoundError:
            new_data = data_to_write

        new_data.to_parquet(self._filepath, index=False)
