from typing import Sequence

import numpy as np
import pandas as pd

import config
import src.cosmo.constants as consts
from src.datalib import dataprocess
from src.protocols import coords


class Empire:
    """
    A dataset of one snapshot of the Bolshoi-Planck Lambda-CDM N-body
    simulation filled with galaxies by Empire.

    """

    data_dir = config.DATA_DIR
    data = pd.DataFrame()  # for typing purposes, will be set when loaded
    merge_col = ['ID', 'PID']
    H0 = 67.8  # H0 = 70 for stellar masses!
    H0_mass = 67.8
    coord_strat: type[coords.CoordinateStrategy] = coords.Cartesian

    def __init__(
        self,
        redshift: float,
        data_cuts: dict[str, tuple[float, float]] | None = None,
        mass_bins: Sequence[float] | None = None,
    ) -> None:
        self.redshift = redshift
        self.data_cuts = data_cuts
        self.mass_bins = mass_bins
        self.filename = f'empire_{self.redshift:.4f}.parquet'

    @property
    def cf_runner(self):
        from src.tpcf import tpcf as cf

        return cf.SimSnapshotTPCF

    @property
    def cf_cross_runner(self):
        from src.tpcf import tpcf as cf

        return cf.SimSnapshotCrossTPCF

    @property
    def filename_str(self) -> str:
        return self.filename.rsplit('.', maxsplit=1)[0]

    def load(self, mass_H0: float | None = consts.H0, **kwargs) -> pd.DataFrame:
        filepath = self.data_dir.joinpath(self.filename)
        data = pd.read_parquet(filepath, **kwargs)
        # data = dataprocess.check_log(data)

        if mass_H0 is not None:
            data = self.adjust_mass_cosmology(data, new_H0=mass_H0)

        if self.data_cuts is not None:
            data = dataprocess.apply_cuts(data, self.data_cuts)

        self.data = data
        return data

    def adjust_mass_cosmology(
        self, data: pd.DataFrame, new_H0: float = consts.H0
    ) -> pd.DataFrame:
        """
        Adjust the stellar masses using a new Hubble parameter.

        """
        hubble_correction = 2 * np.log10((self.H0_mass / 100) / (new_H0 / 100))
        data['M_star'] = data['M_star'] + hubble_correction

        return data
