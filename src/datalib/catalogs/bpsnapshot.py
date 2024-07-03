import pathlib

import numpy as np
import pandas as pd

import config

from src import galhalo
from src.cosmo import constants as consts
from src.datalib import dataprocess
from src.protocols import coords


class BPSnapshot:
    """
    A dataset of one snapshot of the Bolshoi-Planck Lambda-CDM N-body
    simulation.

    """

    data: pd.DataFrame
    filename: str
    merge_col = ['ID', 'PID']
    # H0 = 100  # H0 = 70 for stellar masses!
    H0_mass = 67.8  # 70
    coord_strat: type[coords.CoordinateStrategy] = coords.Cartesian

    def __init__(
        self,
        redshift: float,
        scatter: float,
        data_dir: pathlib.Path = config.BP_DATA_DIR,
        data_cuts: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.redshift = redshift
        self.scatter = scatter
        self.data_dir = data_dir
        self.data_cuts = data_cuts
        self.filename = galhalo.names.get_base_mock_filename(redshift)

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

        if self.scatter is not None:
            data['M_star'] = data[f'M_star_{self.scatter:.3f}']

        data = dataprocess.check_log(data)

        if mass_H0 is not None and mass_H0 != consts.H0:
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

    def merge_sample_properties(
        self, results_path: pathlib.Path, obs_label: str, sim_label: str
    ) -> pd.DataFrame:
        """
        Merge sampled values of obs_label into the mock.

        """
        samples = galhalo.pathing.load_galhalo_sample(
            results_path, obs_label, sim_label
        )
        self.data = self.data.merge(samples, how='left', on=self.merge_col)

        return self.data


class BPSnapshotZSpace(BPSnapshot):
    """
    A dataset of multiple snapshots of the Bolshoi-Planck Lambda-CDM
    N-body simulation. Different mass ranges are taken from different
    snapshots, and the snapshots have been projected into redshift-
    space.

    """

    def __init__(
        self,
        redshift: float,
        scatter: float,
        data_dir: pathlib.Path = config.BP_DATA_DIR,
        data_cuts: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.redshift = redshift
        self.scatter = scatter
        self.data_dir = data_dir
        self.data_cuts = data_cuts
        self.filename = galhalo.names.get_mock_filename(redshift, self.scatter)

    @property
    def cf_runner(self):
        from src.tpcf import tpcf as cf

        return cf.SimSnapshotZSpaceTPCF

    @property
    def cf_cross_runner(self):
        from src.tpcf import tpcf as cf

        return cf.SimSnapshotZSpaceCrossTPCF

    def load(self, mass_H0: float | None = consts.H0, **kwargs) -> pd.DataFrame:
        filepath = self.data_dir.joinpath(self.filename)
        data = pd.read_parquet(filepath, **kwargs)
        data = data.dropna(subset=['M_star'])
        data = dataprocess.check_log(data)

        if mass_H0 is not None and mass_H0 != consts.H0:
            data = self.adjust_mass_cosmology(data, new_H0=mass_H0)

        if self.data_cuts is not None:
            data = dataprocess.apply_cuts(data, self.data_cuts)

        self.data = data
        return data

    def merge_sample_properties(
        self, results_path: pathlib.Path, obs_label: str, sim_label: str
    ) -> pd.DataFrame:
        samples = galhalo.pathing.load_galhalo_sample(
            results_path, obs_label, sim_label
        )
        self.data = self.data.merge(samples, how='left', on=self.merge_col)

        return self.data
