import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import config
import src.cosmo.constants as consts
from src.datalib import dataprocess, volumes
from src.protocols import coords


class MPAJHU:
    """
    A dataset of Data Release 7 of the Sloan Digital Sky Survey.
    Stellar masses and star formation rates are from the Max Planck
    Institute for Astrophysics and Johns Hopkins University
    (MPA-JHU) catalog.

    """

    data_dir = config.DATA_DIR
    data = pd.DataFrame()  # for typing purposes, will be set when loaded
    filename = 'mpa_jhu.parquet'
    merge_col = ['ID_MPA']
    base_cuts: Mapping = dict(
        M_star=(6, 14),
        SFR=(-9, 8),
    )
    H0 = 70
    random_file = config.RANDOM
    coord_strat: type[coords.CoordinateStrategy] = coords.LightCone

    def __init__(
        self,
        data_cuts: dict[str, tuple[float, float]] | None = None,
        mass_bins: Sequence[float] | None = None,
        min_redshifts: Sequence[float] | None = None,
        volume: Sequence[volumes.Volume] | None = None,
    ) -> None:
        self.data_cuts = data_cuts
        self.mass_bins = mass_bins
        self.min_redshifts = min_redshifts
        self.volume = volume

    @property
    def cf_runner(self):
        from src.tpcf import tpcf as cf

        return cf.ObsLightConeTPCF

    @property
    def cf_cross_runner(self):
        from src.tpcf import tpcf as cf

        return cf.ObsLightConeCrossTPCF

    @property
    def filename_str(self) -> str:
        return self.filename.rsplit('.', maxsplit=1)[0]

    def load(
        self,
        mass_H0: float | None = consts.H0,
        apply_base_cuts: bool = True,
        cut_sdss_mgs: bool = True,
        cut_sdss_rects: bool = True,
        completeness: int = config.MPA_COMPLETENESS,
        unprocessed: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load MPA-JHU data. The Hubble parameter value used for the stellar masses
        is adjusted using 'mass_H0' (the value used in the catalog is
        H0 = 70 km/s/Mpc). The completeness condition should be given as an
        integer: either 0 for no completeness, 1 for Yang+2012 volume 1, or 2 for
        Yang+2012 volume 2 (van den Bosch+2008 condition).

        """
        filepath = self.data_dir.joinpath(self.filename)
        data = pd.read_parquet(filepath, **kwargs)

        if unprocessed:
            print('Using unprocessed data.')
            return data

        data = dataprocess.check_log(data)

        if mass_H0 is not None:
            data = self.adjust_mass_cosmology(data, new_H0=mass_H0)
        else:
            warnings.warn('Masses are not cosmology-corrected.')

        if apply_base_cuts:
            data = dataprocess.apply_cuts(data, self.base_cuts)

        if cut_sdss_mgs:
            data = dataprocess.cut_sdss_mgs(data)

        if cut_sdss_rects:
            data = dataprocess.cut_sdss_rects(data)

        if self.volume is not None:
            data = self.cut_to_volume(data)
        elif completeness == 0:
            data = dataprocess.apply_cuts(data, self.data_cuts)
            warnings.warn('Data might be incomplete.')
        else:
            mass_bins = dataprocess.get_mass_bins(self.data_cuts, self.mass_bins)
            data = dataprocess.apply_cuts(data, self.data_cuts)

            if completeness == 1:
                data = dataprocess.make_complete_vol1(data, mass_bins)
            elif completeness == 2:
                data = dataprocess.make_complete_vol2(
                    data, mass_bins, self.min_redshifts
                )
            else:
                raise ValueError(
                    f"Invalid completeness method specified (received {completeness})."
                )

        self.data = data
        return data

    def adjust_mass_cosmology(
        self, data: pd.DataFrame, new_H0: float = consts.H0
    ) -> pd.DataFrame:
        """
        Adjust the stellar masses using a new Hubble parameter.

        """
        hubble_correction = 2 * np.log10((self.H0 / 100) / (new_H0 / 100))
        data['M_star'] = data['M_star'] + hubble_correction

        return data

    def cut_to_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        assert self.volume is not None
        logM = data['M_star'].to_numpy()
        z = data['z_obs'].to_numpy()
        completeness_cond = dataprocess.cut_to_volume(logM, z, self.volume)

        return data[completeness_cond]
