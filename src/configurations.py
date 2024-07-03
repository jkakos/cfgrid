from typing import Any, Sequence

import numpy as np
import pandas as pd

from src import datalib
from src.galhalo import pathing as ghpath
from src.protocols import binning, conditionals, properties


class YbinScheme:
    """
    Track the sets of different parameters that would define a ybin
    scheme when forming grids.

    """

    def __init__(
        self,
        num_bins: int,
        bin_strategy: type[binning.BinningStrategy],
        property_: type[properties.Property],
    ) -> None:
        self._num_bins = num_bins
        self._bin_strategy = bin_strategy
        self._property_ = property_

    @property
    def num_bins(self) -> int:
        return self._num_bins

    @property
    def bin_strategy(self) -> type[binning.BinningStrategy]:
        return self._bin_strategy

    @property
    def property_(self) -> type[properties.Property]:
        return self._property_


class Configuration:
    dirname: str
    dataset_type: datalib.DataSet
    volume: list[datalib.volumes.Volume]

    def __init__(self) -> None:
        self.dataset = self.dataset_type(**self.get_load_settings())

    def get_load_settings(self): ...

    def get_nsat_filename(self) -> str:
        return self.dataset.filename_str

    def load(self) -> pd.DataFrame:
        return self.dataset.load()

    def set_mass_bins(self, mass_bins: Sequence[float]) -> None:
        self.mass_bins = list(mass_bins)


class MPAConfig(Configuration):
    dataset_type = datalib.MPAJHU
    ybin_schemes: list[YbinScheme] = [
        # YbinScheme(3, binning.DeltaMSBins, properties.DeltaMS),
        YbinScheme(4, binning.DeltaMSBins, properties.DeltaMS),
        # YbinScheme(6, binning.DeltaMSBins, properties.DeltaMS),
        # YbinScheme(5, binning.PercentileBins, properties.Size),
        # YbinScheme(5, binning.PercentileBins, properties.TForm),
    ]
    centrals: list[type[conditionals.Conditional]] = [
        conditionals.Centrals,
        conditionals.TempelCentrals,
        conditionals.FRodriguezCentrals,
    ]
    satellites: list[type[conditionals.Conditional]] = [
        conditionals.Satellites,
        conditionals.TempelSatellites,
        conditionals.FRodriguezSatellites,
    ]

    def __init__(self) -> None:
        self.mass_bins = datalib.volumes.mass_bins_from_volume(self.volume)
        self.dataset = self.dataset_type(**self.get_load_settings())

    def get_load_settings(self) -> dict[str, Any]:
        return dict(volume=self.volume)


class BPConfig(Configuration):
    scatter: float
    scale_factors: list[float]
    save_redshift: float
    dataset_type = datalib.BPSnapshotZSpace
    mass_bins: list[float]
    gal_property: str
    halo_property: str
    gal_config: MPAConfig
    volume: list[datalib.volumes.Volume]
    ybin_schemes: list[YbinScheme] = [
        # YbinScheme(3, binning.DeltaMSBins, properties.DeltaMS),
        YbinScheme(4, binning.DeltaMSBins, properties.DeltaMS),
        # YbinScheme(5, binning.PercentileBins, properties.Size),
        # YbinScheme(5, binning.PercentileBins, properties.TForm),
    ]
    centrals: list[type[conditionals.Conditional]] = [conditionals.Centrals]
    satellites: list[type[conditionals.Conditional]] = [conditionals.Satellites]
    cond_label: str = centrals[0].label

    def __init__(self) -> None:
        self.volume = self.gal_config.volume
        self.mass_bins = datalib.volumes.mass_bins_from_volume(self.volume)
        self.dataset = self.dataset_type(**self.get_load_settings())

    @property
    def scatter_str(self) -> str:
        return f'{self.scatter:.3f}'

    def get_nsat_filename(self) -> str:
        """
        Get the filename prefix to use for Nsat results.

        """
        return f'{self.dataset.filename_str}_{self.halo_property}'

    def get_load_settings(self) -> dict[str, Any]:
        return dict(redshift=self.save_redshift, scatter=self.scatter)

    def load(self, galhalo: bool = True) -> pd.DataFrame:
        """
        Load BP data. If 'galhalo' is True, then load galhalo sample
        results.

        """
        # self.dataset = self.dataset(**self.get_load_settings())
        self.data = self.dataset.load()

        if galhalo:
            self.dataset.merge_sample_properties(
                ghpath.get_galhalo_sample_path(self),
                self.gal_property,
                self.halo_property,
            )
            self.data = self.dataset.data

        return self.data

    def set_galhalo_props(self, gal: str, halo: str) -> None:
        """
        Sets the gal and halo properties for the galhalo model. These
        should be given as the 'file_label' attributes of different
        galaxy or halo property objects.

        """
        self.gal_property = gal
        self.halo_property = halo

    def set_scatter(self, scatter: float) -> None:
        self.scatter = scatter


class EmpireConfig(Configuration):
    dirname = 'empire'
    dataset_type = datalib.Empire
    redshift = 0.0
    obs_config: MPAConfig
    mass_bins: list[float]
    volume: list[datalib.volumes.Volume]
    ybin_schemes: list[YbinScheme] = [
        # YbinScheme(3, binning.DeltaMSBins, properties.DeltaMS),
        YbinScheme(4, binning.DeltaMSBins, properties.DeltaMS),
        # YbinScheme(5, binning.PercentileBins, properties.Size),
        # YbinScheme(5, binning.PercentileBins, properties.TForm),
    ]
    centrals: list[type[conditionals.Conditional]] = [conditionals.Centrals]
    satellites: list[type[conditionals.Conditional]] = [conditionals.Satellites]

    def __init__(self) -> None:
        volume = self.obs_config.volume
        self.mass_bins = datalib.volumes.mass_bins_from_volume(volume)
        self.volume = [
            datalib.volumes.Volume(
                (self.mass_bins[0], self.mass_bins[-1]), (self.redshift, self.redshift)
            )
        ]
        self.dataset = self.dataset_type(**self.get_load_settings())

    def get_load_settings(self) -> dict[str, Any]:
        return dict(
            redshift=self.redshift, data_cuts=dict(M_star=self.volume[0].mass_lims)
        )

    def load_nsat(self) -> pd.DataFrame:
        """
        Load data for use in Nsat calculations. Create a column that
        designates group IDs using 'ID' and 'PID'.

        """
        data = self.load()
        group_id_label = self.centrals[0].group_id_label
        data[group_id_label] = data['ID']
        data.loc[self.satellites[0](data).value, group_id_label] = data['PID']
        self.data = data

        return self.data


class MPAConfigVolume1(MPAConfig):
    dirname = 'mpa_volume1'
    volume = datalib.volumes.VOLUME1


class MPAConfigVolume2(MPAConfig):
    dirname = 'mpa_volume2'
    volume = datalib.volumes.VOLUME2


class MPAConfigVolume3(MPAConfig):
    dirname = 'mpa_volume3'
    volume = datalib.volumes.VOLUME3

    def __init__(self) -> None:
        self.mass_bins = datalib.volumes.mass_bins_from_volume(self.volume)
        self.dataset = self.dataset_type(**self.get_load_settings())

    def load_nsat(self) -> pd.DataFrame:
        """
        Load data for use in Nsat calculations. This expands the volume
        slightly for satellites so groups do not get cut by the volume
        edges.

        """
        redshift_expansion = 0.01
        zmin, zmax = self.volume[0].redshift_lims
        load_settings = self.get_load_settings()
        load_settings['volume'] = [
            datalib.volumes.Volume(
                self.volume[0].mass_lims,
                (zmin - redshift_expansion, zmax + redshift_expansion),
            )
        ]
        self.dataset = self.dataset_type(**load_settings)
        self.data = self.dataset.load()

        return self.data


class BPConfigVolume1(BPConfig):
    dirname = 'bp_volume1'
    scale_factors = [0.98712, 0.98712, 0.97193, 0.94156, 0.91119]
    scatter = 0.15
    save_redshift = 0.0
    gal_config: MPAConfig = MPAConfigVolume1()

    def get_sample_mass_bins(self) -> list[float]:
        """
        Get bins to use for galhalo sample procedure.

        """
        self.sample_mass_bins = [
            *np.linspace(self.mass_bins[0], self.mass_bins[1], 6 + 1)[1:-1],
            *np.linspace(self.mass_bins[1], self.mass_bins[2], 6 + 1)[:-1],
            *np.linspace(self.mass_bins[2], self.mass_bins[3], 4 + 1)[:-1],
            *np.linspace(self.mass_bins[3], self.mass_bins[4], 4 + 1)[:-1],
            *np.linspace(self.mass_bins[4], self.mass_bins[5], 4 + 1)[:-1],
        ]
        return self.sample_mass_bins


class BPConfigVolume3(BPConfig):
    dirname = 'bp_volume3'
    scale_factors = [0.98712]
    scatter = 0.15
    save_redshift = 0.04
    gal_config: MPAConfig = MPAConfigVolume3()

    def get_sample_mass_bins(self) -> list[float]:
        """
        Get bins to use for galhalo sample procedure.

        """
        m_low, m_high = self.volume[0].mass_lims
        self.sample_mass_bins = list(np.linspace(m_low, m_high, 36)[1:-1])
        return self.sample_mass_bins

    def load_nsat(self) -> pd.DataFrame:
        """
        Load data for use in Nsat calculations. Create a column that
        designates group IDs using 'ID' and 'PID'.

        """
        data = self.load()
        group_id_label = self.centrals[0].group_id_label
        data[group_id_label] = data['ID']
        data.loc[self.satellites[0](data).value, group_id_label] = data['PID']
        self.data = self.dataset.data

        return self.data


class EmpireConfigVolume1(EmpireConfig):
    # dirname = 'empire_volume1'
    obs_config: MPAConfig = MPAConfigVolume1()


class EmpireConfigVolume3(EmpireConfig):
    # dirname = 'empire_volume3'
    obs_config: MPAConfig = MPAConfigVolume3()
