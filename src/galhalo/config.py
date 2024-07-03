from typing import Protocol

import pandas as pd

from src import configurations, datalib
from src.protocols import binning, conditionals, properties


class GalHaloConfig(Protocol):
    dirname: str
    scale_factors: list[float]
    scatter: float
    save_redshift: float
    dataset: datalib.DataSet
    mass_bins: list[float]
    gal_config: configurations.MPAConfig
    volume: list[datalib.volumes.Volume]
    centrals: list[type[conditionals.Conditional]]
    satellites: list[type[conditionals.Conditional]]
    cond_label: str

    @property
    def scatter_str(self) -> str: ...

    def get_sample_mass_bins(self) -> list[float]: ...

    def load(self, galhalo: bool = True) -> pd.DataFrame: ...

    def set_galhalo_props(self, gal: str, halo: str) -> None: ...

    def set_scatter(self, scatter: float) -> None: ...


# =========================================================
# DATA SETUP
# =========================================================
CONFIG_TYPE: type[GalHaloConfig] = configurations.BPConfigVolume1
CONFIG = CONFIG_TYPE()

# =========================================================
# BINNING
# =========================================================
MASS = properties.Mstar
NUM_SIZE_BINS = 5
NUM_DELTAMS_BINS = 4
NUM_TFORM_BINS = 5

# Only Yang centrals are used here because the stellar mass relations
# were determined based on Yang satellite fractions.
GAL_CENTRALS: type[conditionals.Conditional] = conditionals.Centrals

# (Galaxy property, Halo property, if correlation is reversed)
PAIRINGS: dict[
    tuple[type[properties.GalHaloProperty], type[properties.Property]], bool
] = {
    # (properties.Size, properties.AccretionRate): False,
    # (properties.Size, properties.Concentration): True,
    # (properties.Size, properties.Spin): False,
    # (properties.Size, properties.Vpeak): True,
    (properties.SSFR, properties.AccretionRate): False,
    (properties.SSFR, properties.Concentration): True,
    (properties.SSFR, properties.Vpeak): True,
    # (properties.TForm, properties.AccretionRate): False,
    # (properties.TForm, properties.Concentration): True,
    # (properties.TForm, properties.Vpeak): True,
    # (properties.SSFR, properties.Mvir): True,
    # (properties.SSFR, properties.Vmax): True,
    # (properties.SSFR, properties.Mpeak): True,
}
YBIN_SCHEMES: dict[
    tuple[type[properties.GalHaloProperty], type[properties.Property]],
    configurations.YbinScheme,
] = {
    (properties.Size, properties.AccretionRate): configurations.YbinScheme(
        NUM_SIZE_BINS, binning.PercentileBins, properties.Size
    ),
    (properties.Size, properties.Concentration): configurations.YbinScheme(
        NUM_SIZE_BINS, binning.PercentileBins, properties.Size
    ),
    (properties.Size, properties.Spin): configurations.YbinScheme(
        NUM_SIZE_BINS, binning.PercentileBins, properties.Size
    ),
    (properties.Size, properties.Vpeak): configurations.YbinScheme(
        NUM_SIZE_BINS, binning.PercentileBins, properties.Size
    ),
    (properties.SSFR, properties.AccretionRate): configurations.YbinScheme(
        NUM_DELTAMS_BINS, binning.DeltaMSBins, properties.DeltaMS
    ),
    (properties.SSFR, properties.Concentration): configurations.YbinScheme(
        NUM_DELTAMS_BINS, binning.DeltaMSBins, properties.DeltaMS
    ),
    (properties.SSFR, properties.Vpeak): configurations.YbinScheme(
        NUM_DELTAMS_BINS, binning.DeltaMSBins, properties.DeltaMS
    ),
    (properties.TForm, properties.AccretionRate): configurations.YbinScheme(
        NUM_TFORM_BINS, binning.PercentileBins, properties.TForm
    ),
    (properties.TForm, properties.Concentration): configurations.YbinScheme(
        NUM_TFORM_BINS, binning.PercentileBins, properties.TForm
    ),
    (properties.TForm, properties.Vpeak): configurations.YbinScheme(
        NUM_TFORM_BINS, binning.PercentileBins, properties.TForm
    ),
    (properties.SSFR, properties.Mvir): configurations.YbinScheme(
        NUM_DELTAMS_BINS, binning.DeltaMSBins, properties.DeltaMS
    ),
    (properties.SSFR, properties.Vmax): configurations.YbinScheme(
        NUM_DELTAMS_BINS, binning.DeltaMSBins, properties.DeltaMS
    ),
    (properties.SSFR, properties.Mpeak): configurations.YbinScheme(
        NUM_DELTAMS_BINS, binning.DeltaMSBins, properties.DeltaMS
    ),
}
