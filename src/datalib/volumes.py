from dataclasses import dataclass
from typing import Sequence


@dataclass
class Volume:
    mass_lims: tuple[float, float]
    redshift_lims: tuple[float, float]


# Completeness from number densities down to 50% (see Yang et al. 2012)
VOLUME1: list[Volume] = [
    Volume((10.000, 10.375), (0.020, 0.065)),
    Volume((10.375, 10.750), (0.020, 0.085)),
    Volume((10.750, 11.000), (0.020, 0.125)),
    Volume((11.000, 11.250), (0.060, 0.155)),
    Volume((11.250, 11.500), (0.095, 0.190)),
]
# Completeness from van den Bosch et al. 2008
VOLUME2: list[Volume] = [
    Volume((10.000, 10.375), (0.020, 0.060)),
    Volume((10.375, 10.750), (0.020, 0.086)),
    Volume((10.750, 11.000), (0.020, 0.124)),
    Volume((11.000, 11.250), (0.020, 0.156)),
    Volume((11.250, 11.500), (0.020, 0.196)),
]
# Completeness from number densities down to 50% for single volume
VOLUME3: list[Volume] = [
    Volume((10.0, 12.0), (0.020, 0.080)),
]


def mass_bins_from_volume(volume: Sequence[Volume]) -> list[float]:
    """
    Extract the mass bins from a volume.

    """
    mass_bins = []
    for i, vol in enumerate(volume):
        mass_bins.append(vol.mass_lims[0])
        if i == len(volume) - 1:
            mass_bins.append(vol.mass_lims[-1])

    return mass_bins
