from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class BinningStrategy(Protocol):
    file_label: str

    def __init__(self, x: npt.NDArray) -> None: ...

    def get_bins(self, n: int) -> list[float]:
        """
        Get bin edges (excluding the endpoints) that break the data into
        'n' bins.

        """
        ...


class EvenlySpacedBins:
    file_label = 'evenbins'

    def __init__(self, x: npt.NDArray) -> None:
        self.x = x

    def __str__(self):
        return 'Evenly-Spaced Bins'

    def __format__(self, format_spec):
        return f'{str(self):{format_spec}}'

    def get_bins(self, n: int) -> list[float]:
        """
        Get the binning that breaks 'x' into 'n' bins that all have
        equal width.

        """
        bins = np.linspace(min(self.x), max(self.x), n + 1)

        return list(bins[1:-1])


class PercentileBins:
    file_label = 'percbins'

    def __init__(self, x: npt.NDArray) -> None:
        self.x = x

    def __str__(self):
        return 'Percentile Bins'

    def __format__(self, format_spec):
        return f'{str(self):{format_spec}}'

    def get_bins(self, n: int) -> list[float]:
        """
        Get the binning that breaks 'x' into 'n' bins each containing
        an equal number of points.

        """
        base_percentile = 100 / n
        pbins = base_percentile * np.arange(1, n)

        return list(np.percentile(self.x, pbins))


class DeltaMSBins:
    file_label = 'deltaMS'
    upper_ms_upper_lim = 0.25
    ms_center = 0.0
    bottom_ms_upper_lim = -0.25
    green_valley_upper_lim = -0.45
    quenched_upper_lim = -1
    bins = {
        2: [green_valley_upper_lim],
        3: [quenched_upper_lim, green_valley_upper_lim],
        4: [quenched_upper_lim, green_valley_upper_lim, ms_center],
        5: [
            quenched_upper_lim,
            green_valley_upper_lim,
            bottom_ms_upper_lim,
            upper_ms_upper_lim,
        ],
        6: [
            quenched_upper_lim,
            green_valley_upper_lim,
            bottom_ms_upper_lim,
            ms_center,
            upper_ms_upper_lim,
        ],
    }

    def __init__(self, x: npt.NDArray = np.empty(0)) -> None:
        self.x = x

    def __str__(self):
        return 'Delta MS Bins'

    def __format__(self, format_spec):
        return f'{str(self):{format_spec}}'

    def get_bins(self, n: int) -> list[float]:
        """
        Get the binning based on the number of bins given. Since delta
        MS is a relative measure calculated using the data, the bins
        are fixed and do not need to be calculated.

        """
        bins = self.bins.get(n, None)

        if bins is None:
            raise ValueError(
                f"'n' must be among {list(self.bins.keys())} for delta MS"
                f" bins ({n=} was given)."
            )

        return bins
