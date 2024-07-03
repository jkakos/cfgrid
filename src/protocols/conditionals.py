from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd


class Conditional(Protocol):
    label: str
    group_id_label: str
    legend_label: str
    value: npt.NDArray

    def __init__(self, data: pd.DataFrame) -> None: ...

    @staticmethod
    def get_cond(data: pd.DataFrame) -> npt.NDArray:
        """Return the condition array with which to filter the data."""
        ...


class BaseConditional:
    def __format__(self, format_spec):
        return f'{str(self):{format_spec}}'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__class__)


class AllGalaxies(BaseConditional):
    label = 'all'
    group_id_label = 'none'
    legend_label = 'All Galaxies'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = self.get_cond(data)

    def __str__(self):
        return 'All Galaxies'

    @staticmethod
    def get_cond(data: pd.DataFrame) -> npt.NDArray:
        return np.ones(len(data), dtype=bool)


class Centrals(BaseConditional):
    label = 'centrals'
    group_id_label = 'yang_group_id'
    legend_label = 'Yang+ centrals'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = self.get_cond(data)

    def __str__(self):
        return 'Yang Centrals'

    @staticmethod
    def get_cond(data: pd.DataFrame) -> npt.NDArray:
        pid = data.get('PID')
        if data.get('central') is not None:  # TNG300
            return data['central'].to_numpy(copy=True) == 1

        if pid is not None:  # simulation
            return data['PID'].to_numpy(copy=True) == -1
        elif data.get('sat') is not None:
            return data['sat'].to_numpy(copy=True) == 0
        else:  # mpa
            return data['most_massive'].to_numpy(copy=True) == 1


class Satellites(BaseConditional):
    label = 'satellites'
    group_id_label = 'yang_group_id'
    legend_label = 'Yang+ satellites'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = self.get_cond(data)

    def __str__(self):
        return 'Yang Satellites'

    @staticmethod
    def get_cond(data: pd.DataFrame) -> npt.NDArray:
        pid = data.get('PID')

        if data.get('sat') is not None:  # TNG300
            return data['sat'].to_numpy(copy=True) == 1

        if pid is not None:  # simulation
            return data['PID'].to_numpy(copy=True) != -1
        elif data.get('sat') is not None:
            return data['sat'].to_numpy(copy=True) == 1
        else:  # mpa
            return data['most_massive'].to_numpy(copy=True) == 2


class TempelCentrals(BaseConditional):
    label = 'tempel_centrals'
    group_id_label = 'Tempel_group_id'
    legend_label = 'Tempel+ centrals'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = self.get_cond(data)

    def __str__(self):
        return 'Tempel Centrals'

    @staticmethod
    def get_cond(data: pd.DataFrame) -> npt.NDArray:
        return data['Tempel_most_massive'].to_numpy(copy=True) == 1


class TempelSatellites(BaseConditional):
    label = 'tempel_satellites'
    group_id_label = 'Tempel_group_id'
    legend_label = 'Tempel+ satellites'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = self.get_cond(data)

    def __str__(self):
        return 'Tempel Satellites'

    @staticmethod
    def get_cond(data: pd.DataFrame) -> npt.NDArray:
        return data['Tempel_most_massive'].to_numpy(copy=True) == 0


class FRodriguezCentrals(BaseConditional):
    label = 'frodriguez_centrals'
    group_id_label = 'FRod_group_id'
    legend_label = 'Rodriguez+ centrals'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = self.get_cond(data)

    def __str__(self):
        return 'Rodriguez Centrals'

    @staticmethod
    def get_cond(data: pd.DataFrame) -> npt.NDArray:
        return data['FRod_most_massive'].to_numpy(copy=True) == 1


class FRodriguezSatellites(BaseConditional):
    label = 'frodriguez_satellites'
    group_id_label = 'FRod_group_id'
    legend_label = 'Rodriguez+ satellites'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = self.get_cond(data)

    def __str__(self):
        return 'Rodriguez Satellites'

    @staticmethod
    def get_cond(data: pd.DataFrame) -> npt.NDArray:
        return data['FRod_most_massive'].to_numpy(copy=True) == 0
