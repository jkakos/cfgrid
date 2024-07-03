import copy
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

import config
from src import cosmo


@runtime_checkable
class Property(Protocol):
    label: str
    file_label: str
    full_label: str
    value: npt.NDArray

    def __init__(self, data: pd.DataFrame) -> None: ...

    def get_good_cond(self) -> npt.NDArray:
        """
        Get a bool array that chooses only the 'good' values in that
        given quantity. 'Good' values are ones that are not, e.g.,
        less than 0 when logs are being used.

        """
        ...


@runtime_checkable
class GalHaloProperty(Property, Protocol):
    """
    A Property that has bins for purposes of modeling that property
    in dark matter halos.

    """

    bins: list[float]


class BaseProperty:
    def __str__(self):
        return self.__class__.__name__

    def __format__(self, format_spec):
        return f'{str(self):{format_spec}}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return self.value == other.value

    def __hash__(self):
        return hash(self.__class__)


def standardize(*properties: Property) -> tuple[list[Property], npt.NDArray[np.bool_]]:
    """
    Take any number of Property objects, each with value arrays of
    length N, and apply a boolean conditional such that all value
    arrays consist of only 'good' values for their respective property.
    Each value array should end with length M where M <= N. Copies of
    the properties with updated value arrays are returned.

    """
    good_cond_arr = np.array([x.get_good_cond() for x in properties])
    all_good = np.logical_and.reduce(good_cond_arr)
    prop_copy = copy.deepcopy(properties)

    for property_ in prop_copy:
        property_.value = property_.value[all_good]

    return list(prop_copy), all_good


# ---------------------------------------------------------------------
# Halo properties
# ---------------------------------------------------------------------
class AccretionRate(BaseProperty):
    label = r'$\dot{M}_{1T{\rm dyn}}$'
    file_label = 'acc_rate_1tdyn'
    # full_label = r'$\log(\dot{M}_{1T_{\rm dyn}}/h^{-1}M_\odot{\rm yr}^{-1})$'
    full_label = r'$\dot{M}_{1T_{\rm dyn}}~[h^{-1}M_\odot{\rm yr}^{-1}]$'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = data['Acc_Rate_1*Tdyn'].to_numpy(copy=True)

        # accrate = data['Acc_Rate_1*Tdyn'].to_numpy(copy=True)
        # bad_accrate = (accrate <= 0)
        # accrate[~bad_accrate] = np.log10(accrate[~bad_accrate])
        # accrate[bad_accrate] = -99

        # return accrate

    def get_good_cond(self) -> npt.NDArray:
        # return quantity != -99
        return ~np.isnan(self.value)


class Concentration(BaseProperty):
    label = 'Cvir'
    file_label = 'Cvir'
    full_label = r'$\log(C_{\rm vir})$'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = np.log10((data['Rvir'] / data['Rs']).to_numpy(copy=True))

    def get_good_cond(self) -> npt.NDArray:
        return np.ones(len(self.value), dtype=bool)


class FormationRedshift(BaseProperty):
    label = 'zform'
    file_label = 'zform'
    full_label = r'$\log(z_{\rm form})$'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = np.log10(1 / data['Halfmass_Scale'] - 1).to_numpy(copy=True)

    def get_good_cond(self) -> npt.NDArray:
        return np.ones(len(self.value), dtype=bool)


class Mpeak(BaseProperty):
    label = r'$M_{\rm peak}$'
    file_label = 'Mpeak'
    full_label = r'$\log(M_{\rm peak}/M_\odot)$'

    def __init__(self, data: pd.DataFrame, log: bool = True) -> None:
        h = cosmo.constants.H0 / 100
        print(f'Using h = {h:.3f} with Mpeak')

        if not log:
            self.value = data['Mpeak'].to_numpy(copy=True) / h
        else:
            self.value = np.log10(data['Mpeak'] / h).to_numpy(copy=True)

    def get_good_cond(self) -> npt.NDArray:
        return self.value > 0


class Mvir(BaseProperty):
    label = r'$M_{\rm vir}$'
    file_label = 'Mvir'
    full_label = r'$\log(M_{\rm vir}/M_\odot)$'

    def __init__(self, data: pd.DataFrame, log: bool = True) -> None:
        h = cosmo.constants.H0 / 100
        print(f'Using h = {h:.3f} with Mvir')

        if not log:
            self.value = data['Mvir'].to_numpy(copy=True) / h
        else:
            self.value = np.log10(data['Mvir'] / h).to_numpy(copy=True)

    def get_good_cond(self) -> npt.NDArray:
        return self.value > 0


class SpecificAccretionRate(BaseProperty):
    label = r's$\dot{M}_{1T{\rm dyn}}$'
    file_label = 'sAcc_rate_1tdyn'
    full_label = r'$\dot{M}_{1T_{\rm dyn}}/M~[{\rm yr}^{-1}]$'

    def __init__(self, data: pd.DataFrame) -> None:
        accrate = data['Acc_Rate_1*Tdyn'].to_numpy(copy=True)
        mvir = data['Mvir'].to_numpy(copy=True)
        saccrate = accrate / mvir

        self.value = saccrate

    def get_good_cond(self) -> npt.NDArray:
        return self.value != -99


class Spin(BaseProperty):
    label = 'spin'
    file_label = 'spin'
    full_label = r'$\log({\rm Spin})$'

    def __init__(self, data: pd.DataFrame) -> None:
        spin = data['Spin'].to_numpy(copy=True)
        bad_spin = spin <= 0
        spin[~bad_spin] = np.log10(spin[~bad_spin])
        spin[bad_spin] = -99

        self.value = spin

    def get_good_cond(self) -> npt.NDArray:
        return self.value != -99


class Vmax(BaseProperty):
    label = 'Vmax'
    file_label = 'Vmax'
    full_label = r'$\log(V_{\rm max}/{\rm km~s}^{-1})$'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = np.log10(data['Vmax'].to_numpy(copy=True))

    def get_good_cond(self) -> npt.NDArray:
        return np.ones(len(self.value), dtype=bool)


class Vpeak(BaseProperty):
    label = 'Vpeak'
    file_label = 'Vpeak'
    full_label = r'$\log(V_{\rm peak}/{\rm km~s}^{-1})$'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = np.log10(data['Vpeak'].to_numpy(copy=True))

    def get_good_cond(self) -> npt.NDArray:
        return np.ones(len(self.value), dtype=bool)


class Vpeak2(BaseProperty):
    label = 'Vpeak^2'
    file_label = 'Vpeak2'
    full_label = r'$\log(V^2_{\rm peak}/{\rm km}^2 {\rm s}^{-2})$'

    def __init__(self, data: pd.DataFrame) -> None:
        self.value = np.log10((data['Vpeak'].to_numpy(copy=True)) ** 2)

    def get_good_cond(self) -> npt.NDArray:
        return np.ones(len(self.value), dtype=bool)


# ---------------------------------------------------------------------
# Galaxy properties
# ---------------------------------------------------------------------
class Age(BaseProperty):
    label = 'age'
    file_label = 'age'
    full_label = r'$\log({\rm Age/yr})$'

    def __init__(self, data: pd.DataFrame) -> None:
        pid = data.get('PID')

        if pid is not None:  # Empire
            self.value = np.log10(data['age_star'] * 10**9).to_numpy(copy=True)
        else:  # mpa
            self.value = data['age'].to_numpy(copy=True)

    def get_good_cond(self) -> npt.NDArray:
        return self.value > 0


class DeltaMS(BaseProperty):
    label = r'$\Delta$MS'
    file_label = 'deltaMS'
    full_label = r'$\Delta{\rm MS}$'
    value: npt.NDArray
    tpcf_bins = [-1, -0.45, 0]

    def __init__(self, data: pd.DataFrame, log: bool = True) -> None:
        if data.get('Delta_MS') is not None:
            self.value = data['Delta_MS'].to_numpy(copy=True)
        else:
            logM = Mstar(data, log=True).value
            logSSFR = SSFR(data, log=True).value

            ms = self.ms_fit(logM)
            self.value = logSSFR - ms

    @staticmethod
    def ms_fit(mass: float | npt.NDArray) -> float | npt.NDArray:
        """
        Calculate the sSFR that corresponds to the main sequence at a
        given stellar mass.

        """
        ssfr = (
            config.MS_PSI0
            - np.log10(1 + 10 ** (config.MS_GAMMA * (mass - config.MS_M0)))
            - mass
        )

        return ssfr

    def get_good_cond(self) -> npt.NDArray:
        return self.value != -99


class Mstar(BaseProperty):
    label = r'$M_*$'
    file_label = 'Mstar'
    full_label = r'$\log(M_*/M_\odot)$'

    def __init__(self, data: pd.DataFrame, log: bool = True) -> None:
        logM = data['M_star'].to_numpy(copy=True)

        if not log:
            self.value = 10.0**logM
        else:
            self.value = logM

    def get_good_cond(self) -> npt.NDArray:
        # Doesn't exactly check the right thing if log was passed
        # as True, but should still work since a galaxy won't have
        # a mass as low as logM = 0.
        return self.value > 0


class Size(BaseProperty):
    label = 'Size'
    file_label = 'size'
    full_label = r'$\log(R_{\rm chlr}/{\rm kpc})$'
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # [0.4, 0.6, 0.8]
    tpcf_bins = [0.25, 0.45, 0.65]  # [0.4, 0.6, 0.8]

    def __init__(self, data: pd.DataFrame) -> None:
        pid = data.get('PID')
        sample_size = data.get(self.label)

        if pid is None:  # mpa
            rhlr = data['Rhlr'].to_numpy(copy=True)
            ellipticity = data['e'].to_numpy(copy=True)
            size = np.sqrt(1 - ellipticity) * rhlr
        elif sample_size is not None:  # sampled sizes
            size = data[self.label].to_numpy(copy=True)
            self.value = size
            return None
        else:  # Empire
            size = data['R_obs_5000'].to_numpy(copy=True)

        bad_sizes = (size <= 0) | np.isnan(size)
        size[~bad_sizes] = np.log10(size[~bad_sizes])
        size[bad_sizes] = -99

        self.value = size

    def get_good_cond(self) -> npt.NDArray:
        return self.value != -99


class SFR(BaseProperty):
    label = 'SFR'
    file_label = 'sfr'
    full_label = r'$\log({\rm SFR}/M_\odot{\rm yr}^{-1})$'

    def __init__(self, data: pd.DataFrame, log: bool = True) -> None:
        if data.get('SFR_obs') is not None:
            logSFR = data['SFR_obs'].to_numpy(copy=True)
        else:
            logSFR = data['SFR'].to_numpy(copy=True)

        if not log:
            self.value = 10.0**logSFR
        else:
            self.value = logSFR

    def get_good_cond(self) -> npt.NDArray:
        return self.value != -99


class SSFR(BaseProperty):
    label = 'sSFR'
    file_label = 'ssfr'
    full_label = r'$\log({\rm sSFR/yr}^{-1})$'
    delta_ms: npt.NDArray | None = None
    bins = [
        -12.3,
        -12.05,
        -11.8,
        -11.55,
        -11.3,
        -11.05,
        -10.8,
        -10.55,
        -10.3,
        -10.05,
        -9.8,
        -9.55,
    ]  # [-11.0, -10.5, -10.0]
    tpcf_bins = [-11.0, -10.5, -10.0]

    def __init__(self, data: pd.DataFrame, log: bool = True) -> None:
        pid = data.get('PID')

        if pid is not None and data.get('sSFR') is not None:  # simulation
            logSSFR = data['sSFR'].to_numpy(copy=True)
        elif data.get('ssfr') is not None:
            logSSFR = np.log10(data['ssfr'].to_numpy(copy=True))
        else:
            logSSFR = SFR(data, log=True).value - Mstar(data, log=True).value

        if not log:
            self.value = 10.0**logSSFR
        else:
            self.value = logSSFR

    def get_good_cond(self) -> npt.NDArray:
        return self.value != -99


class TForm(BaseProperty):
    label = 'Tform'
    file_label = 'tform'
    full_label = r'$\log(T_{\rm form}/{\rm yr})$'
    bins = [
        9.5,
        9.55,
        9.6,
        9.65,
        9.7,
        9.75,
        9.8,
        9.85,
        9.9,
        9.95,
        10.0,
        10.05,
    ]  # [9.8, 9.9, 10.0]
    tpcf_bins = [9.8, 9.9, 10.0]

    def __init__(self, data: pd.DataFrame) -> None:
        pid = data.get('PID')

        if pid is not None:  # simulation
            t_form = data[self.label].to_numpy(copy=True)
        else:  # mpa
            z = data['z_obs'].to_numpy(copy=True)
            uni_age = cosmo.quantities.age_of_universe(z, H0=cosmo.constants.H0)
            gal_age = Age(data).value

            bad_ages = gal_age == -99

            t_form = uni_age - 10**gal_age
            t_form[~bad_ages] = np.log10(t_form[~bad_ages])
            t_form[bad_ages] = -99

        self.value = t_form

    def get_good_cond(self) -> npt.NDArray:
        return self.value != -99


class Velocity2(BaseProperty):
    label = 'v^2'
    file_label = 'v2'
    full_label = r'$\log(M_*/R_{\rm chlr})$'

    def __init__(self, data: pd.DataFrame) -> None:
        size_cls = Size(data)
        size = size_cls.value
        good_sizes = size_cls.get_good_cond()
        mass = Mstar(data).value
        v2 = mass - size
        v2[~good_sizes] = -99

        self.value = v2

    def get_good_cond(self) -> npt.NDArray:
        return self.value != -99
