import copy
from typing import Any, Optional, Protocol

import pandas as pd
import numpy as np
import numpy.typing as npt

import config
from src import cosmo
from src.cosmo import constants as consts
from src.cosmo.distances import comoving_distance
from src.datalib import dataprocess
from src.protocols import conditionals
from src.tpcf import utils as cfu


H0 = 100  # Keep distances in Mpc/h despite set H0 value in cosmo.constants


class TpcfRunner(Protocol):
    data: pd.DataFrame
    settings: dict

    def run(
        self,
        cond: npt.NDArray,
        return_mean: bool = False,
        return_median: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        """
        Calculate the two-point correlation function. For each of
        return_mean and return_median flagged as True, those statistics
        will be returned after xi and xi_err in the order of xi_mean
        before xi_median if both are returned.

        """
        ...


class CrossTpcfRunner(Protocol):
    data1: pd.DataFrame
    data2: pd.DataFrame
    settings: dict

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]: ...

    def set_data(
        self,
        cond1: Optional[conditionals.Conditional],
        cond2: Optional[conditionals.Conditional],
    ): ...

    def run(
        self,
        cond1: Optional[npt.NDArray] = None,
        cond2: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        """
        Calculate the two-point correlation function. For each of
        return_mean and return_median flagged as True, those statistics
        will be returned after xi and xi_err in the order of xi_mean
        before xi_median if both are returned.

        """
        ...


class SimSnapshotTPCF:
    def __init__(self, data: pd.DataFrame, settings: dict) -> None:
        self.data = data
        self.settings = settings

    def get_data_coords(
        self, cond: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        XD = self.data['X'].to_numpy()
        YD = self.data['Y'].to_numpy()
        ZD = self.data['Z'].to_numpy()

        if cond is not None:
            XD = XD[cond]
            YD = YD[cond]
            ZD = ZD[cond]

        return XD, YD, ZD

    def get_random_coords(
        self,
        XD: npt.NDArray,
        YD: npt.NDArray,
        ZD: npt.NDArray,
        sample_size: int = 10**6,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        XR, YR, ZR = cfu.box_random(XD, YD, ZD, sample_size)

        return XR, YR, ZR

    def run(
        self,
        cond: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        XD, YD, ZD = self.get_data_coords(cond)
        XR, YR, ZR = self.get_random_coords(XD, YD, ZD)

        # fmt: off
        ret = cfu.calc_tpcf(
            XD, YD, ZD,
            self.settings['bins'],
            X2=XR, Y2=YR, Z2=ZR,
            bootstraps=self.settings['bootstraps'],
            nthreads=self.settings['nthreads'],
            return_mean=return_mean,
            return_median=return_median,
            pimax=self.settings['pimax'],
            periodic=False,
        )
        # fmt: on

        return ret


class SimLightConeTPCF:
    def __init__(self, data: pd.DataFrame, settings: dict) -> None:
        self.data = data
        self.settings = settings

    def get_data_coords(
        self, cond: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        RAD = self.data['RA'].to_numpy()
        DECD = self.data['DEC'].to_numpy()
        CZD = self.data['z_obs'].to_numpy()

        if cond is not None:
            RAD = RAD[cond]
            DECD = DECD[cond]
            CZD = CZD[cond]

        return RAD, DECD, CZD

    def get_random_coords(
        self,
        RAD: npt.NDArray,
        DECD: npt.NDArray,
        CZD: npt.NDArray,
        sample_size: int = 10**6,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        RAR, DECR, CZR = cfu.sdss_random(RAD, DECD, CZD, sample_size)

        return RAR, DECR, CZR

    def run(
        self,
        cond: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        return_counts: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        RAD, DECD, CZD = self.get_data_coords(cond)
        RAR, DECR, CZR = self.get_random_coords(RAD, DECD, CZD)

        CZD = comoving_distance(CZD, H0=H0, Om=consts.Om, Ol=consts.Ol)
        CZR = comoving_distance(CZR, H0=H0, Om=consts.Om, Ol=consts.Ol)

        # fmt: off
        ret = cfu.calc_tpcf(
            RAD, DECD, CZD,
            self.settings['bins'],
            RA2=RAR, DEC2=DECR, CZ2=CZR,
            bootstraps=self.settings['bootstraps'],
            nthreads=self.settings['nthreads'],
            return_mean=return_mean,
            return_median=return_median,
            return_counts=return_counts,
            pimax=self.settings['pimax'],
        )
        # fmt: on

        return ret


class SimLightConeSphereTPCF:
    def __init__(self, data: pd.DataFrame, settings: dict) -> None:
        self.data = data
        self.settings = settings

    def get_data_coords(
        self, cond: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        RAD = self.data['RA'].to_numpy()
        DECD = self.data['DEC'].to_numpy()
        CZD = self.data['z_obs'].to_numpy()

        if cond is not None:
            RAD = RAD[cond]
            DECD = DECD[cond]
            CZD = CZD[cond]

        return RAD, DECD, CZD

    def get_random_coords(
        self,
        RAD: npt.NDArray,
        DECD: npt.NDArray,
        CZD: npt.NDArray,
        sample_size: int = 10**6,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        RAR, DECR, CZR = cfu.sphere_random(RAD, DECD, CZD, sample_size)

        return RAR, DECR, CZR

    def run(
        self,
        cond: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        return_counts: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        RAD, DECD, CZD = self.get_data_coords(cond)
        RAR, DECR, CZR = self.get_random_coords(RAD, DECD, CZD)

        CZD = comoving_distance(CZD, H0=H0, Om=consts.Om, Ol=consts.Ol)
        CZR = comoving_distance(CZR, H0=H0, Om=consts.Om, Ol=consts.Ol)

        # fmt: off
        ret = cfu.calc_tpcf(
            RAD, DECD, CZD,
            self.settings['bins'],
            RA2=RAR, DEC2=DECR, CZ2=CZR,
            bootstraps=self.settings['bootstraps'],
            nthreads=self.settings['nthreads'],
            return_mean=return_mean,
            return_median=return_median,
            return_counts=return_counts,
            pimax=self.settings['pimax'],
        )
        # fmt: on

        return ret


class ObsLightConeTPCF:
    def __init__(self, data: pd.DataFrame, settings: dict) -> None:
        self.data = data
        self.settings = settings
        self.random = None

    def get_data_coords(
        self, cond: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        RAD = self.data['RA'].to_numpy()
        DECD = self.data['DEC'].to_numpy()
        CZD = self.data['z_obs'].to_numpy()

        if cond is not None:
            RAD = RAD[cond]
            DECD = DECD[cond]
            CZD = CZD[cond]

        return RAD, DECD, CZD

    def get_random_coords(
        self,
        CZD: npt.NDArray,
        sample_size: int = 10**6,
        process: bool = True,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        random = self.load_random(process=process)
        rand_inds = np.random.randint(0, len(random), size=sample_size)
        RAR = random['RA'].iloc[rand_inds].to_numpy()
        DECR = random['DEC'].iloc[rand_inds].to_numpy()
        CZR = cfu.generate_random_sample(CZD, sample_size)

        return RAR, DECR, CZR

    @staticmethod
    def load_random(process: bool = True, **kwargs) -> pd.DataFrame:
        """
        Read the random catalog and apply standard cuts that were
        applied to the observations.

        """
        filepath = config.DATA_DIR.joinpath(config.RANDOM)
        random = pd.read_csv(
            filepath, names=['RA', 'DEC', 'unknown1', 'unknown2'], sep='\s+', **kwargs
        )

        if process:
            random = dataprocess.cut_sdss_mgs(random)
            random = dataprocess.cut_sdss_rects(random)

        return random

    def store_random(self, process=True, sample_size=10**6) -> None:
        random = self.load_random(process=process)
        rand_inds = np.random.randint(0, len(random), size=sample_size)
        self.random = random.iloc[rand_inds]

    def run(
        self,
        cond: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        return_counts: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        process: bool = kwargs.get('process', True)

        bins = self.settings.get('bins')
        bootstraps = self.settings.get('bootstraps')
        nthreads = self.settings.get('nthreads')
        pimax = self.settings.get('pimax')

        RAD, DECD, ZD = self.get_data_coords(cond)
        CZD = comoving_distance(ZD, H0=H0, Om=consts.Om, Ol=consts.Ol)

        if self.random is None:
            # CZD is already a distance, so CZR will be also
            RAR, DECR, CZR = self.get_random_coords(CZD, process=process)
        else:
            RAR = self.random['RA'].to_numpy()
            DECR = self.random['DEC'].to_numpy()
            CZR = cfu.generate_random_sample(CZD, len(RAR))

        # fmt: off
        ret = cfu.calc_tpcf(
            RAD, DECD, CZD, bins,
            RA2=RAR, DEC2=DECR, CZ2=CZR,
            bootstraps=bootstraps,
            nthreads=nthreads,
            return_mean=return_mean,
            return_median=return_median,
            return_counts=return_counts,
            pimax=pimax,
        )
        # fmt: on

        return ret


class SimSnapshotZSpaceTPCF:
    def __init__(self, data: pd.DataFrame, settings: dict) -> None:
        self.data = data
        self.settings = settings

    def get_data_coords(
        self, cond: Optional[npt.NDArray] = None
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        RAD = self.data['RA'].to_numpy()
        DECD = self.data['DEC'].to_numpy()
        ZD = self.data['z_obs'].to_numpy()

        if cond is not None:
            RAD = RAD[cond]
            DECD = DECD[cond]
            ZD = ZD[cond]

        return RAD, DECD, ZD

    def get_random_coords(
        self,
        RAD: npt.NDArray,
        DECD: npt.NDArray,
        CZD: npt.NDArray,
        sample_size: int = 10**6,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        XD, YD, ZD = cosmo.coords.convert_spherical_to_cartesian(RAD, DECD, CZD, H0=H0)
        XR, YR, ZR = cfu.box_random(XD, YD, ZD, sample_size)
        CZR, RAR, DECR = cosmo.coords.convert_cartesian_to_spherical(XR, YR, ZR)

        return RAR, DECR, CZR

    def run(
        self,
        cond: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        return_counts: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        bins = self.settings.get('bins')
        bootstraps = self.settings.get('bootstraps')
        nthreads = self.settings.get('nthreads')
        pimax = self.settings.get('pimax')

        # CZD is initially a redshift, but CZR is calculated as a distance
        RAD, DECD, CZD = self.get_data_coords(cond)
        RAR, DECR, CZR = self.get_random_coords(RAD, DECD, CZD)
        CZD = comoving_distance(CZD, H0=H0, Om=consts.Om, Ol=consts.Ol)

        # fmt: off
        ret = cfu.calc_tpcf(
            RAD, DECD, CZD, bins,
            RA2=RAR, DEC2=DECR, CZ2=CZR,
            bootstraps=bootstraps,
            nthreads=nthreads,
            return_mean=return_mean,
            return_median=return_median,
            return_counts=return_counts,
            pimax=pimax,
        )
        # fmt: on

        return ret


class ObsLightConeCrossTPCF:
    def __init__(self, data: pd.DataFrame, settings: dict) -> None:
        self.data1 = data
        self.data2 = data
        self.settings = settings

    def store_random(self, process: bool = True, sample_size: int = 10**5) -> None:
        random = ObsLightConeTPCF.load_random(process=process)
        rand_inds = np.random.randint(0, len(random), size=sample_size)
        self.random = random.iloc[rand_inds]

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.data1, self.data2

    def set_data(
        self,
        cond1: Optional[conditionals.Conditional],
        cond2: Optional[conditionals.Conditional],
    ) -> None:
        if cond1 is not None:
            self.data1 = self.data1[cond1.value]

        if cond2 is not None:
            self.data2 = self.data2[cond2.value]

    def run(
        self,
        cond1: Optional[npt.NDArray] = None,
        cond2: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        return_counts: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        process: bool = kwargs.get('process', True)

        self.runner1 = ObsLightConeTPCF(copy.deepcopy(self.data1), self.settings)
        self.runner2 = ObsLightConeTPCF(copy.deepcopy(self.data2), self.settings)

        RAD1, DECD1, ZD1 = self.runner1.get_data_coords(cond1)
        CZD1 = comoving_distance(ZD1, H0=H0, Om=consts.Om, Ol=consts.Ol)

        RAD2, DECD2, ZD2 = self.runner2.get_data_coords(cond2)
        CZD2 = comoving_distance(ZD2, H0=H0, Om=consts.Om, Ol=consts.Ol)

        if self.random is None:
            RAR1, DECR1, CZR1 = self.runner1.get_random_coords(
                CZD1, sample_size=10**5, process=process
            )
            RAR2, DECR2, CZR2 = self.runner2.get_random_coords(
                CZD2, sample_size=10**5, process=process
            )
        else:
            RAR1 = self.random['RA'].to_numpy()
            DECR1 = self.random['DEC'].to_numpy()
            CZR1 = cfu.generate_random_sample(CZD1, len(RAR1))

            RAR2 = self.random['RA'].to_numpy()
            DECR2 = self.random['DEC'].to_numpy()
            CZR2 = cfu.generate_random_sample(CZD2, len(RAR2))

        # fmt: off
        ret = cfu.calc_cross_tpcf(
            RAD1, DECD1, CZD1,
            RAD2, DECD2, CZD2,
            RAR1, DECR1, CZR1,
            RAR2, DECR2, CZR2,
            self.settings['bins'],
            box=False,
            bootstraps=self.settings['bootstraps'],
            nthreads=self.settings['nthreads'],
            return_mean=return_mean,
            return_median=return_median,
            return_counts=return_counts,
            pimax=self.settings['pimax'],
        )
        # fmt: on

        return ret


class SimSnapshotCrossTPCF:
    def __init__(self, data: pd.DataFrame, settings: dict) -> None:
        self.data1 = data
        self.data2 = data
        self.settings = settings

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.data1, self.data2

    def set_data(
        self,
        cond1: Optional[conditionals.Conditional],
        cond2: Optional[conditionals.Conditional],
    ) -> None:
        if cond1 is not None:
            self.data1 = self.data1[cond1.value]

        if cond2 is not None:
            self.data2 = self.data2[cond2.value]

    def run(
        self,
        cond1: Optional[npt.NDArray] = None,
        cond2: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        return_counts: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        self.runner1 = SimSnapshotTPCF(copy.deepcopy(self.data1), self.settings)
        self.runner2 = SimSnapshotTPCF(copy.deepcopy(self.data2), self.settings)

        XD1, YD1, ZD1 = self.runner1.get_data_coords(cond1)
        XR1, YR1, ZR1 = self.runner1.get_random_coords(XD1, YD1, ZD1)

        XD2, YD2, ZD2 = self.runner2.get_data_coords(cond2)
        XR2, YR2, ZR2 = self.runner2.get_random_coords(XD2, YD2, ZD2)

        # fmt: off
        ret = cfu.calc_cross_tpcf(
            XD1, YD1, ZD1,
            XD2, YD2, ZD2,
            XR1, YR1, ZR1,
            XR2, YR2, ZR2,
            self.settings['bins'],
            box=True,
            bootstraps=self.settings['bootstraps'],
            nthreads=self.settings['nthreads'],
            return_mean=return_mean,
            return_median=return_median,
            return_counts=return_counts,
            pimax=self.settings['pimax'],
            periodic=False,
        )
        # fmt: on

        return ret


class SimSnapshotZSpaceCrossTPCF:
    def __init__(self, data: pd.DataFrame, settings: dict) -> None:
        self.data1 = data
        self.data2 = data
        self.settings = settings

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.data1, self.data2

    def set_data(
        self,
        cond1: Optional[conditionals.Conditional],
        cond2: Optional[conditionals.Conditional],
    ) -> None:
        if cond1 is not None:
            self.data1 = self.data1[cond1.value]

        if cond2 is not None:
            self.data2 = self.data2[cond2.value]

    def run(
        self,
        cond1: Optional[npt.NDArray] = None,
        cond2: Optional[npt.NDArray] = None,
        return_mean: bool = False,
        return_median: bool = False,
        return_counts: bool = False,
        **kwargs: Any,
    ) -> tuple[npt.NDArray, npt.NDArray, Optional[npt.NDArray], Optional[npt.NDArray]]:
        self.runner1 = SimSnapshotZSpaceTPCF(copy.deepcopy(self.data1), self.settings)
        self.runner2 = SimSnapshotZSpaceTPCF(copy.deepcopy(self.data2), self.settings)

        # CZD is initially a redshift, but CZR is calculated as a distance
        RAD1, DECD1, CZD1 = self.runner1.get_data_coords(cond1)
        RAR1, DECR1, CZR1 = self.runner1.get_random_coords(RAD1, DECD1, CZD1)
        CZD1 = comoving_distance(CZD1, H0=H0, Om=consts.Om, Ol=consts.Ol)

        RAD2, DECD2, CZD2 = self.runner2.get_data_coords(cond2)
        RAR2, DECR2, CZR2 = self.runner2.get_random_coords(RAD2, DECD2, CZD2)
        CZD2 = comoving_distance(CZD2, H0=H0, Om=consts.Om, Ol=consts.Ol)

        # fmt: off
        ret = cfu.calc_cross_tpcf(
            RAD1, DECD1, CZD1,
            RAD2, DECD2, CZD2,
            RAR1, DECR1, CZR1,
            RAR2, DECR2, CZR2,
            self.settings['bins'],
            box=False,
            bootstraps=self.settings['bootstraps'],
            nthreads=self.settings['nthreads'],
            return_mean=return_mean,
            return_median=return_median,
            return_counts=return_counts,
            pimax=self.settings['pimax'],
        )
        # fmt: on

        return ret
