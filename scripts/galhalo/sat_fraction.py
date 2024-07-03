import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import datalib, galhalo
from src.protocols import conditionals


def main() -> None:
    """
    Compute the satellite fraction as a function of mass in MPA-JHU using
    the Yang+2012 group catalog.

    """
    M_LOW = 9.5
    M_HIGH = 12
    BIN_WIDTH = 0.1
    mass_bins = list(
        np.arange(10 * M_LOW + 10 * BIN_WIDTH, 10 * M_HIGH, 10 * BIN_WIDTH) / 10
    )
    volume = (
        [datalib.volumes.Volume((9.500, 10.000), (0.020, 0.035))]
        + datalib.volumes.VOLUME1
        + [datalib.volumes.Volume((11.500, 12.000), (0.095, 0.190))]
    )
    mpa_zcut = datalib.MPAJHU(volume=volume).load()

    mass = []
    sat_frac = []
    ngal = []
    fig, ax = plt.subplots(figsize=(10, 4), ncols=2, constrained_layout=True)

    for mlow, mhigh in zip([M_LOW, *mass_bins], [*mass_bins, M_HIGH]):
        mpa_mbin = mpa_zcut.query("M_star > @mlow and M_star < @mhigh")
        num_sats = sum(conditionals.Satellites(mpa_mbin).value)
        num_gal = len(mpa_mbin)
        mass.append((mlow + mhigh) / 2)
        sat_frac.append(num_sats / num_gal) if num_gal else sat_frac.append(-1)
        ngal.append(num_gal)

        ax[0].scatter(mpa_mbin['z_obs'], mpa_mbin['M_star'], s=5, edgecolor='none')
        ax[0].text(
            np.median(mpa_mbin['z_obs']),
            np.median(mpa_mbin['M_star']),
            f'{sat_frac[-1]:.4f}',
            ha='center',
            va='center',
        )
        ax[1].scatter(mass[-1], sat_frac[-1], s=10, edgecolor='none')

    df = pd.DataFrame.from_dict({'logMs': mass, 'f_sat': sat_frac, 'ngal': ngal})
    df.to_csv(config.DATA_DIR.joinpath('sat_fraction_vol1.csv'), index=False)
    print(df)
    ax[0].set(xlabel=r'$z$', ylabel=r'$M_*$', xlim=(0, 0.3))
    ax[1].set(xlabel=r'$M_*$', ylabel=r'$f_{\rm sat}$', ylim=(0, 0.5))
    ax[1].plot(df['logMs'], df['f_sat'], markersize=3, color='red')

    plt.show()


def compare_obs_and_sim_sat_frac() -> None:
    halo_config = galhalo.config.CONFIG
    gal_config = halo_config.gal_config
    mpa = gal_config.load()
    bp = halo_config.load(galhalo=False)

    fig, ax = plt.subplots(constrained_layout=True)
    mass_bins = halo_config.mass_bins
    m_center = []
    mpa_sat_frac = []
    bp_sat_frac = []

    for mb1, mb2 in zip(mass_bins[:-1], mass_bins[1:]):
        mpa_mb = mpa.query("M_star > @mb1 and M_star < @mb2")
        bp_mb = bp.query("M_star > @mb1 and M_star < @mb2")
        m_center.append(0.5 * (mb1 + mb2))
        mpa_sat_frac.append(sum(conditionals.Satellites(mpa_mb).value) / len(mpa_mb))
        bp_sat_frac.append(sum(conditionals.Satellites(bp_mb).value) / len(bp_mb))

    print(f'{"Stellar Mass":>14}', f'{"MPA Sat frac":>14}', f'{"BP Sat frac":>14}')
    for m, mpasf, bpsf in zip(m_center, mpa_sat_frac, bp_sat_frac):
        print(f'{m:>14.4f}', f'{mpasf:>14.4f}', f'{bpsf:>14.4f}')

    ax.plot(m_center, mpa_sat_frac, color='tab:red', label='MPA')
    ax.plot(m_center, bp_sat_frac, color='tab:blue', label='BP')
    ax.set(xlabel=r'$M_*$', ylabel='Satellite fraction')
    ax.legend()
    plt.show()
