import os
import pathlib
from typing import Callable, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import CubicSpline
from tqdm import tqdm

import config
from src import galhalo
from src.figlib import save


def get_stellar_mass(
    data: pd.DataFrame,
    centrals_prop_name: str,
    satellitess_prop_name: str,
    interp_centrals: Callable,
    interp_satellites: Callable,
    scatter: float,
    seed: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """
    Calculate stellar mass from mass relations and add scatter. If seed
    is provided, it will be used as the random seed for determining the
    scatter.

    """
    m_star = np.empty(len(data), dtype=np.float64)
    satellites = data['pid'] != -1
    m_vir = np.log10(data.loc[~satellites, centrals_prop_name].to_numpy())
    m_peak = np.log10(data.loc[satellites, satellitess_prop_name].to_numpy())

    if seed is not None:
        rng = np.random.default_rng(seed)
        cen_scatter = rng.normal(0, scatter, size=len(m_vir))
        sat_scatter = rng.normal(0, scatter, size=len(m_peak))
    else:
        cen_scatter = np.random.normal(0, scatter, size=len(m_vir))
        sat_scatter = np.random.normal(0, scatter, size=len(m_peak))

    m_star[~satellites] = interp_centrals(m_vir) + cen_scatter
    m_star[satellites] = interp_satellites(m_peak) + sat_scatter

    return m_star


def plot_mass_relation(
    data: pd.DataFrame,
    xs: tuple[npt.NDArray, npt.NDArray],
    interps: tuple[Callable, Callable],
    labels: tuple[str, str],
    sat_cond: npt.NDArray,
    path: pathlib.Path,
    filename: str,
) -> None:
    """
    Plot the stellar mass - halo property relations for centrals (left)
    and satellites (right).

    """
    fig, axes = plt.subplots(
        figsize=(10, 5), sharex=True, sharey=True, ncols=2, constrained_layout=True
    )
    x1, x2 = xs
    interp1, interp2 = interps
    label1, label2 = labels
    axes[0].scatter(
        np.log10(data.loc[~sat_cond, label1]),
        data.loc[~sat_cond, 'M_star'],
        s=1,
        edgecolor='none',
        rasterized=True,
    )
    axes[0].plot(x1, interp1(x1), color='k')
    axes[1].scatter(
        np.log10(data.loc[sat_cond, label2]),
        data.loc[sat_cond, 'M_star'],
        s=1,
        edgecolor='none',
        rasterized=True,
    )
    axes[1].plot(x2, interp2(x2), color='k')
    axes[0].set(xlabel=label1, ylabel='M_star')
    axes[1].set(xlabel=label2)
    save.savefig(fig, filename, path=path)


def _main(
    scale_factors: Sequence[float],
    halo_property: Literal['velocity', 'mass'],
    plot_results: bool = False,
) -> None:
    """
    Assign stellar masses to Bolshoi-Planck dark matter halos according
    to some mass relation based on Mvir (Mpeak) or Vmax (Vpeak) for
    centrals (satellites). Instances of 'Mhalo', 'm_vir', or 'm_peak'
    are placeholder names and will represent either a mass or a
    velocity depending on the choice from 'halo_property'. A base
    dataframe will be stored that contains all the halo properties for
    halos with acceptable stellar masses. If different relations with
    different scatters are present, separate files with the stellar
    masses will be saved for each amount of scatter.

    """
    APPROX_FILE_LENGTH = 12_200_000
    CHUNKSIZE = 10**5
    HALO_PROPERTIES = {
        'mass': ['logMhalo', 'mvir', 'Mpeak'],
        'velocity': ['logvmax', 'vmax', 'Vpeak'],
    }
    OUTPUT_RENAME = {
        'id': 'ID',
        'pid': 'PID',
        'mvir': 'Mvir',
        'rvir': 'Rvir',
        'rs': 'Rs',
        'vrms': 'Vrms',
        'vmax': 'Vmax',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'vx': 'Vx',
        'vy': 'Vy',
        'vz': 'Vz',
    }

    files = [f'hlist_{scale_factor:.5f}.list' for scale_factor in scale_factors]
    redshifts = [1 / sf - 1 for sf in scale_factors]
    property_names = HALO_PROPERTIES[halo_property]

    base_df_path = config.BASE_MOCK_DIR
    mstar_path = config.MSTAR_MOCK_DIR
    base_df_path.mkdir(parents=True, exist_ok=True)
    mstar_path.mkdir(parents=True, exist_ok=True)

    scatters = galhalo.utils.get_scatters(config.SIM_DIR)
    scatter_floats = [float(scatter) for scatter in scatters]

    interps_cen = []
    interps_sat = []
    min_mass_cen = []
    max_mass_cen = []
    min_mass_sat = []
    max_mass_sat = []

    for scatter in scatters:
        msmh_cen = pd.read_csv(
            config.SIM_DIR.joinpath(f'cen_galaxy_halo_{scatter}.dat'),
            sep='\s+',
            index_col=False,
            skipinitialspace=True,
            usecols=['#logMs', 'logMhalo', 'logvmax'],
        )
        msmh_sat = pd.read_csv(
            config.SIM_DIR.joinpath(f'sat_galaxy_halo_{scatter}.dat'),
            sep='\s+',
            index_col=False,
            skipinitialspace=True,
            usecols=['#logMs', 'logMhalo', 'logvmax'],
        )
        msmh_cen = msmh_cen.query("logMhalo < 16 and logvmax < 3.5")
        msmh_sat = msmh_sat.query("logMhalo < 16 and logvmax < 3.38")

        logMs_cen = msmh_cen['#logMs'].to_numpy()
        logMhalo_cen = msmh_cen[property_names[0]].to_numpy()
        logMs_sat = msmh_sat['#logMs'].to_numpy()
        logMhalo_sat = msmh_sat[property_names[0]].to_numpy()
        interp_cen = CubicSpline(logMhalo_cen, logMs_cen, extrapolate=False)
        interp_sat = CubicSpline(logMhalo_sat, logMs_sat, extrapolate=False)

        interps_cen.append(interp_cen)
        interps_sat.append(interp_sat)
        min_mass_cen.append(min(logMhalo_cen))
        max_mass_cen.append(max(logMhalo_cen))
        min_mass_sat.append(min(logMhalo_sat))
        max_mass_sat.append(max(logMhalo_sat))

    MIN_MASS_CEN = min(min_mass_cen)
    MAX_MASS_CEN = max(max_mass_cen)
    MIN_MASS_SAT = min(min_mass_sat)
    MAX_MASS_SAT = max(max_mass_sat)

    if plot_results:
        fig_path = mstar_path.joinpath('mass_relations')
        fig_path.mkdir(parents=True, exist_ok=True)
        label1 = 'Vmax' if halo_property == 'velocity' else 'Mvir'
        label2 = 'Vpeak' if halo_property == 'velocity' else 'Mpeak'
        x1 = np.linspace(MIN_MASS_CEN, MAX_MASS_CEN, 100)
        x2 = np.linspace(MIN_MASS_SAT, MAX_MASS_SAT, 100)

    existing_base_files = [f for f in os.listdir(base_df_path)]
    existing_mstar_files = [f for f in os.listdir(mstar_path)]
    bp_paths = [config.BP_CATALOG_DIR.joinpath(f) for f in files]

    for bp_path, redshift in zip(bp_paths, redshifts):
        data_chunks: list[list[pd.DataFrame]] = [[] for _ in scatters]
        base_df_filename = galhalo.names.get_base_mock_filename(redshift)

        if base_df_filename in existing_base_files:
            data = pd.read_parquet(base_df_path.joinpath(base_df_filename))
            m_vir = np.log10(data[OUTPUT_RENAME[property_names[1]]])
            m_peak = np.log10(data[property_names[2]])
            sats = data[OUTPUT_RENAME['pid']] != -1

            for i, scatterf in enumerate(scatter_floats):
                mstar_filename = galhalo.names.get_mstar_mock_filename(
                    redshift, scatterf
                )
                if mstar_filename in existing_mstar_files:
                    continue

                # Keep only halos within the limits of the fit
                filtered_data = data[
                    (~sats & (m_vir >= min_mass_cen[i]) & (m_vir <= max_mass_cen[i]))
                    | (sats & (m_peak >= min_mass_sat[i]) & (m_peak <= max_mass_sat[i]))
                ].copy()

                m_star = get_stellar_mass(
                    filtered_data,
                    property_names[1],
                    property_names[2],
                    interps_cen[i],
                    interps_sat[i],
                    scatterf,
                    seed=i + int(scatterf * 10**5),
                )
                data_chunks[i].append(
                    pd.DataFrame.from_dict(
                        {
                            'ID': filtered_data['id'],
                            'PID': filtered_data['pid'],
                            'M_star': m_star,
                        }
                    )
                )
        else:
            with open(bp_paths[0]) as f:
                line = f.readline()

            header = line.strip().lstrip('# ').split()
            header = [x.rsplit('(', maxsplit=1)[0] for x in header]

            base_df = []
            chunk_num = -1

            with pd.read_csv(
                bp_path, names=header, comment='#', sep='\s+', chunksize=CHUNKSIZE
            ) as reader:
                for data in tqdm(reader, total=APPROX_FILE_LENGTH // CHUNKSIZE):
                    chunk_num += 1
                    m_vir = np.log10(data[property_names[1]])
                    m_peak = np.log10(data[property_names[2]])
                    sats = data['pid'] != -1

                    base_df.append(
                        data[
                            (~sats & (m_vir >= MIN_MASS_CEN) & (m_vir <= MAX_MASS_CEN))
                            | (
                                sats
                                & (m_peak >= MIN_MASS_SAT)
                                & (m_peak <= MAX_MASS_SAT)
                            )
                        ].copy()
                    )

                    for i, scatterf in enumerate(scatter_floats):
                        # Keep only halos within the limits of the fit
                        filtered_data = data[
                            (
                                ~sats
                                & (m_vir >= min_mass_cen[i])
                                & (m_vir <= max_mass_cen[i])
                            )
                            | (
                                sats
                                & (m_peak >= min_mass_sat[i])
                                & (m_peak <= max_mass_sat[i])
                            )
                        ].copy()

                        m_star = get_stellar_mass(
                            filtered_data,
                            property_names[1],
                            property_names[2],
                            interps_cen[i],
                            interps_sat[i],
                            scatterf,
                            seed=chunk_num + int(scatterf * 10**5),
                        )
                        data_chunks[i].append(
                            pd.DataFrame.from_dict(
                                {
                                    'ID': filtered_data['id'],
                                    'PID': filtered_data['pid'],
                                    'M_star': m_star,
                                }
                            )
                        )

            # Save the base properties for later merging
            base_df_out = pd.concat(base_df)
            base_df_out = base_df_out.rename(columns=OUTPUT_RENAME)
            base_df_out.to_parquet(
                base_df_path.joinpath(base_df_filename),
                index=False,
            )

        # Save the stellar masses to merge with base properties later
        for i, scatter in enumerate(scatters):
            if not data_chunks[i]:
                continue

            data_out = pd.concat(data_chunks[i])
            data_out.to_parquet(
                mstar_path.joinpath(
                    galhalo.names.get_mstar_mock_filename(redshift, scatter)
                ),
                index=False,
            )

            # Plot M_star vs halo property to see relation with scatter
            if plot_results:
                data_fig = base_df_out.merge(data_out, how='left', on=['ID', 'PID'])
                data_fig = data_fig.dropna(subset='M_star')
                sats = data_fig['PID'] != -1

                plot_mass_relation(
                    data_fig,
                    (x1, x2),
                    (interps_cen[i], interps_sat[i]),
                    (label1, label2),
                    sat_cond=sats.to_numpy(),
                    path=fig_path,
                    filename=f'm_relation_{redshift:.4f}_{scatter}',
                )


def main() -> None:
    _main(
        galhalo.config.CONFIG.scale_factors, halo_property='velocity', plot_results=True
    )
