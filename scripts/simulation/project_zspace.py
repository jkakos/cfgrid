from typing import Sequence

import numpy as np
import pandas as pd

import config
from src import cosmo
from src import datalib
from src import galhalo


BOX_SIZE = 250
Z_CUT = True
WRITE_FILE = True


def assemble_and_project(
    scale_factors: Sequence[float],
    volumes: Sequence[datalib.volumes.Volume],
    save_redshift: float,
) -> None:
    """
    Project Bolshoi-Planck snapshots into redshift space.
    'scale_factors' should be a list of scale factors that will
    identify which BP snapshots to use. There must be a scale factor
    that corresponds to each volume in 'volumes'. 'save_redshift' is
    used differentiate mocks that correspond to different volumes.

    """
    if len(scale_factors) != len(volumes):
        raise ValueError(
            f"Number of scale factors ({len(scale_factors)}) must equal number of"
            f" volumes ({len(volumes)})."
        )

    data_dir = config.BP_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    origin = (0, BOX_SIZE / 2, BOX_SIZE / 2)
    scatters = [f'{float(s):.3f}' for s in galhalo.utils.get_scatters(config.SIM_DIR)]
    redshift_interp = cosmo.distances.redshift_interp(z=0.3, z0=0.0, H0=100)
    redshifts = [1 / sf - 1 for sf in scale_factors]
    obs_redshifts = {i: vol.redshift_lims for (i, vol) in enumerate(volumes)}

    for scatter in scatters:
        print(f'Scatter: {scatter}')
        snapshots = []
        for i, (redshift, volume) in enumerate(zip(redshifts, volumes)):
            print(f'Redshift: {redshift:.4f}')
            mmin, mmax = volume.mass_lims
            zmin_obs, zmax_obs = volume.redshift_lims
            bp = pd.read_parquet(
                config.BASE_MOCK_DIR.joinpath(
                    galhalo.names.get_base_mock_filename(redshift)
                )
            )
            mass_df = pd.read_parquet(
                config.MSTAR_MOCK_DIR.joinpath(
                    galhalo.names.get_mstar_mock_filename(redshift, scatter)
                )
            )
            bp = bp.merge(mass_df, how='left', on=['ID', 'PID']).dropna(subset='M_star')
            mass_cond = (bp['M_star'] > mmin) & (bp['M_star'] < mmax)
            bp = bp[mass_cond]
            ngal_scatter = sum(mass_cond)

            # Shift the origin of the halos in the simulation box
            bp['X_shifted'] = bp['X'].copy()
            bp['Y_shifted'] = bp['Y'].copy()
            bp['Z_shifted'] = bp['Z'].copy()
            bp['Y_shifted'] -= origin[1]
            bp['Z_shifted'] -= origin[2]

            # X axis is taken to be the long axis, so the shift will be
            # redshift-dependent
            cdist_to_box = cosmo.distances.comoving_distance(redshift, z0=0, H0=100)
            bp['X_shifted'] += cdist_to_box

            if (s := sum(bp['X_shifted'] == bp['X'])) != (len(bp) - ngal_scatter):
                raise ValueError(
                    "Check mass inequalities to ensure all halos are being updated."
                    f" (sum(X == Xz) = {s}, should be 0)"
                )

            # Convert to spherical coordinates to apply redshift-space distortions
            r, ra, dec = cosmo.coords.convert_cartesian_to_spherical(
                bp['X_shifted'].to_numpy(),
                bp['Y_shifted'].to_numpy(),
                bp['Z_shifted'].to_numpy(),
            )
            z_cos = np.round(redshift_interp(r), 6)

            positions = np.array([bp['X_shifted'], bp['Y_shifted'], bp['Z_shifted']])
            velocities = np.array([bp['Vx'], bp['Vy'], bp['Vz']])
            direction = positions / np.linalg.norm(positions, axis=0)
            v_los = np.array(
                [np.dot(x, v) for (x, v) in zip(direction.T, velocities.T)]
            )
            z_obs = np.round(z_cos + (v_los / cosmo.constants.c) * (1 + z_cos), 6)

            # Convert back to Cartesian coordinates that are in redshift space
            Xz, Yz, Zz = cosmo.coords.convert_spherical_to_cartesian(
                ra,
                dec,
                z_obs,
            )
            bp = bp.assign(RA=ra, DEC=dec, z_obs=z_obs, Xz=Xz, Yz=Yz, Zz=Zz)

            if Z_CUT:
                # Remove values if they are outside the redsfhit range of their
                # mass bin
                print(f'Number of galaxies before cut: {len(bp)}')
                bp = bp.query("z_obs > @zmin_obs and z_obs < @zmax_obs")
                print(f'Number of galaxies after cut: {len(bp)}')

            zmin = min(bp['z_obs'])
            zmax = max(bp['z_obs'])
            print(
                'Volume redshift limits:'
                f' {obs_redshifts[i][0]:.4f} - {obs_redshifts[i][1]:.4f}'
            )
            print(f'Resultant mock redshift limits: {zmin:.4f} - {zmax:.4f}')

            snapshots.append(bp)

        if Z_CUT:
            filename = f'bp_snapshot_{save_redshift:.4f}_{scatter}_zcut.parquet'
        else:
            filename = f'bp_snapshot_{save_redshift:.4f}_{scatter}.parquet'

        if WRITE_FILE:
            data_out = pd.concat(snapshots)
            data_out = data_out.drop(columns=['X_shifted', 'Y_shifted', 'Z_shifted'])
            if (s1 := bp['ID'].isna().sum()) or (s2 := bp['M_star'].isna().sum()):
                raise ValueError(
                    "Something is wrong with the merging or projecting process. Found"
                    f" {s1} nans in ID and {s2} nans in M_star."
                )
            data_out.to_parquet(data_dir.joinpath(filename), index=False)


def main() -> None:
    """
    See docstring of 'assemble_and_project'.

    """
    halo_config = galhalo.config.CONFIG
    assemble_and_project(
        halo_config.scale_factors, halo_config.volume, halo_config.save_redshift
    )
