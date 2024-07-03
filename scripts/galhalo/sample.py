import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import config
from src import galhalo
from src.figlib import save
from src.protocols import properties


def sample(
    obs_x: properties.Property,
    obs_y: properties.Property,
    sim_data: pd.DataFrame,
    sim_x: properties.Property,
    sim_y: properties.Property,
    obs_xbins: Sequence[float],
    mass_bins: Sequence[float],
    reverse: bool,
    path: pathlib.Path,
) -> dict[str, npt.NDArray]:
    """
    Run the main galaxy property sampling function for halos.

    """
    num_xbins = len(obs_xbins) + 1

    obs_xcoords = np.digitize(obs_x.value, bins=obs_xbins)
    sim_xcoords = np.digitize(sim_x.value, bins=obs_xbins)
    sampled_obs = np.full(len(sim_x.value), -1.0)

    # Make figures
    fig, ax = plt.subplots(
        figsize=(8, 4), ncols=2, sharex=True, sharey=True, constrained_layout=True
    )
    inds = np.arange(len(sampled_obs))

    for x in range(num_xbins):
        obs_xcond = obs_xcoords == x
        sim_xcond = sim_xcoords == x
        samples = np.random.choice(obs_y.value[obs_xcond], size=sum(sim_xcond))

        sample_rank_order = np.argsort(samples)
        sim_y_rank_order = np.argsort(sim_y.value[sim_xcond])
        sim_y_match_order = sim_y_rank_order[::-1] if reverse else sim_y_rank_order
        sampled_obs[inds[sim_xcond][sim_y_match_order]] = samples[sample_rank_order]

    for mb_low, mb_high in zip(mass_bins[:-1], mass_bins[1:]):
        mbin_cond = (obs_x.value > mb_low) & (obs_x.value < mb_high)
        mbin_cond2 = (sim_x.value > mb_low) & (sim_x.value < mb_high)
        x_ = obs_x.value[mbin_cond]
        y_ = obs_y.value[mbin_cond]
        ax[0].hist2d(
            x_,
            y_,
            bins=[np.linspace(mb_low, mb_high, 10), np.linspace(min(y_), max(y_), 40)],
            range=[[min(x_), max(x_)], [min(y_), max(y_)]],
            cmap='turbo',
            density=True,
            zorder=1,
            rasterized=True,
        )
        ax[1].hist2d(
            sim_x.value[mbin_cond2],
            sampled_obs[mbin_cond2],
            bins=[np.linspace(mb_low, mb_high, 10), np.linspace(min(y_), max(y_), 40)],
            range=[[min(x_), max(x_)], [min(y_), max(y_)]],
            cmap='turbo',
            density=True,
            zorder=1,
            rasterized=True,
        )
        ax[0].axvline(x=mb_high, color='grey', ls='--')
        ax[1].axvline(x=mb_high, color='grey', ls='--')

    ax[0].set(
        xlim=(min(obs_x.value), max(obs_x.value)),
        ylim=(min(obs_y.value), max(obs_y.value)),
    )
    ax[1].set(
        xlim=(min(obs_x.value), max(obs_x.value)),
        ylim=(min(obs_y.value), max(obs_y.value)),
    )
    ax[0].set(xlabel=obs_x.full_label, ylabel=obs_y.full_label)
    ax[1].set(
        xlabel=obs_x.full_label,
        ylabel=sim_y.full_label,
    )

    for a in ax:
        a.minorticks_on()
        a.tick_params(which='both', direction='in')

    save.savefig(
        fig, filename=f'{obs_y.file_label}_{sim_y.file_label}', path=path, ext='pdf'
    )
    plt.close('all')

    sampled_results = {
        'ID': sim_data['ID'].to_numpy(),
        'PID': sim_data['PID'].to_numpy(),
        obs_y.label: sampled_obs,
    }
    return sampled_results


def main() -> None:
    halo_config = galhalo.config.CONFIG
    gal_config = halo_config.gal_config
    obs_data = gal_config.load()
    obs_xbins = halo_config.get_sample_mass_bins()

    # Get the scatters
    scatters = [float(s) for s in galhalo.utils.get_scatters(config.SIM_DIR)]
    scatters.sort()

    mass_bins = halo_config.mass_bins

    # Apply only to central galaxies and halos
    galaxy_centrals = galhalo.config.GAL_CENTRALS
    halo_centrals = halo_config.centrals[0]
    filtered_obs_data = obs_data[galaxy_centrals.get_cond(obs_data)]

    for (gal, halo), reverse in list(galhalo.config.PAIRINGS.items()):
        print(gal.file_label, halo.file_label)

        for scatter in scatters:
            print(f'Scatter: {scatter:.3f}')
            halo_config = type(galhalo.config.CONFIG)()
            halo_config.set_scatter(scatter)
            sim_data = halo_config.load(galhalo=False)
            filtered_sim_data = sim_data[halo_centrals.get_cond(sim_data)]

            obs_x_property = properties.Mstar(filtered_obs_data)
            obs_y_property = gal(filtered_obs_data)
            sim_x_property = properties.Mstar(filtered_sim_data)
            sim_y_property = halo(filtered_sim_data)

            # Apply mass limits for selected simulation mass
            sim_mass_cond = (sim_x_property.value >= mass_bins[0]) & (
                sim_x_property.value <= mass_bins[-1]
            )
            filtered_sim_data = filtered_sim_data[sim_mass_cond]
            sim_x_property.value = sim_x_property.value[sim_mass_cond]
            sim_y_property.value = sim_y_property.value[sim_mass_cond]

            # Get good x, y
            [obs_x, obs_y], *_ = properties.standardize(obs_x_property, obs_y_property)
            [sim_x, sim_y], sim_all_good = properties.standardize(
                sim_x_property, sim_y_property
            )
            sim_data_good = filtered_sim_data.loc[sim_all_good]

            # Create pathing
            path = galhalo.pathing.get_galhalo_sample_path(halo_config)
            sampled_results = sample(
                obs_x=obs_x,
                obs_y=obs_y,
                sim_data=sim_data_good,
                sim_x=sim_x,
                sim_y=sim_y,
                obs_xbins=obs_xbins,
                mass_bins=mass_bins,
                reverse=reverse,
                path=path.parent,
            )
            df = pd.DataFrame.from_dict(
                {
                    'ID': sampled_results['ID'],
                    'PID': sampled_results['PID'],
                    obs_y.label: sampled_results[obs_y.label],
                }
            )
            galhalo.pathing.save_galhalo_sample(
                path, obs_y.file_label, sim_y.file_label, df
            )
