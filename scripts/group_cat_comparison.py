from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import configurations, datalib
from src.figlib import colors as fcolors
from src.figlib import save
from src.protocols import conditionals


def main() -> None:
    configuration = configurations.MPAConfigVolume1()
    data = configuration.load()
    mass_bins = datalib.volumes.mass_bins_from_volume(configuration.volume)

    fig, ax = plt.subplots(constrained_layout=True)
    fig2, ax2 = plt.subplots(constrained_layout=True)

    mass: list[np.float64] = []
    yang: list[np.float64] = []
    tempel: list[np.float64] = []
    frod: list[np.float64] = []
    all_ratio: list[np.float64] = []
    two_ratio: list[np.float64] = []
    all_or_two_ratio: list[np.float64] = []

    for mb1, mb2 in zip(mass_bins[:-1], mass_bins[1:]):
        data_cut = data.query('M_star > @mb1 and M_star < @mb2')

        yang_c = conditionals.Centrals.get_cond(data_cut)
        yang_s = conditionals.Satellites.get_cond(data_cut)
        tempel_c = conditionals.TempelCentrals.get_cond(data_cut)
        tempel_s = conditionals.TempelSatellites.get_cond(data_cut)
        frod_c = conditionals.FRodriguezCentrals.get_cond(data_cut)
        frod_s = conditionals.FRodriguezSatellites.get_cond(data_cut)

        yang_id = yang_c | yang_s
        tempel_id = tempel_c | tempel_s
        frod_id = frod_c | frod_s
        all_id = yang_id & tempel_id & frod_id
        yang_tempel_two_id = yang_id & tempel_id & ~frod_id
        yang_frod_two_id = yang_id & ~tempel_id & frod_id
        tempel_frod_two_id = ~yang_id & tempel_id & frod_id
        two_id = yang_tempel_two_id + yang_frod_two_id + tempel_frod_two_id
        yang_one_id = yang_id & ~tempel_id & ~frod_id
        tempel_one_id = ~yang_id & tempel_id & ~frod_id
        frod_one_id = ~yang_id & ~tempel_id & frod_id
        one_id = yang_one_id + tempel_one_id + frod_one_id
        zero_id = ~yang_id & ~tempel_id & ~frod_id
        print(
            "ID'd by two catalogs  - ",
            f'Y+T ({sum(yang_tempel_two_id)}) |',
            f'Y+FR ({sum(yang_frod_two_id)}) |',
            f'T+FR ({sum(tempel_frod_two_id)}) |',
            f'Total two ID: {sum(two_id)} ({(sum(two_id) / len(two_id)):.2%})',
        )
        print(
            "ID'd by one catalog  - ",
            f'Y ({sum(yang_one_id)}) |',
            f'T ({sum(tempel_one_id)}) |',
            f'FR ({sum(frod_one_id)}) |',
            f'Total one ID: {sum(one_id)} ({(sum(one_id) / len(one_id)):.2%})',
        )
        print(f'No ID: {sum(zero_id)} ({(sum(zero_id) / len(zero_id)):.2%})', '\n')

        all_agree_c = sum(yang_c & tempel_c & frod_c)
        all_agree_s = sum(yang_s & tempel_s & frod_s)

        all_agree_c_w_id = sum(yang_c[all_id] & tempel_c[all_id] & frod_c[all_id])
        all_agree_s_w_id = sum(yang_s[all_id] & tempel_s[all_id] & frod_s[all_id])

        two_agree_c_w_id = (
            sum(yang_c[yang_tempel_two_id] & tempel_c[yang_tempel_two_id])
            + sum(yang_c[yang_frod_two_id] & frod_c[yang_frod_two_id])
            + sum(tempel_c[tempel_frod_two_id] & frod_c[tempel_frod_two_id])
        )
        two_agree_s_w_id = (
            sum(yang_s[yang_tempel_two_id] & tempel_s[yang_tempel_two_id])
            + sum(yang_s[yang_frod_two_id] & frod_s[yang_frod_two_id])
            + sum(tempel_s[tempel_frod_two_id] & frod_s[tempel_frod_two_id])
        )

        df = pd.DataFrame.from_dict(
            {
                'Yang': [sum(yang_c), sum(yang_s)],
                'Tempel': [sum(tempel_c), sum(tempel_s)],
                'FRodriguez': [sum(frod_c), sum(frod_s)],
                'All_Agree': [all_agree_c, all_agree_s],
                'All_Agree_w_ID': [all_agree_c_w_id, all_agree_s_w_id],
                'Two_Agree_w_ID': [two_agree_c_w_id, two_agree_s_w_id],
                'All_or_Two_w_ID': [
                    all_agree_c_w_id + two_agree_c_w_id,
                    all_agree_s_w_id + two_agree_s_w_id,
                ],
            },
            orient='index',
            columns=['Centrals', 'Satellites'],
        )

        df['Identifications'] = df['Centrals'] + df['Satellites']
        df['Unassigned'] = len(data_cut) - df['Identifications']
        df.loc['All_Agree_w_ID', 'Unassigned'] = (
            sum(all_id) - df.loc['All_Agree_w_ID', 'Identifications']
        )
        df['C/Total'] = df['Centrals'] / df['Identifications']
        df['C/S'] = df['Centrals'] / df['Satellites']

        df = df.astype(
            {
                'Centrals': np.int64,
                'Satellites': np.int64,
                'Identifications': np.int64,
                'Unassigned': np.int64,
                'C/Total': np.float64,
                'C/S': np.float64,
            }
        )
        w = 2
        col_space: dict[Any, Any] = {
            'Centrals': len('Centrals') + w,
            'Satellites': len('Satellites') + w,
            'Identifications': len('Identifications') + w,
            'Unassigned': len('Unassigned') + w,
            'C/Total': len('C/Total') + w,
            'C/S': len('C/S') + w + 2,
        }
        formatters: dict[Any, Any] = {
            'Centrals': lambda s: f'{s:,}',
            'Satellites': lambda s: f'{s:,}',
            'Identifications': lambda s: f'{s:,}',
            'Unassigned': lambda s: f'{s:,}',
            'C/Total': lambda s: f'{s:.3f}',
            'C/S': lambda s: f'{s:.3f}',
        }
        print(f'Mass range: {mb1} - {mb2}')
        print(df.to_string(col_space=col_space, formatters=formatters), '\n')

        mass.append(np.mean([mb1, mb2]))
        yang.append(df.at['Yang', 'C/S'])
        tempel.append(df.at['Tempel', 'C/S'])
        frod.append(df.at['FRodriguez', 'C/S'])

        all_ratio.append((all_agree_c_w_id + all_agree_s_w_id) / sum(all_id))
        two_ratio.append((two_agree_c_w_id + two_agree_s_w_id) / sum(two_id))
        all_or_two_ratio.append(
            (all_agree_c_w_id + all_agree_s_w_id + two_agree_c_w_id + two_agree_s_w_id)
            / (sum(all_id) + sum(two_id))
        )

    ratios = [yang, tempel, frod]
    colors = ['k', fcolors.MAGENTA, fcolors.LIGHT_BLUE]
    labels = [
        'Yang et al. (2012)',
        'Tempel et al. (2017)',
        r'Rodriguez \& Merch$\acute{\rm a}$n (2020)',
    ]
    markers = ['o', 's', '^']
    for ratio, color, label, marker in zip(ratios, colors, labels, markers):
        ax.plot(
            mass,
            ratio,
            color=color,
            label=label,
            marker=marker,
            markersize=4,
        )

    ax.set(
        xlabel=r'$\log(M_*/M_\odot)$',
        ylabel=r'$N_{\rm cen}/N_{\rm sat}$',
        xlim=(mass_bins[0] - 0.05, mass_bins[-1] + 0.05),
        xticks=[10, 10.5, 11, 11.5],
    )
    ax.legend(loc='upper left', fontsize=7)

    ax2.plot(mass, all_ratio, marker='.', color='k')
    ax2.set(
        xlabel=r'$\log(M_*/M_\odot)$',
        ylabel=r'$f_{\rm match}$',
        xlim=(mass_bins[0], mass_bins[-1]),
        ylim=(0.65, 1),
    )

    print(f'All_Agree_w_ID %:         {[f"{x:.2%}" for x in all_ratio]}')
    print(f'Two_Agree_w_ID %:         {[f"{x:.2%}" for x in two_ratio]}')
    print(f'All_or_Two_Agree_w_ID %:  {[f"{x:.2%}" for x in all_or_two_ratio]}')

    save.savefig(
        fig,
        'centrals_per_satellite',
        config.RESULTS_DIR.joinpath(configuration.dirname, config.MISC_DIRNAME),
    )
    save.savefig(
        fig2,
        'fraction_matched',
        config.RESULTS_DIR.joinpath(configuration.dirname, config.MISC_DIRNAME),
    )
