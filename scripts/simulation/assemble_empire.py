import os
import pathlib

import numpy as np
import pandas as pd

import config


def main() -> None:
    files_dir = config.EMPIRE_CATALOG_DIR
    filenames = os.listdir(files_dir)
    files = [
        pathlib.Path(files_dir, file) for file in filenames if file.endswith('.dat')
    ]

    data = pd.concat(
        [pd.read_csv(file, sep='\s+') for file in files], ignore_index=True
    )
    data = data.rename(
        {
            '#Tree_root_ID': 'Tree_root_ID',
            'id_halo': 'ID',
            'pid': 'PID',
            'redshift': 'z_obs',
            # 'Ms_obs': 'M_star',
        },
        axis=1,
    )
    redshift = data.at[0, 'z_obs']
    data['M_star'] = 0.787 * np.log10(data['Ms_obs']) + 2.169
    data = data.query("M_star > 8.3").copy()
    data.loc[data['PID'] == 0, 'PID'] = -1
    # data['M_star'] = np.log10(
    #     data['M_star']
    # )  # Edit this to the new column then cut > 10^8.3

    if not all(data['z_obs'] == redshift):
        raise ValueError("Not all redshifts are the same in Empire.")

    zero_sfr = data['SFR_obs'] == 0
    data.loc[~zero_sfr, 'SFR_obs'] = np.log10(data.loc[~zero_sfr, 'SFR_obs'])
    data.loc[zero_sfr, 'SFR_obs'] = -20 + data.loc[zero_sfr, 'M_star']
    data.to_parquet(
        config.DATA_DIR.joinpath(f'empire_{redshift:.4f}.parquet'), index=False
    )


"""
Header:

Tree_root_ID id_halo pid upid redshift Mvir M200c M500c vmax rvir Mpeak vpeak rpeak
sf_eff dMcooling_dt Mcooling facc_hot Ms Ms_insitu Ms_exitu ICL num_mm_1to4 num_mm_1to10
num_mm_1to100 MH2 MHI Mgas dMgas_dt Re SFR MFUV LFIR Ms_expected Ms_50_expected MBH BHAR
MBH_merger M_BH_expected X Y Z Ms_obs SFR_obs MBH_obs u SFR_halo_ID BHAR_ID

"""
