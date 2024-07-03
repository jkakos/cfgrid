from . import utils


def get_base_mock_filename(redshift: float) -> str:
    """
    Sets the standard filename structure for Bolshoi-Planck snapshots.

    """
    return f'bp_snapshot_{redshift:.4f}.parquet'


def get_mstar_mock_filename(redshift: float, scatter: float | str) -> str:
    """
    Sets the standard filename structure for stellar masses assigned to
    Bolshoi-Planck snapshots.

    """
    if isinstance(scatter, str):
        scatter = utils.validate_scatter(scatter)

    return f'bp_snapshot_{redshift:.4f}_{scatter:.3f}.parquet'


def get_mock_filename(redshift: float, scatter: float | str) -> str:
    """
    Sets the standard filename structure for Bolshoi-Planck mock SDSS
    catalogs that have been projected into redshift space and cut in
    stellar mass and redshift.

    """
    if isinstance(scatter, str):
        scatter = utils.validate_scatter(scatter)

    return f'bp_snapshot_{redshift:.4f}_{scatter:.3f}_zcut.parquet'


def get_galhalo_sample_filename(obs_label: str, sim_label: str) -> str:
    """
    Return the file name used for the galhalo sample procedure. The
    form used in this function will set the general pattern used.

    """
    return f'{obs_label}_{sim_label}.parquet'
