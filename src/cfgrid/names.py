from src import cfgrid


def _get_filename(base: str, *suffix: str) -> str:
    """
    Join together a 'base' string and any given suffixes with
    underscores to create a single file name.

    """
    if not len(suffix):
        return base

    return f"{base}_{'_'.join(suffix)}"


def get_tpcf_filename(base: str, *suffix: str, num_ybins: int = 4) -> str:
    """
    Get the file name used to store tpcf results. Any given suffix will
    be added to the end of the file name.

    """
    prefix = _get_filename(base, *suffix)
    return f'{prefix}_ybins{num_ybins}.csv'


def get_tpcf_comp_filename(
    base: str, comparison: cfgrid.constants.COMPARISON_OPTIONS, *suffix: str
) -> str:
    """
    Get the file name used to store tpcf results for some comparison
    (e.g. the whole mass bin or only main sequence galaxies). Any given
    suffix will be added after the base file name.

    """
    cfgrid.utils.validate_comparison(comparison)
    prefix = _get_filename(base, *suffix)
    return f"{prefix}_{comparison}.csv"


def get_comp_from_filename(comp_filename: str) -> cfgrid.constants.COMPARISON_OPTIONS:
    """
    Pull off the last portion of a comparison file name to extract the
    comparison.

    """
    comparison = comp_filename.split('_')[-1].split('.')[0]
    valid_comparison = cfgrid.utils.validate_comparison(comparison)
    return valid_comparison


def get_plane_filename(*suffix: str, num_ybins: int = 4) -> str:
    """
    Get the file name used to save plane figures. This uses the
    'get_tpcf_filename' function to ensure it follows the same format.

    """
    tpcf_filename = get_tpcf_filename('plane', *suffix, num_ybins=num_ybins)
    return tpcf_filename.split('.')[0]


def get_results_suffix(col: int, row: int | None = None) -> str:
    """
    Get the suffix that will be used to store and access saved grid
    results based on a 'col' and an optional 'row' position in the
    grid.

    """
    suffix = f'{col}'

    if row is not None:
        suffix = f'{suffix}_{row}'

    return suffix
