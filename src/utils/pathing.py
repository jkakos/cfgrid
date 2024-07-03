import pathlib


def _get_path(base: pathlib.Path, *suffix: str) -> pathlib.Path:
    """
    Create and/or get path where figures and results will be stored.

    """
    path = base.joinpath(*suffix)
    path.mkdir(parents=True, exist_ok=True)

    return path


def get_results_path_from_path(base: pathlib.Path, *suffix: str) -> pathlib.Path:
    """
    Create and/or get path where results will be stored.

    """
    return _get_path(base, *suffix, 'results')


def get_path(
    base_level: pathlib.Path,
    dataset: str,
    *other: str,
) -> pathlib.Path:
    """
    This differentiates different sets of results which will generally
    only differ in the choice of volumes (which is determined by the
    mass limits and binning). Other modifications could be made here if
    more differentiation is required.

    """
    path_levels = [dataset]

    for level in other:
        path_levels.append(level)

    return _get_path(base_level, *path_levels)


def get_results_path(
    base_level: pathlib.Path,
    dataset: str,
    *other: str,
) -> pathlib.Path:
    """
    Run the given arguments through the pathing pipeline to get the
    results path. See 'utils.pathing.get_path'.

    """
    return get_results_path_from_path(get_path(base_level, dataset, *other))
