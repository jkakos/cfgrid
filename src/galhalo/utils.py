import os
import pathlib
import re


def validate_scatter(scatter: str) -> float:
    try:
        scatter_float = float(scatter)
    except ValueError as e:
        raise ValueError(
            f"'scatter' must represent a number if given as a string ({scatter} was"
            " given)."
        ) from e

    return scatter_float


def get_scatters(path: pathlib.Path) -> list[str]:
    """
    Search through all files of the form 'cen_<decimal number>.dat' to
    extract the decimal numbers which will correspond to different
    amounts of scatter.

    """
    filenames = [
        f for f in os.listdir(path) if f.startswith('cen') and f.endswith('.dat')
    ]
    scatters = [re.search(r'\d+\.\d+', f) for f in filenames]
    if None in scatters:
        raise ValueError('A file with no scatter in its name was found.')

    return [s.group() for s in scatters if s is not None]
