import pathlib
from typing import Sequence

import matplotlib.pyplot as plt

import config
from src.figlib.config import RC_PARAMS


def savefig(
    fig: plt.Figure,
    filename: str,
    path: str | Sequence[str] | pathlib.Path | None = None,
    ext: str = 'pdf',
    dpi: int = 240,
) -> None:
    """
    Create a directory (if it does not exist) and save a figure to it.

    """
    filename_ext = f'{filename}.{ext}'

    if not isinstance(path, pathlib.Path):
        base_path = config.RESULTS_DIR

        if path is None:
            fig.savefig(base_path.joinpath(filename_ext), dpi=dpi)
            return

        if isinstance(path, str):
            path = (path,)

        dir_path = base_path.joinpath(*path)

    else:
        dir_path = path

    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_path.joinpath(filename_ext), dpi=dpi)


plt.rcParams.update(RC_PARAMS)

# This is used to make the vertical arrow on grid figures. This could be removed
# (along with the arrow) if needed.
plt.rc('text', usetex=True)
plt.rc(
    'text.latex',
    preamble=(
        r'\usepackage{wasysym}'
        r'\def\arrowtext#1#2{\hbox to#1{\arrowtextA\ #2 \arrowtextA\kern2pt\llap{$\RHD$}}}'
        r'\def\arrowtextA{\leaders\vrule height2.9pt depth-2.1pt\hfil}'
    ),
)
