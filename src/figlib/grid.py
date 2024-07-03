import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .config import FIGWIDTH


class Grid:
    def __init__(
        self,
        figsize: tuple[float, float] = (6, 4),
        nrows: int = 1,
        ncols: int = 1,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xfontsize: int = 10,
        yfontsize: int = 10,
        sharex: bool = True,
        sharey: bool = True,
        constrained_layout: bool = True,
    ) -> None:
        self.fig, self.ax = plt.subplots(
            figsize=figsize,
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            constrained_layout=constrained_layout,
        )

        self.ax = self.setup_axes(
            self.ax, nrows, ncols, xlabel, ylabel, xfontsize, yfontsize
        )

    @staticmethod
    def setup_axes(
        axes: npt.NDArray[plt.axes],
        nrows: int,
        ncols: int,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xfontsize: int = 10,
        yfontsize: int = 10,
    ) -> npt.NDArray[plt.axes]:
        """
        Apply default settings to the subplots to limit where axis
        ticks are shown and ensure all subplots share the same x
        and y limits.

        """
        try:
            axes_ = axes.flatten()
        except AttributeError:
            axes_ = np.array([axes])

        n_subplots = nrows * ncols
        for i, ax in enumerate(axes_):
            ax.tick_params(
                which='both',
                direction='in',
                axis='both',
                labelleft=False,
                labelbottom=False,
            )

            ax.minorticks_on()

        for i in range(nrows):
            axes_[i * ncols].set_ylabel(ylabel, fontsize=yfontsize)
            axes_[i * ncols].tick_params(axis='y', labelleft=True)

        for i in range(n_subplots - ncols, n_subplots):
            axes_[i].set_xlabel(xlabel, fontsize=xfontsize)
            axes_[i].tick_params(axis='x', labelbottom=True)

        return axes_.reshape(nrows, ncols)


def get_grid_size(num_xbins: int, num_ybins: int) -> tuple[float, float]:
    fig_width = 2 * FIGWIDTH if num_xbins > 2 else FIGWIDTH
    fig_height = (1.0 + (num_ybins - 4) / 4) * FIGWIDTH

    if fig_height < (0.6 * FIGWIDTH):
        fig_height = 0.6 * FIGWIDTH

    return fig_width, fig_height


def get_grid_adjustments(
    fig_width: float, fig_height: float, right_label: bool = False
) -> tuple[float, float, float, float]:
    """
    Set fixed sizes for the figure margins to ensure labels have
    sufficient space. All numbers here are in the units matplotlib uses
    (currently inches).

    """
    fig_left = 0.50 / fig_width
    fig_bottom = 0.40 / fig_height
    fig_top = (fig_height - 0.16) / fig_height

    if right_label:
        fig_right = (fig_width - 0.25) / fig_width
    else:
        fig_right = (fig_width - 0.04) / fig_width

    return fig_left, fig_right, fig_top, fig_bottom


def add_spanning_ylabel(
    fig: plt.Figure,
    top: float,
    bottom: float,
    ylabel: str,
) -> None:
    """
    Add 'ylabel' that is anchored to the center of all the axes in the
    vertical direction of a 2d grid.

    """
    width, height = fig.get_size_inches()
    outer_label_ypos = bottom + 0.5 * (top - bottom)
    fig.text(
        0.04 / width,
        outer_label_ypos / height,
        ylabel,
        ha='left',
        va='center',
        fontsize=10,
        rotation=90,
    )


def add_right_vertical_arrow(
    fig: plt.Figure,
    top: float,
    bottom: float,
    ylabel: str,
    height: float,
) -> None:
    fig_width, fig_height = fig.get_size_inches()
    outer_label_ypos = bottom + 0.5 * (top - bottom)
    fig.text(
        (fig_width - 0.04) / fig_width,
        outer_label_ypos / fig_height,
        fr'\arrowtext{{{height}in}}{{\rm Increasing {ylabel}}}',
        ha='right',
        va='center',
        fontsize=10,
        rotation=90,
    )
