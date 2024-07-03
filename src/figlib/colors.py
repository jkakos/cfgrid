import colorsys

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import numpy.typing as npt

from src.protocols import binning


"""
The colors used here are derived from the following.

colors = ['#D81B60', '#1E88E5', '#FFCB05', '#00274C', '#5B9F49']
colors = [color_adj.lighten(color_adj.intensify(c, -0.1), 0.1) for c in colors]

ms_colors = colors = plt.cm.turbo(np.array([0.9, 0.4, 0.28, 0.21, 0.14, 0.07]))
gv = plt.cm.terrain(np.array([0.25]))

The 'colors' were lightened with 0.1 and intensified with -0.1. The
'ms_colors' came from the turbo color map with green replaced by one
from the terrain color map.

"""

MAGENTA = '#DB3370'
LIGHT_BLUE = '#3D93DF'
YELLOW = '#F4CA29'
DARK_BLUE = '#063D71'
GREEN = '#6AAC58'

MS_Q = '#C32503'
MS_GV = '#01CC66'
MS_BMS = '#1ECBDA'
MS_LMS = '#3BA0FD'
MS_UMS = '#4773EB'
MS_HSF = '#4143A7'


def get_colors(num_ybins: int, delta_MS: bool = False) -> npt.NDArray | list[str]:
    """
    Get a set of colors based on the 'turbo' color map. If 'delta_MS'
    is True, colors will be used from a preset of colors.

    """
    colors: npt.NDArray | list[str]
    if delta_MS:
        if num_ybins not in binning.DeltaMSBins.bins.keys():
            raise ValueError(
                f"'num_ybins' must be among {list(binning.DeltaMSBins.bins.keys())}"
                f" for delta MS bins ({num_ybins=} was given)."
            )

        if num_ybins == 6:
            colors = [MS_Q, MS_GV, MS_BMS, MS_LMS, MS_UMS, MS_HSF]
        elif num_ybins == 5:
            colors = [MS_Q, MS_GV, MS_BMS, MS_UMS, MS_HSF]
        elif num_ybins == 4:
            colors = [MS_Q, MS_GV, MS_LMS, MS_HSF]
        elif num_ybins == 3:
            colors = [MS_Q, MS_GV, MS_UMS]
        elif num_ybins == 2:
            colors = [MS_Q, MS_UMS]
    else:
        colors = plt.cm.turbo(np.linspace(0.9, 0.1, num_ybins + 1))
        colors = np.delete(colors, len(colors) // 2, axis=0).reshape(num_ybins, 4)

    return colors


def get_qualitative_colors(n: int) -> list[str]:
    """
    Get n qualitative colors. These are meant to be distinct from each
    other and not be sequential or perceptually uniform.

    """
    colors = [MAGENTA, LIGHT_BLUE, YELLOW, DARK_BLUE, GREEN]
    if n > len(colors):
        raise ValueError(
            f"Not enough preset colors. Only {len(colors)} are set but {n} requested."
        )
    return colors[:n]


def rgb_to_hls(color) -> tuple[float, float, float]:
    """
    Take a color and turn it into a set of hue, luminosity, and
    saturation. Colors can be given as matplotlib color strings, hex
    code strings, or a tuple of (r, g, b) values.

    """
    try:
        c = mc.cnames[color]
    except:
        c = color

    hue, lum, sat = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return hue, lum, sat


def lighten(color, amount=0.5):
    """
    Lightens the given color by adding amount * (1-luminosity). Amount
    should be a number between -1 and 1, where positive values lighten
    and negative values darken. If 0 is given, no lightening will
    occur. If 1 (or -1) is given, the lightening will be maximized and
    the color will return white (or black). For abs(amount) between 0
    and 1, the difference between the color's luminosity and 1 (or -1)
    will be linearly scaled by the given amount.

    Example scalings:
    lum=0.7, amount=0.5
    - Available room to lighten -> 1 - lum = 0.3
    - Lightening by 50% of available -> lum + 0.5*0.3 = 0.85

    lum=0.7, amount=-0.5
    - Available room to darken -> lum = 0.7
    - Darkening by 50% of available -> lum - 0.5*0.7 = 0.35

    Example colors:
    >> lighten('g', 0.3)
    >> lighten('#F034A3', 0.6)
    >> lighten((0.3, 0.55, 0.1), -0.5)

    """
    if amount < -1 or amount > 1:
        raise ValueError(f"Must give amount from -1 to +1 ({amount=} given).")

    hue, lum, sat = rgb_to_hls(color)

    if amount >= 0:
        new_lum = lum + amount * (1 - lum)
    else:
        new_lum = lum * (1 + amount)

    return colorsys.hls_to_rgb(hue, new_lum, sat)


def intensify(color, amount=0.5):
    """
    Works the same way as 'lighten' but for color saturation. See the
    'lighten' documentation for details.

    """
    if amount < -1 or amount > 1:
        raise ValueError(f"Must give amount from -1 to +1 ({amount=} given).")

    hue, lum, sat = rgb_to_hls(color)

    if amount >= 0:
        new_sat = sat + amount * (1 - sat)
    else:
        new_sat = sat * (1 + amount)

    return colorsys.hls_to_rgb(hue, lum, new_sat)
