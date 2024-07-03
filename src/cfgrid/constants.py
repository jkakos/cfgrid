from typing import Literal


CF_BIN_LABELS = {'rp': 'rp', 'rp_min': 'rp_min', 'rp_max': 'rp_max'}
CF_RESULTS_LABELS = {
    'wp': 'wp',
    'err': 'err',
    'mean': 'mean',
    'median': 'median',
    'counts': 'counts',
}

# Base strings used as prefixes to group results and figures
AUTO_BASE = 'auto_tpcf'
CROSS_BASE = 'cross_tpcf'
CROSS_ALL_SAT_BASE = 'all_sat_cross_tpcf'
CROSS_ALL_SAT_BELOW_BASE = 'all_sat_below_cross_tpcf'
COUNTS_BASE = 'counts'

RP_LABEL = r'$r_{\rm p}~[h^{-1}{\rm Mpc}]$'

# Used for comparison options in cfgrid figures
COMPARISON_LABELS = {'xbin': 'Mass bin', 'ms': 'MS'}
COMPARISON_OPTIONS = Literal['xbin', 'ms']
