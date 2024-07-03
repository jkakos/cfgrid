SUBSAMPLES = {
    'all': "M_star > @mlow and M_star < @mhigh",
    'Q': "delta_ms_bin == 0",
    'GVQ': "delta_ms_bin in [0, 1]",
    'GV': "delta_ms_bin == 1",
    'BMS': "delta_ms_bin == 2",
    'LMS': "delta_ms_bin == 3",
    'UMS': "delta_ms_bin == 4",
    'SFMS': "delta_ms_bin in [3, 4]",
    'HSF': "delta_ms_bin == 5",
    'SF': "delta_ms_bin > 1",
    'nonHSF': "delta_ms_bin in [2, 3, 4]",
}
RESULTS_LABELS = {'logMs': 'logMs', 'Ns': 'Ns', 'err_Ns': 'err_Ns', 'n_gals': 'n_gals'}
