import warnings

import Corrfunc.mocks as cfm
import Corrfunc.theory as cft
import numpy as np
import scipy.stats
from Corrfunc.utils import convert_3d_counts_to_cf
from scipy.ndimage import gaussian_filter as gf
from tqdm import tqdm


# fmt: off
def generate_random_sample(y, sample_size, nbins=200, width=2):
    """
    Generates a random sample based one the 1d distribution of y.

    Parameters
    ----------

    y : array-like
        Data that determines the shape of the random sample distribution.

    sample_size : int
        Number of points to sample from the new distribution.

    nbins : int
        Number of bins to use when making y histogram.

    width : int
        Amount of smoothing applied to the y distribution.

    Returns
    -------

    random_sample : array-like
        New sample of random values

    """
    if len(y) == 0:
        print('No data')
        return -1 * np.ones(sample_size)

    counts, bins = np.histogram(y, bins=nbins, density=True)
    pdf = scipy.stats.rv_histogram([gf(counts, width), bins])

    random_sample = pdf.rvs(size=sample_size)
    return random_sample


def generate_random_sdss_sky(ra, dec, sample_size):
    """
    Create a randomly distributed mock in ra and dec using the sky
    footprint of SDSS.

    """
    ra2 = np.random.uniform(min(ra), max(ra), size=sample_size)
    dec2 = np.random.uniform(min(dec), max(dec), size=sample_size)

    southern_limit = dec2 > 0
    western_limit = dec2 > -2.555556 * (ra2 - 131)
    eastern_limit = dec2 > 1.70909 * (ra2 - 235)
    northern_limit = dec2 < (
        180/np.pi * np.arcsin((0.93232*np.sin(np.pi/180 * (ra2-95.9)))
            / np.sqrt(1 - (0.93232*np.cos(np.pi/180 * (ra2-95.9)))**2)))

    within_limits = (
        southern_limit &
        western_limit &
        eastern_limit &
        northern_limit
    )

    return ra2[within_limits], dec2[within_limits]


def box_random(x, y, z, sample_size, nbins=200, width=2):
    """
    Generate a real-space random catalog assuming a box geometry.

    """
    xr = generate_random_sample(x, sample_size, nbins=nbins, width=width)
    yr = generate_random_sample(y, sample_size, nbins=nbins, width=width)
    zr = generate_random_sample(z, sample_size, nbins=nbins, width=width)

    # xr = np.random.uniform(min(x), max(x), size=sample_size)
    # yr = np.random.uniform(min(y), max(y), size=sample_size)
    # zr = np.random.uniform(min(z), max(z), size=sample_size)

    return xr, yr, zr


def sphere_random(ra, dec, z, sample_size, nbins=200, width=2):
    """
    Generate a redshift-space random catalog assuming a spherical geometry.

    """
    rar = generate_random_sample(ra, sample_size, nbins=nbins, width=width)
    decr = generate_random_sample(dec, sample_size, nbins=nbins, width=width)
    zr = generate_random_sample(z, sample_size, nbins=nbins, width=width)

    return rar, decr, zr


def sdss_random(ra, dec, z, sample_size, nbins=200, width=2):
    """
    Generate a random SDSS catalog using (ra, dec, z).

    """
    rar, decr = generate_random_sdss_sky(ra, dec, sample_size)
    zr = generate_random_sample(z, len(rar), nbins=nbins, width=width)

    return rar, decr, zr


def plot_smoothed_pdfs(y1, y2, y3, sample_size=None):
    """
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=3, ncols=1, sharey=False)
    axes = axes.flatten()

    if sample_size is None:
        if len(y1) > 5000:
            sample_size = 5 * len(y1)
        else:
            sample_size = 10 * len(y1)

    _plot_smoothed_pdfs(axes[0], y1, sample_size)
    _plot_smoothed_pdfs(axes[1], y2, sample_size)
    _plot_smoothed_pdfs(axes[2], y3, sample_size)

    axes[0].set_xlabel('X')
    axes[1].set_xlabel('Y')
    axes[2].set_xlabel('Z')
    axes[2].legend(ncol=3)

    plt.tight_layout()
    plt.show()


def _plot_smoothed_pdfs(ax, y, sample_size, nbins=200, width=2):
    """
    """
    counts, bins, _ = ax.hist(y, bins=200, color='k', histtype='stepfilled',
        alpha=0.3, density=True, label='Data')
    pdf = scipy.stats.rv_histogram([gf(counts, width), bins])
    adjusted_bins = bins[:-1] + (bins[1]-bins[0])/2
    ax.plot(adjusted_bins, pdf.pdf(adjusted_bins), color='b', alpha=0.5,
        label='Smoothed PDF')
    ax.hist(pdf.rvs(size=sample_size), bins=nbins, color='r', histtype='step',
        density=True, label='New Sample')


def get_auto_counts(DD, RR, DR, pimax, bins):
    dd = np.empty(len(bins)-1, dtype=np.int64)
    rr = np.empty(len(bins)-1, dtype=np.int64)
    dr = np.empty(len(bins)-1, dtype=np.int64)

    for i in range(len(bins) - 1):
        idx1 = i * pimax
        idx2 = (i+1) * pimax

        if not all([isinstance(idx, int) for idx in [idx1, idx2]]):
            raise ValueError(
                "'pimax' must be given as an integer for indexing to "
                "work properly following Corrfunc's example."
            )

        dd[i] = np.sum(DD['npairs'][idx1:idx2])
        rr[i] = np.sum(RR['npairs'][idx1:idx2])
        dr[i] = np.sum(DR['npairs'][idx1:idx2])

    return dd, rr, dr


def get_cross_counts(D1D2, D1R2, D2R1, R1R2, pimax, bins):
    d1d2 = np.empty(len(bins)-1, dtype=np.int64)
    d1r2 = np.empty(len(bins)-1, dtype=np.int64)
    d2r1 = np.empty(len(bins)-1, dtype=np.int64)
    r1r2 = np.empty(len(bins)-1, dtype=np.int64)

    for i in range(len(bins) - 1):
        idx1 = i * pimax
        idx2 = (i+1) * pimax

        d1d2[i] = np.sum(D1D2['npairs'][idx1:idx2], dtype=np.int64)
        d1r2[i] = np.sum(D1R2['npairs'][idx1:idx2], dtype=np.int64)
        d2r1[i] = np.sum(D2R1['npairs'][idx1:idx2], dtype=np.int64)
        r1r2[i] = np.sum(R1R2['npairs'][idx1:idx2], dtype=np.int64)

    return d1d2, d1r2, d2r1, r1r2


def counts_to_wp_LS(ND, NR, DD, RR, DR, pimax, bins):
    xi = np.empty(len(bins)-1, dtype=np.float64)
    m = NR / ND

    for i in range(len(bins) - 1):
        idx1 = i * pimax
        idx2 = (i+1) * pimax

        if not all([isinstance(idx, int) for idx in [idx1, idx2]]):
            raise ValueError(
                "'pimax' must be given as an integer for indexing to "
                "work properly following Corrfunc's example."
            )

        dd = np.sum(DD['npairs'][idx1:idx2])
        rr = np.sum(RR['npairs'][idx1:idx2])
        dr = np.sum(DR['npairs'][idx1:idx2])

        xi[i] = (m*m*dd/rr - 2*m*dr/rr + 1)

    wp = 2 * xi * pimax

    return wp


def counts_to_wp_cross(ND1, ND2, NR1, NR2, D1D2, D1R2, D2R1, R1R2, pimax, bins):
    xi = np.empty(len(bins)-1, dtype=np.float64)

    for i in range(len(bins) - 1):
        idx1 = i * pimax
        idx2 = (i+1) * pimax

        d1d2 = np.sum(D1D2['npairs'][idx1:idx2], dtype=np.int64) / (ND1*ND2)
        d1r2 = np.sum(D1R2['npairs'][idx1:idx2], dtype=np.int64) / (ND1*NR2)
        d2r1 = np.sum(D2R1['npairs'][idx1:idx2], dtype=np.int64) / (ND2*NR1)
        r1r2 = np.sum(R1R2['npairs'][idx1:idx2], dtype=np.int64) / (NR1*NR2)

        xi[i] = (d1d2 - d1r2 - d2r1 + r1r2) / r1r2

    wp = 2 * xi * pimax

    return wp


def calc_pair_counts(X, Y, Z, autocorr, bins, nthreads=1, **kwargs):
    """
    Calculates 3D or projected 2D pair counts.

    Parameters
    ----------

    X, Y, Z : array_like
        3D coordinates of points in the data set.

    autocorr : bool
        Whether to calculate an auto-correlation or cross-correlation
        function.

    bins : array_like
        Binning to use when finding pairs.

    nthreads : int, optional
        Number of OpenMP threads to use.

    Returns
    -------

    counts : Numpy array
        Array of pair counts per each bin.

    """
    if 'pimax' in kwargs:
        pimax = kwargs['pimax']
        del kwargs['pimax']
    else:
        pimax = None


    # make this work with 3d

    if 'periodic' not in kwargs:
        return cfm.DDrppi_mocks(autocorr, 2, nthreads, pimax, bins, X, Y, Z,
            is_comoving_dist=True, **kwargs)

    if pimax is not None:
        results = cft.DDrppi(autocorr, nthreads, pimax, bins, X, Y, Z, **kwargs)
        # counts = np.array([x[4] for x in results])

    else:
        results = cft.DD(autocorr, nthreads, bins, X, Y, Z, **kwargs)
        # counts = np.array([x[3] for x in results])

    return results


def calc_tpcf(
    X, Y, Z,
    bins,
    X2=None, Y2=None, Z2=None,
    RA2=None, DEC2=None, CZ2=None,
    bootstraps=0,
    nthreads=1,
    return_mean=False,
    return_median=False,
    return_counts=False,
    RR=None,
    **kwargs
):
    """
    Calculates the two-point correlation function. If `pimax` is
    provided as a kwarg, the projected 2D correlation function
    will be calculated with a projected distance `pimax`.

    Parameters
    ----------

    X, Y, Z : array_like
        3D coordinates of points in the data set.

    bins : array_like
        Binning to use when finding pairs.

    spherical : bool, optional
        If true, assume the coordinates are given as ra, dec, z.

    cond : array of booleans, optional
        Selects a subset of X, Y, Z according to this condition.

    bootstraps : int, optional
        Number of bootstrap iterations used to estimate errors.

    nthreads : int, optional
        Number of OpenMP threads to use.

    Returns
    -------

    xi_LS : array_like
        Two-point correlation function for the given bins.

    xi_LS_err : array_like
        Standard deviation of two-point correlation function bootstraps
        for the given bins.

    xi_LS_mean : array_like, optional
        Mean of two-point correlation function bootstraps for the
        given bins.

    xi_LS_median : array_like, optional
        Median of two-point correlation function bootstraps for the
        given bins.

    """
    if bootstraps < 0:
        raise ValueError("'bootstraps' must be non-negative")

    box = True if any(coord is not None for coord in (X2, Y2, Z2)) else False
    lightcone = True if any(coord is not None for coord in (RA2, DEC2, CZ2)) else False

    # Make sure exactly 1 set of random coordinates is provided.
    if (box and lightcone) or (not box and not lightcone):
        raise ValueError(
            "The random must be given as either (X2, Y2, Z2) or (RA2, DEC2, CZ2)."
        )

    pimax = kwargs.get('pimax')

    XD, YD, ZD = X, Y, Z
    XR, YR, ZR = (X2, Y2, Z2) if box else (RA2, DEC2, CZ2)

    # Auto pair counts for data and random catalog
    autocorr = True
    DD = calc_pair_counts(XD, YD, ZD, autocorr, bins, nthreads=nthreads, **kwargs)
    RR = calc_pair_counts(XR, YR, ZR, autocorr, bins, nthreads=nthreads, **kwargs)

    # Cross pair counts between data and random catalog
    autocorr = False
    if box:
        DR = calc_pair_counts(XD, YD, ZD, autocorr, bins, nthreads=nthreads,
            X2=XR, Y2=YR, Z2=ZR, **kwargs)
    elif lightcone:
        DR = calc_pair_counts(XD, YD, ZD, autocorr, bins, nthreads=nthreads,
            RA2=XR, DEC2=YR, CZ2=ZR, **kwargs)

    xi = []
    dd_counts = []
    rr_counts = []
    dr_counts = []

    ND = len(XD)
    NR = len(XR)

    if pimax is not None:
        xi.append(counts_to_wp_LS(ND, NR, DD, RR, DR, pimax, bins))
        dd, rr, dr = get_auto_counts(DD, RR, DR, pimax, bins)
        dd_counts.append(dd)
        rr_counts.append(rr)
        dr_counts.append(dr)
        # xi.append(convert_rp_pi_counts_to_wp(ND, ND, NR, NR, DD, DR, DR, RR,
        #     len(bins)-1, pimax, estimator='LS'))
    else:
        xi.append(convert_3d_counts_to_cf(ND, ND, NR, NR, DD, DR, DR, RR,
            estimator='LS'))

    if bootstraps > 0:
        for _ in tqdm(range(bootstraps)):
            sample_inds = np.random.randint(0, len(XD), len(XD))
            x = XD[sample_inds]
            y = YD[sample_inds]
            z = ZD[sample_inds]

            autocorr = True
            DD = calc_pair_counts(x, y, z, autocorr, bins, nthreads=nthreads,
                **kwargs)

            autocorr = False
            if box:
                DR = calc_pair_counts(x, y, z, autocorr, bins, nthreads=nthreads,
                    X2=XR, Y2=YR, Z2=ZR, **kwargs)
            elif lightcone:
                DR = calc_pair_counts(x, y, z, autocorr, bins, nthreads=nthreads,
                    RA2=XR, DEC2=YR, CZ2=ZR, **kwargs)

            if pimax is not None:
                xi.append(counts_to_wp_LS(ND, NR, DD, RR, DR, pimax, bins))
                dd, rr, dr = get_auto_counts(DD, RR, DR, pimax, bins)
                dd_counts.append(dd)
                rr_counts.append(rr)
                dr_counts.append(dr)

                # xi.append(convert_rp_pi_counts_to_wp(ND, ND, NR, NR, DD, DR, DR, RR,
                #     len(bins)-1, pimax, estimator='LS'))
            else:
                xi.append(convert_3d_counts_to_cf(ND, ND, NR, NR, DD, DR, DR, RR,
                    estimator='LS'))

    # Ignore runtimewarnings when axis is all nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)

        xi_err = np.nanstd(xi, axis=0)
        ret = [xi[0], xi_err, None, None]

        if return_mean:
            xi_mean = np.nanmean(xi, axis=0)
            ret[2] = xi_mean

        if return_median:
            xi_median = np.nanmedian(xi, axis=0)
            ret[3] = xi_median

        if return_counts:
            ret.append(dd_counts[0])
            ret.append(rr_counts[0])
            ret.append(dr_counts[0])
            ret.append(np.nanmean(dd_counts, axis=0))
            ret.append(np.nanmean(rr_counts, axis=0))
            ret.append(np.nanmean(dr_counts, axis=0))
            ret.append(np.nanmedian(dd_counts, axis=0))
            ret.append(np.nanmedian(rr_counts, axis=0))
            ret.append(np.nanmedian(dr_counts, axis=0))

    return tuple(ret)


def calc_cross_tpcf(
    XD1, YD1, ZD1,
    XD2, YD2, ZD2,
    XR1, YR1, ZR1,
    XR2, YR2, ZR2,
    bins,
    box: bool,
    bootstraps=0,
    nthreads=1,
    return_mean=False,
    return_median=False,
    return_counts=False,
    **kwargs
):
    """
    Calculates the two-point correlation function. If `pimax` is
    provided as a kwarg, the projected 2D correlation function
    will be calculated with a projected distance `pimax`.

    Parameters
    ----------

    X, Y, Z : array_like
        3D coordinates of points in the data set.

    bins : array_like
        Binning to use when finding pairs.

    cond : array of booleans, optional
        Selects a subset of X, Y, Z according to this condition.

    bootstraps : int, optional
        Number of bootstrap iterations used to estimate errors.

    nthreads : int, optional
        Number of OpenMP threads to use.

    Returns
    -------

    xi_LS : array_like
        Two-point correlation function for the given bins.

    xi_LS_err : array_like
        Standard deviation of two-point correlation function bootstraps
        for the given bins.

    xi_LS_mean : array_like, optional
        Mean of two-point correlation function bootstraps for the
        given bins.

    xi_LS_median : array_like, optional
        Median of two-point correlation function bootstraps for the
        given bins.

    """
    if bootstraps < 0:
        raise ValueError('`bootstraps` must be non-negative!')

    pimax = kwargs.get('pimax')
    ND1 = len(XD1)
    ND2 = len(XD2)
    NR1 = len(XR1)
    NR2 = len(XR2)

    # Cross pair counts
    autocorr = False
    if box:
        D1D2 = calc_pair_counts(XD1, YD1, ZD1, autocorr, bins, nthreads=nthreads,
            X2=XD2, Y2=YD2, Z2=ZD2, **kwargs)
        D1R2 = calc_pair_counts(XD1, YD1, ZD1, autocorr, bins, nthreads=nthreads,
            X2=XR2, Y2=YR2, Z2=ZR2, **kwargs)
        D2R1 = calc_pair_counts(XD2, YD2, ZD2, autocorr, bins, nthreads=nthreads,
            X2=XR1, Y2=YR1, Z2=ZR1, **kwargs)
        R1R2 = calc_pair_counts(XR1, YR1, ZR1, autocorr, bins, nthreads=nthreads,
            X2=XR2, Y2=YR2, Z2=ZR2, **kwargs)
    else:
        D1D2 = calc_pair_counts(XD1, YD1, ZD1, autocorr, bins, nthreads=nthreads,
            RA2=XD2, DEC2=YD2, CZ2=ZD2, **kwargs)
        D1R2 = calc_pair_counts(XD1, YD1, ZD1, autocorr, bins, nthreads=nthreads,
            RA2=XR2, DEC2=YR2, CZ2=ZR2, **kwargs)
        D2R1 = calc_pair_counts(XD2, YD2, ZD2, autocorr, bins, nthreads=nthreads,
            RA2=XR1, DEC2=YR1, CZ2=ZR1, **kwargs)
        R1R2 = calc_pair_counts(XR1, YR1, ZR1, autocorr, bins, nthreads=nthreads,
            RA2=XR2, DEC2=YR2, CZ2=ZR2, **kwargs)

    xi = []
    d1d2_counts = []
    d1r2_counts = []
    d2r1_counts = []
    r1r2_counts = []

    xi.append(
        counts_to_wp_cross(
            ND1, ND2, NR1, NR2, D1D2, D1R2, D2R1, R1R2, pimax, bins
        )
    )
    d1d2, d1r2, d2r1, r1r2 = get_cross_counts(D1D2, D1R2, D2R1, R1R2, pimax, bins)
    d1d2_counts.append(d1d2)
    d1r2_counts.append(d1r2)
    d2r1_counts.append(d2r1)
    r1r2_counts.append(r1r2)

    # xi.append((d1d2 - d1r2 - d2r1 + r1r2) / r1r2)

    # if pimax is not None:
    #     xi.append(counts_to_wp_LS(ND, NR, DD, RR, DR, pimax, bins))
    # else:
    #     xi.append(convert_3d_counts_to_cf(ND, ND, NR, NR, DD, DR, DR, RR,
    #         estimator='LS'))

    if bootstraps > 0:
        for _ in tqdm(range(bootstraps)):
            # Sample new points
            sample_d1_inds = np.random.randint(0, len(XD1), len(XD1))
            sample_d2_inds = np.random.randint(0, len(XD2), len(XD2))

            xd1 = XD1[sample_d1_inds]
            yd1 = YD1[sample_d1_inds]
            zd1 = ZD1[sample_d1_inds]

            xd2 = XD2[sample_d2_inds]
            yd2 = YD2[sample_d2_inds]
            zd2 = ZD2[sample_d2_inds]

            # Cross pair counts
            autocorr = False
            if box:
                D1D2 = calc_pair_counts(xd1, yd1, zd1, autocorr, bins, nthreads=nthreads,
                    X2=xd2, Y2=yd2, Z2=zd2, **kwargs)
                D1R2 = calc_pair_counts(xd1, yd1, zd1, autocorr, bins, nthreads=nthreads,
                    X2=XR2, Y2=YR2, Z2=ZR2, **kwargs)
                D2R1 = calc_pair_counts(xd2, yd2, zd2, autocorr, bins, nthreads=nthreads,
                    X2=XR1, Y2=YR1, Z2=ZR1, **kwargs)
            else:
                D1D2 = calc_pair_counts(xd1, yd1, zd1, autocorr, bins, nthreads=nthreads,
                    RA2=xd2, DEC2=yd2, CZ2=zd2, **kwargs)
                D1R2 = calc_pair_counts(xd1, yd1, zd1, autocorr, bins, nthreads=nthreads,
                    RA2=XR2, DEC2=YR2, CZ2=ZR2, **kwargs)
                D2R1 = calc_pair_counts(xd2, yd2, zd2, autocorr, bins, nthreads=nthreads,
                    RA2=XR1, DEC2=YR1, CZ2=ZR1, **kwargs)

            xi.append(
                counts_to_wp_cross(
                    ND1, ND2, NR1, NR2, D1D2, D1R2, D2R1, R1R2, pimax, bins
                )
            )
            d1d2, d1r2, d2r1, r1r2 = get_cross_counts(
                D1D2, D1R2, D2R1, R1R2, pimax, bins
            )
            d1d2_counts.append(d1d2)
            d1r2_counts.append(d1r2)
            d2r1_counts.append(d2r1)
            r1r2_counts.append(r1r2)

            # if pimax is not None:
            #     xi.append(counts_to_wp_LS(ND, NR, DD, RR, DR, pimax, bins))

            #     # xi.append(convert_rp_pi_counts_to_wp(ND, ND, NR, NR, DD, DR, DR, RR,
            #     #     len(bins)-1, pimax, estimator='LS'))
            # else:
            #     xi.append(convert_3d_counts_to_cf(ND, ND, NR, NR, DD, DR, DR, RR,
            #         estimator='LS'))

    # Ignore runtimewarnings when axis is all nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
    #     xi_mean = np.nanmean(xi, axis=0)
    #     xi_err = np.nanstd(xi, axis=0)
        xi_err = np.nanstd(xi, axis=0)
        ret = [xi[0], xi_err, None, None]

        if return_mean:
            xi_mean = np.nanmean(xi, axis=0)
            ret[2] = xi_mean

        if return_median:
            xi_median = np.nanmedian(xi, axis=0)
            ret[3] = xi_median

        if return_counts:
            ret.append(d1d2_counts[0])
            ret.append(d1r2_counts[0])
            ret.append(d2r1_counts[0])
            ret.append(r1r2_counts[0])
            ret.append(np.nanmean(d1d2_counts, axis=0))
            ret.append(np.nanmean(d1r2_counts, axis=0))
            ret.append(np.nanmean(d2r1_counts, axis=0))
            ret.append(np.nanmean(r1r2_counts, axis=0))
            ret.append(np.nanmedian(d1d2_counts, axis=0))
            ret.append(np.nanmedian(d1r2_counts, axis=0))
            ret.append(np.nanmedian(d2r1_counts, axis=0))
            ret.append(np.nanmedian(r1r2_counts, axis=0))

    # return xi[0], xi_err
    # # return xi_mean, xi_err
    return tuple(ret)


def calc_ratio_errors(x, y, sigma_x, sigma_y):
    """Propagates the uncertainty in x and y to the ratio of x and y."""
    return np.abs(x/y) * np.sqrt((sigma_x/x)**2 + (sigma_y/y)**2)
