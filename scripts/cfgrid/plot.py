from typing import Callable, cast, Optional, Sequence

from src import cfgrid, configurations, galhalo
from src.protocols import conditionals, properties
from src.utils import split


def plane() -> None:
    """
    Make a scatter plot of x and y showing the bins used for the grid.

    """
    configuration = cfgrid.config.CONFIG
    xbins = cfgrid.config.XBINS
    data = configuration.load()

    for cond_type in cfgrid.config.PLANE_CONDITIONALS:
        cond_obj = cond_type(data)
        filtered_data = data[cond_obj.value]
        path = cfgrid.pathing.get_cfgrid_path(configuration.dirname, cond_type.label)

        for ybin_scheme in configuration.ybin_schemes:
            y_property = ybin_scheme.property_(filtered_data)
            ybin_strategy = ybin_scheme.bin_strategy
            num_ybins = ybin_scheme.num_bins

            delta_ms = isinstance(y_property, properties.DeltaMS)

            if delta_ms:
                x_property = properties.Mstar(filtered_data)
                ssfr: properties.Property = properties.SSFR(filtered_data)
                # Use only good values of properties
                [x, y, ssfr], _ = properties.standardize(x_property, y_property, ssfr)
            else:
                x_property = cfgrid.config.X_PROPERTY(filtered_data)
                # Use only good values of properties
                [x, y], _ = properties.standardize(x_property, y_property)

            # Get ybins
            ybins = split.get_ybins(
                x=x.value,
                y=y.value,
                xbins=xbins[1:-1],
                ybin_strategy=ybin_strategy,
                num_ybins=num_ybins,
            )

            if delta_ms:
                mass = cast(properties.Mstar, x)
                delta_MS = cast(properties.DeltaMS, y)
                ssfr = cast(properties.SSFR, ssfr)
                cfgrid.plot.plane_delta_ms(
                    mass=mass,
                    ssfr=ssfr,
                    delta_ms=delta_MS,
                    xbins=xbins,
                    ybins=ybins,
                    path=path,
                )
            else:
                cfgrid.plot.plane(
                    x_property=x,
                    y_property=y,
                    xbins=xbins,
                    ybins=ybins,
                    path=path,
                )


def plane_galhalo() -> None:
    """
    Make a scatter plot of x and y using galhalo models showing the
    bins used for the grid.

    """
    halo_config = galhalo.config.CONFIG
    xbins = cfgrid.config.XBINS
    centrals = halo_config.centrals[0]

    path = galhalo.pathing.get_galhalo_path(halo_config)

    for gal, halo in galhalo.config.PAIRINGS:
        ybin_scheme = galhalo.config.YBIN_SCHEMES[(gal, halo)]
        halo_config = galhalo.config.CONFIG_TYPE()
        halo_config.set_galhalo_props(gal.file_label, halo.file_label)
        data = halo_config.load()
        cond_obj = centrals(data)
        filtered_data = data[cond_obj.value]
        y_property = ybin_scheme.property_(filtered_data)
        ybin_strategy = ybin_scheme.bin_strategy
        num_ybins = ybin_scheme.num_bins

        delta_ms = isinstance(y_property, properties.DeltaMS)

        if delta_ms:
            x_property = properties.Mstar(filtered_data)
            ssfr: properties.Property = properties.SSFR(filtered_data)
            # Use only good values of properties
            [x, y, ssfr], _ = properties.standardize(x_property, y_property, ssfr)
        else:
            x_property = galhalo.config.MASS(filtered_data)
            # Use only good values of properties
            [x, y], _ = properties.standardize(x_property, y_property)

        # Get ybins
        ybins = split.get_ybins(
            x=x.value,
            y=y.value,
            xbins=xbins[1:-1],
            ybin_strategy=ybin_strategy,
            num_ybins=num_ybins,
        )

        if delta_ms:
            mass = cast(properties.Mstar, x)
            delta_MS = cast(properties.DeltaMS, y)
            ssfr = cast(properties.SSFR, ssfr)
            cfgrid.plot.plane_delta_ms(
                mass=mass,
                ssfr=ssfr,
                delta_ms=delta_MS,
                xbins=xbins,
                ybins=ybins,
                path=path,
            )
        else:
            cfgrid.plot.plane(
                x_property=x,
                y_property=y,
                xbins=xbins,
                ybins=ybins,
                path=path,
            )


def _grid(
    func: Callable,
    base: str,
    conds: Sequence[type[conditionals.Conditional]],
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
    autocorr: bool = True,
) -> None:
    configuration = cfgrid.config.CONFIG
    xbins = cfgrid.config.XBINS

    for cond_type in conds:
        for ybin_scheme in configuration.ybin_schemes:
            y_property = ybin_scheme.property_
            num_ybins = ybin_scheme.num_bins
            path = cfgrid.pathing.get_cfgrid_path(
                configuration.dirname, cond_type.label
            )
            results_filename = cfgrid.names.get_tpcf_filename(
                base, y_property.file_label, num_ybins=num_ybins
            )

            if comparison is not None:
                valid_comparison = cfgrid.utils.validate_comparison(comparison)
                comp_filename = cfgrid.names.get_tpcf_comp_filename(
                    base, valid_comparison
                )
            else:
                comp_filename = None

            if comparison != 'ms' or (
                comparison == 'ms' and y_property == properties.DeltaMS
            ):
                func(
                    path=path,
                    results_filename=results_filename,
                    prop=y_property,
                    num_xbins=len(xbins) - 1,
                    num_ybins=num_ybins,
                    xbins=xbins,
                    col_label=cfgrid.config.X_PROPERTY.full_label,
                    rp_scale=True,
                    autocorr=autocorr,
                    comp_filename=comp_filename,
                )


def _grid_all(
    func: Callable,
    base: str,
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
) -> None:
    """
    Plot a grid of correlation functions showing all galaxies, centrals
    only, and satellites only, with centrals and satellites being
    scaled based on their relative counts of galaxies.

    """
    configuration = cfgrid.config.CONFIG
    xbins = cfgrid.config.XBINS

    for ybin_scheme in configuration.ybin_schemes:
        y_property = ybin_scheme.property_
        num_ybins = ybin_scheme.num_bins

        # Get different results paths
        all_path = cfgrid.pathing.get_cfgrid_path(
            configuration.dirname, conditionals.AllGalaxies.label
        )
        centrals_path = cfgrid.pathing.get_cfgrid_path(
            configuration.dirname, conditionals.Centrals.label
        )
        satellites_path = cfgrid.pathing.get_cfgrid_path(
            configuration.dirname, conditionals.Satellites.label
        )

        # Get results file names
        results_filename = cfgrid.names.get_tpcf_filename(
            base, y_property.file_label, num_ybins=num_ybins
        )
        counts_filename = cfgrid.names.get_tpcf_filename(
            cfgrid.constants.COUNTS_BASE, y_property.file_label, num_ybins=num_ybins
        )

        if comparison is not None:
            valid_comparison = cfgrid.utils.validate_comparison(comparison)
            comp_filename = cfgrid.names.get_tpcf_comp_filename(base, valid_comparison)
        else:
            comp_filename = None

        if comparison != 'ms' or (
            comparison == 'ms' and y_property == properties.DeltaMS
        ):
            func(
                path=[all_path, centrals_path, satellites_path],
                results_filename=results_filename,
                counts_filename=counts_filename,
                prop=y_property,
                num_xbins=len(xbins) - 1,
                num_ybins=num_ybins,
                xbins=xbins,
                col_label=cfgrid.config.X_PROPERTY.full_label,
                rp_scale=True,
                comp_filename=comp_filename,
            )


def _condensed_grid_multi(
    func: Callable,
    base: str,
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
    autocorr: bool = True,
) -> None:
    """
    Plot a grid of correlation functions that shows results of the
    Yang, Tempel, and Rodriguez group catalogs.

    """
    configuration = cfgrid.config.CONFIG
    xbins = cfgrid.config.XBINS
    conds: list[conditionals.Conditional] = [
        conditionals.Centrals,
        conditionals.TempelCentrals,
        conditionals.FRodriguezCentrals,
    ]
    row_labels = [
        'Yang et al. (2012)',
        'Tempel et al. (2017)',
        r'Rodriguez \&' + '\n' + r'Merch$\acute{\rm a}$n (2020)',
    ]

    for ybin_scheme in configuration.ybin_schemes:
        y_property = ybin_scheme.property_
        num_ybins = ybin_scheme.num_bins
        path = [
            cfgrid.pathing.get_cfgrid_path(configuration.dirname, cond_type.label)
            for cond_type in conds
        ]
        results_filename = cfgrid.names.get_tpcf_filename(
            base, y_property.file_label, num_ybins=num_ybins
        )

        if comparison is not None:
            valid_comparison = cfgrid.utils.validate_comparison(comparison)
            comp_filename = cfgrid.names.get_tpcf_comp_filename(base, valid_comparison)
        else:
            comp_filename = None

        if comparison != 'ms' or (
            comparison == 'ms' and y_property == properties.DeltaMS
        ):
            func(
                path=path,
                results_filename=results_filename,
                prop=y_property,
                num_xbins=len(xbins) - 1,
                num_ybins=num_ybins,
                xbins=xbins,
                col_label=cfgrid.config.X_PROPERTY.full_label,
                row_labels=row_labels,
                rp_scale=True,
                autocorr=autocorr,
                comp_filename=comp_filename,
            )


def _grid_galhalo(
    func: Callable,
    base: str,
    cond: type[conditionals.Conditional],
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
    autocorr: bool = True,
) -> None:
    halo_config = galhalo.config.CONFIG
    gal_config = halo_config.gal_config
    xbins = cfgrid.config.XBINS

    for gal, halo in list(galhalo.config.PAIRINGS.keys()):
        ybin_scheme = galhalo.config.YBIN_SCHEMES[(gal, halo)]
        y_property = ybin_scheme.property_
        num_ybins = ybin_scheme.num_bins

        gal_path = cfgrid.pathing.get_cfgrid_path(gal_config.dirname, cond.label)
        halo_path = galhalo.pathing.get_galhalo_tpcf_path(halo_config).parent

        gal_results_filename = cfgrid.names.get_tpcf_filename(
            base, y_property.file_label, num_ybins=num_ybins
        )
        halo_results_filename = cfgrid.names.get_tpcf_filename(
            f"{galhalo.constants.GALHALO_BASE}_{base}",
            y_property.file_label,
            halo.file_label,
            num_ybins=num_ybins,
        )

        if comparison is not None:
            valid_comparison = cfgrid.utils.validate_comparison(comparison)
            gal_comp_filename = cfgrid.names.get_tpcf_comp_filename(
                base, valid_comparison
            )
            if comparison == 'ms':
                halo_comp_filename = cfgrid.names.get_tpcf_comp_filename(
                    f"{galhalo.constants.GALHALO_BASE}_{base}_{halo.file_label}",
                    valid_comparison,
                )
            else:
                halo_comp_filename = cfgrid.names.get_tpcf_comp_filename(
                    f"{galhalo.constants.GALHALO_BASE}_{base}",
                    valid_comparison,
                )
        else:
            gal_comp_filename = None
            halo_comp_filename = None

        if comparison != 'ms' or (
            comparison == 'ms' and y_property == properties.DeltaMS
        ):
            func(
                path=(gal_path, halo_path),
                results_filename=(gal_results_filename, halo_results_filename),
                prop=(y_property, halo),
                num_xbins=len(xbins) - 1,
                num_ybins=num_ybins,
                xbins=xbins,
                col_label=cfgrid.config.X_PROPERTY.full_label,
                rp_scale=True,
                autocorr=autocorr,
                comp_filename=(gal_comp_filename, halo_comp_filename),
            )


def _grid_galhalo_multi(
    base: str,
    cond: type[conditionals.Conditional],
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
    autocorr: bool = True,
) -> None:
    import copy
    import matplotlib.pyplot as plt
    from src.figlib import legend, save
    from src.figlib import colors as fcolors

    halo_config = galhalo.config.CONFIG
    gal_config = halo_config.gal_config

    rp_scale = True
    xbins = cfgrid.config.XBINS
    pairings = list(galhalo.config.PAIRINGS.keys())
    gals = [p[0] for p in pairings]
    halos = [p[1] for p in pairings]

    if not all(g == gals[0] for g in gals):
        raise ValueError("All gal properties must be the same. Check galhalo pairings.")

    gal = gals[0]
    halo_paths = [
        galhalo.pathing.get_galhalo_tpcf_path(halo_config).parent for _ in pairings
    ]

    ybin_scheme = galhalo.config.YBIN_SCHEMES[(gal, halos[0])]
    y_property = ybin_scheme.property_
    num_ybins = ybin_scheme.num_bins

    gal_path = cfgrid.pathing.get_cfgrid_path(gal_config.dirname, cond.label)

    gal_results_filename = cfgrid.names.get_tpcf_filename(
        base, y_property.file_label, num_ybins=num_ybins
    )
    halo_results_filenames = [
        cfgrid.names.get_tpcf_filename(
            f"{galhalo.constants.GALHALO_BASE}_{base}",
            y_property.file_label,
            halo.file_label,
            num_ybins=num_ybins,
        )
        for halo in halos
    ]

    if comparison is not None:
        valid_comparison = cfgrid.utils.validate_comparison(comparison)
        gal_comp_filename = cfgrid.names.get_tpcf_comp_filename(base, valid_comparison)
    else:
        gal_comp_filename = None

    if comparison != 'ms' or (comparison == 'ms' and y_property == properties.DeltaMS):
        fig, axes = cfgrid.plot._grid_compare(
            path=(gal_path, *halo_paths),
            results_filename=(gal_results_filename, *halo_results_filenames),
            prop=(y_property, *halos),
            num_xbins=len(xbins) - 1,
            num_ybins=num_ybins,
            xbins=xbins,
            col_label=cfgrid.config.X_PROPERTY.full_label,
            rp_scale=True,
            autocorr=autocorr,
            comp_filename=(gal_comp_filename, None),
        )

        if autocorr:
            ylim = (1.001, 1900)
        else:
            ylim = (3, 9000)

        axes[0, 0].set_ylim(ylim)

        points_kwargs, _, line_kwargs_, _ = cfgrid.plot.get_grid_compare_plot_kwargs()
        line_kwargs = []
        colors = ['k', fcolors.LIGHT_BLUE, fcolors.MAGENTA]
        linestyles = ['-.', '--', '-']
        for i in range(len(halos)):
            _line_kwargs = copy.deepcopy(line_kwargs_)
            _line_kwargs['color'] = colors[i]
            _line_kwargs['ls'] = linestyles[i]
            line_kwargs.append(_line_kwargs)

        plot_kwargs = [points_kwargs]
        for lk in line_kwargs:
            plot_kwargs.append(lk)

        fig = legend.add_galhalo_legend(fig, halos, plot_kwargs)

        # Set the figure file name
        fig_filename = f"multi_{halo_results_filenames[-1].split('.')[0]}"

        if gal_comp_filename is not None:
            fig_filename = f'{fig_filename}_{comparison}'
        if not rp_scale:
            fig_filename = f'{fig_filename}_norp'

        save.savefig(fig, filename=fig_filename, path=halo_paths[0])
        plt.close('all')


def _compare_obs_and_empire(
    base: str,
    conds: Sequence[type[conditionals.Conditional]],
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
    autocorr: bool = True,
    fig_filename_base: str = 'mpa_comparison',
) -> None:
    """
    Plot a grid of correlations functions that shows both MPA and
    Empire results.

    """
    empire_config = configurations.EmpireConfigVolume1()
    mpa_config = empire_config.obs_config
    xbins = cfgrid.config.XBINS

    for cond_type in conds:
        for ybin_scheme in mpa_config.ybin_schemes:
            y_property = ybin_scheme.property_
            num_ybins = ybin_scheme.num_bins
            mpa_path = cfgrid.pathing.get_cfgrid_path(
                mpa_config.dirname, cond_type.label
            )
            empire_path = cfgrid.pathing.get_cfgrid_path(
                empire_config.dirname, cond_type.label
            )
            results_filename = cfgrid.names.get_tpcf_filename(
                base, y_property.file_label, num_ybins=num_ybins
            )

            if comparison is not None:
                valid_comparison = cfgrid.utils.validate_comparison(comparison)
                comp_filename = cfgrid.names.get_tpcf_comp_filename(
                    base, valid_comparison
                )
            else:
                comp_filename = None

            if comparison != 'ms' or (
                comparison == 'ms' and y_property == properties.DeltaMS
            ):
                cfgrid.plot.grid_empire(
                    path=(mpa_path, empire_path),
                    results_filename=(results_filename, results_filename),
                    prop=(y_property, y_property),
                    num_xbins=len(xbins) - 1,
                    num_ybins=num_ybins,
                    xbins=xbins,
                    col_label=cfgrid.config.X_PROPERTY.full_label,
                    rp_scale=True,
                    autocorr=autocorr,
                    comp_filename=(comp_filename, comp_filename),
                    fig_filename=fig_filename_base,
                )


def auto_corr() -> None:
    _grid(
        cfgrid.plot.plot_grid,
        cfgrid.constants.AUTO_BASE,
        cfgrid.config.AUTO_CONDITIONALS,
        comparison='xbin',
        autocorr=True,
    )


def auto_corr_ms() -> None:
    _grid(
        cfgrid.plot.plot_grid,
        cfgrid.constants.AUTO_BASE,
        cfgrid.config.AUTO_CONDITIONALS,
        comparison='ms',
        autocorr=True,
    )


def cross_corr_all_satellites() -> None:
    _grid(
        cfgrid.plot.plot_grid,
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        cfgrid.config.CONFIG.centrals,
        comparison='xbin',
        autocorr=False,
    )


def cross_corr_all_satellites_ms() -> None:
    _grid(
        cfgrid.plot.plot_grid,
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        cfgrid.config.CONFIG.centrals,
        comparison='ms',
        autocorr=False,
    )


def condensed_grid() -> None:
    _grid(
        cfgrid.plot.condensed_grid,
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        cfgrid.config.CONFIG.centrals,
        comparison='xbin',
        autocorr=False,
    )


def condensed_grid_ms() -> None:
    _grid(
        cfgrid.plot.condensed_grid,
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        cfgrid.config.CONFIG.centrals,
        comparison='ms',
        autocorr=False,
    )


def condensed_grid_multi() -> None:
    _condensed_grid_multi(
        cfgrid.plot.condensed_grid_multi,
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        comparison='xbin',
        autocorr=False,
    )


def condensed_grid_multi_ms() -> None:
    _condensed_grid_multi(
        cfgrid.plot.condensed_grid_multi,
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        comparison='ms',
        autocorr=False,
    )


def condensed_grid_multi_auto_ms() -> None:
    _condensed_grid_multi(
        cfgrid.plot.condensed_grid_multi,
        cfgrid.constants.AUTO_BASE,
        comparison='ms',
        autocorr=True,
    )


def all_corr_ms() -> None:
    _grid_all(
        cfgrid.plot.grid_all_corr,
        cfgrid.constants.AUTO_BASE,
        # comparison='ms',
        comparison=None,
    )


def auto_corr_galhalo() -> None:
    _grid_galhalo(
        cfgrid.plot.grid_galhalo,
        cfgrid.constants.AUTO_BASE,
        galhalo.config.GAL_CENTRALS,
        comparison='xbin',
        autocorr=True,
    )


def auto_corr_ms_galhalo() -> None:
    _grid_galhalo(
        cfgrid.plot.grid_galhalo,
        cfgrid.constants.AUTO_BASE,
        galhalo.config.GAL_CENTRALS,
        comparison='ms',
        autocorr=True,
    )


def cross_corr_all_satellites_galhalo() -> None:
    _grid_galhalo(
        cfgrid.plot.grid_galhalo,
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        galhalo.config.GAL_CENTRALS,
        comparison='xbin',
        autocorr=False,
    )


def cross_corr_all_satellites_ms_galhalo() -> None:
    _grid_galhalo(
        cfgrid.plot.grid_galhalo,
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        galhalo.config.GAL_CENTRALS,
        comparison='ms',
        autocorr=False,
    )


def auto_corr_ms_galhalo_multi() -> None:
    _grid_galhalo_multi(
        cfgrid.constants.AUTO_BASE,
        galhalo.config.GAL_CENTRALS,
        comparison=None,
        autocorr=True,
    )


def cross_corr_all_satellites_ms_galhalo_multi() -> None:
    _grid_galhalo_multi(
        cfgrid.constants.CROSS_ALL_SAT_BASE,
        galhalo.config.GAL_CENTRALS,
        # comparison='ms',
        comparison=None,
        autocorr=False,
    )


def compare_auto_corr_obs_and_empire() -> None:
    empire_config = configurations.EmpireConfigVolume1
    conds = [
        *empire_config.centrals,
        conditionals.AllGalaxies,
        *empire_config.satellites,
    ]
    _compare_obs_and_empire(
        base=cfgrid.constants.AUTO_BASE,
        conds=conds,
        comparison='xbin',
        autocorr=True,
    )


def compare_auto_corr_obs_and_empire_ms() -> None:
    empire_config = configurations.EmpireConfigVolume1
    conds = [
        *empire_config.centrals,
        conditionals.AllGalaxies,
        *empire_config.satellites,
    ]
    _compare_obs_and_empire(
        base=cfgrid.constants.AUTO_BASE,
        conds=conds,
        comparison='ms',
        autocorr=True,
    )


def compare_cross_all_satellites_obs_and_empire() -> None:
    empire_config = configurations.EmpireConfigVolume1
    _compare_obs_and_empire(
        base=cfgrid.constants.CROSS_ALL_SAT_BASE,
        conds=empire_config.centrals,
        comparison='xbin',
        autocorr=False,
    )


def compare_cross_all_satellites_obs_and_empire_ms() -> None:
    empire_config = configurations.EmpireConfigVolume1
    _compare_obs_and_empire(
        base=cfgrid.constants.CROSS_ALL_SAT_BASE,
        conds=empire_config.centrals,
        comparison='ms',
        autocorr=False,
    )


def _bias_multi(
    func: Callable,
    base: str,
    conds: Sequence[type[conditionals.Conditional]],
    cond_labels: Optional[Sequence[str]] = None,
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
) -> None:
    configuration = cfgrid.config.CONFIG
    xbins = cfgrid.config.XBINS

    for ybin_scheme in configuration.ybin_schemes:
        y_property = ybin_scheme.property_
        num_ybins = ybin_scheme.num_bins
        path = [
            cfgrid.pathing.get_cfgrid_path(configuration.dirname, cond_type.label)
            for cond_type in conds
        ]
        results_filename = cfgrid.names.get_tpcf_filename(
            base, y_property.file_label, num_ybins=num_ybins
        )

        if comparison is not None:
            valid_comparison = cfgrid.utils.validate_comparison(comparison)
            comp_filename = cfgrid.names.get_tpcf_comp_filename(base, valid_comparison)
        else:
            comp_filename = None

        if comparison != 'ms' or (
            comparison == 'ms' and y_property == properties.DeltaMS
        ):
            func(
                path=path,
                results_filename=results_filename,
                prop=properties.DeltaMS,
                num_xbins=len(xbins) - 1,
                num_ybins=num_ybins,
                xbins=xbins,
                col_label=cfgrid.config.X_PROPERTY.full_label,
                cond_labels=cond_labels,
                comp_filename=comp_filename,
            )


def auto_corr_bias() -> None:
    _grid(
        cfgrid.plot.grid_bias,
        cfgrid.constants.AUTO_BASE,
        cfgrid.config.AUTO_CONDITIONALS,
        comparison='xbin',
        autocorr=True,
    )


def auto_corr_bias_multi() -> None:
    conds: list[type[conditionals.Conditional]] = [
        conditionals.AllGalaxies,
        conditionals.Centrals,
        conditionals.TempelCentrals,
        conditionals.FRodriguezCentrals,
    ]
    cond_labels = [
        'All Galaxies',
        'Yang et al. (2012)',
        'Tempel et al. (2017)',
        r'Rodriguez \&' + '\n' + r'Merch$\acute{\rm a}$n (2020)',
    ]
    _bias_multi(
        cfgrid.plot.grid_bias_multi,
        cfgrid.constants.AUTO_BASE,
        conds,
        cond_labels,
        comparison='xbin',
    )


def _bias_compare(
    func: Callable,
    base: str,
    conds: Sequence[type[conditionals.Conditional]],
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
    cond_labels: Optional[Sequence[str]] = None,
) -> None:
    configuration = cfgrid.config.CONFIG
    xbins = cfgrid.config.XBINS

    for ybin_scheme in configuration.ybin_schemes:
        y_property = ybin_scheme.property_
        num_ybins = ybin_scheme.num_bins
        path = [
            cfgrid.pathing.get_cfgrid_path(configuration.dirname, cond_type.label)
            for cond_type in conds
        ]
        results_filename = cfgrid.names.get_tpcf_filename(
            base, y_property.file_label, num_ybins=num_ybins
        )

        if comparison is not None:
            valid_comparison = cfgrid.utils.validate_comparison(comparison)
            comp_filename = cfgrid.names.get_tpcf_comp_filename(base, valid_comparison)
        else:
            comp_filename = None

        if comparison != 'ms' or (
            comparison == 'ms' and y_property == properties.DeltaMS
        ):
            func(
                path=path,
                results_filename=results_filename,
                num_xbins=len(xbins) - 1,
                num_ybins=num_ybins,
                xbins=xbins,
                col_label=cfgrid.config.X_PROPERTY.full_label,
                cond_labels=cond_labels,
                comp_filename=comp_filename,
                bias_sq=True,
            )


def auto_corr_bias_compare() -> None:
    conds: list[type[conditionals.Conditional]] = [
        conditionals.Centrals,
        conditionals.TempelCentrals,
        conditionals.FRodriguezCentrals,
        # conditionals.AllGalaxies,
    ]
    cond_labels = [
        'Yang Centrals',
        'Tempel Centrals',
        r'Rodriguez \& Merch$\acute{\rm a}$n Centrals',
        # 'All Galaxies',
    ]
    for cond, label in zip(conds, cond_labels):
        _bias_compare(
            cfgrid.plot.grid_bias_compare,
            cfgrid.constants.AUTO_BASE,
            [cond, conditionals.AllGalaxies],
            cond_labels=[label, 'All Galaxies'],
            comparison='xbin',
        )
