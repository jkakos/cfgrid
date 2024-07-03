import copy
from typing import Mapping, Optional, Protocol, Sequence

import config
from src import cfgrid, datalib, galhalo
from src.protocols import binning, conditionals, properties
from src.tpcf import tpcf as cf
from src.utils import output, split


class YbinScheme(Protocol):
    num_bins: int
    bin_strategy: type[binning.BinningStrategy]
    property_: type[properties.Property]


def get_comparisons(
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
) -> tuple[bool, bool]:
    xbin = False
    ms = False

    if comparison == 'xbin':
        xbin = True
    elif comparison == 'ms':
        ms = True

    if xbin and ms:
        raise ValueError("'xbin' and 'ms' comparisons cannot both be given.")

    return xbin, ms


def get_auto_runner(dataset: datalib.DataSet) -> cf.TpcfRunner:
    """
    Set up an auto correlation 2pcf runner based on the type of data
    set being used.

    """
    cf_runner: cf.TpcfRunner

    if isinstance(dataset, datalib.ObsLightConeDataSet):
        obs_lc_runner = dataset.cf_runner(dataset.data, config.tpcf_settings)
        obs_lc_runner.store_random(process=True, sample_size=10**6)
        cf_runner = obs_lc_runner
    elif isinstance(dataset, datalib.SimSnapshotDataSet):
        sim_snapshot_runner = dataset.cf_runner(dataset.data, config.tpcf_settings)
        cf_runner = sim_snapshot_runner

    return cf_runner


def get_cross_runner(dataset: datalib.DataSet) -> cf.CrossTpcfRunner:
    """
    Set up a cross correlation 2pcf runner based on the type of data
    set being used.

    """
    cf_cross_runner: cf.CrossTpcfRunner

    if isinstance(dataset, datalib.ObsLightConeDataSet):
        obs_lc_cross_runner = dataset.cf_cross_runner(
            dataset.data, config.tpcf_settings
        )
        obs_lc_cross_runner.store_random(process=True, sample_size=10**5)
        cf_cross_runner = obs_lc_cross_runner
    elif isinstance(dataset, datalib.SimSnapshotDataSet):
        sim_snapshot_cross_runner = dataset.cf_cross_runner(
            dataset.data, config.tpcf_settings
        )
        cf_cross_runner = sim_snapshot_cross_runner

    return cf_cross_runner


def _auto_corr(
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
) -> None:
    """
    Calculate the auto correlations in a grid.

    """
    xbin, ms = get_comparisons(comparison)
    configuration = cfgrid.config.CONFIG
    xbins = cfgrid.config.XBINS
    data = configuration.load()

    for cond_type in cfgrid.config.AUTO_CONDITIONALS:
        cond_obj = cond_type(data)
        filtered_data = data[cond_obj.value]

        cf_runner = get_auto_runner(configuration.dataset)
        cf_runner.data = filtered_data
        results_path = cfgrid.pathing.get_cfgrid_tpcf_path(
            configuration.dirname, cond_obj.label
        )

        for ybin_scheme in configuration.ybin_schemes:
            x_property = cfgrid.config.X_PROPERTY(filtered_data)
            cf_runner_copy = copy.deepcopy(cf_runner)

            if xbin:
                num_ybins = 1
                ybins: Mapping[int, Sequence[float]] = {0: []}

                # Use only good values of properties
                [x], all_good = properties.standardize(x_property)
                prop = x
            else:
                if ms:
                    y_property: properties.Property = properties.DeltaMS(filtered_data)
                    num_ybins = 1
                    ybins = {0: []}
                else:
                    ybin_strategy = ybin_scheme.bin_strategy
                    num_ybins = ybin_scheme.num_bins
                    y_property = ybin_scheme.property_(filtered_data)

                # Use only good values of properties
                [x, y], all_good = properties.standardize(x_property, y_property)
                prop = y

                if ms:
                    # Use only main sequence galaxies
                    main_sequence_gals = (
                        y.value > binning.DeltaMSBins.bottom_ms_upper_lim
                    ) & (y.value < binning.DeltaMSBins.upper_ms_upper_lim)
                    x.value = x.value[main_sequence_gals]
                    y.value = y.value[main_sequence_gals]
                    all_good = all_good & main_sequence_gals
                else:
                    # Get ybins
                    ybins = split.get_ybins(
                        x=x.value,
                        y=y.value,
                        xbins=xbins[1:-1],
                        ybin_strategy=ybin_strategy,
                        num_ybins=num_ybins,
                    )

            cf_runner_copy.data = cf_runner.data[all_good]

            # Set and print current settings
            output_settings = dict(
                Configuration=configuration.__class__.__name__,
                num_xbins=len(xbins) - 1,
                num_ybins=num_ybins,
                Conditional=cond_obj,
                Property=prop,
            )

            # Get path and file names
            if comparison is not None:
                tpcf_filename = cfgrid.names.get_tpcf_comp_filename(
                    cfgrid.constants.AUTO_BASE, comparison
                )
                cts_filename = cfgrid.names.get_tpcf_comp_filename(
                    cfgrid.constants.COUNTS_BASE, comparison
                )
                output_settings['Comparison'] = comparison
            else:
                tpcf_filename = cfgrid.names.get_tpcf_filename(
                    cfgrid.constants.AUTO_BASE,
                    y_property.file_label,
                    num_ybins=num_ybins,
                )
                cts_filename = cfgrid.names.get_tpcf_filename(
                    cfgrid.constants.COUNTS_BASE,
                    y_property.file_label,
                    num_ybins=num_ybins,
                )
                if isinstance(ybin_strategy, binning.BinningStrategy):
                    output_settings['ybin_strategy'] = ybin_strategy.__name__

            output.print_settings(header='SETTINGS', length=35, **output_settings)

            # Run calculation
            cfgrid.calc.auto_corr(
                x_property=x,
                y_property=prop,
                cf_runner=cf_runner_copy,
                xbins=xbins,
                ybins=ybins,
                path_tpcf=results_path.joinpath(tpcf_filename),
                path_counts=results_path.joinpath(cts_filename),
            )

            if comparison is not None:
                break


def auto_corr() -> None:
    """
    Calculate the auto correlations in a grid.

    """
    _auto_corr()


def auto_corr_xbin() -> None:
    """
    Calculate the auto correlations of each whole xbin in the grid.

    """
    _auto_corr(comparison='xbin')


def auto_corr_ms() -> None:
    """
    Calculate the auto correlations of main sequence galaxies in each
    xbin in the grid.

    """
    _auto_corr(comparison='ms')


def _cross_corr_all_satellites(
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
) -> None:
    """
    Calculate the cross correlations of centrals with all satellites in
    the same xbin in a grid.

    """
    xbin, ms = get_comparisons(comparison)
    configuration = cfgrid.config.CONFIG
    xbins = cfgrid.config.XBINS
    data = configuration.load()

    centrals: list[conditionals.Conditional] = [
        cond_type(data) for cond_type in configuration.centrals
    ]
    satellites: list[conditionals.Conditional] = [
        cond_type(data) for cond_type in configuration.satellites
    ]

    for cent, sat in zip(centrals, satellites):
        cf_cross_runner = get_cross_runner(configuration.dataset)
        cf_cross_runner.set_data(cent, sat)
        data1, data2 = cf_cross_runner.get_data()
        results_path = cfgrid.pathing.get_cfgrid_tpcf_path(
            configuration.dirname, cent.label
        )

        for ybin_scheme in configuration.ybin_schemes:
            x_property1 = cfgrid.config.X_PROPERTY(data1)
            x_property2 = cfgrid.config.X_PROPERTY(data2)
            cf_cross_runner_copy = copy.deepcopy(cf_cross_runner)

            if xbin:
                num_ybins = 1
                ybins: Mapping[int, Sequence[float]] = {0: []}

                # Use only good values of properties
                [x1], all_good1 = properties.standardize(x_property1)
                [x2], all_good2 = properties.standardize(x_property2)
                y1, y2 = x1, x2
            else:
                if ms:
                    y_property1: properties.Property = properties.DeltaMS(data1)
                    y_property2: properties.Property = properties.DeltaMS(data2)
                    num_ybins = 1
                    ybins = {0: []}
                else:
                    y_property1 = ybin_scheme.property_(data1)
                    y_property2 = ybin_scheme.property_(data2)
                    num_ybins = ybin_scheme.num_bins
                    ybin_strategy = ybin_scheme.bin_strategy

                # Use only good values of properties
                [x1, y1], all_good1 = properties.standardize(x_property1, y_property1)
                [x2, y2], all_good2 = properties.standardize(x_property2, y_property2)

                if ms:
                    # Use only main sequence centrals
                    main_sequence_gals = (
                        y1.value > binning.DeltaMSBins.bottom_ms_upper_lim
                    ) & (y1.value < binning.DeltaMSBins.upper_ms_upper_lim)
                    x1.value = x1.value[main_sequence_gals]
                    y1.value = y1.value[main_sequence_gals]
                    all_good1 = all_good1 & main_sequence_gals
                else:
                    # Get ybins
                    ybins = split.get_ybins(
                        x=x1.value,
                        y=y1.value,
                        xbins=xbins[1:-1],
                        ybin_strategy=ybin_strategy,
                        num_ybins=num_ybins,
                    )

            cf_cross_runner_copy.data1 = cf_cross_runner.data1[all_good1]
            cf_cross_runner_copy.data2 = cf_cross_runner.data2[all_good2]

            # Set and print current settings
            output_settings = dict(
                Configuration=configuration.__class__.__name__,
                num_xbins=len(xbins) - 1,
                num_ybins=num_ybins,
                Centrals=cent,
                Satellites=sat,
                Property=y1,
            )

            # Get path and file names
            if comparison is not None:
                tpcf_filename = cfgrid.names.get_tpcf_comp_filename(
                    cfgrid.constants.CROSS_ALL_SAT_BASE, comparison
                )
                output_settings['Comparison'] = comparison
            else:
                tpcf_filename = cfgrid.names.get_tpcf_filename(
                    cfgrid.constants.CROSS_ALL_SAT_BASE,
                    y1.file_label,
                    num_ybins=num_ybins,
                )
                if isinstance(ybin_strategy, binning.BinningStrategy):
                    output_settings['ybin_strategy'] = ybin_strategy.__name__

            output.print_settings(header='SETTINGS', length=35, **output_settings)

            # Run calculation
            cfgrid.calc.cross_corr_all_satellites(
                x_property=(x1, x2),
                y_property=(y1, y2),
                cf_runner=cf_cross_runner_copy,
                xbins=xbins,
                ybins=ybins,
                path_tpcf=results_path.joinpath(tpcf_filename),
                below=False,
            )

            if comparison is not None:
                break


def cross_corr_all_satellites() -> None:
    """
    Calculate the cross correlations of centrals with all satellites in
    the same xbin in a grid.

    """
    _cross_corr_all_satellites()


def cross_corr_all_satellites_xbin() -> None:
    """
    Calculate the cross correlations of all centrals in each xbin with
    all satellites in the same xbin in the grid.

    """
    _cross_corr_all_satellites(comparison='xbin')


def cross_corr_all_satellites_ms() -> None:
    """
    Calculate the cross correlations of main sequence centrals in each
    xbin with all satellites in the same xbin in the grid.

    """
    _cross_corr_all_satellites(comparison='ms')


def _auto_corr_galhalo(
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
) -> None:
    """
    Calculate the auto correlations in a grid of halos using
    galaxy-sampled properties.

    """
    xbin, ms = get_comparisons(comparison)
    halo_config = galhalo.config.CONFIG
    mass_bins = halo_config.mass_bins
    cond_type = halo_config.centrals[0]

    # Get results path
    results_path = galhalo.pathing.get_galhalo_tpcf_path(halo_config)

    for gal, halo in list(galhalo.config.PAIRINGS.keys()):
        halo_config = galhalo.config.CONFIG_TYPE()
        halo_config.set_galhalo_props(gal.file_label, halo.file_label)

        if ms and gal != properties.SSFR:
            continue

        if xbin:
            data = halo_config.load(galhalo=False)
        else:
            data = halo_config.load()

        cond_obj = cond_type(data)
        filtered_data = data[cond_obj.value]

        cf_runner = get_auto_runner(halo_config.dataset)
        cf_runner.data = filtered_data
        x_property = galhalo.config.MASS(filtered_data)

        if xbin:
            num_ybins = 1
            ybins: Mapping[int, Sequence[float]] = {0: []}

            # Use only good values of properties
            [x], all_good = properties.standardize(x_property)
            prop = x
        else:
            if ms:
                y_property: properties.Property = properties.DeltaMS(filtered_data)
                num_ybins = 1
                ybins = {0: []}
            else:
                ybin_scheme = galhalo.config.YBIN_SCHEMES[(gal, halo)]
                y_property = ybin_scheme.property_(filtered_data)
                ybin_scheme = galhalo.config.YBIN_SCHEMES[(gal, halo)]
                ybin_strategy = ybin_scheme.bin_strategy
                num_ybins = ybin_scheme.num_bins

            # Use only good values of properties
            [x, y], all_good = properties.standardize(x_property, y_property)
            prop = y

            if ms:
                # Use only main sequence galaxies
                main_sequence_gals = (
                    y.value > binning.DeltaMSBins.bottom_ms_upper_lim
                ) & (y.value < binning.DeltaMSBins.upper_ms_upper_lim)
                x.value = x.value[main_sequence_gals]
                y.value = y.value[main_sequence_gals]
                all_good = all_good & main_sequence_gals
            else:
                # Get ybins
                ybins = split.get_ybins(
                    x=x.value,
                    y=y.value,
                    xbins=mass_bins[1:-1],
                    ybin_strategy=ybin_strategy,
                    num_ybins=num_ybins,
                )

        cf_runner.data = cf_runner.data[all_good]

        # Set and print current settings
        output_settings = dict(
            Configuration=halo_config.__class__.__name__,
            num_xbins=len(mass_bins) - 1,
            num_ybins=num_ybins,
            Conditional=cond_obj,
            Property=prop,
        )

        # Get path and file names
        if comparison is not None:
            tpcf_args = [
                galhalo.constants.GALHALO_BASE,
                comparison,
                cfgrid.constants.AUTO_BASE,
            ]
            cts_args = [
                galhalo.constants.GALHALO_BASE,
                comparison,
                cfgrid.constants.COUNTS_BASE,
            ]
            if ms:
                tpcf_args.append(halo.file_label)
                cts_args.append(halo.file_label)

            tpcf_filename = cfgrid.names.get_tpcf_comp_filename(
                *tpcf_args  # type: ignore
            )
            cts_filename = cfgrid.names.get_tpcf_comp_filename(
                *cts_args  # type: ignore
            )
            output_settings['Comparison'] = comparison
        else:
            tpcf_filename = cfgrid.names.get_tpcf_filename(
                galhalo.constants.GALHALO_BASE,
                cfgrid.constants.AUTO_BASE,
                y_property.file_label,
                halo.file_label,
                num_ybins=num_ybins,
            )
            cts_filename = cfgrid.names.get_tpcf_filename(
                galhalo.constants.GALHALO_BASE,
                cfgrid.constants.COUNTS_BASE,
                y_property.file_label,
                halo.file_label,
                num_ybins=num_ybins,
            )
            if isinstance(ybin_strategy, binning.BinningStrategy):
                output_settings['ybin_strategy'] = ybin_strategy.__name__

        output.print_settings(header='SETTINGS', length=35, **output_settings)

        # Run calculation
        cfgrid.calc.auto_corr(
            x_property=x,
            y_property=prop,
            cf_runner=cf_runner,
            xbins=mass_bins,
            ybins=ybins,
            path_tpcf=results_path.joinpath(tpcf_filename),
            path_counts=results_path.joinpath(cts_filename),
        )

        if xbin:
            break


def auto_corr_galhalo() -> None:
    """
    Calculate the auto correlations in a grid using galaxy-sampled
    properties.

    """
    _auto_corr_galhalo()


def auto_corr_xbin_galhalo() -> None:
    """
    Calculate the auto correlations of each whole xbin in the grid of
    halos.

    """
    _auto_corr_galhalo(comparison='xbin')


def auto_corr_ms_galhalo() -> None:
    """
    Calculate the auto correlations of main sequence galaxies in each
    xbin in the grid of halos using galaxy-sampled properties.

    """
    _auto_corr_galhalo(comparison='ms')


def _cross_corr_all_satellites_galhalo(
    comparison: Optional[cfgrid.constants.COMPARISON_OPTIONS] = None,
) -> None:
    """
    Calculate the cross correlations of centrals with all satellites in
    the same xbin in a grid of halos using galaxy-sampled properties.

    """
    xbin, ms = get_comparisons(comparison)
    halo_config = galhalo.config.CONFIG
    mass_bins = halo_config.mass_bins

    # Get results path
    results_path = galhalo.pathing.get_galhalo_tpcf_path(halo_config)

    for gal, halo in list(galhalo.config.PAIRINGS.keys()):
        halo_config = galhalo.config.CONFIG_TYPE()
        halo_config.set_galhalo_props(gal.file_label, halo.file_label)

        if ms and gal != properties.SSFR:
            continue

        if xbin:
            data = halo_config.load(galhalo=False)
        else:
            data = halo_config.load()

        centrals = halo_config.centrals[0](data)
        satellites = halo_config.satellites[0](data)
        cf_cross_runner = get_cross_runner(halo_config.dataset)
        cf_cross_runner.set_data(centrals, satellites)
        data1, data2 = cf_cross_runner.get_data()
        x_property1 = galhalo.config.MASS(data1)
        x_property2 = galhalo.config.MASS(data2)

        if xbin:
            num_ybins = 1
            ybins: Mapping[int, Sequence[float]] = {0: []}

            # Use only good values of properties
            [x1], all_good1 = properties.standardize(x_property1)
            [x2], all_good2 = properties.standardize(x_property2)
            cf_cross_runner.data1 = cf_cross_runner.data1[all_good1]
            cf_cross_runner.data2 = cf_cross_runner.data2[all_good2]
            y1, y2 = x1, x2
        else:
            if ms:
                y_property1: properties.Property = properties.DeltaMS(data1)
                y_property2: properties.Property = properties.DeltaMS(data2)
                num_ybins = 1
                ybins = {0: []}
            else:
                ybin_scheme = galhalo.config.YBIN_SCHEMES[(gal, halo)]
                ybin_strategy = ybin_scheme.bin_strategy
                num_ybins = ybin_scheme.num_bins
                y_property1 = ybin_scheme.property_(data1)
                y_property2 = ybin_scheme.property_(data2)

            # Use only good values of properties
            [x1, y1], all_good1 = properties.standardize(x_property1, y_property1)
            [x2, y2], all_good2 = properties.standardize(x_property2, y_property2)

            if ms:
                # Use only main sequence centrals
                main_sequence_gals = (
                    y1.value > binning.DeltaMSBins.bottom_ms_upper_lim
                ) & (y1.value < binning.DeltaMSBins.upper_ms_upper_lim)
                x1.value = x1.value[main_sequence_gals]
                y1.value = y1.value[main_sequence_gals]
                all_good1 = all_good1 & main_sequence_gals
            else:
                # Get ybins
                ybins = split.get_ybins(
                    x=x1.value,
                    y=y1.value,
                    xbins=mass_bins[1:-1],
                    ybin_strategy=ybin_strategy,
                    num_ybins=num_ybins,
                )

            cf_cross_runner.data1 = cf_cross_runner.data1[all_good1]
            cf_cross_runner.data2 = cf_cross_runner.data2[all_good2]

        # Set and print current settings
        output_settings = dict(
            Configuration=halo_config.__class__.__name__,
            num_xbins=len(mass_bins) - 1,
            num_ybins=num_ybins,
            Centrals=centrals,
            Satellites=satellites,
            Property=y1,
        )
        if comparison is not None:
            output_settings['Comparison'] = comparison

        # Get path and file names
        if xbin:
            assert comparison is not None
            tpcf_filename = cfgrid.names.get_tpcf_comp_filename(
                galhalo.constants.GALHALO_BASE,
                comparison,
                cfgrid.constants.CROSS_ALL_SAT_BASE,
            )
        elif ms:
            assert comparison is not None
            tpcf_filename = cfgrid.names.get_tpcf_comp_filename(
                galhalo.constants.GALHALO_BASE,
                comparison,
                cfgrid.constants.CROSS_ALL_SAT_BASE,
                halo.file_label,
            )
        else:
            tpcf_filename = cfgrid.names.get_tpcf_filename(
                galhalo.constants.GALHALO_BASE,
                cfgrid.constants.CROSS_ALL_SAT_BASE,
                y1.file_label,
                halo.file_label,
                num_ybins=num_ybins,
            )
            if isinstance(ybin_strategy, binning.BinningStrategy):
                output_settings['ybin_strategy'] = ybin_strategy.__name__

        output.print_settings(header='SETTINGS', length=35, **output_settings)

        # Run calculation
        cfgrid.calc.cross_corr_all_satellites(
            x_property=(x1, x2),
            y_property=(y1, y2),
            cf_runner=cf_cross_runner,
            xbins=mass_bins,
            ybins=ybins,
            path_tpcf=results_path.joinpath(tpcf_filename),
        )

        if xbin:
            break


def cross_corr_all_satellites_galhalo() -> None:
    """
    Calculate the cross correlations of centrals with all satellites in
    the same xbin in a grid of halos using galaxy-sampled properties.

    """
    _cross_corr_all_satellites_galhalo()


def cross_corr_all_satellites_xbin_galhalo() -> None:
    """
    Calculate the cross correlations of all centrals in each xbin with
    all satellites in the same xbin in the grid.

    """
    _cross_corr_all_satellites_galhalo(comparison='xbin')


def cross_corr_all_satellites_ms_galhalo() -> None:
    """
    Calculate the cross correlations of main sequence centrals in each
    xbin with all satellites in the same xbin in the grid.

    """
    _cross_corr_all_satellites_galhalo(comparison='ms')
