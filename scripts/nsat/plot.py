from src import nsatlib
from src.protocols import properties


MIN_COUNTS = 50


def nsat() -> None:
    nsatlib.plot.plot_nsat(min_cts=MIN_COUNTS)


def nsat_two_panel() -> None:
    nsatlib.plot.plot_nsat_two_panel(min_cts=MIN_COUNTS)


def nsat_group_cats() -> None:
    nsatlib.plot.plot_nsat_group_cats(min_cts=MIN_COUNTS)


def nsat_comparison_mpa_bp() -> None:
    halo_props: list[type[properties.Property]] = [
        properties.Vpeak,
        properties.Concentration,
        properties.AccretionRate,
    ]
    nsatlib.plot.plot_nsat_comparison(halo_props, min_cts=MIN_COUNTS)


def nsat_comparison_mpa_empire() -> None:
    nsatlib.plot.plot_nsat_empire(mpa_min_cts=MIN_COUNTS, empire_min_cts=25)
