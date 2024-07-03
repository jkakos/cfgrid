from typing import cast, get_args

from src import cfgrid


def validate_comparison(comparison: str) -> cfgrid.constants.COMPARISON_OPTIONS:
    comp_options = get_args(cfgrid.constants.COMPARISON_OPTIONS)
    if comparison not in comp_options:
        raise ValueError(
            f"'comparison' must be among {comp_options} ({comparison=} was given)."
        )
    return cast(cfgrid.constants.COMPARISON_OPTIONS, comparison)
