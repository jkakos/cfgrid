from typing import Sequence

from src import denslib
import src.denslib.environment as denv


def calc(
    configuration: denslib.config.DensityConfig,
    env_measure: Sequence[denv.EnvironmentMeasure],
    params: dict[str, Sequence[float]],
    ddp_mass_bins: tuple[float, float],
) -> None:
    """
    Calculate different densities as determined by 'env_measure'.

    """
    denshandler = denslib.densityhandler.DensityHandler(configuration)

    # Instantiate the density calculator
    calculator = denslib.calculator.DensityCalculator(
        configuration.dataset.coord_strat(configuration.dataset.data, ddp_mass_bins)
    )

    radius = params.get('radius')
    if radius is not None:
        for r in radius:
            calculator.run(denshandler, env_measure, radius=r)

    rp_max = params.get('rp_max')
    if rp_max is not None:
        for rp in rp_max:
            for rpi in params['rpi_max']:
                calculator.run(
                    denshandler,
                    env_measure,
                    rp_max=rp,
                    rpi_max=rpi,
                )


def main() -> None:
    params: dict[str, Sequence[float]] = dict(
        radius=denslib.config.RADIUS,
        rp_max=denslib.config.RP_MAX,
        rpi_max=denslib.config.RPI_MAX,
    )
    configuration = denslib.config.CONFIG
    data = configuration.load()

    env_measures: list[denv.EnvironmentMeasure] = [
        denv.TotalGalaxies(),
        denv.TotalMass(data),
        denv.TotalSFR(data),
        denv.TotalSSFR(data),
    ]

    calc(
        configuration=configuration,
        env_measure=env_measures,
        params=params,
        ddp_mass_bins=denslib.config.DDP_MASS_BINS,
    )
