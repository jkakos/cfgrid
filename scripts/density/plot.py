from src import denslib
import src.denslib.environment as denv


def main() -> None:
    configuration = denslib.config.CONFIG
    denshandler = denslib.densityhandler.DensityHandler(configuration)

    # Use denshandler to load multiple types of densities
    data, [num_dens, mass_dens] = denshandler.load(
        [
            denv.Density(denv.DensityType.NUMBER, radius=4),
            denv.Density(denv.DensityType.STELLARMASS, rp_max=1, rpi_max=8),
        ]
    )
    print(data)
    print(num_dens.value)
    print(mass_dens.value)
