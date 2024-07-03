# cfgrid

This project is broken into two main directories: `src/` and `scripts/`. `scripts/` contains various files and functions that are called in `run.py` to do different calculations or make figures. Some functions are identical to others but have slightly different inputs; it was set up this way to make it easy to run different calculations or figures without having to constantly change function arguments. `src/` contains the more generalized code that is used by `scripts/`.

## `src/`
This contains the bulk of the code broken into different packages. The `cfgrid/`, `galhalo/`, `nsatlib/`, and `denslib/` packages all have similar structures containing files such as: `config.py`, `constants.py`, `names.py`, `pathing.py`, `utils.py`, `calc.py`, and `plot.py`. They will be described in the next sub-section for `cfgrid/`, but the same general ideas apply to the other packages.

### `cfgrid/`
This package controls the main correlation function grid style calculations and figures.

* `calc.py`: Contains the main functions for calculating auto- and cross-correlations within the grid scheme as well as saving the results.

* `config.py`: Controls various settings when the grid correlations functions are being calculated or plotted. The main thing is the configuration being used, which contains information about the galaxy properties being looped over, the number of bins and binning scheme for the galaxy properties, and the different conditionals being looped over (e.g., all  galaxies or central galaxies only, which group catalog to use, etc.).

* `constants.py`: Contains constants that will be used in table headers when storing results as well strings that will be used as the bases for file and figure names to differentiate results with different configurations.

* `names.py`: This sets the naming conventions when saving or loading results.

* `pathing.py`: This sets the locations of where results are stored.

* `plot.py`: Contains the main functions for plotting different correlation function grid figures or other relevant figures.

* `utils.py`: Contains a function that validates one of the `cfgrid` arguments.

### `complib/`
This package contains for calculating and plotting different completeness conditions in observed data.

### `cosmo/`
This package contains different calculations and settings for the cosmology.

* `constants.py`: Controls different cosmological parameters that will be used in calculations.

* `coords.py`: Contains functions for converting between spherical (ra, dec, z) and Cartesian (x, y, z) coordinates

* `distances.py`: Contains various cosmological distance measures (e.g., comoving distance).

* `gsmf.py`: Contains a fit of the galaxy stellar mass function from Dragomir et al. 2018.

* `quantities.py`: Contains a function to calculate the age of the universe at a given redshift, but could potentially have additional quantities in the future.

### `datalib/`
This package is where any data sets that will be used should be set up.

* `catalogs/`: This is where different data set classes are stored. They should follow the minimum structure defined in `datalib/datasets.py`. The main data set is `MPAJHU` in `mpajhu.py` which loads Data Release 7 of the Sloan Digital Sky Survey (SDSS) with stellar masses and star formation rates (SFRs) from the Max Planck Institute for Astrophysics and Johns Hopkins University (MPA-JHU) catalog.

* `constants.py`: This sets different catalogs that can be looked up directly using `datalib.<catalog>` as opposed to, e.g., `datalib.catalogs.mpajhu.MPAJHU`. Any new catalogs added should also be added to `CATALOGS` such that `CATALOGS[CatalogClassName] = catalog_filename`.

* `dataprocess.py`: This contains various data processing functions that can be applied when a data set is loaded. The main processes are applying given data cuts and cutting for completeness in observational data. There are two completeness functions included: one following 'Volume1' of Appendix A in Yang et al. (2012) and one following equation A8 in van den Bosch et al. (2008).

* `datasets.py`: This sets up different protocols for different types of data sets.

* `volumes.py`: This holds the mass and redshift limits used for different completeness conditions.

### `denslib/`
This package is used to calculate, save, and load density estimates.

* `calculator.py`: This sets up a density calculator class that runs different types of density calculations.

* `densityhandler.py`: This sets up a class used save and load density results for a given configuration. Loaded results are merged into a configurations data.

* `environment.py`: This defines different types of environmental quantities to measure when calculating densities (e.g., total number of galaxies or total stellar mass). This also sets up the `Density` class which is passed to a `DensityHandler` to load these different types of densities.

* `proj_neighbors.py`: These are functions used to calculated densities in projected cylindrical volumes defined by the distance perpendicular (`rp_max`) to and parallel (`rpi_max`) to the line of sight.

* `sphere.py`: These are functions used to calculate densities in spherical volumes defined by a `radius`.

### `figlib/`
This package contains some general functions for figures that will be made.

* `colors.py`: This contains some functions for selecting and adjusting colors.

* `config.py`: This defines some settings to use by default when making figures.

* `completeness.py`: This sets up a plotting function to plot volume-limited samples and a completeness limit used to determine the volumes.

* `grid.py`: This contains the main structure for setting up a grid plot.

* `legend.py`: This contains functions for creating different legends for the main grid figures.

* `mask.py`: This defines a masking procedure to remove points from a figure that have errors of a certain size.

* `save.py`: This contains a function for saving figures.

### `galhalo/`
This package is used to model galaxy properties for dark matter halos.

* `config.py`: This contains settings for the galaxy-halo modeling. The `PAIRINGS` dictionary denotes whether the correlation between a galaxy property and halo property should be reversed (i.e., if the halo property is larger, the galaxy property should be smaller).

### `nsatlib/`
This package is used to calculate the average number of satellites per group (Nsat) as a function of central galaxy specific star formation rate (sSFR).

* `config.py`: This contains the main settings for Nsat calculations. `CENTRAL_MASS_BIN_SIZE` sets the size of the step in central galaxy stellar mass as Nsat is calculated as a function of stellar mass. `WINDOW_SIZE` sets the size of the window to consider when calculating Nsat in a stellar mass bin. `GAL_PROPERTY` and `HALO_PROPERTY` are used to load a specific galhalo model.

### `protocols/`
This package stores different kinds of protocols that are used for structural typing throughout the project as well as classes of those types.

* `binning.py`: This contains various binning strategies used for binning the grid.

* `conditionals.py`: This contains different conditionals for selecting subsets of a data set - whether all galaxies, central galaxies, or satellite galaxies. There are different group catalogs usable for SDSS observations.

* `coords.py`: This contains different methods for accessing Cartesian coordinates from a data set. If the data set is a light cone, its (ra, dec, z) coordinates will be converted to (x, y, z) using the `cosmo` package.

* `properties.py`: This contains different classes for accessing galaxy and halo properties from a data set.

### `tests/`
This package contains various `pytest` tests.

### `tpcf/`
This package is used to calculate two-point auto- and cross-correlation functions for different kinds of data sets (observations vs simulations and light cones vs simulation snapshots). Simulation positions should be in comoving Mpc/h.

### `utils/`
This package contains a few useful functions.

* `output.py`: This contains functions for printing settings or results in an organized way.

* `pathing.py`: This sets up general pathing functions used throughout the project. `get_path` will retrieve a path and create the entire path to a directory if it does not exist. `get_results_path` will create and retrieve a directory `results/` within the provided directory.

* `split.py`: This contains functions for applying binning strategies and recovering the coordinates that point to cells in the grid to which data points will be assigned.


## `scripts/`
This contains different scripts that are run to do various calculations or create figures. Most packages tend to have a `calc.py` and `plot.py` for running calculation functions or plotting functions.

### `cfgrid/`
Calculate and plot results of auto- and cross-correlation functions as a function of different galaxy properties at a fixed stellar mass.

* `calc.py`: Functions with `_xbin` or `_ms` are used to calculate comparison lines (either the correlation function of the entire mass bin or central main sequence galaxies). Functions ending in `_galhalo` are used to run the same calculations as the observations but using galhalo models.

* `plot.py`: (See above.) Functions with `_multi` should only be used with observed correlation functions as they compare results from the Yang, Tempel, and Rodriguez & Merchan group catalogs. `all_corr_ms` plots the correlation functions of all, central, and satellite galaxies separately on the same figure, with the central and satellite correlation functions being scaled based on their relative counts (see equation 7 of Kakos et al. 2024).

### `completeness/`
Calculate and check the completeness of different volumes determined by a set of mass bins.

* `calc.py`: Use this to find complete volumes for a given set of mass bins. These can then be added `src.datalib.volumes` to create new load settings for new configurations.

* `gsmf.py`: Use this to calculate the galaxy stellar mass function (GSMF) of a set of volumes to check their completeness. This uses the best-fit parameters for low-redshift SDSS galaxies from Dragomir et al. (2018).

### `density/`
Calculate different types of densities using different kinds of volumes.

### `galhalo/`
Run the galhalo models and make various plots to visualize how different properties are related within them.

* `plot.py`: Use this to generate different colored scatter plots to see how various galaxy and halo properties are related in the galhalo models.

* `sample.py`: Use this to run the galhalo model procedure of sampling observed galaxy properties and assigning them to dark matter halos.

* `sat_fraction.py`: Calculate the satellite fraction of SDSS galaxies as a function of stellar mass. This is used to determine the stellar-to-halo mass relation to add stellar masses to dark matter halos.

* `shmr.py`: Use this to plot the stellar-to-halo mass relation (SHMR) for star-forming and quiescent galaxies as determined by a galhalo model and compare to estimates from the literature.

* `ssfr_mstar`: Use this to make plots of the sSFR - stellar mass plane colored by different halo properties.

### `nsat/`
Calculate and plot results of the average number of satellites per group as a function of central galaxy sSFR.

### `simulation/`
This package contains functions for creating mock catalogs that galhalo models will be run on.

* `assemble_bp.py`: Use this to run the two main functions for creating a mock.

* `assembly_empire.py`: Use this to piece together a full Empire mock catalog from the individual sub-files.

* `mass_assign.py`: Use this to assign stellar masses to different Bolshoi-Planck (BP) snapshots that correspond to a set of observed volumes. The snapshots are cut in stellar mass to the range that is valid given the used stellar-to-halo mass relation.

* `project_zspace.py`: Use this to take the BP snapshots, project them into redshift space, and then cut them in stellar mass and redshift to correspond to observed volumes.

 `group_cat_comparison.py`: This creates two figures and prints some statistics for comparing the results of different SDSS group catalogs.

`ms_fit.py`: Use this to find the median values of sSFR as a function of stellar mass for star-forming galaxies. These are used to create a smooth function that will define the center of the main sequence and all other delta MS bins.