import scripts


# =========================================================
# COMPLETENESS
# =========================================================
# ------ calculate completeness
# scripts.completeness.calc.completeness_vol1()
# scripts.completeness.calc.completeness_vol3()

# ------ plot completeness results
# scripts.completeness.plot.vol1()
# scripts.completeness.plot.vol2()
# scripts.completeness.plot.vol3()
# scripts.completeness.plot.vol1_with_hist()
# scripts.completeness.plot.vol2_with_hist()
# scripts.completeness.plot.vol3_with_hist()
# scripts.completeness.plot.vol1_completeness_density()
# scripts.completeness.plot.print_volume1_stats()
# scripts.completeness.plot.print_volume2_stats()
# scripts.completeness.plot.print_volume3_stats()
# scripts.completeness.plot.sample_vol2_completeness_points()

# ------ plot GSMF of complete volumes
# scripts.completeness.gsmf.vol1()
# scripts.completeness.gsmf.vol2()
# scripts.completeness.gsmf.vol3()

# =========================================================
# CFGRID
# =========================================================
# ------ cfgrid calc
# scripts.cfgrid.calc.auto_corr()
# scripts.cfgrid.calc.auto_corr_xbin()
# scripts.cfgrid.calc.auto_corr_ms()
# scripts.cfgrid.calc.cross_corr_all_satellites()
# scripts.cfgrid.calc.cross_corr_all_satellites_xbin()
# scripts.cfgrid.calc.cross_corr_all_satellites_ms()

# ------ cfgrid plot
# scripts.cfgrid.plot.plane()
# scripts.cfgrid.plot.auto_corr()
# scripts.cfgrid.plot.auto_corr_ms()
# scripts.cfgrid.plot.cross_corr_all_satellites()
# scripts.cfgrid.plot.cross_corr_all_satellites_ms()
# scripts.cfgrid.plot.condensed_grid()
# scripts.cfgrid.plot.condensed_grid_ms()
# scripts.cfgrid.plot.condensed_grid_multi()
# scripts.cfgrid.plot.condensed_grid_multi_ms()
# scripts.cfgrid.plot.condensed_grid_multi_auto_ms()
# scripts.cfgrid.plot.all_corr_ms()
# scripts.cfgrid.plot.compare_auto_corr_obs_and_empire()
# scripts.cfgrid.plot.compare_auto_corr_obs_and_empire_ms()
# scripts.cfgrid.plot.compare_cross_all_satellites_obs_and_empire()
# scripts.cfgrid.plot.compare_cross_all_satellites_obs_and_empire_ms()
# scripts.cfgrid.plot.auto_corr_bias()
# scripts.cfgrid.plot.auto_corr_bias_compare()
# scripts.cfgrid.plot.auto_corr_bias_multi()

# =========================================================
# SIMULATIONS
# =========================================================
# ------ put together mock catalogs
# scripts.simulation.assemble_bp.main()
# scripts.simulation.assemble_empire.main()

# =========================================================
# GALHALO
# =========================================================
# ------ satellite fraction
# scripts.galhalo.sat_fraction.main()
# scripts.galhalo.sat_fraction.compare_obs_and_sim_sat_frac()

# ------ sample observed values to make galhalo models
# scripts.galhalo.sample.main()

# ------ cfgrid calc
# scripts.cfgrid.calc.auto_corr_galhalo()
# scripts.cfgrid.calc.auto_corr_xbin_galhalo()
# scripts.cfgrid.calc.auto_corr_ms_galhalo()
# scripts.cfgrid.calc.cross_corr_all_satellites_galhalo()
# scripts.cfgrid.calc.cross_corr_all_satellites_xbin_galhalo()
# scripts.cfgrid.calc.cross_corr_all_satellites_ms_galhalo()

# ------ cfgrid plot
# scripts.cfgrid.plot.plane_galhalo()
# scripts.cfgrid.plot.auto_corr_galhalo()
# scripts.cfgrid.plot.auto_corr_ms_galhalo()
# scripts.cfgrid.plot.cross_corr_all_satellites_galhalo()
# scripts.cfgrid.plot.cross_corr_all_satellites_ms_galhalo()
# scripts.cfgrid.plot.auto_corr_ms_galhalo_multi()
# scripts.cfgrid.plot.cross_corr_all_satellites_ms_galhalo_multi()

# ------ galhalo stellar-to-halo mass relation
# scripts.galhalo.shmr.shmr()
# scripts.galhalo.shmr.inverted_shmr()
# scripts.galhalo.shmr.both()
# scripts.galhalo.shmr.empire()
# scripts.galhalo.shmr.formation_redshift()

# ------ scatter plots of galhalo results
# scripts.galhalo.plot.scatter_ssfr()
# scripts.galhalo.plot.plot_all()
# scripts.galhalo.plot.plot_quarters()
# scripts.galhalo.ssfr_mstar.plot_concentration()
# scripts.galhalo.ssfr_mstar.plot_delta_v()
# scripts.galhalo.ssfr_mstar.plot_delta_v_over_v()
# scripts.galhalo.ssfr_mstar.plot_halfmass_scale()
# scripts.galhalo.ssfr_mstar.plot_scale_of_last_mm()
# scripts.galhalo.ssfr_mstar.plot_specific_accretion_rate()
# scripts.galhalo.ssfr_mstar.plot_spin()
# scripts.galhalo.ssfr_mstar.plot_tidal_force()
# scripts.galhalo.ssfr_mstar.plot_t_over_u()

# =========================================================
# NSAT
# =========================================================
# ------ calculate Nsat
# scripts.nsat.calc.main()

# ------ plot Nsat results
# scripts.nsat.plot.nsat()
# scripts.nsat.plot.nsat_two_panel()
# scripts.nsat.plot.nsat_group_cats()
# scripts.nsat.plot.nsat_comparison_mpa_bp()
# scripts.nsat.plot.nsat_comparison_mpa_empire()

# =========================================================
# OTHER
# =========================================================
# ------ calculate main sequence values for fitting
# scripts.ms_fit.main()

# ------ calculate stats to compare different MPA group catalogs
# scripts.group_cat_comparison.main()

# ------ calculate density
# scripts.density.calc.main()
