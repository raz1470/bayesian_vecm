# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

First public release.

### Model

- `BayesianVECM`: a Bayesian vector error correction model in PyMC, sampled with
  nutpie. Posterior uncertainty on every parameter and every derived quantity.
- Deterministic terms: `"n"`, `"co"`, `"ci"`, `"lo"` and `"li"` (a constant or
  linear trend inside or outside the cointegrating relations).
- Priors set through a plain dict, including a regularised horseshoe on the
  short-run lags (`priors={"Gamma": {"dist": "Horseshoe"}}`).
- Exogenous drivers in the short run (`exog`) and inside the long-run relations
  (`exog_coint`, entering at t-1 like the lagged levels, as in Johansen's model
  and statsmodels), with `exog_lags` for drivers that act with a delay.
- `adstock=`: geometric carryover on chosen drivers.
- `saturation=`: logistic diminishing returns on chosen drivers, with
  `saturation_curve()`.
- `fading=`: lets a share of each listed driver's lift fade with an estimated
  half-life (`fade_share`, `fade_half_life`), instead of assuming every lift in
  the brand metrics lasts for good. `exog_irf()` and forecasts include the fade.
- `contemporaneous=` and `excluded=`: same-week links between endogenous series,
  identified by exclusion restrictions, with an instrument check at fit time and
  `same_week_links()`.

### Results

- `irf()`: generalised and Cholesky impulse responses (with `order=`).
- `exog_irf()`: the response to a one-period rise in each driver.
- `fevd()` and `granger_causality()`, including a direct vs via-link split for
  drivers.
- `sample_posterior_predictive()`: forecasts and counterfactual paths for chosen
  driver plans, with deterministic terms carried forward.
- `summary()`, `fittedvalues`, `resid`, `var_rep`, `test_normality()` and
  `test_whiteness()`.

### Tests before fitting

- `check_stationarity()`: ADF tests that sort series into `endog` and `exog`.
- `select_coint_rank()`: the Johansen trace test.
- `granger_pretest()`: Toda-Yamamoto tests of drivers, delays, links and
  feedback, and a same-week instrument test.

### Simulated data

- `simulate_brand_data()`: weekly brand data with a known answer, with options
  for trending drivers, carryover, saturation, same-week links, total sales,
  fading effects (`exog_fade`, `exog_fade_share`) and channels switched off for
  a spell (`exog_dark`).

### Documentation

- Documentation site with a user guide, API reference and model details.
- The walkthrough article, *How Long Does Brand Last?*
- Nine guide notebooks, including a plug-in notebook for response curves on
  your own data, with how much to spend on a campaign and what happens to sales
  if a channel goes dark, and a guide to whether brand lifts last.
- An agent skill for running the analysis with an analyst new to VECMs.

### Dependencies

- Requires Python 3.12+, PyMC 6, ArviZ 1.1, nutpie, NumPy 2, SciPy, pandas,
  xarray, PyTensor, statsmodels, matplotlib (PyMC's progress bar needs it) and
  zarr (to save fits).
- The package ships type information (`py.typed`).

[0.1.0]: https://github.com/raz1470/bayesian_vecm/releases/tag/v0.1.0
