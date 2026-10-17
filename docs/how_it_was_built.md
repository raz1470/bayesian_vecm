# How it was built

The package was built one feature at a time. Each step came with a notebook
that explains the feature and checks it on data with a known answer. The
notebooks are kept in build order. Each one describes the package as it was
at that step, with a note at the top where something has changed since.

Start with the [user guide](user_guide.md) to use the package. Read these to
see why each piece works the way it does.

| Notebook | Feature | What it shows |
|---|---|---|
| [01. Data utilities](build_notebooks/01_data_utilities_walkthrough.ipynb) | `validate_endog`, `difference`, `lag_matrix` | The three preprocessing helpers and why a VECM needs each |
| [02. Cointegration design](build_notebooks/02_cointegration_design_walkthrough.ipynb) | `cointegration_design` | Aligning the regression matrices, and where deterministic terms go |
| [03. Model skeleton](build_notebooks/03_bayesian_vecm_skeleton_walkthrough.ipynb) | `BayesianVECM` | The public API before any sampling, and the identification problem for β |
| [04. First PyMC model](build_notebooks/04_first_pymc_model_walkthrough.ipynb) | `fit`, `summary` | The first fit, with β pinned so the posterior is identified |
| [05. Posterior predictive](build_notebooks/05_posterior_predictive_walkthrough.ipynb) | `sample_posterior_predictive` | Forecasts whose bands widen with the horizon |
| [06. Wider graph](build_notebooks/06_wider_graph_walkthrough.ipynb) | `coint_rank > 1`, `deterministic` | Higher rank and the five deterministic codes |
| [07. Impulse responses](build_notebooks/07_irf_walkthrough.ipynb) | `irf` | Generalised and Cholesky responses, and why they do not decay |
| [08. Output methods](build_notebooks/08_output_methods_walkthrough.ipynb) | `fittedvalues`, `resid`, `var_rep`, `test_normality`, `test_whiteness` | Post-fit outputs and residual checks |
| [09. Exogenous drivers](build_notebooks/09_exog_brand_marketing.ipynb) | `exog`, `exog_future` | Brand spend as a driver, and a counterfactual forecast |
| [10. Horseshoe prior](build_notebooks/10_horseshoe_prior_walkthrough.ipynb) | `priors={"Gamma": {"dist": "Horseshoe"}}`, `select_coint_rank` | Shrinking spare lags, and choosing the rank before fitting |
| [11. Simulated brand data](build_notebooks/11_simulated_brand_data.ipynb) | `simulate_brand_data`, `BrandData`, `BrandTruth` | The shared brand system with known parameters |
| [12. Pretests](build_notebooks/12_pretests_walkthrough.ipynb) | `check_stationarity`, `granger_pretest` | Which series trend and which help predict which, before fitting |
| [13. What drives what](build_notebooks/13_causality_and_fevd_walkthrough.ipynb) | `granger_causality`, `fevd` | Posterior evidence on each link, and the sources of forecast uncertainty |

## Not covered here

Some later features have no build notebook. The practitioner guide and the
API reference cover them:

- `exog_lags` (driver effects that arrive late) and `exog_irf` (the response
  to one more unit of a driver): [practitioner guide](guide/practitioner_walkthrough.ipynb)
  §9b.
- Forecasting the levels with drivers: practitioner guide §9c.
- A trending driver (`exog_unit_root`): notebook 11 §4 and practitioner guide
  §3a.
- `exog_coint` (a driver inside the long-run relation):
  {py:class}`~bayesian_vecm.BayesianVECM`.

## Running them

The notebooks are in `notebooks/build/` with their outputs saved. Each one
runs in a few minutes with `FAST_SAMPLING = True`. A weekly CI job runs them
all to check that none has stopped working.

```{toctree}
:hidden:
:glob:

build_notebooks/*
```
