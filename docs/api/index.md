# API reference

Everything below is importable from `bayesian_vecm`.

| Step | Object |
|---|---|
| Sort the series | {func}`~bayesian_vecm.check_stationarity` |
| What helps predict what, before fitting | {func}`~bayesian_vecm.granger_pretest` |
| Choose the rank | {func}`~bayesian_vecm.select_coint_rank` |
| Fit and read the model | {class}`~bayesian_vecm.BayesianVECM` |
| Simulated data with known parameters | {func}`~bayesian_vecm.simulate_brand_data` |

```{toctree}
:maxdepth: 1

model
pretests
simulation
```
