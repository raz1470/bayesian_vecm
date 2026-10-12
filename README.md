# bayesian_vecm

Bayesian Vector Error Correction Models (VECM) in Python, built on [PyMC](https://www.pymc.io/).

A VECM captures the long-run cointegrating relationships between time series variables and the short-run dynamics that pull them back to equilibrium when they drift apart. This package brings a fully Bayesian treatment to that framework: posterior uncertainty over all parameters, horseshoe shrinkage for over-specified lag structures, and impulse responses with credible bands.

The primary motivation is **brand marketing**. Brand metrics and sales move together over years, so brand spend that lifts awareness can lift sales for good. An MMM whose effects fade within weeks misses that long-run tail. See [Capturing the long-term causal effect of brand marketing](https://medium.com/@raz1470/capturing-the-long-term-causal-effect-of-brand-marketing-bc577621a627) for the use case.

---

## Resources

| | |
|---|---|
| 📖 [Walkthrough](https://raz1470.github.io/bayesian_vecm/walkthrough.html) | How long does brand last? The method, the tests and the results on simulated brand data |
| 📓 [Practitioner Walkthrough](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/01_brand_vecm_walkthrough.ipynb) | End-to-end notebook: synthetic brand data → rank selection → fit → GIRF fan charts → counterfactual ROI |
| 🧭 [Choosing the rank and the lags](docs/choosing_rank_and_lags.md) | Which tests to use and how often they find the truth on simulated data |

---

## Installation

```bash
pip install bayesian_vecm
```

Requires Python 3.12+.

---

## Quick start

The examples use the simulated brand data that ships with the package. Swap in your own weekly data frames: `endog` holds the log brand metrics and sales, `exog` the drivers.

```python
from bayesian_vecm import simulate_brand_data

data = simulate_brand_data()
endog, exog = data.endog, data.exog
```

### 1. Choose the lags and the rank

Pick the lag order with the HQ criterion on a levels VAR, then run the Johansen trace test with the same lags. [Choosing the rank and the lags](docs/choosing_rank_and_lags.md) explains both choices and how far to trust them.

```python
from statsmodels.tsa.api import VAR
from bayesian_vecm import select_coint_rank

orders = VAR(endog, exog=exog).select_order(maxlags=8, trend="c")
k_ar_diff = max(orders.selected_orders["hqic"] - 1, 1)

result = select_coint_rank(endog, k_ar_diff=k_ar_diff, det_order=0)
print(result)
#    H0: r <=    Trace stat   Crit val (5%)  Reject?
#      r <= 0        61.915          29.796      yes
#      r <= 1        21.374          15.494      yes
#      r <= 2         1.085           3.841       no
#   Recommended rank: 2
```

### 2. Fit the model

```python
from bayesian_vecm import BayesianVECM

model = BayesianVECM(
    k_ar_diff=k_ar_diff,
    coint_rank=result.rank,
    deterministic="co",  # constant outside the relations, so the series can drift up
)
model.fit(endog, exog=exog, target_accept=0.90)
```

`exog` enters in the same week only, through the coefficient matrix `B`.

### 3. Inspect the posterior

```python
model.summary(var_names=["alpha", "beta"])
```

`beta` holds the long-run relationships. `alpha` says how fast each series closes a gap in each relationship.

### 4. Responses to shocks and to drivers

```python
irf = model.irf(steps=52, method="girf")  # shocks to the brand metrics
response = model.exog_irf(steps=260)      # a one-week, one-unit rise in each driver
```

Both are `xarray.DataArray`s over `(chain, draw, horizon, response_variable, ...)`. The last dimension is `shock_variable` for `irf` and `exog_variable` for `exog_irf`. A GIRF for one shock equals the Cholesky IRF with that series ordered first. With log `endog`, the responses are log changes.

### 5. Counterfactual forecast

```python
import numpy as np

steps = 52
baseline = np.zeros((steps, exog.shape[1]))  # every driver at its average
uplift = baseline.copy()
uplift[:26, exog.columns.get_loc("brand_spend")] = exog["brand_spend"].std()

low = model.sample_posterior_predictive(steps, exog_future=baseline, random_seed=0)
high = model.sample_posterior_predictive(steps, exog_future=uplift, random_seed=0)

gain = high.posterior_predictive["y"] - low.posterior_predictive["y"]  # log uplift
```

Use the same `random_seed` for both paths so the shared shocks cancel.

---

## Horseshoe priors

When the lag order is unclear, set `k_ar_diff` generously and let a regularised horseshoe prior shrink the lags you do not need:

```python
model = BayesianVECM(
    k_ar_diff=3,
    coint_rank=2,
    deterministic="co",
    priors={"Gamma": {"dist": "Horseshoe"}},
)
model.fit(endog, exog=exog, target_accept=0.99)
```

- On the simulated data, with three lags where one is true, the two spare lags shrink to a mean absolute value of 0.008. The default Normal prior leaves them at 0.07.
- Use `target_accept=0.99`. At 0.90 it gives 28 to 97 divergences. At 0.99 it gives 1 or 2.
- Only the lag columns are shrunk. The constant or trend keeps the Normal(0, 0.5) prior.
- Optional kwargs: `tau_scale` (default 1.0), `slab_scale` (default 2.0), `slab_df` (default 4.0).

The default Normal(0, 0.5) prior on `Gamma` applies when `priors` is omitted. With the lag order from HQ it samples cleanly at `target_accept=0.90`.

---

## Preparing variables

| Variable | Where it goes | Treatment |
|---|---|---|
| Trending brand metrics and sales (organic sales, awareness, consideration) | `endog` | Log levels |
| Brand channels (spend, reach) | `exog` | Deviation from the average, in round units such as £1,000s or 100,000 people |
| Stationary drivers (interest rate, consumer confidence) | `exog` | Deviation from the average |
| Launches and events | `exog` | 0/1 dummies |

- Use `check_stationarity` to sort the series: trending (I(1)) series go in `endog`, stationary (I(0)) ones in `exog`.
- Logs make the entries of `beta` elasticities and turn forecast gaps into percentage uplifts.
- The model is linear in the drivers, so it shows no diminishing returns. Its returns are marginal returns near current spend.
- A driver must not respond to the brand metrics. If budgets follow last month's sales, spend is not exogenous and its effect absorbs that feedback.

---

## Development

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Install all dependencies (runtime + dev)
uv sync --all-extras

# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Build distributions
uv build
```

---

## License

MIT — see [LICENSE](LICENSE).
