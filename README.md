# bayesian_vecm

Bayesian Vector Error Correction Models (VECM) in Python, built on [PyMC](https://www.pymc.io/).

A VECM captures the long-run cointegrating relationships between time series variables and the short-run dynamics that pull them back to equilibrium when they drift apart. This package brings a fully Bayesian treatment to that framework — posterior uncertainty over all parameters, horseshoe shrinkage for over-specified lag structures, and Generalised Impulse Response Functions with credible bands.

The primary motivation is **brand marketing**: brand spend and revenue are often cointegrated over years, and standard MMMs miss this long-run tail entirely. See [Capturing the long-term causal effect of brand marketing](https://medium.com/@raz1470/capturing-the-long-term-causal-effect-of-brand-marketing-bc577621a627) for the use case.

---

## Resources

| | |
|---|---|
| 📖 [VECM Guide](https://raz1470.github.io/bayesian_vecm/vecm_guide.html) | Visual explainer: cointegration, error correction, GIRFs, and the Bayesian treatment — with diagrams and collapsible maths |
| 📓 [Practitioner Walkthrough](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/01_brand_vecm_walkthrough.ipynb) | End-to-end notebook: synthetic brand data → rank selection → horseshoe fit → GIRF fan charts → counterfactual ROI |

---

## Installation

```bash
pip install bayesian_vecm
```

Requires Python 3.12+.

---

## Quick start

### 1. Select the cointegration rank

Before fitting, use the Johansen trace test to find how many long-run equilibrium relationships exist in your data:

```python
import pandas as pd
from bayesian_vecm import select_coint_rank

endog = pd.read_csv("brand_data.csv", index_col="date", parse_dates=True)

result = select_coint_rank(endog, det_order=1, k_ar_diff=2)
print(result)
# Johansen cointegration rank test (trace statistic, 5% critical value)
#   H0: r <=    Trace stat   Crit val (5%)  Reject?
#   r <= 0        42.3            29.8          yes
#   r <= 1        11.2            15.5           no   ← recommended rank = 1
```

### 2. Fit the model

```python
from bayesian_vecm import BayesianVECM

model = BayesianVECM(
    k_ar_diff=2,
    coint_rank=result.rank,
    deterministic="ci",   # trend inside cointegration space — correct for trending brand data
)
model.fit(endog)
```

Pass exogenous variables (e.g. brand spend) as contemporaneous drivers:

```python
model.fit(endog, exog=exog_df)
```

### 3. Inspect the posterior

```python
model.summary(var_names=["alpha", "beta"])
```

`beta` is the long-run equilibrium equation. `alpha` is the error-correction loading — how fast each variable returns to equilibrium after a shock.

### 4. Impulse Response Functions

```python
irf = model.irf(steps=52, method="girf")  # Generalised IRFs — order-invariant
```

`irf` is an `xarray.DataArray` with shape `(chain, draw, horizon, response, shock)`. Plot 80%/95% HDI bands with ArviZ:

```python
import arviz as az
import matplotlib.pyplot as plt

organic_sales_response = irf.sel(response_variable="organic_sales", shock_variable="brand_spend")
hdi = az.hdi(organic_sales_response, prob=0.80, dim=["chain", "draw"])

plt.plot(organic_sales_response.mean(["chain", "draw"]))
plt.fill_between(range(53), hdi.sel(ci_bound="lower"), hdi.sel(ci_bound="upper"), alpha=0.3)
plt.title("Brand spend shock → organic sales (52 weeks)")
```

### 5. Counterfactual forecast

```python
# High brand spend scenario
high_spend = ...  # shape (steps, m)
low_spend = ...

high = model.sample_posterior_predictive(steps=52, exog_future=high_spend)
low  = model.sample_posterior_predictive(steps=52, exog_future=low_spend)

uplift = high["posterior_predictive"]["y"] - low["posterior_predictive"]["y"]
# → Bayesian ROI: revenue uplift with full credible interval
```

---

## Horseshoe priors

With real marketing data, the right lag order `k` is unknown. Setting `k_ar_diff` generously and applying a horseshoe prior shrinks irrelevant lag coefficients toward zero while preserving genuine short-run dynamics — more principled than hard lag selection via information criteria.

```python
model = BayesianVECM(
    k_ar_diff=3,          # generous; horseshoe shrinks spurious lags
    coint_rank=1,
    deterministic="ci",
    priors={"Gamma": {"dist": "Horseshoe"}},
)
```

Optional kwargs: `tau_scale` (default 1.0), `slab_scale` (default 2.0), `slab_df` (default 4.0). The default Normal(0, 0.5) prior is unchanged when `priors` is omitted.

---

## Variable naming

All continuous endogenous and exogenous variables should be **log-transformed** before fitting. This gives elasticity interpretations on the parameters and natural diminishing returns on the brand response curve.

| Variable type | Treatment |
|---|---|
| I(1) trending (organic sales, brand awareness, consideration) | Endogenous — log-transform |
| I(0) stationary (interest rate, consumer confidence) | Exogenous via `exog=` |
| Brand spend | Exogenous via `exog=` — log-transform |
| Feature launch dates | Binary dummies via `exog=` — no log |

Use `select_coint_rank` to decide: I(1) variables belong in the endogenous system; I(0) variables go in `exog`.

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

After cloning, install the pre-commit hooks:

```bash
pre-commit install
```

---

## License

MIT — see [LICENSE](LICENSE).
