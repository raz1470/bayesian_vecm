# bayesian_vecm

Bayesian Vector Error Correction Models (VECM) in Python, built on [PyMC](https://www.pymc.io/).

A VECM captures the long-run cointegrating relationships between time series variables and the short-run dynamics that pull them back to equilibrium when they drift apart. This package brings a fully Bayesian treatment to that framework: posterior uncertainty over all parameters, horseshoe shrinkage for over-specified lag structures, and impulse responses with credible bands.

The primary motivation is **brand marketing**. Brand metrics and sales move together over years, so brand spend that lifts awareness can lift sales for good. An MMM whose effects fade within weeks misses that long-run tail. See [Capturing the long-term causal effect of brand marketing](https://medium.com/@raz1470/capturing-the-long-term-causal-effect-of-brand-marketing-bc577621a627) for the use case.

---

## Resources

| | |
|---|---|
| 📚 [Documentation](https://raz1470.github.io/bayesian_vecm/) | User guide, API reference and model details |
| 📖 [Walkthrough](https://raz1470.github.io/bayesian_vecm/walkthrough.html) | How long does brand last? The method, the tests and the results on simulated brand data |
| 📓 [Practitioner Walkthrough](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/01_brand_vecm_walkthrough.ipynb) | End-to-end notebook: synthetic brand data → rank selection → fit → GIRF fan charts → counterfactual ROI |
| ⚖️ [Comparison with statsmodels](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/02_statsmodels_comparison.ipynb) | Same estimates as statsmodels' VECM, interval coverage on 30 seeds, and the return per £ with intervals |
| 📈 [Carryover and saturation](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/03_carryover_and_saturation.ipynb) | `adstock=` and `saturation=` against a linear same-week fit: effect by week, response curves, switching spend off and doubling it |
| 🔗 [Same-week links](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/04_same_week_links.ipynb) | Does awareness move consideration within the week? Why regression fails, the instruments, the test, the model and when it goes wrong |
| 🧩 [Putting it together](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/05_putting_it_together.ipynb) | One model with adstock, saturation and a same-week link: the response curve for brand spend in £ and the marginal return per £ |
| 🧮 [Base sales from total sales](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/06_base_sales_from_total_sales.ipynb) | A UCM base, its smoothed level or total sales: which one the VECM needs, on 130 fits where the true base is known |
| 📊 [Base sales from an MMM](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/07_base_sales_from_an_mmm.ipynb) | With carryover and saturation: why an MMM's trend fails as base sales, what to use instead, and the one-step VECM on total sales. Needs `pymc-marketing` |
| 🤖 [Agent skill](skills/README.md) | Instructions for an AI agent to run the analysis with an analyst who has never used a VECM: workflow, theory, base sales, checks and a funnel-metric research protocol |
| 🔧 [How it was built](https://raz1470.github.io/bayesian_vecm/how_it_was_built.html) | One notebook per feature, in build order, each checked on data with a known answer |
| 🧭 [Choosing the rank and the lags](docs/choosing_rank_and_lags.md) | Which tests to use and how often they find the truth on simulated data |
| 🔎 [The Granger pretest](docs/granger_pretest.md) | What helps predict what before fitting: drivers, delays, links and feedback |
| 🗺️ [Future work](docs/future_work.md) | What was tested and left out of the first release, and what to check first |

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

### 1. Check what helps predict what

Run the Granger pretest on the levels before choosing anything. It tests the drivers against the brand metrics in both directions. [The Granger pretest](docs/granger_pretest.md) explains each table and how often it finds the truth.

```python
from bayesian_vecm import granger_pretest

result = granger_pretest(endog, exog[["brand_spend", "interest_rate", "consumer_confidence"]])
print(result)
```

Feedback from a brand metric to spend means budgets follow performance. Spend is then not exogenous and its effect in `B` is biased.

### 2. Choose the lags and the rank

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

### 3. Fit the model

```python
from bayesian_vecm import BayesianVECM

model = BayesianVECM(
    k_ar_diff=k_ar_diff,
    coint_rank=result.rank,
    deterministic="co",  # constant outside the relations, so the series can drift up
)
model.fit(endog, exog=exog, target_accept=0.90)
```

`exog` enters in the same week by default, through the coefficient matrix `B`. If the delay table in step 1 shows that a driver acts late, add its past values with `exog_lags`:

```python
model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co", exog_lags=1)
```

- `B` then has one block per lag. Read the effect summed over lags from `exog_irf()` or the exog table of `granger_causality()`.
- Lags of a flighted driver are highly correlated, so the split between lags is noisy. Keep `exog_lags` at 1 or 2.
- Monthly data usually needs `exog_lags=1`. Spend late in a month moves next month's average.

If a channel's effect carries over for weeks (adstock), use `adstock` instead of many free lags:

```python
model = BayesianVECM(
    k_ar_diff=1,
    coint_rank=2,
    deterministic="co",
    adstock=["brand_spend", "pr_reach", "sponsorship_reach"],
    adstock_max_lag=8,
)
```

- Each listed channel gets a total effect per series (`B_total`) and one geometric decay (`adstock_alpha`, flat prior), shared across series. `B` keeps its usual layout, so `exog_irf()`, forecasts and `granger_causality()` work as before.
- Drivers not listed (here the macro variables) keep `exog_lags`.
- On simulated weekly data with carryover, ignoring it missed about half of each channel's total effect. Adstock and `exog_lags=8` both got the totals roughly right, with adstock's intervals about 40% narrower. With no carryover in the data, adstock estimated decays near zero at no cost. `tools/adstock_check.py` has the details.

If one more unit of a channel buys less as spend grows (saturation), add `saturation`. The listed columns of `exog` must be **raw, non-negative spend**, not deviations from the average:

```python
model = BayesianVECM(
    k_ar_diff=1,
    coint_rank=2,
    deterministic="co",
    adstock=["brand_spend", "pr_reach", "sponsorship_reach"],
    saturation=["brand_spend", "pr_reach", "sponsorship_reach"],
)
model.fit(endog, exog=exog_raw_spend)
curve = model.saturation_curve("brand_spend", [0, 500, 1_000, 2_000])
```

- Each listed channel acts through logistic saturation `tanh(lam * a / 2)` of its spend after adstock, scaled by its in-sample maximum. `saturation_lam` sets how fast it bends.
- `B_total` is the effect of one more unit at mean spend. `exog_irf()` and `granger_causality()` report that; `exog_irf(at=...)` values another spend level and `saturation_curve()` the whole curve. Forecasts use the exact curve.
- On simulated weekly data with saturation (18 seeds), a linear fit was close at mean spend but understated the loss from switching brand spend off by 60% and overstated the gain from doubling it by 120%. The saturation fit got both within about 10%, with 0 divergences.
- With no saturation in the data, the prior still bends the curve, so effects far from current spend are overstated by up to 30% for brand spend. `tools/saturation_check.py` has the details.

If one brand metric may move another within the same week, estimate the link with `contemporaneous` and identify it with `excluded`. Run the same arguments through `granger_pretest` first:

```python
links = {"brand_consideration": ["brand_awareness"]}
excluded = {"brand_consideration": ["brand_spend", "pr_reach", "sponsorship_reach"]}

granger_pretest(endog, exog, contemporaneous=links, excluded=excluded).same_week
model = BayesianVECM(
    k_ar_diff=1, coint_rank=2, deterministic="co", contemporaneous=links, excluded=excluded
)
model.fit(endog, exog=exog)
model.same_week_links()
```

- `excluded` names the drivers that reach an effect only through its causes. They act as instruments: the link is the ratio of a driver's effect on the effect to its effect on the cause. Least squares cannot do this because the brand metrics share shocks. With no link in the data it reports about 0.3.
- The shocks keep a full covariance, so the model is the usual one with `B = A0⁻¹ B_s`. `B`, `exog_irf()` and forecasts keep their meaning. Shock impulse responses still need GIRF or Cholesky.
- `fit()` reruns the instrument check and warns when the instruments are weak (first-stage F below 10) or disagree (Sargan p below 0.05). `same_week_links()` reports both. The drivers table of `granger_causality()` splits each effect into `direct` and `via_links`.
- On 20 simulated weekly data sets per truth the posterior tracked 2SLS (correlation 0.96 to 0.99) with 0 divergences. Tying a driver's effects together through the link halved the interval on brand spend's same-week effect on consideration.
- **A wrong exclusion biases the result.** With brand spend acting on consideration directly, the link came out at 0.33 against 0.2 and Sargan caught it in only 10% of data sets. On monthly data a same-month link also holds the lagged weekly effects inside the month, and the instruments are often weak. `tools/same_week_check.py` and the [same-week guide](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/04_same_week_links.ipynb) have the details.

### 4. Inspect the posterior

```python
model.summary(var_names=["alpha", "beta"])
```

`beta` holds the long-run relationships. `alpha` says how fast each series closes a gap in each relationship.

### 5. Responses to shocks and to drivers

```python
irf = model.irf(steps=52, method="girf")  # shocks to the brand metrics
response = model.exog_irf(steps=260)  # a one-week, one-unit rise in each driver
```

Both are `xarray.DataArray`s over `(chain, draw, horizon, response_variable, ...)`. The last dimension is `shock_variable` for `irf` and `exog_variable` for `exog_irf`. A GIRF for one shock equals the Cholesky IRF with that series ordered first. With log `endog`, the responses are log changes.

### 6. Counterfactual forecast

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
| Trending drivers from outside the brand (a policy interest rate) | `exog` | Weekly change: `x.diff()` |
| Launches and events | `exog` | 0/1 dummies |

- Use `check_stationarity` to sort the series: trending (I(1)) series go in `endog`, stationary (I(0)) ones in `exog`. A trending driver the brand cannot move and that does not trend with the brand goes in `exog` as its change. `simulate_brand_data(exog_unit_root=("interest_rate",))` simulates one.
- Logs make the entries of `beta` elasticities and turn forecast gaps into percentage uplifts.
- The model is linear in the drivers, so it shows no diminishing returns. Its returns are marginal returns near current spend.
- A driver must not respond to the brand metrics. If budgets follow last month's sales, spend is not exogenous and its effect absorbs that feedback.
- With total sales only, take the short-term effects out with a UCM and keep the seasonality and the noise in. The [base sales guide](https://nbviewer.org/github/raz1470/bayesian_vecm/blob/main/notebooks/guides/06_base_sales_from_total_sales.ipynb) compares the options. `simulate_brand_data(total_sales=True)` simulates total sales with its parts.

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
