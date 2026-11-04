# bayesian_vecm

Bayesian Vector Error Correction Models (VECM) in Python, built on [PyMC](https://www.pymc.io/).

A VECM captures the long-run relationships between trending series and the short-run dynamics that pull them back when they drift apart. This package fits it in full Bayes, so every parameter, impulse response and return has an interval.

The main use is **brand marketing**: brand metrics and sales move together over years, so spend that lifts awareness can lift sales for good. A media mix model whose effects fade within weeks misses that tail. The method follows Peter Cain's work:

- Cain, P. M. (2022). [Modelling short- and long-term marketing effects in the consumer purchase journey](https://doi.org/10.1016/j.ijresmar.2021.06.006). *International Journal of Research in Marketing*, 39(1), 96-116. Brand metrics tied to base sales by cointegration, and same-week links between them estimated by instruments.
- Cain, P. M. (2025). [Long-term advertising effects: The Adstock illusion](https://ideas.repec.org/a/aza/ama000/y2025v11i1p23-42.html). *Applied Marketing Analytics*, 11(1), 23-42. Why a long adstock cannot stand in for brand building.

## Resources

| | |
|---|---|
| 📚 [Documentation](https://raz1470.github.io/bayesian_vecm/) | User guide, API reference and model details |
| 📖 [Walkthrough](https://raz1470.github.io/bayesian_vecm/walkthrough.html) | How Long Does Brand Last? The method step by step, on simulated data |
| 🔌 [Response curves on your data](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/08_response_curves_on_your_data.ipynb) | Plug-in notebook: set your columns in one cell, get a response curve and return per £ for each channel |
| 📓 [Practitioner walkthrough](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/01_brand_vecm_walkthrough.ipynb) | Data → rank → fit → impulse responses → counterfactual ROI |
| ⚖️ [Comparison with statsmodels](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/02_statsmodels_comparison.ipynb) | Same estimates as statsmodels, with interval coverage |
| 📈 [Carryover and saturation](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/03_carryover_and_saturation.ipynb) | `adstock=` and `saturation=` |
| 🔗 [Same-week links](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/04_same_week_links.ipynb) | `contemporaneous=` and `excluded=` |
| 🧩 [Putting it together](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/05_putting_it_together.ipynb) | Adstock, saturation and a same-week link in one model |
| 🧮 [Base sales from total sales](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/06_base_sales_from_total_sales.ipynb) | Which base the VECM needs when you only have total sales |
| 📊 [Base sales from an MMM](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/07_base_sales_from_an_mmm.ipynb) | Why an MMM's trend fails as base sales. Needs `pymc-marketing` |
| 🤖 [Agent skill](skills/README.md) | Instructions for an AI agent to run the analysis with an analyst |
| 🧭 [Choosing the rank and the lags](docs/choosing_rank_and_lags.md) | Which tests to use and how far to trust them |
| 🔎 [The Granger pretest](docs/granger_pretest.md) | What helps predict what, before fitting |
| 🔧 [How it was built](https://raz1470.github.io/bayesian_vecm/how_it_was_built.html) | One notebook per feature, each checked on data with a known answer |
| 🗺️ [Future work](docs/future_work.md) | What was tested and left out of the first release |

## Installation

```bash
pip install bayesian_vecm
```

Requires Python 3.12+.

## Quick start

`endog` holds the log brand metrics and sales, `exog` the drivers, both weekly.

```python
from statsmodels.tsa.api import VAR
from bayesian_vecm import BayesianVECM, granger_pretest, select_coint_rank, simulate_brand_data

data = simulate_brand_data()
endog, exog = data.endog, data.exog

# 1. What helps predict what (drivers, delays, feedback)
print(granger_pretest(endog, exog))

# 2. Lags by HQ, rank by Johansen
orders = VAR(endog, exog=exog).select_order(maxlags=8, trend="c")
k_ar_diff = max(orders.selected_orders["hqic"] - 1, 1)
rank = select_coint_rank(endog, k_ar_diff=k_ar_diff, det_order=0).rank

# 3. Fit
model = BayesianVECM(k_ar_diff=k_ar_diff, coint_rank=rank, deterministic="co")
model.fit(endog, exog=exog, target_accept=0.90)

# 4. Results
model.summary(var_names=["alpha", "beta"])  # beta: long-run relations, alpha: speed back
irf = model.irf(steps=52, method="girf")     # shocks to the brand metrics
response = model.exog_irf(steps=260)         # a one-week rise in each driver
```

### Options

| Argument | Use it when |
|---|---|
| `exog_lags=1` | A driver acts with a delay (monthly data usually needs it) |
| `adstock=[...]` | A channel's effect carries over for weeks |
| `saturation=[...]` | One more £ buys less as spend grows. Pass raw, non-negative spend |
| `contemporaneous=`, `excluded=` | One brand metric moves another in the same week. A wrong exclusion biases the link |
| `priors={"Gamma": {"dist": "Horseshoe"}}` | The lag order is unclear. Fit with `target_accept=0.99` |

For a counterfactual, call `sample_posterior_predictive(steps, exog_future=..., random_seed=0)` on a baseline and an uplift path with the same seed, and take the difference. The guides show each option on simulated data with a known answer.

## Preparing variables

| Variable | Where | Treatment |
|---|---|---|
| Trending brand metrics and sales | `endog` | Log levels |
| Brand channels (spend, reach) | `exog` | Deviation from the average, in round units (raw spend with `saturation`) |
| Stationary drivers | `exog` | Deviation from the average |
| Trending outside drivers | `exog` | Weekly change |
| Launches and events | `exog` | 0/1 dummies |

- `check_stationarity` sorts trending series into `endog` and stationary ones into `exog`.
- A driver must not respond to the brand metrics. If budgets follow sales, spend's effect absorbs that feedback.
- With total sales only, see the [base sales guide](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/06_base_sales_from_total_sales.ipynb).

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check . && uv run ruff format .
uv build
```

## License

MIT. See [LICENSE](LICENSE).
