# Choosing the rank and the lags

`BayesianVECM` needs three choices before it can fit: the number of lags
(`k_ar_diff`), the number of long-run relationships (`coint_rank`) and the
deterministic terms (`deterministic`). None of them is estimated by the
sampler. This note sets out how to make each one and how far the usual tests
can be trusted.

The figures come from `tools/rank_lag_check.py`. It runs the tests on 200
seeds of `simulate_brand_data()`, where the truth is known: one lag of weekly
changes (`k_ar_diff=1`, a VAR(2) in levels) and rank 2.

## The order of work

1. Check that each series has a unit root (`check_stationarity`).
2. Choose the lags.
3. Choose the rank with the same `k_ar_diff`.
4. Fit with `deterministic` matched to the rank test.
5. Check the fit and fit the next rank up and down if the test was close.

The lags come first because the rank test depends on them.

## 1. Lags

Fit a VAR in levels and let an information criterion pick the order `p`.
Then `k_ar_diff = p - 1`.

```python
from statsmodels.tsa.api import VAR

orders = VAR(endog, exog=exog).select_order(maxlags=8, trend="c")
k_ar_diff = max(orders.selected_orders["hqic"] - 1, 1)
```

Share of seeds that pick the true order:

| Criterion | 156 weeks | 260 weeks |
|---|---|---|
| AIC | 0.90 | 0.84 to 0.90 |
| BIC | 0.48 to 0.74 | 0.92 to 1.00 |
| HQ | 0.94 to 0.97 | 0.99 to 1.00 |

The ranges cover the test with and without `exog`.

- **Use HQ.** It is right in at least 94% of seeds at both sample sizes.
- AIC errs long. At 260 weeks it picks three or more lags in 8% to 16% of seeds.
- BIC errs short on three years of data. With `exog` it drops to one lag in half the seeds.
- Pass `exog` to the test when the model will have it.
- Keep at least one lag of differences. Weekly brand series nearly always
  have short-run dynamics.
- Too many lags costs precision and too few biases Γ and the impulse
  responses. With a few hundred weeks, err one lag long rather than short.

### Setting lags generously with the horseshoe prior

`priors={"Gamma": {"dist": "Horseshoe"}}` is meant to let you set
`k_ar_diff` long and shrink the lags you do not need. It does not sample
well at present, so use the default Normal prior with the HQ choice.

Simulated data, 2 chains, 1,000 draws, `target_accept=0.90`:

| `k_ar_diff` | Prior | Divergences | Max r-hat | Min bulk ESS |
|---|---|---|---|---|
| 1 | Normal | 0 | 1.003 | 850 |
| 1 | Horseshoe | 253 | 1.072 | 30 |
| 3 | Normal | 0 | 1.007 | 902 |
| 3 | Horseshoe | 303 | 1.128 | 21 |

The Normal prior with `k_ar_diff=3` still samples cleanly. The two spare lags
come out near zero but not at it: their mean absolute coefficient is 0.07.

## 2. Rank

```python
from bayesian_vecm import select_coint_rank

result = select_coint_rank(endog, k_ar_diff=k_ar_diff, det_order=0)
print(result)
```

Share of seeds that pick the true rank of 2:

| `k_ar_diff` | 156 weeks, `det_order=0` | 260 weeks, `det_order=0` | 260 weeks, `det_order=1` |
|---|---|---|---|
| 1 | 0.45 | 0.81 | 0.39 |
| 2 | 0.38 | 0.81 | 0.38 |
| 3 | 0.30 | 0.80 | 0.36 |
| 4 | 0.33 | 0.80 | 0.33 |

- **Three years is not enough to trust the test.** At 156 weeks it finds the
  true rank in under half the seeds. It mostly picks one relationship too few
  (about 40% of seeds).
- **Five years is.** At 260 weeks it finds the true rank in about 80% of
  seeds. When it misses it picks one too many (about 16%).
- **`det_order=0` for drifting data.** `det_order=1` adds a trend to the test
  and picks rank 3 in about 60% of seeds. Rank 3 with three series says every
  series is stationary, which the unit-root tests contradict.
- Extra lags in the test lower the hit rate a little. Use the `k_ar_diff`
  from step 1.
- The test ignores `exog`, so its critical values assume a model without
  drivers. Treat the result as a guide, not a verdict.

### Use theory as well as the test

The rank is `K` minus the number of shared trends. For brand data, start from
the funnel:

- One shared trend (demand or awareness) and every other series tied to it
  gives `r = K - 1`. This is the simulated case.
- If two series wander apart for good, such as a price index and sales, they
  need a trend each and the rank falls.

When the test and the theory disagree, or the trace statistic sits close to
its critical value, fit both ranks and compare:

- **β and α.** A relationship with signs that make no business sense, or an α
  near zero for every series, suggests the rank is too high.
- **Hold-out forecast.** Refit without the last year and compare the error.
  The walkthrough does this for the chosen model.
- **The answer you need.** Check whether the return per channel
  (`exog_irf()`) changes between the two ranks. If it does not, the choice
  matters less.

`coint_rank` must be at least 1. If the test says 0, the series are not tied
together and a VAR in differences is the better model.

## 3. Deterministic terms

| Data | `deterministic` | `det_order` in the rank test |
|---|---|---|
| Levels drift upwards | `"co"` | `0` |
| Levels mean-revert to a fixed gap, no drift | `"ci"` | `0` (closest available) |
| No drift and the gaps sit at zero | `"n"` | `-1` |

The simulated data drifts, so the guide uses `"co"`. With `"ci"` the model
cannot produce a drift.

## Checklist

- [ ] Unit roots confirmed for every series.
- [ ] `k_ar_diff` from HQ on a levels VAR with `exog`, at least 1.
- [ ] Rank from `select_coint_rank` with the same `k_ar_diff` and `det_order=0`.
- [ ] Rank checked against the number of shared trends you expect.
- [ ] With under four years of data, or a close trace test, the next rank fitted as well.
- [ ] `deterministic` matched to the drift in the data.
- [ ] After the fit: residual checks (`test_whiteness`, `test_normality`). Autocorrelated residuals point to too few lags.
