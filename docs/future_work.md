# Future work

Ideas that were tested or scoped and left out of the first release, with the reason. Each one lists what to check first if it is picked up.

## Same-week links between brand metrics

**The question.** Does awareness move consideration within the same week? The fitted model cannot say. `B` and `Gamma` only cover drivers and past growth, and a GIRF or Cholesky IRF assumes an order instead of estimating one.

**Why plain regression fails.** The brand metrics share shocks to a common trend. On simulated data with no same-week link, least squares of consideration growth on awareness growth gives 0.30 (5-95%: 0.23 to 0.38) over 200 seeds of 260 weeks.

**What works.** Instruments, as in Cain (2022). Two-stage least squares, with a driver that moves the cause but reaches the effect only through it as the instrument. On the same data the marketing channels instrument awareness: 2SLS gives 0.002 (-0.12 to 0.12), with a mean first-stage F of 83.

**The test (built).** `granger_pretest(contemporaneous=..., excluded=...)` adds a `same_week` table: per link OLS, 2SLS, first-stage F, Sargan and a verdict ("same-week link", "no clear evidence", "weak instruments", "instruments disagree" or "not identified"). `simulate_brand_data(contemporaneous=...)` gives a non-zero truth. On 200 seeds of 260 weeks:

| | Link 0 | Link 0.2 |
|---|---|---|
| Weekly, awareness → consideration: 2SLS mean | 0.00 | 0.20 |
| Weekly: reported as a link | 5% | 70% |
| Monthly (58 months): 2SLS mean | 0.33 | 0.49 |
| Monthly: first-stage F mean | 15 | 15 |

A monthly link is not zero even when the weekly one is. The weekly lagged effect of awareness lands inside the month. Consideration → sales on monthly data has weak instruments in 31% to 44% of seeds. See `tools/same_week_pretest_check.py`.

**The model (built).** `BayesianVECM(contemporaneous=..., excluded=...)` estimates the links:

```
A0 Δy_t = α β' y_{t-1} + Γ Δy_{t-1} + B_s X_t + u_t,    u_t ~ N(0, Σ_u), Σ_u full
```

- `A0` has ones on the diagonal and minus each link off it. `Σ_u` stays full: a diagonal one would turn the shared shocks into a link, as Cholesky does.
- With `Σ_u` full the model is the reduced form with `B = A0⁻¹ B_s`. The links are identified only by the zeros in `B_s`. No `log|det A0|` term is needed.
- `tools/same_week_check.py` (160 fits, 0 divergences): the posterior mean tracks 2SLS with a correlation of 0.96 to 0.99 on weekly data. A wrong exclusion biased the link from 0.2 to 0.33 and Sargan caught it in 10% of data sets. Weak monthly instruments left the posterior close to the prior.

- **Built since:** `fit()` reruns the instrument check and warns on weak instruments (first-stage F below 10) or disagreeing ones (Sargan p below 0.05). `same_week_links()` reports both numbers. The drivers table of `granger_causality()` splits each effect into `direct` and `via_links`.

**Still open.** Cycles between links (awareness and consideration moving each other in the same week). They need `log|det A0|` and more exclusions.

## Structure on lagged driver effects

`exog_lags` gives every driver and lag its own coefficient. With more lags and drivers, most of these are zero, and two ways to use that were considered.

**A horseshoe prior on `B` (built and rejected).** It used the same regularised horseshoe as `Gamma`. On 10 weekly seeds with `exog_lags=2`, where brand spend acts over weeks 0, 1 and 2 (0.009, 0.0054, 0.0036) and nothing else lags:

| | Normal (default) | Horseshoe |
|---|---|---|
| Error on the entries that are zero | 0.0035 | 0.0008 |
| Brand spend total effect (truth 0.018) | 0.0172 (-4%) | 0.0141 (-21%) |
| 90% coverage of the total | 90% | 70% |
| Divergences (10 fits) | 0 | 405 (82 with scales matched to the data) |

It finds the zeros, but it also shrinks small real lag effects, so the total effect of a driver, the number users report, is biased down. The default Normal prior already gets the total right.

**Adstock (built).** The preferred option was implemented as `adstock=`: `B_l = b * w^l / sum(w^i)`, so `b` (`B_total`) is the total effect, with one decay per driver shared across the equations. A flat `Beta(1, 1)` prior on the decay replaced the pymc-marketing default `Beta(1, 3)`, which pulled the decays and totals down by about 10%. See `tools/adstock_check.py` and the README.

**Logistic saturation (built).** `saturation=` applies `tanh(lam * a / 2)` after adstock to raw spend scaled by its maximum. Sampling the effect at full saturation put the posterior on a ridge with `lam`, so `B_total` is sampled as the effect of one more unit at mean spend and the full-saturation effect is derived. The default prior on `lam` is `Gamma(2, 0.3)`: the pymc-marketing default `Gamma(3, 1)` pulled `lam` down by 17% to 41% and flattened the curve. See `tools/saturation_check.py`.

**Still open: a prior that allows no saturation.** With linear truth the `Gamma(2, 0.3)` prior keeps `lam` near 2, so the fit bends a straight line. Effects far from current spend come out up to 30% too large for brand spend and about 55% for sponsorship. A prior with more mass near zero, or a check of whether the curve is needed, would fix this.

**Still open: a horseshoe grouped by driver.** All lags of a driver share one local scale, so a whole driver can be dropped without zeroing its lags one by one. Only worth it if many drivers with free lags become common.

## A PyMC prior-sampling bug

In PyMC 6.0 with PyTensor 3.0.2, random draws from `HalfCauchy(beta)` use `1 / beta` as the scale. The log density is correct, so fitting is not affected. Prior draws (`pm.draw`, `sample_prior_predictive`) of the `Gamma` horseshoe are wrong when `tau_scale` is not 1. The default `tau_scale=1` is unaffected. Report it upstream and add a note to the horseshoe docs, or check for a fixed release.
