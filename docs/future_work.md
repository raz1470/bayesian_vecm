# Future work

Ideas that were tested or scoped and left out of the first release, with the reason. Each one lists what to check first if it is picked up.

## Same-week links between brand metrics

**The question.** Does awareness move consideration within the same week? The fitted model cannot say. `B` and `Gamma` only cover drivers and past growth, and a GIRF or Cholesky IRF assumes an order instead of estimating one.

**Why plain regression fails.** The brand metrics share shocks to a common trend. On simulated data with no same-week link, least squares of consideration growth on awareness growth gives 0.30 (5-95%: 0.23 to 0.38) over 200 seeds of 260 weeks.

**What works.** Two-stage least squares, with a driver that moves the cause but reaches the effect only through it as the instrument. On the same data the marketing channels instrument awareness: 2SLS gives 0.002 (-0.12 to 0.12), with a mean first-stage F of 83.

**The plan.** An instrumental-variable table in `granger_pretest`:

- an `excluded=` argument naming, for each link, the drivers that do not act on the effect directly;
- per link: OLS, 2SLS, first-stage F and a Sargan test, or "not identified" when no driver is excluded;
- a same-week link option in `simulate_brand_data`, so the test has a non-zero truth to find;
- hit rates on weekly and monthly data, and a docs section with the three conditions an instrument must meet: it moves the cause, it is not set in response to the same week's shock, and it reaches the effect only through the cause.

**Check first.** Run the 2SLS check on 60 months. Monthly instruments may be too weak. If the first-stage F is below about 10, ship only the "not identified / weak instruments" diagnostics and the docs section.

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

**Options if this is picked up:**

1. **Adstock-style lags (preferred).** `B_l = b * w_l(theta)`, with geometric weights that sum to one, so `b` is the total effect and gets its own prior. One decay rate per driver, shared across the equations. `B` keeps its shape, so `exog_irf`, `granger_causality` and forecasting need no change. Check first: when a driver has no effect, `theta` is not identified and may cause divergences.
2. **A horseshoe grouped by driver.** All lags of a driver share one local scale, so a whole driver can be dropped without zeroing its lags one by one.

## A PyMC prior-sampling bug

In PyMC 6.0 with PyTensor 3.0.2, random draws from `HalfCauchy(beta)` use `1 / beta` as the scale. The log density is correct, so fitting is not affected. Prior draws (`pm.draw`, `sample_prior_predictive`) of the `Gamma` horseshoe are wrong when `tau_scale` is not 1. The default `tau_scale=1` is unaffected. Report it upstream and add a note to the horseshoe docs, or check for a fixed release.
