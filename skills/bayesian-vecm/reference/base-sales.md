# Base sales

The long-term model often runs on **base sales**: sales without the short-term effect of marketing, usually taken from a media mix model (MMM). Cain (2022) does this. It is also where the work most often goes wrong before the VECM sees any data.

## What base sales has to be

The VECM explains the slow movement in sales through the brand metrics. So the sales series must still contain that slow movement, and nothing that is counted elsewhere.

1. **Actual sales minus the short-term marketing and promotion contributions**, using the MMM's own adstock. Nothing else removed.
2. **The noise kept in.** It is data minus contributions, not a fitted curve. A smooth base makes the model look surer than it is and distorts how the shared shocks split.
3. **The level and the long-run trend kept in.** That is the brand effect being measured.
4. **Seasonality handled one way for every series.** Either kept in with seasonal terms as drivers, or removed by the same method from all series.
5. **The same frequency, dates and breaks** as the brand metrics.

If short-term media is left in, the VECM counts it again. If the trend is taken out, there is nothing left for the VECM to find.

## Approaches to question

None of these is wrong for an MMM. But the base they produce answers "what is not short-term marketing", while the VECM needs "everything that is not short-term marketing, unsmoothed". Ask the user how their base was built and check it against this table.

| How the base was made | The problem |
|---|---|
| A time-varying intercept in a Bayesian MMM (Gaussian process, random walk, splines: Meridian, pymc-marketing and others) | It absorbs slow brand-driven growth as "baseline", so brand gets no credit for it. Its smoothness is set by a prior chosen for fit, not by the data. |
| Prophet or Robyn trend | Changepoints and smoothing are imposed. Using the trend component as base removes the noise. |
| A linear or polynomial time trend | Assumes growth is fixed, so the VECM cannot explain it with brand. Leaves spurious structure in the residual. |
| Hodrick-Prescott filter | Known to create cycles that are not in the data (Hamilton 2018). Poor at the ends of the sample. |
| Moving average | Shifts timing, distorts the ends and smooths away the noise. |
| UCM smoothed level | Principled, but the smoothed level is still an estimate with the noise removed. Use the recipe below instead. |
| MMM intercept only (a constant) | Not a base at all for this purpose. |

What to say to the user: "Your MMM's base is fine for the MMM. For this model we need sales with only the short-term marketing taken out, and nothing smoothed."

## Building base sales with a UCM

When there is no usable base, build one with an unobserved components model from `statsmodels` (`pip install statsmodels`).

```python
import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Short-term drivers: adstocked media with short decays, promotions, price.
# The long-term part of media stays in the level for the VECM to explain.
short_term = build_short_term_drivers(raw)  # (T, m) DataFrame, columns named

ucm = UnobservedComponents(
    np.log(raw["sales"]),
    level="lldtrend",  # random-walk level, drift, noise
    freq_seasonal=[{"period": 52, "harmonics": 3}],  # weekly data
    exog=short_term,
).fit(disp=False)

coefs = ucm.params[[f"beta.{c}" for c in short_term.columns]]
log_base = np.log(raw["sales"]) - short_term @ coefs.to_numpy()
```

Rules:

- **The level must be stochastic** (`"lldtrend"`: a random-walk level with drift and noise), never `"dtrend"` (a fixed trend). A fixed trend imposes the growth the VECM should explain.
- **Take only the driver contributions out.** Keep the level and the irregular (noise). Keep or remove the seasonal part, matching the other series.
- **Never use `ucm.level["smoothed"]` as base sales.** It is the smoothed curve.
- **Short decays for the media adstock** (a few weeks). Long decays push brand effects into the short-term contribution.
- Print `ucm.summary()` and check the media coefficients have sensible signs before using the base.

Parameter names follow `statsmodels`' `beta.<column>` convention. Check `ucm.params.index` if they differ.

## Checks on any base sales series

Run these on a base someone else built, and on your own.

1. **Too smooth?** Compare the week-to-week changes of the base with those of total sales.

   ```python
   def smoothness(log_series):
       changes = pd.Series(log_series).diff().dropna()
       return {"autocorr": changes.autocorr(), "sd": changes.std()}


   print(smoothness(log_base), smoothness(np.log(raw["sales"])))
   ```

   Smoothing makes changes follow each other. On simulated weekly sales the autocorrelation of the changes is about 0.35. A 9-week moving average of the same series gives 0.93 and an exponential smooth 0.90. An autocorrelation above about 0.7, or changes much less variable than in total sales, means the base is a fitted curve and not data.

2. **Does it trend and move with the brand metrics?** `check_stationarity` should call it a trending series. `select_coint_rank` with the brand metrics should find ties.

3. **Any media left in?** In `granger_pretest(endog, exog)`, the delay table should not show this week's spend predicting base sales strongly beyond its path through the brand metrics. A big same-week spend effect on base sales means short-term media is still in it.

4. **Does it add up?** Base plus the short-term contributions should rebuild total sales, apart from noise. Plot both.

5. **How much does the decomposition matter?** If the MMM is Bayesian, build base sales from a few posterior draws (say five) and refit the VECM on each. If the return per £ moves a lot between them, report that spread.

## The alternative: no base sales

Give the VECM **total sales** and put the spend in as drivers. The short-term effects are then estimated in `B` (with `adstock=` for carryover). Nothing is counted twice and no MMM is needed.

- **Pros:** simpler. One model. No dependence on how the MMM built its base.
- **Cons:** the short-term model is cruder than a full MMM. Promotions, price and distribution need their own drivers.
- **When to prefer it:** there is no trustworthy base, or the base fails the smoothness check and cannot be rebuilt.

Recommend the user's existing base when it passes the checks, the UCM recipe when it does not, and total sales when neither is possible.
