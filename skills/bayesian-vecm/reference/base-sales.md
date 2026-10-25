# Base sales

The long-term model often runs on **base sales**: sales without the short-term effect of marketing, usually taken from a media mix model (MMM). Cain (2022) does this. It is also where the work most often goes wrong before the VECM sees any data.

## What base sales has to be

The VECM explains the slow movement in sales through the brand metrics. So the sales series must still contain that slow movement, and nothing that is counted elsewhere.

1. **Actual sales minus the short-term marketing and promotion contributions**, using the MMM's own adstock. Nothing else removed.
2. **The noise kept in.** It is data minus contributions, not a fitted curve. A smooth base makes the model look surer than it is and distorts how the shared shocks split.
3. **The level and the long-run trend kept in.** That is the brand effect being measured.
4. **Seasonality handled one way for every series.** Either kept in with seasonal terms as drivers, or removed by the same method from all series. When only sales are seasonal, the best route so far is to keep it in the base and give the sales equation the seasonal terms (see below). Subtracting a UCM's seasonal estimate adds a slow error to the base, because over a few years the level and an annual cycle are hard to tell apart. Treat this as provisional and say so to the user: it has been tested on smooth annual patterns only, not on holiday spikes or seasonal brand metrics.
5. **The same frequency, dates and breaks** as the brand metrics.

If short-term media is left in, the VECM counts it again. If the trend is taken out, there is nothing left for the VECM to find.

## Approaches to question

None of these is wrong for an MMM. But the base they produce answers "what is not short-term marketing", while the VECM needs "everything that is not short-term marketing, unsmoothed". Ask the user how their base was built and check it against this table.

| How the base was made | The problem |
|---|---|
| A time-varying intercept in a Bayesian MMM (Gaussian process, random walk, splines: Meridian, pymc-marketing and others) | It absorbs slow brand-driven growth as "baseline", and it is a smooth curve. In simulation (`tools/mmm_trend_check.py`, PyMC-Marketing HSGP at lengthscales of 26, 78 and 156 weeks) the long-run return still came out right, but `alpha` was about 93% too slow, the sales shocks 85% too small and 4 in 5 fits found sales leading the brand metrics. The lengthscale did not fix it. Use the MMM's total minus its contributions instead. |
| Prophet or Robyn trend | Changepoints and smoothing are imposed. Using the trend component as base removes the noise. |
| A linear or polynomial time trend | Assumes growth is fixed, so the VECM cannot explain it with brand. Leaves spurious structure in the residual. |
| Hodrick-Prescott filter | Known to create cycles that are not in the data (Hamilton 2018). Poor at the ends of the sample. |
| Moving average | Shifts timing, distorts the ends and smooths away the noise. |
| UCM smoothed level | Gives the right long-run return, but it is two-sided: each week uses later weeks. When the UCM finds real noise, `alpha` comes out too slow and direction tests find sales leading the brand metrics (7 of 10 simulated data sets at 5% noise). Use the recipe below instead. |
| MMM intercept only (a constant) | Not a base at all for this purpose. |

What to say to the user: "Your MMM's base is fine for the MMM. For this model we need sales with only the short-term marketing taken out, and nothing smoothed."

## Building base sales with a UCM

When there is no usable base, build one with an unobserved components model from `statsmodels` (`pip install statsmodels`).

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Short-term drivers: performance media, promotions, price, and brand
# spend's same-week hit. Linear and same-week only: a UCM cannot fit
# carryover or saturation (that needs a Bayesian MMM).
short_term = build_short_term_drivers(raw)  # (T, m) DataFrame, columns named
period = 365.25 / 7  # weekly data

ucm = UnobservedComponents(
    raw["sales"],  # in £ if the parts add up in £; use np.log(...) if they multiply
    level="lldtrend",  # random-walk level, drift, noise
    freq_seasonal=[{"period": period, "harmonics": 2}],
    stochastic_freq_seasonal=[False],  # a fixed annual pattern
    exog=short_term,
).fit(disp=False)

coefs = ucm.params[[f"beta.{c}" for c in short_term.columns]]
base = raw["sales"] - short_term @ coefs.to_numpy()  # seasonality and noise stay in
log_base = np.log(base)

# The seasonal swing stays in the base, so the VECM's sales equation gets the
# harmonics, as weekly changes, and the brand metric equations do not.
angle = 2 * np.pi * np.arange(len(raw)) / period
harmonics = pd.DataFrame(
    {f"{f.__name__}{h}": f(h * angle) for h in (1, 2) for f in (np.sin, np.cos)},
    index=raw.index,
).diff()
exog_full = pd.concat([exog, harmonics.add_prefix("d_")], axis=1).iloc[1:]
excluded = {m: list(harmonics.add_prefix("d_").columns) for m in brand_metric_names}
# BayesianVECM(..., excluded=excluded).fit(endog.iloc[1:], exog=exog_full)
```

Rules:

- **The level must be stochastic** (`"lldtrend"`: a random-walk level with drift and noise), never `"dtrend"` (a fixed trend). A fixed trend imposes the growth the VECM should explain.
- **Take only the driver contributions out.** Keep the level and the irregular (noise). Keep the seasonal part in and model it in the sales equation, unless every series is deseasonalised the same way.
- **Do not use `ucm.level["smoothed"]` as base sales.** It is a two-sided smoothed curve: the return comes out right but the short-run dynamics and the direction tests do not.
- **Short decays for the media adstock** (a few weeks). Long decays push brand effects into the short-term contribution.
- Print `ucm.summary()` and check the media coefficients have sensible signs before using the base.

Parameter names follow `statsmodels`' `beta.<column>` convention. Check `ucm.params.index` if they differ.

## What the simulation shows

Guide 06 (`notebooks/guides/06_base_sales_from_total_sales.ipynb`, `tools/ucm_base_check.py`) builds total sales from a known base plus short-term effects, seasonality and weekly noise outside the brand system. It runs 10 data sets at 0%, 2% and 5% noise, for 130 fits.

- **The long-run return on brand spend hardly depends on the base.** The UCM base, the smoothed level and total sales all came within 6% of the truth, with 90% to 100% coverage. Cointegration with consideration carries it.
- **The short-run quantities do.** Noise outside the brand system goes into the sales shocks (twice too large at 2% noise) and makes `alpha` look too fast. Do not report `alpha`, half-lives or shock sizes as brand facts when the base carries such noise.
- **Keeping the seasonality in looks better, provisionally.** Subtracting the UCM's seasonal estimate made the week-4 lift 18% to 28% too low at low noise; keeping it in fixed that. `tools/seasonality_check.py` held this up with 1 to 3 harmonics (truth 2), with 5 and 8 years of data and with a drifting seasonal pattern (week-4 lift -11% at strong drift against -29% to -39%). At 3 years both routes were noisy. With more years the gap closes: removing the seasonal estimate was -6% at 8 years.
- **The smoothed level is the riskiest choice** when the UCM's noise variance is not zero: wrong dynamics and false direction findings. When it is zero the smoothed level and the recipe give the same series.
- **Performance spend that follows demand** (`tools/demand_feedback_check.py`). When search spend rose with consideration growth (correlation up to 0.7), the UCM's search coefficient came out 3% to 10% too high. The long-run return moved by 1% or less and the base still cointegrated with consideration in 9 of 10 data sets. Here search is 7% of sales; a channel that is a much larger share of sales would move more demand out of the base. Check that the base still cointegrates with consideration (`select_coint_rank`) before going further.
- **Seasonality kept in the base confuses the rank test.** `select_coint_rank` has no seasonal terms, so with the seasonal swing still in base sales it found the true rank in only 2 to 4 of 10 data sets (8 of 10 with the seasonal part removed). Run the rank test on a deseasonalised copy and fit the VECM on the base with seasonality in.
- **With carryover and saturation, take the base from a Bayesian MMM** (`tools/mmm_trend_check.py`, 5 data sets, PyMC-Marketing). Total minus the MMM's channel and control contributions behaved like the UCM recipe: return within 3%, and the week-4 lift within 4% with the seasonality kept in. The MMM's own trend failed as in the table above. Total sales straight into the VECM, with `adstock=` on the weekly changes of the channels, also got the return within 3% and no false direction findings, but the lift built too fast (week 4 +30%) because it cannot fit saturation.
- **Not yet tested:** brand metrics that are seasonal too, and seasonality that is not smooth (holiday spikes).

## Category demand and competitors

Not yet tested by simulation. This is reasoning from one real-data case, so present it as a hypothesis to check, not a finding.

The pattern: interest in the whole category rises (a new kind of product catches on). It lifts the brand's consideration and its competitors' together. Performance spend, especially search, follows the same demand through bids and budgets. Then:

- **The performance effect looks large.** Performance spend appears to drive consideration, but both follow category demand.
- **The base can lose its tie to the brand metrics.** The UCM or MMM credits performance with demand-driven sales and subtracts them. The part of sales that moves with consideration goes with them.
- **A series may be missing.** Sales can depend on category demand times the brand's share. The brand's consideration alone mixes the two, so it may not tie to sales on its own.

Signs: consideration moving with a competitor's, a large performance effect on a brand metric, or base sales and consideration with no tie.

What to try, cheapest first:

1. Run the rank test on total sales (no UCM) against consideration, with and without a competitor's consideration or a category series. Deseasonalise first. This separates "the base build caused it" from "the category is missing".
2. Put the category in the system: a category demand series (category search interest), or a competitor's consideration, in `endog`.
3. Re-test performance spend with the category in. Use `granger_pretest` for its direction: does consideration help predict spend?
4. In the UCM, enter category demand as a regressor but **keep its contribution in the base**. Take out only what the VECM should not explain: performance, promotions and brand spend's same-week hit.
5. Consider total sales straight into the VECM, with performance spend as a driver.

A competitor's brand metric may itself respond to the brand's spend. Treat it as a category measure and check it.

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
