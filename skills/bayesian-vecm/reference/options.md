# Model options: what to choose and when

Each option is a decision. Recommend one, explain it in a sentence and record it in the decision log.

## Lags (`k_ar_diff`)

How many past weeks of growth enter the model. Pick the order of a VAR in levels with the Hannan-Quinn criterion and subtract one.

```python
from statsmodels.tsa.api import VAR

orders = VAR(endog, exog=exog).select_order(maxlags=8, trend="c")
k_ar_diff = max(orders.selected_orders["hqic"] - 1, 1)
```

- HQ picked the true order in at least 94% of simulated data sets. AIC errs long and BIC errs short.
- Keep at least 1. With a few hundred weeks, one lag too many costs less than one too few.
- If unsure, set it one or two higher and shrink with a horseshoe prior: `priors={"Gamma": {"dist": "Horseshoe"}}` with `target_accept=0.99`.

Tell the user: "This sets how many past weeks the model looks at for momentum. The data suggests one."

## Rank (`coint_rank`)

How many long-run ties.

```python
from bayesian_vecm import select_coint_rank

result = select_coint_rank(endog, k_ar_diff=k_ar_diff, det_order=0)
print(result)  # stop at the first row that does not reject
```

- Check against the funnel. K series sharing one underlying trend give K - 1 ties.
- With under four years of data or a trace statistic close to its critical value, fit the next rank as well and compare the return per £. If it barely moves, the choice matters less.
- The model needs at least 1. If the test says 0, the series are not tied and a VECM is the wrong tool. Say so.

## Deterministic terms (`deterministic`)

| The data | Use |
|---|---|
| Series grow over time (most brands) | `"co"` |
| No growth, the ties settle at a fixed gap | `"ci"` |

The default in the class is `"n"` (none). Always set it explicitly.

## Carryover: `adstock=[channels]`

Use when spend keeps working for weeks after it runs, which is the norm for brand media.

```python
model = BayesianVECM(
    k_ar_diff=1,
    coint_rank=2,
    deterministic="co",
    adstock=["brand_tv", "pr_reach", "sponsorship"],
    adstock_max_lag=8,
)
```

- One decay and one total effect per channel. The half-life is `log(0.5) / log(decay)` weeks.
- Without it a same-week model missed about half of each channel's total on simulated data. With no carryover in the data it estimates decays near zero at no cost. So when unsure, include it.
- The pretest's delay table is the evidence: spend that still predicts a metric beyond the same week.

## Diminishing returns: `saturation=[channels]`

Use when the question involves big budget moves, or spend varied a lot over the sample.

- The saturated channels must be **raw, non-negative spend**, not deviations from the average.
- `model.saturation_curve(channel, levels)` gives the effect of holding spend at each level. `exog_irf(at=...)` values one more unit at another spend level.
- On simulated data a linear fit understated the loss from switching brand spend off by 60% and overstated doubling by 120%. The saturation fit got both within about 10%.
- **Cost:** with no saturation in the data the prior still bends the curve a little. Effects far from current spend came out up to 30% too large for brand spend. Compare with a linear fit before relying on a far-away figure.

Tell the user: "This lets each extra £ buy a bit less than the last. It matters when you ask about big changes to the budget."

## Same-week links: `contemporaneous=` and `excluded=`

Use only when the question needs it (how fast the funnel moves) **and** the user can defend the exclusions.

```python
links = {"consideration": ["awareness"]}
excluded = {"consideration": ["brand_tv", "pr_reach", "sponsorship"]}

print(granger_pretest(endog, exog, contemporaneous=links, excluded=excluded).same_week)
model = BayesianVECM(..., contemporaneous=links, excluded=excluded)
```

- `excluded` lists the drivers that reach the effect **only through** the cause. They are the instruments.
- Ask the business question for each: "Could this channel change consideration other than by building awareness?" A brand awareness campaign: plausibly no. A campaign with a price or product message: probably yes, so it cannot be excluded.
- Read the pretest verdict first:
  - **"same-week link" or "no clear evidence":** fit with the link.
  - **"weak instruments" (first-stage F below 10):** leave it out. The posterior would mostly repeat the prior.
  - **"instruments disagree" (Sargan):** at least one exclusion is wrong. Revisit with the user.
- `fit()` repeats the check and warns. `same_week_links()` reports the link with its first-stage F and Sargan p.
- **A wrong exclusion biases the link and Sargan often misses it.** On simulated data a wrong exclusion turned a true 0.2 into 0.33 and Sargan flagged it 10% of the time. The business argument matters more than the test.
- On monthly data the link also contains lagged weekly effects. Say so when reporting it.
- Leaving links out is safe for the return per £: `exog_irf()` and forecasts use the total effects either way.

## Shock impulse responses: `method=`

- `model.irf(52, method="girf")` (default): no ordering needed. Use this.
- `model.irf(52, method="cholesky", order=["awareness", "consideration", "sales"])`: assumes the order. With awareness first in a funnel it matches GIRF. Show it only as a check.

## Lagged drivers: `exog_lags=`

For drivers without adstock that act with a delay, and for monthly data (`exog_lags=1`). Keep it at 1 or 2: lags of a flighted driver are highly correlated.

## Sampling settings

- `chains=4`, `draws=1000`, `tune=1000` for a final run. 2 chains are fine for a first look.
- `target_accept=0.9` by default. 0.95 with adstock, saturation or links. 0.99 with a horseshoe prior.
- Always set `random_seed` so the user can reproduce the numbers.
