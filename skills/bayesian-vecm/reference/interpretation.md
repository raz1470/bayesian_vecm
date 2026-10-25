# Interpretation: from outputs to business statements

Every output gets one plain sentence with its range. Use 90% intervals. Templates for the sentences are given with each output.

## The long-run ties (β) and who leads (α)

```python
print(model.summary(var_names=["alpha", "beta"]))
causality = model.granger_causality()
print(causality)
```

- **β:** "In the long run, sales move about 1.25% for every 1% move in consideration."
- **α:** "When sales fall behind consideration, they close about a quarter of the gap each week. Half the gap is gone in about 2.5 weeks."  Half-life in weeks: `log(0.5) / log(1 + alpha)` for a negative α on the series' own tie.
  When sales are base sales built from total sales, give α and the half-life as rough only: noise in the base that the brand system does not explain makes α look faster (see `pitfalls.md`). The long-run return is the robust number.
- **Weak exogeneity:** "Awareness does not adjust to the others: it leads. Changes to awareness are where lasting change enters the system."

## What drives what

`model.granger_causality()` gives four tables.

| Table | Plain question | Sentence |
|---|---|---|
| `short_run` | Does last week's change in X help predict this week's change in Y? | "Past awareness helps predict consideration (evidence 0.99)." |
| `long_run` | Does the level of X pull on Y? | "Consideration's level pulls sales towards it." |
| `weak_exogeneity` | Does the series adjust at all? | "Awareness leads." |
| `exog` | Does a driver move a metric? | "Brand TV moves awareness. Its effect on consideration is all through awareness" (from `direct` and `via_links` when the model has links). |

Always add: "These are prediction results, not proof of cause." The evidence score says **whether** there is a link, not how big it is. Use the return per £ for size.

## How long a shock lasts

```python
irf = model.irf(52, method="girf")
path = irf.sel(response_variable="sales", shock_variable="awareness")
```

"After a typical one-week jump in awareness, sales are about 1.1% higher a year later (between 0.9% and 1.2%). The lift does not fade, because sales are tied to awareness through consideration."

## The return per £

```python
WEEKS = {"1 year": 52, "5 years": 260}
mult = model.exog_irf(steps=260)  # log change per unit, by week
sales = mult.sel(response_variable="sales", exog_variable="brand_tv")
sales = sales.stack(sample=("chain", "draw")).transpose("sample", "horizon").values
recent_sales = float(np.exp(endog["sales"].iloc[-52:]).mean())  # £ a week
cum = (np.expm1(sales) * recent_sales).cumsum(axis=1)
cost_of_one_unit = 1_000.0  # £ per unit of brand_tv
for label, w in WEEKS.items():
    per_pound = cum[:, w - 1] / cost_of_one_unit
    print(label, np.percentile(per_pound, [5, 50, 95]).round(2))
```

"One more £1 of brand TV for one week returns about £2.20 in the next year and £12.50 over five years (90% range £9.30 to £16.00). Less than a fifth arrives in the first year, which is why a short-term MMM undervalues it."

- The unit cost must match the units of the driver. Ask the user.
- With saturation, this is the return at average spend. Use `exog_irf(steps, at={"brand_tv": level})` for another level.
- Report the year-1 share: it is the bridge to the MMM figure the business already knows.

## The response curve (with `saturation=`)

```python
levels = np.linspace(0, 2.5, 11) * exog_raw["brand_tv"].mean()
curve = model.saturation_curve("brand_tv", levels)  # effect on weekly growth
```

For a curve in £, compare forecasts with the spend held at each level against average spend, using the same `random_seed` so the noise cancels. Guide 05 in the repo has the full code (`fit_revenue`).

Sentences:

- "Near current spend the next £1 returns about £10 over five years. At twice current spend it returns under £1."
- "No week in the data had spend below £420, so the figure for switching off is a guess at the shape of the curve. Its range is wide: £400k to £440k of lost revenue."

## Same-week links

```python
print(model.same_week_links(ci_prob=0.9))
```

"In the same week, a 1% rise in awareness growth comes with about a 0.17% rise in consideration growth (between 0.06% and 0.29%). This relies on brand TV, PR and sponsorship reaching consideration only through awareness."

Always name the exclusion the link rests on.

## Forecasts

```python
fc = model.sample_posterior_predictive(52, exog_future=future_drivers, random_seed=0)
```

- `exog_future` needs every driver for every future week, in the same units as in the fit (deviations, raw spend for saturated channels, Fourier terms if used).
- For a scenario, forecast the plan and a baseline with the same `random_seed` and compare. Report the difference, not the levels.

## How to present uncertainty

- "About £2.20, very likely between £1.60 and £2.80" beats "£2.20 ± 0.60".
- Round to what the interval supports. Do not report £2.2187.
- When an interval includes zero, say "no clear effect", not "no effect".
