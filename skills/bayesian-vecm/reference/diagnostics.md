# Diagnostics: check before reading any result

Run every check after every fit. Report them to the user in one line: "The fit is healthy: no warnings from the sampler and the residuals look like noise." Only then read the results.

## 1. The sampler

```python
summary = model.summary().apply(pd.to_numeric, errors="coerce")
print("divergences", int(model.idata_.sample_stats["diverging"].sum()))
print("max r-hat  ", summary["r_hat"].max())
print("min ESS    ", summary["ess_bulk"].min())
```

| Check | Good | Tolerable for a first look | What it means |
|---|---|---|---|
| Divergences | 0 | a handful with a horseshoe prior | The sampler hit a region it could not explore. Results near it are unreliable. |
| Max r-hat | 1.01 or below | up to about 1.03 | The chains agree with each other. |
| Min bulk ESS | several hundred | 100 or more | Enough independent draws for stable intervals. |

Fixes, in order:

1. Raise `target_accept` (0.95, then 0.99).
2. More `tune` and `draws` (2,000 each).
3. Check the data: a series with a near-zero variance, an unhandled break, a driver on a very different scale.
4. Simplify: fewer lags, drop saturation on a channel with little spend variation, drop a weak same-week link.
5. If it still fails: stop and say the model needs a specialist to look at it.

## 2. Same-week links (if any)

`fit()` warns about weak or disagreeing instruments. Read `model.same_week_links()`: first-stage F above 10 and Sargan p above 0.05. If not, see `options.md`.

## 3. Residuals

```python
print(model.test_whiteness(lags=10))  # leftover autocorrelation
print(model.test_normality())
```

- Autocorrelated residuals point to too few lags or a missing driver (often seasonality).
- Non-normal residuals usually mean outliers or breaks. Plot the residuals and look for the weeks responsible.

## 4. The long-run ties make sense

```python
print(model.summary(var_names=["alpha", "beta"]))
```

- **β:** each tie should read sensibly. "Sales about 1.25 times consideration in logs" is plausible. A negative tie between sales and awareness is not.
- **α:** at least one series should adjust to each tie (α clearly away from zero). If none does, the rank is probably too high.
- **The tie over time:** plot `endog.values @ beta_mean`. Each line should wander around a level, not trend. A trending line means a wrong rank or a series in the wrong group.

## 5. Out-of-sample check (for anything the business will act on)

Refit without the last year and forecast it with the drivers that actually ran. The forecast should beat a random walk and stay inside its interval. Guide 01 in the repo has the code (section 9c).

## 6. Sensitivity

Before the write-up, refit with the main judgement calls changed one at a time:

- the next rank up or down;
- one more lag;
- with and without adstock or saturation, if they were a close call.

Report the return per £ under each. If it moves less than its own interval, say the answer is robust to that choice. If it moves more, report the range and the reason.
