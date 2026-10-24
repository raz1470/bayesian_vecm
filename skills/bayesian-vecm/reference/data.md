# Data: what is needed and how to prepare it

## What the model needs

| Item | Minimum | Better | Why |
|---|---|---|---|
| Frequency | Monthly | Weekly | Weekly keeps the short run and same-week links. Monthly blurs them (see below). |
| Length | 4 years weekly or 60 months | 5 years or more | The rank test finds the right number of ties in about 80% of cases at 260 weeks and under half at 156. |
| Brand metrics | One, plus sales | Awareness and consideration, plus sales | The funnel is what the model traces. |
| Drivers | Brand spend | Every brand activity: TV, PR, sponsorship, social, plus price and macro | Leaving out a channel that moves with the others biases the ones left in. On monthly simulated data, leaving out PR and sponsorship put brand spend 40% too high. |
| Units | Known | Spend in £, reach in people | The return per £ needs the cost of one unit. |
| History notes | Known breaks | A dated list of changes | A tracker supplier change or a website rebuild looks like a brand effect. |

Ask the user for anything missing with `templates/data-request.md`.

## Which series go where

**Brand metrics and sales (`endog`)** trend over years and are expected to be tied to each other. Typical examples are sales (or base sales), awareness and consideration. Preference and share of search can belong here too.

**Drivers (`exog`)** are set from outside the brand system and do not respond to it within the period. Typical examples are media spend or reach by channel and price. Distribution, macro series and seasonality terms belong here too.

Questions that decide the borderline cases:

- **Is the budget set by looking at recent sales or brand health?** Then spend responds to the system and its effect is biased. The pretest's feedback table checks for this in the data. A "no" from the business is still worth recording.
- **Does the metric trend?** A brand metric that does not trend cannot share a long-run tie. Check with `check_stationarity`.
- **Is it a cause or an outcome?** Web visits and branded search often respond to the same shocks as sales. Treat them as brand metrics (`endog`), not drivers. `research/funnel-metrics.md` covers how to judge a new metric.

## Preparing the series

```python
import numpy as np
import pandas as pd

raw = pd.read_csv("brand_data.csv", index_col="week", parse_dates=True)

# Brand metrics and sales: logs, so effects read as % changes.
endog = np.log(raw[["sales", "awareness", "consideration"]])

# Drivers: deviations from their average, so 0 means "a normal week".
channels = ["brand_tv", "pr_reach", "sponsorship"]
macro = ["interest_rate", "consumer_confidence"]
exog = raw[channels + macro].copy()
exog[channels] = exog[channels] - exog[channels].mean()
```

Then:

- **A driver that trends** (most interest rates, price indices): enter its weekly change. `check_stationarity` reports which drivers need this.

  ```python
  exog["interest_rate"] = exog["interest_rate"].diff()
  endog, exog = endog.iloc[1:], exog.iloc[1:]
  ```

- **Channels that will saturate** (`saturation=`): keep them as **raw, non-negative spend**, not deviations. The curve needs real spend levels.
- **Awareness and consideration as percentages:** use the log of the percentage. Check that they are not near 0% or 100%, where logs behave badly.
- **Seasonality:** the model has no seasonal term. Either add Fourier terms as drivers, or deseasonalise every series the same way (never just one).

  ```python
  t = np.arange(len(exog))
  for k in (1, 2, 3):
      exog[f"sin{k}"] = np.sin(2 * np.pi * k * t / 52.18)
      exog[f"cos{k}"] = np.cos(2 * np.pi * k * t / 52.18)
  ```

  A forecast then needs the same terms for the future weeks.

- **Breaks:** a known one-off (a launch, COVID lockdowns) can enter as a dummy driver. A permanent change in how a metric is measured needs splicing or a shorter sample. Discuss with the user.
- **Missing weeks:** the model needs a complete grid. Interpolate short gaps and note them in the decision log. Long gaps mean a shorter sample.

## Weekly against monthly

Monthly data works for the long run but changes what some numbers mean:

- **α is not a half-life.** Summing weeks into months distorts the adjustment speeds. β and the long-run effect survive aggregation.
- **A same-month link includes lagged weekly effects.** Awareness that moves consideration a week later looks like a same-month link. On simulated data with no weekly link the monthly link was about 0.32.
- **Spend late in a month moves next month's average.** Use `exog_lags=1` with monthly data.
- **Instruments get weak.** The same-week test often reports F below 10 on 60 months.

If both are available, use weekly.
