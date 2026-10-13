# The Granger pretest

`granger_pretest` asks which series help predict which before any model is
fitted. Run it on the levels of the brand metrics and the drivers. It does
not need the cointegration rank.

```python
from bayesian_vecm import granger_pretest

result = granger_pretest(endog, exog[["brand_spend", "interest_rate", "consumer_confidence"]])
print(result)
```

The figures below come from `tools/granger_check.py`. It runs the pretest on
200 seeds of `simulate_brand_data()` with brand spend and the two macro series
as drivers. The truth is a chain: spend moves awareness and awareness moves
consideration. Consideration moves sales. The macro series act on
consideration. Nothing feeds back into the drivers. Every driver effect lands
in the same week.

## What it reports

| Table | Question | Why it matters |
|---|---|---|
| Drivers | Does past X help predict the brand metric? | First look at which drivers belong in `exog` |
| Delay | Does past X still help once this period's drivers are known? | A yes means same-period `exog` misses part of the effect |
| Links | Does past awareness help predict consideration? And so on | Which brand metrics belong in `endog` and how they connect |
| Feedback | Does the past brand metric help predict the driver? | A yes means budgets follow performance and `B` is biased |
| Same period | Do the VAR residuals of two series move together? | Effects inside a period that no lag can show |

## The method

Toda and Yamamoto (1995). Fit a VAR in levels with one lag more than the
order you test. The extra lag absorbs the unit roots. The F test on the
remaining lags is then valid for stationary and cointegrated series alike.

The lag order comes from the Hannan-Quinn criterion. It is capped so that
the test regression keeps at least half its rows as residual degrees of
freedom. Without the cap 60 months of six series picked 7 lags in 141 of 200
seeds. The test then had almost no power.

Monthly data with a yearly pattern needs `seasonal=12`.

## How often it finds the truth

Share of 200 seeds with p below 0.05. True links are in bold. Every other row
should sit near 5%.

| Test | 260 weeks | 60 months |
|---|---|---|
| **Drivers: spend -> awareness** | 72% | 64% |
| **Drivers: interest rate -> consideration** | 95% | 56% |
| **Drivers: confidence -> consideration** | 93% | 52% |
| Drivers: spend -> consideration | 6% | 21% |
| **Links: awareness -> consideration** | 100% | 82% |
| **Links: consideration -> sales** | 100% | 44% |
| Links: reverse directions | 4% to 10% | 4% |
| Feedback: any metric -> any driver | 3% to 7% | 3% to 10% |
| Delay: spend -> awareness | 13% | 56% |
| Delay: macro -> consideration | 4% to 5% | 26% to 31% |

Lags chosen: 1 in 188 weekly seeds and 198 monthly seeds.

## Reading the results

- **Feedback holds its size.** Without feedback in the data the test flags
  one about 5% of the time at both sample shapes. A clear feedback result on
  real data is worth taking seriously.
- **A driver hit can come from persistence.** The simulated spend acts in
  the same week only. Past spend still predicts awareness because spend
  persists from week to week. Check the delay table before concluding that
  spend acts with a lag.
- **Monthly data creates a delay.** Spend late in a month moves awareness in
  the next month's average. The delay test picks this up in 56% of monthly
  seeds even though the weekly effect has no lag. With monthly data expect
  lagged `exog` to matter.
- **60 months finds the strong links and misses weak ones.** Awareness to
  consideration is found in most seeds. Consideration to sales is found in
  fewer than half. A miss is not evidence that a link is absent.
- **Same-period links are found reliably** where the effects are strong.
  Spend and awareness move together in every seed at both sample shapes. The
  direction cannot be read from this table.

## What it cannot show

- **That spend causes awareness.** Granger tests are about prediction. The
  size and value of the effect come from `B` and `exog_irf()` after the fit.
- **Same-period feedback.** If budgets react within the period to that
  period's sales, no lag can reveal it. State how budgets are set as an
  assumption.
