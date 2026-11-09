"""Section 2 chart: a long adstock of brand spend fits a trend it has nothing to do with.

Five years of weekly brand spend, flighted and growing slowly as budgets often
do, and its long adstock (decay 0.97). Separately, a baseline built as a random
walk with upward drift and no link to spend. A regression of the baseline on
the adstock still finds a strong, significant slope.
"""

import numpy as np
import statsmodels.api as sm
from _spec import r, save

rng = np.random.default_rng(3)
W = 260
t = np.arange(W)
on = (t % 13) < 8  # eight weeks on, five off
spend = on * (1.0 + 0.004 * t) * np.exp(rng.normal(0, 0.2, W))
ad = np.zeros(W)
for k in range(1, W):
    ad[k] = 0.97 * ad[k - 1] + 0.03 * spend[k]
base = np.cumsum(0.12 + rng.normal(0, 0.6, W)) + 100.0  # no link to spend

ols = sm.OLS(base, sm.add_constant(ad)).fit()
print(f"slope t = {ols.tvalues[1]:.1f}, R2 = {ols.rsquared:.2f}")

lo, hi = float(min(base.min(), ols.fittedvalues.min())), float(base.max())
pad = 0.08 * (hi - lo)
bar_scale = (hi - lo) / (spend.max() * 3.0)  # spend bars fill the bottom third
save(
    "fig02_adstock_trend",
    {
        "title": "A long adstock fits the trend, not the brand",
        "legend": [
            {"label": "Weekly brand spend", "color": "grey", "kind": "bar"},
            {
                "label": "Baseline sales, built with no link to spend",
                "color": "ink",
                "kind": "line",
            },
            {
                "label": "Long adstock of brand spend, as a regression fits it"
                f" (slope t-statistic {ols.tvalues[1]:.0f})",
                "color": "orange",
                "kind": "line",
            },
        ],
        "height": 280,
        "panels": [
            {
                "x": {
                    "min": 0,
                    "max": W,
                    "ticks": [[52 * k, str(k)] for k in range(6)],
                    "label": "Years",
                    "hover": "week",
                },
                "y": {"min": lo - pad, "max": hi + pad, "ticks": [], "label": ""},
                "layers": [
                    {
                        "type": "bars",
                        "x": r(t),
                        "y": r(lo - pad + spend * bar_scale),
                        "base": lo - pad,
                        "color": "grey",
                    },
                    {
                        "type": "line",
                        "x": r(t),
                        "y": r(base),
                        "color": "ink",
                        "width": 1.8,
                        "name": "Baseline",
                    },
                    {
                        "type": "line",
                        "x": r(t),
                        "y": r(ols.fittedvalues),
                        "color": "orange",
                        "width": 2.4,
                        "name": "Fitted adstock",
                    },
                ],
            }
        ],
    },
)
