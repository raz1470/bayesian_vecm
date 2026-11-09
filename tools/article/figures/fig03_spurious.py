"""Section 4 chart: two unrelated random walks.

Also prints how often regression calls such pairs related.
"""

import numpy as np
import statsmodels.api as sm
from _spec import r, save

rng = np.random.default_rng(2026)
T = 260
sig = 0
for _ in range(1000):
    a, b = rng.standard_normal((2, T)).cumsum(axis=1)
    if sm.OLS(a, sm.add_constant(b)).fit().pvalues[1] < 0.05:
        sig += 1
print(f"share significant at 5%: {sig / 10:.0f}%")

# One illustrative pair: pick the first seed whose slope is clearly 'significant'.
for seed in range(100):
    pair_rng = np.random.default_rng(seed)
    a, b = pair_rng.standard_normal((2, T)).cumsum(axis=1)
    fit = sm.OLS(a, sm.add_constant(b)).fit()
    if fit.rsquared > 0.4 and fit.params[1] > 0:
        break
print(f"seed {seed}: slope {fit.params[1]:.2f}, t {fit.tvalues[1]:.1f}, R2 {fit.rsquared:.2f}")
resid = fit.resid

w = np.arange(len(a))
XA = {
    "min": 0,
    "max": T,
    "ticks": [[52 * k, str(k)] for k in range(6)],
    "label": "Years",
    "hover": "week",
}
save(
    "fig03_spurious",
    {
        "title": "Regression calls them related. They are not.",
        "legend": [
            {"label": "Random walk A", "color": "blue", "kind": "line"},
            {"label": "Random walk B", "color": "orange", "kind": "line"},
            {"label": "Leftover gap", "color": "ink", "kind": "line"},
        ],
        "height": 230,
        "minPanel": 260,
        "panels": [
            {
                "title": "Two walks, built independently",
                "x": XA,
                "y": {"ticks": [], "label": ""},
                "layers": [
                    {
                        "type": "line",
                        "x": r(w),
                        "y": r(a),
                        "color": "blue",
                        "width": 1.6,
                        "name": "Walk A",
                    },
                    {
                        "type": "line",
                        "x": r(w),
                        "y": r(b),
                        "color": "orange",
                        "width": 1.6,
                        "name": "Walk B",
                    },
                ],
            },
            {
                "title": "What regression leaves over: no pull back to zero",
                "x": XA,
                "y": {"ticks": [], "label": "", "zero": True},
                "layers": [
                    {"type": "hline", "y": 0, "dash": True},
                    {
                        "type": "line",
                        "x": r(w),
                        "y": r(resid),
                        "color": "ink",
                        "width": 1.5,
                        "name": "Leftover",
                    },
                ],
            },
        ],
    },
)
