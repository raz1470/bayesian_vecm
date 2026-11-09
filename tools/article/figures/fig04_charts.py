"""Section 5 chart: base sales and consideration trend together; the gap between them does not.

Also prints the Johansen trace test the text quotes (rank 2)."""

import numpy as np
from _spec import r, save

from bayesian_vecm import select_coint_rank, simulate_brand_data

data = simulate_brand_data(exog_unit_root=("interest_rate",))
e = data.endog
B = 1.272
s = 100 * (e["organic_sales"] - e["organic_sales"].mean())
c = 100 * B * (e["brand_consideration"] - e["brand_consideration"].mean())
gap = s - c
print("gap range", round(gap.min(), 1), round(gap.max(), 1))

wk = np.arange(len(e))
lim = float(max(abs(s).max(), abs(c).max())) * 1.05
XA = {
    "min": 0,
    "max": len(e),
    "ticks": [[52 * k, str(k)] for k in range(6)],
    "label": "Years",
    "hover": "week",
}
YA = {"min": -lim, "max": lim, "ticks": "auto", "fmt": "pct", "label": "% from average (log scale)"}
save(
    "fig04_cointegration",
    {
        "title": "Sales and consideration wander together. The gap between them does not.",
        "legend": [
            {"label": "Base sales", "color": "blue", "kind": "line"},
            {"label": "1.27 × consideration", "color": "purple", "kind": "line"},  # noqa: RUF001
            {"label": "The gap", "color": "ink", "kind": "line"},
        ],
        "height": 240,
        "shareY": True,
        "minPanel": 260,
        "panels": [
            {
                "title": "Both trend over five years",
                "x": XA,
                "y": YA,
                "layers": [
                    {
                        "type": "line",
                        "x": r(wk),
                        "y": r(s),
                        "color": "blue",
                        "width": 1.6,
                        "name": "Base sales",
                    },
                    {
                        "type": "line",
                        "x": r(wk),
                        "y": r(c),
                        "color": "purple",
                        "width": 1.6,
                        "name": "1.27 × consideration",  # noqa: RUF001
                    },
                ],
            },
            {
                "title": "The gap: within 8% of zero throughout",
                "x": XA,
                "y": YA,
                "layers": [
                    {"type": "hspan", "lo": -8, "hi": 8},
                    {"type": "hline", "y": 0, "dash": True},
                    {
                        "type": "line",
                        "x": r(wk),
                        "y": r(gap),
                        "color": "ink",
                        "width": 1.5,
                        "name": "Gap",
                    },
                ],
            },
        ],
    },
)

res = select_coint_rank(e, k_ar_diff=1, det_order=0)
stats, crit = np.asarray(res.test_stats), np.asarray(res.crit_vals)
print("trace", stats.round(1), "crit", crit.round(1), "rank", res.rank)
