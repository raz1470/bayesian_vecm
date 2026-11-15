"""Section 8 chart: how the 5-year return and the rank test hold up as data shrinks.

Reads ``tools/data_size_check_results.csv`` (20 seeds; weekly or monthly, 5 or
3 years). Left: the 5-year return on brand spend as a ratio to the truth, the
median across seeds of each fit's median and 89% interval ends. Right: the
share of seeds where the Johansen trace test finds the two ties we built in.
"""

from pathlib import Path

import pandas as pd
from _spec import r, save

CSV = Path(__file__).resolve().parents[2] / "data_size_check_results.csv"
ORDER = ["weekly 5y", "weekly 3y", "monthly 5y", "monthly 3y"]
LABELS = ["Weekly 5y", "Weekly 3y", "Monthly 5y", "Monthly 3y"]

d = pd.read_csv(CSV)
d = d[d["window"] == "5 years"].copy()
for c in ("median", "lower", "upper"):
    d[c] = d[c] / d["truth"]
g = d.groupby("data").agg(
    mid=("median", "median"),
    lo=("lower", "median"),
    hi=("upper", "median"),
    cover=("covered", "mean"),
    rank2=("rank", lambda s: (s == 2).mean()),
    n=("n_periods", "first"),
    seeds=("seed", "nunique"),
)
g = g.loc[ORDER]
print(g.round(2).to_string())

X = list(range(len(ORDER)))
XAXIS = {
    "min": -0.6,
    "max": len(ORDER) - 0.4,
    "ticks": list(zip(X, LABELS, strict=True)),
    "hover": False,
}
tips = [
    f"{lab}: {row.n} periods. Typical interval {row.lo:.2f} to {row.hi:.2f} of the truth; "
    f"held the truth in {row.cover:.0%} of data sets"
    for lab, row in zip(LABELS, g.itertuples(), strict=True)
]

save(
    "fig_data_size",
    {
        "title": "Fewer data points widen the interval. Fewer years hide the ties.",
        "legend": [
            {
                "label": "Typical 89% interval, as a share of the truth",
                "color": "blue",
                "kind": "band",
            },
            {"label": "Truth", "color": "ink", "kind": "dash"},
        ],
        "height": 280,
        "minPanel": 300,
        "panels": [
            {
                "title": "5-year return on brand spend",
                "x": XAXIS,
                "y": {"min": 0, "max": 2, "label": "Model ÷ truth"},
                "layers": [
                    {"type": "hline", "y": 1, "dash": True},
                    {
                        "type": "interval",
                        "x": X,
                        "lo": r(g["lo"]),
                        "mid": r(g["mid"]),
                        "hi": r(g["hi"]),
                        "color": "blue",
                        "tips": tips,
                    },
                ],
            },
            {
                "title": "Rank test finds the two ties",
                "x": XAXIS,
                "y": {"min": 0, "max": 100, "fmt": "pct_plain", "label": "% of data sets"},
                "layers": [
                    {
                        "type": "bars",
                        "x": X,
                        "y": r(100 * g["rank2"]),
                        "color": "grey",
                        "barWidth": 0.55,
                    },
                    *[
                        {
                            "type": "text",
                            "x": x,
                            "y": 100 * v + 4,
                            "text": f"{100 * v:.0f}%",
                            "anchor": "middle",
                        }
                        for x, v in zip(X, g["rank2"], strict=True)
                    ],
                ],
            },
        ],
    },
)
