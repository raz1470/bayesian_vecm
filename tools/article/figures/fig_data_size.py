"""Section 8 charts: how the rank test and the 5-year return hold up as data shrinks.

Reads ``tools/data_size_check_results.csv`` (20 seeds; weekly or monthly, 5 or
3 years; large, medium and small marketing effects). Left: the 5-year return on
brand spend as a ratio to the truth, the median across seeds of each fit's
median and 89% interval ends (``fig_data_size``). The share of data sets where the
Johansen trace test finds the two ties, pooled over effect sizes because they
barely differ (``fig_data_rank``).
"""

from pathlib import Path

import pandas as pd
from _spec import r, save

CSV = Path(__file__).resolve().parents[2] / "data_size_check_results.csv"
ORDER = ["weekly 5y", "weekly 3y", "monthly 5y", "monthly 3y"]
LABELS = ["Weekly 5y", "Weekly 3y", "Monthly 5y", "Monthly 3y"]
EFFECTS = {"large": "blue", "medium": "aqua", "small": "orange"}
OFFSET = {"large": -0.24, "medium": 0.0, "small": 0.24}

d = pd.read_csv(CSV)
d = d[d["window"] == "5 years"].copy()
for c in ("median", "lower", "upper"):
    d[c] = d[c] / d["truth"]
g = d.groupby(["effect", "data"]).agg(
    mid=("median", "median"),
    lo=("lower", "median"),
    hi=("upper", "median"),
    cover=("covered", "mean"),
    rank2=("rank", lambda s: (s == 2).mean()),
    rank_low=("rank", lambda s: (s < 2).mean()),
    n=("n_periods", "first"),
    seeds=("seed", "nunique"),
)
g = g.reindex(pd.MultiIndex.from_product([list(EFFECTS), ORDER]))
print(g.round(2).to_string())

X = list(range(len(ORDER)))
XAXIS = {
    "min": -0.6,
    "max": len(ORDER) - 0.4,
    "ticks": list(zip(X, LABELS, strict=True)),
    "hover": False,
}
TOP = max(2.0, float(g["hi"].max()) * 1.05)
BOTTOM = min(0.0, float(g["lo"].min()) * 1.05)


def tips(effect: str) -> list[str]:
    return [
        f"{lab}, {effect} effect: {row.n} periods. Typical interval {row.lo:.2f} to "
        f"{row.hi:.2f} of the truth; held the truth in {row.cover:.0%} of data sets"
        for lab, row in zip(LABELS, g.loc[effect].itertuples(), strict=True)
    ]


rank = d.groupby("data")["rank"].apply(lambda s: (s == 2).mean()).loc[ORDER]
print(rank.round(2).to_string())

save(
    "fig_data_rank",
    {
        "title": "Five years find the ties. Three years often miss them.",
        "height": 260,
        "minPanel": 300,
        "panels": [
            {
                "title": "Rank test finds the two ties",
                "x": XAXIS,
                "y": {"min": 0, "max": 100, "fmt": "pct_plain", "label": "% of data sets"},
                "layers": [
                    {
                        "type": "bars",
                        "x": X,
                        "y": r(100 * rank),
                        "color": "blue",
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
                        for x, v in zip(X, rank, strict=True)
                    ],
                ],
            }
        ],
    },
)

save(
    "fig_data_size",
    {
        "title": "Smaller effects and fewer data points widen the interval",
        "legend": [
            {"label": f"{e.capitalize()} effect", "color": c, "kind": "bar"}
            for e, c in EFFECTS.items()
        ]
        + [{"label": "Truth", "color": "ink", "kind": "dash"}],
        "height": 320,
        "minPanel": 300,
        "panels": [
            {
                "title": "5-year return on brand spend, typical 89% interval",
                "x": XAXIS,
                "y": {"min": round(BOTTOM, 1), "max": round(TOP, 1), "label": "Model ÷ truth"},
                "layers": [
                    {"type": "hline", "y": 0},
                    {"type": "hline", "y": 1, "dash": True},
                    *[
                        {
                            "type": "interval",
                            "x": X,
                            "offset": OFFSET[e],
                            "lo": r(g.loc[e, "lo"]),
                            "mid": r(g.loc[e, "mid"]),
                            "hi": r(g.loc[e, "hi"]),
                            "color": c,
                            "tips": tips(e),
                        }
                        for e, c in EFFECTS.items()
                    ],
                ],
            }
        ],
    },
)
