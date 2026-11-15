"""Section 9 chart: what survives and what breaks with each way of building base sales.

Reads ``tools/mmm_trend_check_results.csv``, the four arms the article compares
(20 seeds): the true base; total sales minus the MMM's short-term
contributions; total sales straight into the VECM with the short-term drivers
alongside; and the MMM's smooth time-varying intercept. Left: the long-run
lift in sales from brand spend. Middle: how fast sales close their gap to
consideration (alpha of sales). Both as a ratio to the truth, the median across
seeds of each fit's median and 90% interval ends. Also printed for the text:
the share of fits that find sales leading the brand metrics in the long run
(evidence 0.95 or more), a link that doesn't exist.
"""

from pathlib import Path

import pandas as pd
from _spec import r, save

CSV = Path(__file__).resolve().parents[2] / "mmm_trend_check_results.csv"
ARMS = ["true_base", "mmm78_minus", "total_adstock", "mmm78_trend"]
LABELS = ["True base", "Minus MMM", "Total sales", "MMM trend"]

d = pd.read_csv(CSV)
d = d[d["arm"].isin(ARMS)].copy()
for c in ("median", "lower", "upper"):
    d[c] = d[c] / d["truth"]
# Alpha is negative, so dividing by the truth flips the interval ends.
d["lower"], d["upper"] = d[["lower", "upper"]].min(axis=1), d[["lower", "upper"]].max(axis=1)


def ranges(quantity: str) -> pd.DataFrame:
    q = d[d["quantity"] == quantity]
    g = q.groupby("arm").agg(
        mid=("median", "median"),
        lo=("lower", "median"),
        hi=("upper", "median"),
        cover=("covered", "mean"),
        seeds=("seed", "nunique"),
    )
    return g.loc[ARMS]


fits = d.drop_duplicates(["seed", "arm"]).copy()
fits["lead"] = (
    fits[["evidence_sales_to_aware_long", "evidence_sales_to_cons_long"]].max(axis=1) >= 0.95
)
lead = fits.groupby("arm")["lead"].mean().loc[ARMS]
lift, alpha = ranges("lift_permanent"), ranges("alpha_sales")
print("long-run lift\n", lift.round(2).to_string())
print("alpha of sales\n", alpha.round(2).to_string())
print("sales look like the leader\n", lead.round(2).to_string())

X = list(range(len(ARMS)))
XAXIS = {
    "min": -0.6,
    "max": len(ARMS) - 0.4,
    "ticks": list(zip(X, LABELS, strict=True)),
    "hover": False,
}


def interval_panel(title: str, g: pd.DataFrame, ymax: float) -> dict:
    tips = [
        f"{lab}: model {row.mid:.2f} of the truth (interval {row.lo:.2f} to {row.hi:.2f}); "
        f"held the truth in {row.cover:.0%} of data sets"
        for lab, row in zip(LABELS, g.itertuples(), strict=True)
    ]
    return {
        "title": title,
        "x": XAXIS,
        "y": {"min": 0, "max": ymax, "label": "Model ÷ truth"},
        "layers": [
            {"type": "hline", "y": 1, "dash": True},
            {
                "type": "interval",
                "x": X,
                "lo": r(g["lo"].clip(upper=ymax)),
                "mid": r(g["mid"]),
                "hi": r(g["hi"].clip(upper=ymax)),
                "color": "blue",
                "tips": tips,
            },
        ],
    }


save(
    "fig_base_sales",
    {
        "title": "The long run survives a rough base. The speed of sales doesn't.",
        "legend": [
            {
                "label": "Typical 90% interval, as a share of the truth",
                "color": "blue",
                "kind": "band",
            },
            {"label": "Truth", "color": "ink", "kind": "dash"},
        ],
        "height": 270,
        "minPanel": 300,
        "panels": [
            interval_panel("Long-run lift from brand spend", lift, 2.0),
            interval_panel("How fast sales catch up", alpha, 2.0),
        ],
    },
)
