"""Section 10 chart: performance spend that chases demand, with an MMM building the base.

Reads ``tools/perf_feedback_check_results.csv`` (10 seeds; feedback none,
moderate (0.25) and strong (0.5)). Left: how far the MMM over-credits paid
search. Middle: the long-run lift in sales from brand spend from the MMM's
base, as a ratio to the truth. Right: the lift in sales four weeks after
one more unit of brand spend, the same way. Also printed for the text: the
true-base reference, and how often a Granger check flags the feedback
(p < 0.05) using past changes in total sales or in consideration.
"""

from pathlib import Path

import pandas as pd
from _spec import r, save

CSV = Path(__file__).resolve().parents[2] / "perf_feedback_check_results.csv"
STRENGTHS = [0.0, 0.25, 0.5]
NAMES = ["None", "Moderate", "Strong"]

d = pd.read_csv(CSV)
mmm = d[d["arm"] == "mmm78_minus"]
per_fit = mmm.drop_duplicates(["seed", "strength"]).set_index("strength")
credit = per_fit.groupby(level=0)["mmm_contrib_error_paid_search"].median().loc[STRENGTHS]
corr = per_fit.groupby(level=0)["corr_search_demand"].median().loc[STRENGTHS]
flag_sales = (per_fit["granger_p_sales"] < 0.05).groupby(level=0).mean().loc[STRENGTHS]
flag_cons = (per_fit["granger_p_consideration"] < 0.05).groupby(level=0).mean().loc[STRENGTHS]

lift = d[d["quantity"] == "lift_permanent"].copy()
for c in ("median", "lower", "upper"):
    lift[c] = lift[c] / lift["truth"]
rows = [lift[lift["arm"] == "true_base"]] + [
    lift[(lift["arm"] == "mmm78_minus") & (lift["strength"] == s)] for s in STRENGTHS
]
g = pd.DataFrame(
    [
        {
            "mid": x["median"].median(),
            "lo": x["lower"].median(),
            "hi": x["upper"].median(),
            "cover": x["covered"].mean(),
            "seeds": x["seed"].nunique(),
        }
        for x in rows
    ],
    index=["True base", *NAMES],
)
alpha = d[d["quantity"] == "alpha_sales"].copy()
alpha["ratio"] = alpha["median"] / alpha["truth"]
print("search over-credit\n", credit.round(2).to_string())
print("corr search vs demand\n", corr.round(2).to_string())
print("long-run lift\n", g.round(2).to_string())
print("alpha ratio\n", alpha.groupby(["arm", "strength"])["ratio"].median().round(2).to_string())
print(
    "flagged, sales\n", flag_sales.to_string(), "\nflagged, consideration\n", flag_cons.to_string()
)

X3 = [0, 1, 2]
AX3 = {
    "min": -0.6,
    "max": 2.6,
    "ticks": list(zip(X3, NAMES, strict=True)),
    "label": "How hard search chases demand",
    "hover": False,
}

sp = d[(d["arm"] == "mmm78_minus") & (d["quantity"] == "lift_week4")].copy()
for c in ("median", "lower", "upper"):
    sp[c] = sp[c] / sp["truth"]
speed = (
    sp.groupby("strength")
    .agg(mid=("median", "median"), lo=("lower", "median"), hi=("upper", "median"))
    .loc[STRENGTHS]
)
lift3 = g.loc[NAMES]


def interval_panel(title: str, h: pd.DataFrame, what: str) -> dict:
    return {
        "title": title,
        "x": AX3,
        "y": {"min": 0, "max": 2, "label": "Model ÷ truth"},
        "layers": [
            {"type": "hline", "y": 1, "dash": True},
            {
                "type": "interval",
                "x": X3,
                "lo": r(h["lo"]),
                "mid": r(h["mid"]),
                "hi": r(h["hi"]),
                "color": "blue",
                "tips": [
                    f"{n}: {what} {row.mid:.2f} of the truth "
                    f"(interval {row.lo:.2f} to {row.hi:.2f})"
                    for n, row in zip(NAMES, h.itertuples(), strict=True)
                ],
            },
        ],
    }


top = max(0.1, float(credit.max()))
save(
    "fig_perf_feedback",
    {
        "title": "The MMM over-credits search. The brand's long run doesn't notice.",
        "legend": [
            {"label": "MMM credit to search, above the truth", "color": "orange", "kind": "bar"},
            {
                "label": "Typical 90% interval, as a share of the truth",
                "color": "blue",
                "kind": "band",
            },
            {"label": "Truth", "color": "ink", "kind": "dash"},
        ],
        "height": 270,
        "minPanel": 190,
        "panels": [
            {
                "title": "MMM credit to paid search",
                "x": AX3,
                "y": {
                    "min": 0,
                    "max": round(100 * top * 1.25, -1),
                    "fmt": "pct_plain",
                    "label": "% above the truth",
                },
                "layers": [
                    {
                        "type": "bars",
                        "x": X3,
                        "y": r(100 * credit.clip(lower=0)),
                        "color": "orange",
                        "barWidth": 0.55,
                    },
                    *[
                        {
                            "type": "text",
                            "x": x,
                            "y": 100 * max(v, 0) + 2,
                            "text": f"{100 * v:+.0f}%",
                            "anchor": "middle",
                        }
                        for x, v in zip(X3, credit, strict=True)
                    ],
                ],
            },
            interval_panel("Long-run lift from brand spend", lift3, "long-run lift"),
            interval_panel("Lift four weeks after spend", speed, "4-week lift"),
        ],
    },
)
