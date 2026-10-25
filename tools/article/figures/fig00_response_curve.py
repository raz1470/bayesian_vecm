"""§0 chart: response curve for a 13-week brand campaign, revenue within 90 days and 5 years.

Full model (adstock, saturation, same-week link) on guide 08's default data,
set up exactly as guide 08 does. No truth line: the chart illustrates the shape.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_vecm import BayesianVECM, simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
CH = ["brand_spend", "pr_reach", "sponsorship_reach"]
sim = simulate_brand_data(
    exog_unit_root=("interest_rate",),
    exog_adstock={"brand_spend": 0.5, "pr_reach": 0.3, "sponsorship_reach": 0.6},
    exog_saturation={"brand_spend": 5.0, "pr_reach": 3.0, "sponsorship_reach": 5.0},
    contemporaneous={"brand_consideration": {"brand_awareness": 0.2}},
)
endog = sim.endog.copy()
exog = sim.exog.copy()
exog["interest_rate"] = exog["interest_rate"].diff()
other = ["interest_rate", "consumer_confidence"]
exog[other] = exog[other] - exog[other].mean()
endog, exog = endog.iloc[1:], exog.iloc[1:]

links = {"brand_consideration": ["brand_awareness"]}
model = BayesianVECM(
    k_ar_diff=1,
    coint_rank=2,
    deterministic="co",
    adstock=CH,
    saturation=CH,
    contemporaneous=links,
    excluded={"brand_consideration": CH},
)
model.fit(
    endog,
    exog=exog,
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.95,
    random_seed=1,
    progressbar=False,
)


recent_sales = float(np.exp(endog["organic_sales"].iloc[-52:]).mean())
COST, PLAN, HORIZON = 1_000.0, 13, 260
names = list(exog.columns)
j = names.index("brand_spend")
mean_spend = float(exog["brand_spend"].mean())
base_row = exog.mean().to_numpy().copy()
base_row[[names.index(c) for c in other]] = 0.0
MULT = np.linspace(0, 1.5, 13)


def extra_revenue(level):
    base = np.tile(base_row, (HORIZON, 1))
    plan = base.copy()
    plan[:PLAN, j] = level
    paths = []
    for x in (base, plan):
        fc = model.sample_posterior_predictive(HORIZON, exog_future=x, random_seed=0)
        paths.append(
            fc.posterior_predictive["y"].sel(variable="organic_sales").values.reshape(-1, HORIZON)
        )
    weekly = np.expm1(paths[1] - paths[0]) * recent_sales
    return weekly[:, :52].sum(axis=1), weekly.sum(axis=1)


res = [extra_revenue(m * mean_spend) for m in MULT]
# Baseline: brand spend switched off for the 13 weeks, so the curve starts at zero.
short = np.array([r[0] - res[0][0] for r in res])
long = np.array([r[1] - res[0][1] for r in res])
spend = MULT * mean_spend * COST  # £ a week
cost = spend * PLAN
rows = []
for lab, arr in [("1 year", short), ("5 years", long)]:
    med = np.median(arr, axis=1)
    lo, hi = np.percentile(arr, [5.5, 94.5], axis=1)
    for i in range(len(MULT)):
        rows.append(
            {
                "window": lab,
                "spend_per_week": spend[i],
                "cost": cost[i],
                "median": med[i],
                "lo89": lo[i],
                "hi89": hi[i],
                "per_pound": med[i] / cost[i] if cost[i] else np.nan,
            }
        )
tab = pd.DataFrame(rows)
print(tab.round(2).to_string(index=False))
lo_d, hi_d = exog["brand_spend"].min() * COST, exog["brand_spend"].max() * COST
print("data range £/week:", round(lo_d), round(hi_d))

BLUE, AMBER, INK, MUTED = "#1f5fb4", "#7fb0ea", "#1f2328", "#6b7280"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#c9ced6",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
i_avg = int(np.argmin(np.abs(MULT - 1)))
for arr, colour, lab in [
    (long, BLUE, "Earned within 5 years"),
    (short, AMBER, "Earned within 1 year"),
]:
    med = np.median(arr, axis=1)
    lo, hi = np.percentile(arr, [5.5, 94.5], axis=1)
    ax.fill_between(spend, lo / 1e3, hi / 1e3, color=colour, alpha=0.18, lw=0)
    ax.plot(spend, med / 1e3, color=colour, lw=2.2, label=lab)
    ax.plot(spend[i_avg], med[i_avg] / 1e3, "o", color=colour, ms=6, zorder=5)
y1 = np.median(short, axis=1)[i_avg] / 1e3
y5 = np.median(long, axis=1)[i_avg] / 1e3
ax.annotate(
    "",
    xy=(spend[i_avg] + 25, y5),
    xytext=(spend[i_avg] + 25, y1),
    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2),
)
ax.text(
    spend[i_avg] + 45,
    (y1 + y5) / 2,
    f"about {y5 / y1:.0f}x",
    color=INK,
    fontsize=12,
    fontweight="bold",
    va="center",
)
ax.axvline(mean_spend * COST, color=MUTED, lw=0.8, ls="--")
ax.set_xlabel("Brand spend during a 13-week campaign")
ax.set_xticks([0, mean_spend * COST])
ax.set_xticklabels(["none", "current\naverage"])
ax.set_ylabel("Extra revenue against no campaign")
ax.set_yticklabels([])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, spend[-1])
ax.set_ylim(0, None)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
ax.set_title(
    "A brand campaign keeps earning long after it ends",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig00_response_curve.png"), bbox_inches="tight")
