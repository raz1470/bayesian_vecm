"""§8 charts: cumulative return per £1 of brand spend with the truth; all three channels."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayesian_vecm import BayesianVECM, simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
data = simulate_brand_data(exog_unit_root=("interest_rate",))
endog, exog, truth = data.endog, data.exog, data.truth
exog = exog.assign(interest_rate=exog["interest_rate"].diff()).iloc[1:]
endog = endog.iloc[1:]
names = list(endog.columns)
model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co")
model.fit(
    endog,
    exog=exog,
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.90,
    random_seed=42,
    progressbar=False,
)
recent = float(np.exp(endog["organic_sales"].iloc[-52:]).mean())
STEPS = 260
mult = model.exog_irf(steps=STEPS)


def cum(channel, cost):
    s = (
        mult.sel(response_variable="organic_sales", exog_variable=channel)
        .stack(sample=("chain", "draw"))
        .transpose("sample", "horizon")
        .values
    )
    return (np.expm1(s) * recent).cumsum(axis=1) / cost


def true_cum(channel, cost):
    x = np.zeros(truth.b.shape[1])
    x[list(exog.columns).index(channel)] = 1.0
    level = np.zeros(3)
    growth = np.zeros(3)
    path = []
    for h in range(STEPS + 1):
        growth = truth.alpha @ truth.beta.T @ level + truth.gamma @ growth
        growth = growth + (truth.b @ x if h == 0 else 0.0)
        level = level + growth
        path.append(level[names.index("organic_sales")])
    return (np.expm1(np.array(path)) * recent).cumsum() / cost


INK, MUTED, BLUE = "#1f2328", "#6b7280", "#2563eb"
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
XT = ([0, 52, 156, 260], ["0", "1 yr", "3 yrs", "5 yrs"])

# Chart 1: brand spend per £1
c = cum("brand_spend", 1000.0)
t = true_cum("brand_spend", 1000.0)
w = np.arange(1, c.shape[1] + 1)
m = c.mean(axis=0)
lo, hi = np.percentile(c, [5.5, 94.5], axis=0)
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
ax.fill_between(w, lo, hi, color=BLUE, alpha=0.18, lw=0, label="89% interval")
ax.plot(w, m, color=BLUE, lw=2.2, label="Model")
ax.plot(w, t[: len(w)], color=INK, lw=1.2, ls="--", label="Truth")
for wk in (52, 156, 260):
    ax.plot(wk, m[wk - 1], "o", color=BLUE, ms=6, zorder=5)
    ax.annotate(
        f"£{m[wk - 1]:.2f}",
        (wk, m[wk - 1]),
        xytext=(-6, 9),
        textcoords="offset points",
        ha="right",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
print("brand", [round(m[k - 1], 2) for k in (52, 156, 260)], "truth5", round(t[259], 2))
ax.set_xticks(*XT)
ax.set_xlim(0, 262)
ax.set_ylim(0, None)
ax.set_ylabel("Revenue back per £1 (£, cumulative)")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
ax.set_title(
    "One more £1 of brand spend for one week, added up over five years",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig08_return_brand.png"), bbox_inches="tight")

# Chart 2: three channels side by side
CH = [
    ("brand_spend", 1000.0, "Brand spend", "per £1", "#d98a1b"),
    ("pr_reach", 1.0, "PR", "per 100,000 people reached", "#db2777"),
    ("sponsorship_reach", 1.0, "Sponsorship", "per 100,000 people reached", "#0d9488"),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
for ax, (ch, cost, lab, unit, col) in zip(axes, CH, strict=True):
    c = cum(ch, cost)
    t = true_cum(ch, cost)
    m = c.mean(axis=0)
    lo, hi = np.percentile(c, [5.5, 94.5], axis=0)
    ax.fill_between(w, lo, hi, color=col, alpha=0.18, lw=0)
    ax.plot(w, m, color=col, lw=2.2)
    ax.plot(w, t[: len(w)], color=INK, lw=1.1, ls="--")
    ax.set_title(f"{lab}\n{unit}", loc="left", fontsize=11, color=INK)
    ax.set_xticks(*XT)
    ax.set_xlim(0, 262)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    print(lab, round(m[51]), round(m[259]), "truth", round(t[259]))
axes[0].plot([], [], color=INK, ls="--", label="Truth")
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle(
    "Every channel keeps paying for years",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig08_return_channels.png"), bbox_inches="tight")
