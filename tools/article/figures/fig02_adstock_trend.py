"""§2 chart: a long adstock of brand spend fits a trend it has nothing to do with.

Five years of weekly brand spend, flighted and growing slowly as budgets often
do, and its long adstock (decay 0.97). Separately, a baseline built as a random
walk with upward drift and no link to spend. A regression of the baseline on
the adstock still finds a strong, significant slope.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

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

INK, MUTED, RED, GREY = "#1f2328", "#6b7280", "#c0392b", "#d1d5db"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9ced6"})
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
ax2 = ax.twinx()
ax2.bar(t, spend, width=1.0, color=GREY, lw=0, label="Weekly brand spend")
ax2.set_ylim(0, spend.max() * 3.0)
ax2.set_yticks([])
ax.plot(t, base, color=INK, lw=1.8, label="Baseline sales (built with no link to spend)")
ax.plot(
    t,
    ols.fittedvalues,
    color=RED,
    lw=2.2,
    label="Long adstock of brand spend, as a regression fits it",
)
ax.text(
    t[-1],
    ols.fittedvalues[-1],
    f"  slope t = {ols.tvalues[1]:.0f}",
    color=RED,
    va="center",
    fontsize=10,
)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)
ax.set_xticks([0, 52, 104, 156, 208, 260])
ax.set_xticklabels(["0", "1", "2", "3", "4", "5"])
ax.set_xlabel("Years", color=MUTED)
ax.set_yticks([])
for a in (ax, ax2):
    for side in ("top", "right", "left"):
        a.spines[side].set_visible(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h2 + h1, l2 + l1, frameon=False, loc="upper left", fontsize=9.5)
ax.set_title(
    "A long adstock fits the trend, not the brand",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig02_adstock_trend.png"), bbox_inches="tight")
