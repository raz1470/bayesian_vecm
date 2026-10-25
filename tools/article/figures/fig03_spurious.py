"""§3 chart: two unrelated random walks.

Also prints how often regression calls such pairs related.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

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

# --- chart ---

w = np.arange(len(a))
INK, MUTED, C1, C2, RED = "#1f2328", "#6b7280", "#2563eb", "#d98a1b", "#c0392b"
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
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
left.plot(w, a, color=C1, lw=1.5, label="Random walk A")
left.plot(w, b, color=C2, lw=1.5, label="Random walk B")
left.set_title("Two walks, built independently", loc="left", fontsize=11.5, color=INK)
left.legend(frameon=False, loc="upper left")
right.plot(w, resid, color=RED, lw=1.4)
right.axhline(0, color=MUTED, lw=0.8, ls="--")
right.set_title(
    "What regression leaves over: no pull back to zero", loc="left", fontsize=11.5, color=INK
)
for ax in (left, right):
    ax.set_xticks([0, 52, 104, 156, 208, 260])
    ax.set_xticklabels(["0", "1", "2", "3", "4", "5"])
    ax.set_xlabel("Years")
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
fig.suptitle(
    "Regression calls them related (t = 15.9, R² = 0.50). They are not.",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig03_spurious.png"), bbox_inches="tight")
