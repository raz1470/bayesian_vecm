"""§4 charts: sales and 1.27 x consideration with their gap; Johansen trace test."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayesian_vecm import select_coint_rank, simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

data = simulate_brand_data(exog_unit_root=("interest_rate",))
e = data.endog
B = 1.272
s = 100 * (e["organic_sales"] - e["organic_sales"].mean())
c = 100 * B * (e["brand_consideration"] - e["brand_consideration"].mean())
gap = s - c
print("gap range", round(gap.min(), 1), round(gap.max(), 1))

INK, MUTED, BLUE, PURPLE = "#1f2328", "#6b7280", "#2563eb", "#7c3aed"
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
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150, sharey=True)
left.plot(e.index, s, color=BLUE, lw=1.5, label="Base sales")
left.plot(e.index, c, color=PURPLE, lw=1.5, label="1.27 × consideration")  # noqa: RUF001
left.set_title("Both trend over five years", loc="left", fontsize=11.5, color=INK)
left.legend(frameon=False, loc="upper left")
left.set_ylabel("% from average (log scale)")
right.axhspan(-8, 8, color="#f1f2f4", lw=0)
right.plot(e.index, gap, color=INK, lw=1.4)
right.axhline(0, color=MUTED, lw=0.8, ls="--")
right.set_title("The gap: within 8% of zero throughout", loc="left", fontsize=11.5, color=INK)
for ax in (left, right):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
fig.suptitle(
    "Sales and consideration wander together. The gap between them does not.",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig04_cointegration.png"), bbox_inches="tight")

res = select_coint_rank(e, k_ar_diff=1, det_order=0)
stats, crit = np.asarray(res.test_stats), np.asarray(res.crit_vals)
print("trace", stats.round(1), "crit", crit.round(1), "rank", res.rank)
fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
labels = ["None", "At most one", "At most two"]
x = np.arange(3)
colours = [BLUE if st > cv else "#b8c4d6" for st, cv in zip(stats, crit, strict=True)]
ax.bar(x, stats, width=0.55, color=colours, zorder=2)
for i, (st, cv) in enumerate(zip(stats, crit, strict=True)):
    ax.hlines(cv, i - 0.38, i + 0.38, color=INK, lw=2, zorder=3)
    ax.text(i + 0.4, cv, f"critical value {cv:.1f}", va="center", fontsize=9.5, color=INK)
    ax.text(
        i, max(st, cv) + 1.5, f"{st:.1f}", ha="center", fontsize=10.5, color=INK, fontweight="bold"
    )
ax.set_xticks(x)
ax.set_xticklabels([f"{t} lead{'s' if t == 'None' else ''}" for t in labels])
ax.set_xticklabels(["None", "At most one", "At most two"])
ax.set_xlabel("Number of leads tested")
ax.set_ylabel("Trace statistic")
ax.set_xlim(-0.6, 3.0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.set_title(
    "Two leads: the test rejects none and one, then stops",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig04_rank_test.png"), bbox_inches="tight")
