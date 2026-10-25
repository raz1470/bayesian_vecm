"""§2 chart: the three brand series over five years, indexed to 100 (guide 01 data)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayesian_vecm import simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

data = simulate_brand_data(exog_unit_root=("interest_rate",))
lv = np.exp(data.endog)
idx = lv / lv.iloc[0] * 100
print((idx.iloc[-1] - 100).round(0).to_dict())

COL = {
    "organic_sales": ("#2563eb", "Base sales"),
    "brand_awareness": ("#059669", "Awareness"),
    "brand_consideration": ("#7c3aed", "Consideration"),
}
INK, MUTED = "#1f2328", "#6b7280"
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
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
for col, (colour, label) in COL.items():
    ax.plot(idx.index, idx[col], color=colour, lw=1.6)
    ax.annotate(
        f"{label}  +{idx[col].iloc[-1] - 100:.0f}%",
        (idx.index[-1], idx[col].iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=colour,
        fontsize=10.5,
        fontweight="bold",
    )
ax.axhline(100, color="#c9ced6", lw=0.8)
ax.set_ylabel("Index, first week = 100")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.set_title(
    "Sales, awareness and consideration climb together",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig02_brand_series.png"), bbox_inches="tight")
