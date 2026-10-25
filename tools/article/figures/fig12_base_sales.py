"""§12 chart: what survives and what breaks with each way of building base sales (5 seeds)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(str(ROOT / "tools" / "mmm_trend_check_results.csv"))
ARMS = [
    ("true_base", "True base"),
    ("mmm78_keep_season", "MMM: total minus contributions"),
    ("total_adstock", "Total sales straight into the VECM"),
    ("mmm78_trend", "MMM: its smooth trend"),
]
rows = []
for arm, lab in ARMS:
    a = d[d.arm == arm]

    def err(q, a=a):
        rows_q = a[a.quantity == q]
        return 100 * np.median(rows_q["median"] / rows_q["truth"] - 1)

    ev = a.drop_duplicates("seed")[
        [
            "evidence_sales_to_aware_short",
            "evidence_sales_to_cons_short",
            "evidence_sales_to_aware_long",
            "evidence_sales_to_cons_long",
        ]
    ]
    false_dir = int((ev.max(axis=1) >= 0.95).sum())
    rows.append((lab, err("lift_permanent"), err("alpha_sales"), false_dir, a.seed.nunique()))
    print(rows[-1])

INK, MUTED, BLUE, RED = "#1f2328", "#6b7280", "#2563eb", "#c0392b"
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
fig, axes = plt.subplots(
    1, 3, figsize=(13, 3.6), dpi=150, sharey=True, gridspec_kw={"width_ratios": [1, 1, 0.6]}
)
y = np.arange(len(rows))[::-1]
for ax, idx, title in [
    (axes[0], 1, "Long-run return on brand spend"),
    (axes[1], 2, "Adjustment speed of sales (α)"),  # noqa: RUF001
]:
    vals = [r[idx] for r in rows]
    cols = [RED if abs(v) > 25 else BLUE for v in vals]
    ax.barh(y, vals, color=cols, height=0.55)
    for yi, v in zip(y, vals, strict=True):
        ax.text(
            v + (3 if v >= 0 else -3),
            yi,
            (f"{v:+.0f}%" if round(v) != 0 else "0%"),
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=10,
            color=INK,
            fontweight="bold",
        )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-130, 130)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK)
    ax.set_xlabel("Median error against the truth")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(y)
axes[0].set_yticklabels([r[0] for r in rows])
ax = axes[2]
for yi, r in zip(y, rows, strict=True):
    ax.text(
        0.5,
        yi,
        f"{r[3]} of {r[4]}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=RED if r[3] else INK,
    )
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.tick_params(axis="y", length=0)
ax.set_title("Fits finding sales\nleading the brand", loc="left", fontsize=11.5, color=INK)
for side in ("top", "right", "left", "bottom"):
    ax.spines[side].set_visible(False)
fig.suptitle(
    "Every base gets the long-run return. None gets the adjustment speed right.",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig12_base_sales.png"), bbox_inches="tight")
