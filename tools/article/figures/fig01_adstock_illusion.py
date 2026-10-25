"""§1 chart: illustration only. One burst of brand spend and three sales responses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

T = 105
weeks = np.arange(T)
spend = np.zeros(T)
spend[1:5] = 1.0  # a four-week burst


def adstock(x, decay):
    out, a = np.zeros_like(x), 0.0
    for t, v in enumerate(x):
        a = v + decay * a
        out[t] = a
    return out * (1 - decay)  # same total weight per unit of spend


short = adstock(spend, 0.5)
long = adstock(spend, 0.95)
# Brand system: each week of spend lifts the level for good, reached gradually
# as awareness feeds consideration and consideration feeds sales.
step = 1 - np.exp(-np.arange(T) / 10.0)
system = np.convolve(spend, step)[:T] * 0.075

RED, BLUE, INK, MUTED, GREY = "#c0392b", "#1f5fb4", "#1f2328", "#6b7280", "#d7dbe0"
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
ax.bar(weeks, spend * 0.12, width=0.9, color=GREY, label="Brand spend (4-week burst)", zorder=1)
ax.plot(weeks, short, color=RED, lw=2, ls="--", label="Short adstock: gone in weeks")
ax.plot(weeks, long, color=RED, lw=1.8, ls=":", label="Long adstock: fades over a year")
ax.plot(weeks, system, color=BLUE, lw=2.6, label="Brand system: a new level that stays")
ax.set_xlim(0, T - 1)
ax.set_ylim(0, None)
ax.set_xticks([0, 26, 52, 78, 104])
ax.set_xticklabels(["0", "6 months", "1 year", "18 months", "2 years"])
ax.set_yticks([])
ax.set_ylabel("Extra weekly sales")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper right")
ax.set_title(
    "Adstock lets the effect go. Brand building moves the level.",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
ax.text(
    1.0,
    -0.16,
    "Illustration, not an estimate",
    transform=ax.transAxes,
    ha="right",
    color=MUTED,
    fontsize=9,
    style="italic",
)
fig.tight_layout()
fig.savefig(str(OUT / "fig01_adstock_illusion.png"), bbox_inches="tight")
print(round(short.max(), 3), round(long.max(), 3), round(system[-1], 3))
