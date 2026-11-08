"""§1 chart: three ways a campaign can play out. A sketch with made-up shapes, no estimates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

W, START, LEN = 156, 8, 13
t = np.arange(W)
spend = np.where((t >= START) & (t < START + LEN), 1.0, 0.0)

# Adstock: geometric carryover of the burst, back to zero within weeks.
adstock = np.zeros(W)
for k in range(1, W):
    adstock[k] = 0.7 * adstock[k - 1] + spend[k]
# Lasting lift: each week of spend adds a step that stays; the level eases in.
lasting = np.zeros(W)
target = np.cumsum(spend)
for k in range(1, W):
    lasting[k] = lasting[k - 1] + 0.25 * (target[k] - lasting[k - 1])
# Fading lift: the same steps, drained back towards trend with a half-life of a year.
delta = 0.5 ** (1 / 52)
fading = np.zeros(W)
stock = 0.0
for k in range(1, W):
    stock = delta * stock + spend[k]
    fading[k] = fading[k - 1] + 0.25 * (stock - fading[k - 1])

INK, MUTED, GREY = "#1f2328", "#6b7280", "#d1d5db"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c9ced6"})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150, sharey=True)
scale = lasting.max()
panels = [
    ("Adstock", "fades to zero within weeks", adstock, "#c0392b"),
    ("A lasting lift", "steps up and stays", lasting, "#1f5fb4"),
    ("A fading lift", "rises, then drains back", fading, "#7c3aed"),
]
for ax, (title, sub, y, colour) in zip(axes, panels, strict=True):
    ax.bar(t, spend * 0.25 * scale, width=1.0, color=GREY, lw=0)
    ax.plot(t, y / y.max() * scale if title == "Adstock" else y, color=colour, lw=2.4)
    ax.set_title(f"{title}\n", loc="left", fontsize=12, fontweight="bold", color=colour)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=10.5)
    ax.set_xticks([0, 52, 104, 156])
    ax.set_xticklabels(["0", "1", "2", "3"])
    ax.set_xlabel("Years", color=MUTED)
    ax.set_yticks([])
    ax.tick_params(axis="x", colors=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
axes[0].text(START + LEN + 8, 0.2 * scale, "13 weeks of spend", color=MUTED, fontsize=9.5)
axes[0].set_ylabel("Sales above where they\nwould have been", color=INK)
fig.suptitle(
    "The same campaign can fade in weeks, last for good, or drain away over years",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig01_three_ways.png"), bbox_inches="tight")
