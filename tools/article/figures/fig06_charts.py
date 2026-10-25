"""§6 charts: gap-closing curves and alpha against the truth (guide 01 committed outputs)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, BLUE, PURPLE, ORANGE = "#1f2328", "#6b7280", "#2563eb", "#7c3aed", "#d98a1b"
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

# Gap-closing curves
weeks = np.arange(0, 13)
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
for a, colour, label in [
    (0.230, BLUE, "Sales closing the gap to consideration"),
    (0.202, PURPLE, "Consideration closing the gap to awareness"),
]:
    left = 100 * (1 - a) ** weeks
    hl = np.log(0.5) / np.log(1 - a)
    ax.plot(weeks, left, color=colour, lw=2.2, marker="o", ms=4, label=label)
    ax.plot([hl, hl], [0, 50], color=colour, lw=1, ls=":")
    left_side = colour == BLUE
    ax.text(
        hl + (-0.12 if left_side else 0.12),
        25,
        f"{hl:.1f} weeks",
        color=colour,
        ha="right" if left_side else "left",
        fontsize=10,
        fontweight="bold",
    )
ax.axhline(50, color=MUTED, lw=0.8, ls="--")
ax.text(12, 52, "half the gap closed", ha="right", color=MUTED, fontsize=9.5)
ax.set_xlim(0, 12)
ax.set_ylim(0, 105)
ax.set_xlabel("Weeks after the gap opens")
ax.set_ylabel("% of the gap still open")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right")
ax.set_title(
    "Sales and consideration close their gaps within weeks",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig06_gap_closing.png"), bbox_inches="tight")

# Alpha forest
rows = [  # label, mean, lower, upper, truth
    ("Sales, gap 1", -0.230, -0.291, -0.163, -0.24),
    ("Sales, gap 2", 0.010, -0.042, 0.063, 0.0),
    ("Awareness, gap 1", 0.001, -0.036, 0.040, 0.0),
    ("Awareness, gap 2", 0.050, 0.020, 0.081, 0.0),
    ("Consideration, gap 1", 0.018, -0.026, 0.064, 0.0),
    ("Consideration, gap 2", 0.202, 0.164, 0.239, 0.18),
]
fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
for i, (_lab, m, lo, hi, t) in enumerate(rows[::-1]):
    miss = not (lo <= t <= hi)
    c = ORANGE if miss else BLUE
    ax.plot([lo, hi], [i, i], color=c, lw=2.4, solid_capstyle="round")
    ax.plot(m, i, "o", color=c, ms=7, zorder=3)
    ax.plot(t, i, marker="|", color=INK, ms=16, mew=2, zorder=4)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.axvline(0, color=MUTED, lw=0.8, ls="--")
ax.set_xlabel(
    "α: how strongly each series reacts to each gap\n"  # noqa: RUF001
    "(the sign depends on which side of the lead it sits)"
)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.plot([], [], color=BLUE, lw=2.4, marker="o", label="Estimate, 89% interval")
ax.plot([], [], color=ORANGE, lw=2.4, marker="o", label="Interval misses the truth")
ax.plot([], [], color=INK, marker="|", ms=12, mew=2, ls="none", label="Truth")
ax.legend(frameon=False, loc="lower left", fontsize=9.5)
ax.set_title(
    "Two followers, and one near-zero miss for awareness",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig06_alpha_forest.png"), bbox_inches="tight")
