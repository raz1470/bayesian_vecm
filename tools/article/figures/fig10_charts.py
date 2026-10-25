"""§10 charts: exclusion diagram, and OLS against 2SLS for the same-week link.

200 simulated brands with no same-week link.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from bayesian_vecm import simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GREEN, PURPLE, AMBER, RED = (
    "#1f2328",
    "#6b7280",
    "#059669",
    "#7c3aed",
    "#d98a1b",
    "#c0392b",
)
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


def est(seed):
    d = simulate_brand_data(seed=seed)
    y = d.endog.to_numpy()
    x = d.exog.to_numpy()
    dy = np.diff(y, axis=0)
    t = np.arange(1, len(dy))
    y_eff = dy[t, 2]  # consideration growth
    x_cause = dy[t, 1]  # awareness growth
    w_ctrl = np.column_stack([np.ones(len(t)), y[t], dy[t - 1], x[t + 1][:, 3:]])  # controls
    z_inst = x[t + 1][:, :3]  # channels this week: the instruments
    ols = np.linalg.lstsq(np.column_stack([x_cause, w_ctrl]), y_eff, rcond=None)[0][0]
    zw = np.column_stack([z_inst, w_ctrl])
    x_hat = zw @ np.linalg.lstsq(zw, x_cause, rcond=None)[0]
    iv = np.linalg.lstsq(np.column_stack([x_hat, w_ctrl]), y_eff, rcond=None)[0][0]
    return ols, iv


res = np.array([est(s) for s in range(1, 201)])
print("OLS mean", res[:, 0].mean().round(3), np.percentile(res[:, 0], [5, 95]).round(3))
print("IV  mean", res[:, 1].mean().round(3), np.percentile(res[:, 1], [5, 95]).round(3))

fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
bins = np.linspace(-0.35, 0.55, 46)
ax.hist(
    res[:, 1], bins=bins, color=GREEN, alpha=0.75, label="Instrument (brand spend, PR, sponsorship)"
)
ax.hist(res[:, 0], bins=bins, color=MUTED, alpha=0.6, label="Plain regression")
ax.axvline(0, color=INK, lw=1.4, ls="--", label="True link: 0")
ax.set_xlabel("Estimated same-week link from awareness to consideration")
ax.set_yticks([])
ax.set_ylabel("Simulated brands")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper left")
ax.set_title(
    "With no real link, regression finds one. The instrument does not.",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig10_ols_vs_iv.png"), bbox_inches="tight")

# Exclusion diagram
fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")
boxes = {
    "spend": (0.3, 1.6, "Brand spend", "#f7f6f3", "#d6d3cd", INK),
    "aw": (3.9, 1.6, "Awareness", "#ecfdf5", GREEN, "#047857"),
    "co": (7.5, 1.6, "Consideration", "#f5f3ff", PURPLE, "#6d28d9"),
}
for x0, y0, lab, fc, ec, tc in boxes.values():
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), 2.2, 0.8, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec=ec, lw=1.6
        )
    )
    ax.text(
        x0 + 1.1, y0 + 0.4, lab, ha="center", va="center", fontsize=12, fontweight="bold", color=tc
    )


def arrow(a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, **kw))


arrow((2.55, 2.0), (3.85, 2.0), color=INK, lw=1.6)
arrow((6.15, 2.0), (7.45, 2.0), color=GREEN, lw=2.2)
ax.text(6.8, 2.25, "the link", ha="center", color=GREEN, fontsize=10.5, fontweight="bold")
arrow((1.4, 1.55), (8.6, 1.55), color=RED, lw=1.4, ls="--", connectionstyle="arc3,rad=0.28")
ax.text(
    5.0,
    0.35,
    "✕  excluded: brand spend reaches consideration only through awareness",
    ha="center",
    color=RED,
    fontsize=10.5,
)
ax.annotate(
    "",
    xy=(5.0, 2.45),
    xytext=(8.6, 2.45),
    arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.2, ls=":", connectionstyle="arc3,rad=0.35"),
)
ax.text(
    6.8, 3.35, "shared shocks move both in the same week", ha="center", color=MUTED, fontsize=10
)
ax.set_title(
    "The exclusion that identifies the link", loc="left", fontsize=13, fontweight="bold", color=INK
)
fig.savefig(str(OUT / "fig10_exclusion.png"), bbox_inches="tight")
