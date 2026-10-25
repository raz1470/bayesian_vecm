"""§11 chart: how often 90% intervals hold the truth, Bayesian against statsmodels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(str(ROOT / "tools" / "statsmodels_check_results.csv"))
d["bayes"] = (d.bayes_lo <= d.truth) & (d.truth <= d.bayes_hi)
d["sm"] = (d.sm_lo <= d.truth) & (d.truth <= d.sm_hi)
blocks = [
    ("beta", "Leads (β)"),
    ("alpha", "Adjustment (α)"),  # noqa: RUF001
    ("Gamma", "Momentum (Γ)"),
    ("B", "Drivers (B)"),
    ("return", "Return per £"),
]
cov = d.groupby(["sample", "block"])[["bayes", "sm"]].mean()

INK, MUTED, BLUE, GREY = "#1f2328", "#6b7280", "#2563eb", "#b8c4d6"
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
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, sharey=True)
x = np.arange(len(blocks))
for ax, sample, title in [
    (axes[0], "260 weeks", "Five years of weekly data"),
    (axes[1], "60 periods", "Only 60 periods, the length of five years of monthly data"),
]:
    b = [100 * cov.loc[(sample, k), "bayes"] for k, _ in blocks]
    s = [100 * cov.loc[(sample, k), "sm"] for k, _ in blocks]
    ax.bar(x - 0.19, b, 0.36, color=BLUE, label="Bayesian")
    ax.bar(x + 0.19, s, 0.36, color=GREY, label="statsmodels (bootstrap for returns)")
    for xi, v in zip(x + 0.19, s, strict=True):
        if v < 85:
            ax.text(
                xi,
                v - 2,
                f"{v:.0f}%",
                ha="center",
                va="top",
                fontsize=10,
                color=INK,
                fontweight="bold",
            )
    ax.axhline(90, color=INK, lw=1.2, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in blocks], fontsize=9.5)
    ax.set_ylim(40, 100)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("% of 90% intervals that hold the truth")
axes[0].text(4.45, 91, "target 90%", ha="right", va="bottom", fontsize=9.5, color=INK)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
fig.suptitle(
    "On short samples the classical intervals claim too much",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig11_coverage.png"), bbox_inches="tight")
