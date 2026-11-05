"""§13 charts: does brand last? The lasting-only and fading models against the truth.

1. fig13_fade_check.png: the 5-year return from one week of brand spend, as a
   multiple of the truth, for four truths and both models, across five seeds
   (from tools/fade_check_results.csv).
2. fig13_going_dark.png: brand spend off for 13 weeks then back to average, on
   simulated data where every lift fades with a half-life of 52 weeks, for both
   models and the truth (the same set-up as guide 09).
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_vecm import BayesianVECM, simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

INK, MUTED = "#1f2328", "#6b7280"
LASTS, FADES = "#d98a1b", "#1f5fb4"
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

# --- Chart 1: across seeds ---------------------------------------------------
runs = pd.read_csv(ROOT / "tools" / "fade_check_results.csv")
five = runs[runs.quantity == "pulse sales, 5 years"].copy()
for col in ("median", "lo90", "hi90"):
    five[col] = five[col] / five["truth"]
five["held"] = (five.lo90 <= 1) & (five.hi90 >= 1)
SETTINGS = [
    ("permanent", "Every lift\nlasts"),
    ("fade 104", "Fades,\nhalf-life 2 years"),
    ("fade 52", "Fades,\nhalf-life 1 year"),
    ("fade 26", "Fades,\nhalf-life 6 months"),
]
FITS = [
    ("permanent model", "Model: every lift lasts", LASTS),
    ("fading model", "Model: fading=", FADES),
]
fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
x = np.arange(len(SETTINGS))
for k, (fit, label, colour) in enumerate(FITS):
    med, lo, hi, held = [], [], [], []
    for setting, _ in SETTINGS:
        r = five[(five.setting == setting) & (five.fit == fit)]
        med.append(r["median"].median())
        lo.append(r["lo90"].median())
        hi.append(r["hi90"].median())
        held.append(r["held"].mean())
    pos = x + (k - 0.5) * 0.32
    med, lo, hi = map(np.array, (med, lo, hi))
    ax.errorbar(
        pos,
        med,
        yerr=[med - lo, hi - med],
        fmt="o",
        color=colour,
        ms=8,
        lw=2,
        capsize=0,
        label=label,
    )
    for p, m, h in zip(pos, hi, held, strict=True):
        ax.text(p, m + 0.15, f"{h:.0%}", ha="center", va="bottom", fontsize=9, color=colour)
    print(label, np.round(med, 2), held)
ax.axhline(1, color=INK, lw=1, ls="--")
ax.text(len(SETTINGS) - 0.45, 1.05, "truth", color=INK, fontsize=9, va="bottom", ha="right")
ax.set_xticks(x, [s[1] for s in SETTINGS])
ax.set_ylabel("5-year return as a multiple of the truth")
ax.set_ylim(0, None)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
ax.set_title(
    "If lifts fade, assuming they last overstates the 5-year return",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig13_fade_check.png"), bbox_inches="tight")
plt.close(fig)

# --- Chart 2: going dark for a quarter --------------------------------------
CHANNELS = ["brand_spend", "pr_reach", "sponsorship_reach"]
data = simulate_brand_data(exog_fade=dict.fromkeys(CHANNELS, 52.0))
endog, exog, truth = data.endog, data.exog, data.truth
j = list(exog.columns).index("brand_spend")
DARK, FOLLOW = 13, 156
base = np.zeros((FOLLOW, exog.shape[1]))
plan = base.copy()
plan[:DARK, j] = -1.0  # a linear channel here averages one unit: -1 is off
gaps = {}
for label, extra in {"Model: every lift lasts": {}, "Model: fading=": {"fading": CHANNELS}}.items():
    model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co", **extra)
    model.fit(
        endog,
        exog=exog,
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=3,
        progressbar=False,
    )
    paths = []
    for xx in (base, plan):
        fc = model.sample_posterior_predictive(FOLLOW, exog_future=xx, random_seed=0)
        y = fc.posterior_predictive["y"].sel(variable="organic_sales").values
        paths.append(y.reshape(-1, FOLLOW))
    gaps[label] = 100 * np.expm1(paths[1] - paths[0])


def truth_sales(u):
    pi = truth.alpha @ truth.beta.T
    b = truth.b[:, j]
    delta = truth.fade_decay[j]
    y, dy, stock, out = np.zeros(3), np.zeros(3), 0.0, []
    for t in range(len(u)):
        if t > 0:
            stock = delta * stock + u[t - 1]
        dy = pi @ y + truth.gamma @ dy + b * u[t] - truth.fade_share * (1 - delta) * b * stock
        y = y + dy
        out.append(y[0])
    return np.array(out)


off = np.zeros(FOLLOW)
off[:DARK] = -1.0
true_gap = 100 * np.expm1(truth_sales(off))
fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
weeks = np.arange(1, FOLLOW + 1)
ax.axvspan(0, DARK, color="#9ca3af", alpha=0.18, lw=0, label="Brand spend off")
for (label, gap), colour in zip(gaps.items(), [LASTS, FADES], strict=True):
    lo, hi = np.percentile(gap, [5, 95], axis=0)
    ax.fill_between(weeks, lo, hi, color=colour, alpha=0.16, lw=0)
    ax.plot(weeks, np.median(gap, axis=0), color=colour, lw=2.2, label=label)
    print(label, "3 years on:", round(float(np.median(gap[:, -1])), 1))
ax.plot(weeks, true_gap, color=INK, lw=1.2, ls="--", label="Truth: every lift fades")
print("truth 3 years on:", round(float(true_gap[-1]), 1))
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("Weeks after brand spend stops")
ax.set_ylabel("Weekly sales against carrying on (%)")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title(
    "A quarter off: a permanent loss, or a dip that comes back?",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig13_going_dark.png"), bbox_inches="tight")
