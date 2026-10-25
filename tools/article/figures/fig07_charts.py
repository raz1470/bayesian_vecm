"""§7 charts: responses to an awareness shock; the sales response under three orderings."""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayesian_vecm import BayesianVECM, simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
data = simulate_brand_data(exog_unit_root=("interest_rate",))
endog, exog = data.endog, data.exog
exog = exog.assign(interest_rate=exog["interest_rate"].diff()).iloc[1:]
endog = endog.iloc[1:]
model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co")
model.fit(
    endog,
    exog=exog,
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.90,
    random_seed=42,
    progressbar=False,
)

H = 52
girf = model.irf(steps=H, method="girf")
funnel = ["brand_awareness", "brand_consideration", "organic_sales"]
chol_aw = model.irf(steps=H, method="cholesky", order=funnel)
chol_sa = model.irf(
    steps=H, method="cholesky", order=["organic_sales", "brand_awareness", "brand_consideration"]
)


def draws(irf, resp):
    v = irf.sel(response_variable=resp, shock_variable="brand_awareness")
    return 100 * v.stack(sample=("chain", "draw")).transpose("sample", "horizon").values


INK, MUTED, RED = "#1f2328", "#6b7280", "#c0392b"
COL = {
    "organic_sales": ("#2563eb", "Base sales"),
    "brand_awareness": ("#059669", "Awareness"),
    "brand_consideration": ("#7c3aed", "Consideration"),
}
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
h = np.arange(H + 1)

fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
for resp in ["brand_awareness", "brand_consideration", "organic_sales"]:
    d = draws(girf, resp)
    c, lab = COL[resp]
    lo, hi = np.percentile(d, [5.5, 94.5], axis=0)
    ax.fill_between(h, lo, hi, color=c, alpha=0.15, lw=0)
    ax.plot(h, d.mean(axis=0), color=c, lw=2.2)
    ax.text(
        H + 0.8, d.mean(axis=0)[-1], lab, color=c, va="center", fontsize=10.5, fontweight="bold"
    )
sales = draws(girf, "organic_sales").mean(axis=0)
r0 = sales[0]  # same week-0 effect as the VECM
ax.plot(h, r0 * 0.7**h, color=RED, lw=1.4, ls="--")
ax.text(5, r0 * 0.7**5 + 0.06, "Adstock (decay 0.7)", color=RED, fontsize=9.5)
print("sales wk52 mean", round(sales[-1], 2))
ax.axhline(0, color="#c9ced6", lw=0.8)
ax.set_xlim(0, H)
ax.set_xlabel("Weeks after the shock")
ax.set_ylabel("% above where it would have been")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.set_title(
    "A surprise rise in awareness lifts the whole funnel for good",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig07_awareness_shock.png"), bbox_inches="tight")

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
for irf, c, lab, ls, lw in [
    (girf, "#2563eb", "GIRF", "-", 4.5),
    (chol_aw, INK, "Cholesky, awareness first", "--", 1.6),
    (chol_sa, "#d98a1b", "Cholesky, sales first", "-", 2.0),
]:
    m = draws(irf, "organic_sales").mean(axis=0)
    ax.plot(
        h,
        m,
        color=c,
        lw=lw,
        ls=ls,
        label=f"{lab}: {m[-1]:.2f}% at week 52",
        alpha=0.45 if lab == "GIRF" else 1,
    )
ax.axhline(0, color="#c9ced6", lw=0.8)
ax.set_xlim(0, H)
ax.set_ylim(0, None)
ax.set_xlabel("Weeks after the shock")
ax.set_ylabel("Sales, % above where they would have been")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
ax.set_title(
    "The ordering changes the first weeks, not where sales end up",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig07_orderings.png"), bbox_inches="tight")
