"""§7 chart: a surprise in sales fades; a surprise in awareness lasts and carries down the funnel.

Impulse responses from the plain model on the simulated brand, with the shocks
ordered down the funnel (awareness, consideration, sales). A surprise in
awareness then brings the same-week movement that usually comes with it (the
same as a generalised impulse response, Pesaran and Shin 1998); a surprise in
sales is the part no other series shares. The data seed is the first one where the fitted
awareness row of alpha is clearly near zero (both 89% intervals hold zero), so
the chart shows the pattern the truth has rather than a chance miss.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM

from bayesian_vecm import BayesianVECM, simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


def load(seed):
    d = simulate_brand_data(seed=seed, exog_unit_root=("interest_rate",))
    exog = d.exog.assign(interest_rate=d.exog["interest_rate"].diff()).iloc[1:]
    return d.endog.iloc[1:], exog


# Screen seeds quickly with statsmodels: awareness's two alphas both small.
for seed in range(14, 60):
    endog, exog = load(seed)
    res = VECM(endog, exog=exog, k_ar_diff=1, coint_rank=2, deterministic="co").fit()
    i = list(endog.columns).index("brand_awareness")
    if np.all(np.abs(res.tvalues_alpha[i]) < 1.0):
        break
print("data seed", seed)

model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co")
model.fit(
    endog,
    exog=exog,
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.90,
    random_seed=seed,
    progressbar=False,
)
alpha = model.idata_.posterior["alpha"].values[..., i, :]
lo, hi = np.percentile(alpha.reshape(-1, alpha.shape[-1]), [5.5, 94.5], axis=0)
print("awareness alpha 89%:", np.round(lo, 3), np.round(hi, 3))

H = 52
girf = model.irf(
    steps=H, method="cholesky", order=["brand_awareness", "brand_consideration", "organic_sales"]
)
INK, MUTED = "#1f2328", "#6b7280"
COL = {
    "organic_sales": ("#2563eb", "Sales"),
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
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, sharey=True)
for ax, shock, title in (
    (axes[0], "organic_sales", "A surprise in sales"),
    (axes[1], "brand_awareness", "A surprise in awareness"),
):
    v = girf.sel(shock_variable=shock)
    for resp in ["brand_awareness", "brand_consideration", "organic_sales"]:
        d = 100 * v.sel(response_variable=resp).stack(s=("chain", "draw")).transpose("s", "horizon")
        d = d.values
        c, lab = COL[resp]
        lo, hi = np.percentile(d, [5.5, 94.5], axis=0)
        ax.fill_between(h, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(h, np.median(d, axis=0), color=c, lw=2.2, label=lab)
        if shock == "brand_awareness":
            ax.text(
                H + 0.8,
                np.median(d, axis=0)[-1],
                lab,
                color=c,
                va="center",
                fontsize=10.5,
                fontweight="bold",
            )
        print(
            title,
            lab,
            "week 0",
            round(np.median(d[:, 0]), 2),
            "week 52",
            round(np.median(d[:, -1]), 2),
        )
    ax.axhline(0, color="#c9ced6", lw=0.8)
    ax.set_xlim(0, H)
    ax.set_xticks([0, 13, 26, 39, 52])
    ax.set_xlabel("Weeks after the surprise")
    ax.set_title(title, loc="left", fontsize=11.5, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, loc="upper right")
axes[0].set_ylabel("% from where it would have been")
fig.suptitle(
    "Followers get pulled back. The leader doesn't.",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig07_shocks.png"), bbox_inches="tight")
print("saved")
