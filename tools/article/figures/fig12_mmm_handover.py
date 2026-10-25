"""§12 chart: what a Bayesian MMM hands over as base sales (guide 07 setup, PyMC-Marketing 1.3.0).

Run in a separate environment with pymc-marketing==1.3.0 installed.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from bayesian_vecm import simulate_brand_data

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
CHANNELS = ["paid_search", "paid_social", "brand_spend"]
data = simulate_brand_data(
    total_sales=True,
    sales_noise=0.02,
    sales_adstock={"brand_spend": 0.3, "paid_search": 0.3, "paid_social": 0.5},
    sales_saturation={"paid_search": 3.0, "paid_social": 2.0},
)
exog, truth = data.exog, data.truth
total = data.sales["total_sales"]
parts = truth.sales_parts
spend = exog["brand_spend"]
mmm_data = pd.DataFrame(
    {
        "date": total.index,
        "paid_search": data.sales_drivers["paid_search"].to_numpy(),
        "paid_social": data.sales_drivers["paid_social"].to_numpy(),
        "brand_spend": (spend - spend.min()).to_numpy(),
        "promo": data.sales_drivers["promo"].to_numpy(),
    }
)
y = pd.Series(total.to_numpy(), name="y")
mmm = MMM(
    date_column="date",
    channel_columns=CHANNELS,
    control_columns=["promo"],
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
    time_varying_intercept=True,
    model_config={
        "intercept": Prior("Gamma", mu=0.6, sigma=0.3),
        "saturation_beta": Prior("HalfNormal", sigma=0.15, dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=0.2, dims="control"),
        "intercept_tvp_config": HSGPKwargs(m=100, eta_lam=1.0, ls_mu=78, ls_sigma=78 / 4),
    },
)
mmm.build_model(mmm_data, y)
mmm.add_original_scale_contribution_variable(
    var=[
        "channel_contribution",
        "control_contribution",
        "intercept_contribution",
        "yearly_seasonality_contribution",
    ]
)
mmm.fit(
    mmm_data,
    y,
    draws=500,
    tune=500,
    chains=2,
    target_accept=0.9,
    nuts_sampler="nutpie",
    random_seed=14,
    progressbar=False,
)
post = mmm.idata["posterior"]


def mean(n):
    return post[f"{n}_original_scale"].mean(("chain", "draw"))


channels = mean("channel_contribution")
control = mean("control_contribution").sum("control").to_numpy()
minus = total - channels.sum("channel").to_numpy() - control
trend = pd.Series(mean("intercept_contribution").to_numpy(), index=total.index)
truth_base = parts["base"] + parts["seasonality"]

INK, MUTED, BLUE, ORANGE = "#1f2328", "#6b7280", "#2563eb", "#e4572e"
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
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
ax.plot(total.index, total, color="#c9ced6", lw=0.9, label="Total sales")
ax.plot(total.index, truth_base, color=INK, lw=1.6, label="True base")
ax.plot(total.index, minus, color=BLUE, lw=1.1, label="Total minus the MMM's contributions")
ax.plot(total.index, trend, color=ORANGE, lw=2.4, label="The MMM's smooth trend")
ax.set_ylabel("£ a week")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
ax.set_title(
    "The MMM's trend smooths away the weekly movement the VECM needs",
    loc="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
    pad=12,
)
fig.tight_layout()
fig.savefig(str(OUT / "fig12_mmm_handover.png"), bbox_inches="tight")
print("divergences", int(mmm.idata["sample_stats"]["diverging"].sum()))
