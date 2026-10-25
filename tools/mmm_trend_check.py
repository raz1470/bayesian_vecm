"""Which PyMC-Marketing trend gives the VECM a usable base, and is a one-step VECM as good?

Data from ``simulate_brand_data(total_sales=True)``, 260 weeks, 5 seeds, 2%
weekly noise outside the brand system. The short-term part now has carryover
and saturation, which a UCM with linear regressors cannot fit:

* adstock decays: brand spend 0.3, paid search 0.3, paid social 0.5;
* logistic saturation: paid search 3, paid social 2.

Two-stage arms: a PyMC-Marketing ``MMM`` on total sales in £ with geometric
adstock, logistic saturation, two yearly Fourier modes, promo as a control
and an HSGP time-varying intercept. The HSGP lengthscale prior has mean 26,
78 or 156 weeks (standard deviation a quarter of the mean). Channels: paid
search, paid social and brand spend (shifted to start at zero). Three bases
from each fit, posterior means in £:

* ``mmm{ls}_keep_season``: total minus channel and control contributions.
  The VECM gets the seasonal harmonics in the sales equation.
* ``mmm{ls}_minus``: the same minus the seasonal contribution too.
* ``mmm{ls}_trend``: the time-varying intercept itself.

One-step and reference arms:

* ``total_adstock``: log total sales straight into the VECM. The short-term
  drivers and harmonics enter the sales equation as weekly changes, with the
  VECM's own ``adstock=`` on the three channels. Linear in spend, so the
  saturation is not captured.
* ``ucm_keep_season``: the UCM recipe, now misspecified (no carryover or
  saturation).
* ``true_base``: the oracle.

Scored as in ``tools/ucm_base_check.py``. MMM bases are cached in
``tools/mmm_trend_check_bases/`` so a restart does not refit. Results go to
``tools/mmm_trend_check_results.csv``. Needs ``pymc-marketing`` (not a
package dependency); tested with 1.3.0.

    uv run --with pymc-marketing python tools/mmm_trend_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ucm_base_check import evaluate, harmonics, in_sales_equation, score

from bayesian_vecm import simulate_brand_data

warnings.filterwarnings("ignore")

SEEDS = range(5)
NOISE = 0.02
SALES_ADSTOCK = {"brand_spend": 0.3, "paid_search": 0.3, "paid_social": 0.5}
SALES_SATURATION = {"paid_search": 3.0, "paid_social": 2.0}
LENGTHSCALES = (26, 78, 156)
CHANNELS = ["paid_search", "paid_social", "brand_spend"]
MMM_SAMPLING = {"chains": 2, "draws": 500, "tune": 500, "target_accept": 0.9}
HERE = Path(__file__).parent
RESULTS = HERE / "mmm_trend_check_results.csv"
CACHE = HERE / "mmm_trend_check_bases"


def simulate(seed: int):
    return simulate_brand_data(
        seed=seed,
        total_sales=True,
        sales_noise=NOISE,
        sales_adstock=SALES_ADSTOCK,
        sales_saturation=SALES_SATURATION,
    )


def mmm_bases(data, seed: int, ls: float) -> pd.DataFrame:
    """Fit the MMM and return its bases and diagnostics, one row per week."""
    path = CACHE / f"seed{seed}_ls{ls}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)

    from pymc_extras.prior import Prior
    from pymc_marketing.hsgp_kwargs import HSGPKwargs
    from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

    spend = data.exog["brand_spend"]
    x = pd.DataFrame(
        {
            "date": data.sales.index,
            "paid_search": data.sales_drivers["paid_search"].to_numpy(),
            "paid_social": data.sales_drivers["paid_social"].to_numpy(),
            "brand_spend": (spend - spend.min()).to_numpy(),
            "promo": data.sales_drivers["promo"].to_numpy(),
        }
    )
    y = pd.Series(data.sales["total_sales"].to_numpy(), name="y")
    model_config = {
        # Positive priors on the scaled target, as an analyst would set them.
        "intercept": Prior("Gamma", mu=0.6, sigma=0.3),
        "saturation_beta": Prior("HalfNormal", sigma=0.15, dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=0.2, dims="control"),
        "intercept_tvp_config": HSGPKwargs(m=100, eta_lam=1.0, ls_mu=ls, ls_sigma=ls / 4),
    }
    mmm = MMM(
        date_column="date",
        channel_columns=CHANNELS,
        control_columns=["promo"],
        adstock=GeometricAdstock(l_max=8),
        saturation=LogisticSaturation(),
        yearly_seasonality=2,
        time_varying_intercept=True,
        model_config=model_config,
    )
    mmm.build_model(x, y)
    mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
            "control_contribution",
            "intercept_contribution",
            "yearly_seasonality_contribution",
        ]
    )
    start = time.perf_counter()
    mmm.fit(x, y, nuts_sampler="nutpie", random_seed=seed, progressbar=False, **MMM_SAMPLING)
    seconds = time.perf_counter() - start

    post = mmm.idata["posterior"]

    def mean(name):
        return post[f"{name}_original_scale"].mean(("chain", "draw"))

    channels = mean("channel_contribution")
    control = mean("control_contribution").sum("control").to_numpy()
    season = mean("yearly_seasonality_contribution").to_numpy()
    total = data.sales["total_sales"].to_numpy()
    keep = total - channels.sum("channel").to_numpy() - control
    out = pd.DataFrame(
        {
            "keep_season": keep,
            "minus": keep - season,
            "trend": mean("intercept_contribution").to_numpy(),
            **{f"contrib_{c}": channels.sel(channel=c).to_numpy() for c in CHANNELS},
        },
        index=data.sales.index,
    )
    out["mmm_seconds"] = seconds
    out["mmm_diverging"] = int(mmm.idata["sample_stats"]["diverging"].sum())
    out["mmm_ls"] = float(post["intercept_latent_process_raw_ls"].mean())
    CACHE.mkdir(exist_ok=True)
    out.to_csv(path)
    return out


def mmm_arm(data, seed: int, ls: float, kind: str) -> list[dict]:
    bases = mmm_bases(data, seed, ls)
    parts = data.truth.sales_parts
    info = {
        "mmm_seconds": float(bases["mmm_seconds"].iloc[0]),
        "mmm_diverging": int(bases["mmm_diverging"].iloc[0]),
        "mmm_ls": float(bases["mmm_ls"].iloc[0]),
        **{
            f"mmm_contrib_error_{c}": float(bases[f"contrib_{c}"].mean() / parts[c].mean() - 1)
            for c in ("paid_search", "paid_social")
        },
    }
    fit_exog, kwargs = data.exog.copy(), {}
    if kind == "keep_season":
        fit_exog, kwargs["excluded"] = in_sales_equation(data.exog, harmonics(data.sales.index))
    return score(data, bases[kind], fit_exog, kwargs, f"mmm{ls}_{kind}", seed, {}, info)


def one(seed: int, arm: str) -> list[dict]:
    data = simulate(seed)
    if arm.startswith("mmm"):
        ls_text, kind = arm[3:].split("_", 1)
        return mmm_arm(data, seed, int(ls_text), kind)
    if arm == "total_adstock":
        return evaluate(data, "total", seed, {}, total_adstock=CHANNELS)
    return evaluate(data, arm, seed, {})


ARMS = (
    "true_base",
    "ucm_keep_season",
    "total_adstock",
    *(f"mmm{ls}_{kind}" for ls in LENGTHSCALES for kind in ("keep_season", "minus", "trend")),
)


def main() -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS, dtype={"arm": str})
        done = set(zip(prev["seed"], prev["arm"], strict=True))
    for seed in SEEDS:
        for arm in ARMS:
            if (seed, arm) in done:
                continue
            start = time.perf_counter()
            rows = pd.DataFrame(one(seed, arm))
            rows["arm"] = arm
            if RESULTS.exists():
                rows = pd.concat(
                    [pd.read_csv(RESULTS, dtype={"arm": str}), rows], ignore_index=True
                )
            rows.to_csv(RESULTS, index=False)
            print(f"seed {seed} {arm}: {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
