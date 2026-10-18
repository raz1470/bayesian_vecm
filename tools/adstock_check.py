"""Does the adstock option recover carryover effects on simulated brand data?

Three data settings from ``simulate_brand_data``, 260 weeks each:

* ``carryover`` (18 seeds): geometric adstock in the truth, decays
  brand spend 0.5, PR 0.3, sponsorship 0.6.
* ``none`` (5 seeds): no carryover, every effect in the same week.
* ``delayed`` (5 seeds): ``exog_lag_share=0.6``, so 40% of each effect lands
  in the same week and 60% a week later. Geometric decay cannot peak at
  lag 1, so this is the wrong shape for adstock.

Four fits on each data set, all rank 2, one lag of differences, ``"co"``:

* ``same week``: ``exog_lags=0``
* ``free lags``: ``exog_lags=8``, Normal prior on every lag
* ``adstock``: ``adstock`` on the three channels, ``adstock_max_lag=8``,
  default flat ``Beta(1, 1)`` prior on the decay
* ``adstock Beta(1,3)``: the same with a ``Beta(1, 3)`` decay prior (the
  pymc-marketing default, mean 0.25), which this check showed pulls the
  decays and totals down

Each runs 2 chains of 1000 draws after 1000 tuning steps with
``target_accept=0.95``. The fits are not lined up on the same rows: the two
8-lag fits drop 7 more start-up rows than the same-week fit.

Recorded per channel: the total effect on awareness (sum of the lag
coefficients), the decay (adstock fit only), and the extra sales from one
more unit of the channel for one week, summed over 1 and 5 years, in units
of weekly sales. Posterior median and 90% equal-tailed interval for each.

Takes about 90 minutes on two cores. Results go to
``tools/adstock_check_results.csv``; finished (setting, seed, fit) triples
are skipped, so an interrupted run resumes.

    uv run python tools/adstock_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._irf import compute_exog_irf

warnings.filterwarnings("ignore")

CHANNELS = ("brand_spend", "pr_reach", "sponsorship_reach")
DECAYS = {"brand_spend": 0.5, "pr_reach": 0.3, "sponsorship_reach": 0.6}
SETTINGS = {
    "carryover": (range(18), {"exog_adstock": DECAYS}),
    "none": (range(5), {}),
    "delayed": (range(5), {"exog_lag_share": 0.6}),
}
FITS = {
    "same week": {"exog_lags": 0},
    "free lags": {"exog_lags": 8},
    "adstock": {"adstock": list(CHANNELS), "adstock_max_lag": 8},
    "adstock Beta(1,3)": {
        "adstock": list(CHANNELS),
        "adstock_max_lag": 8,
        "priors": {"adstock_alpha": {"dist": "Beta", "alpha": 1.0, "beta": 3.0}},
    },
}
SPEC = {"k_ar_diff": 1, "coint_rank": 2, "deterministic": "co"}
SAMPLING = {"draws": 1000, "tune": 1000, "chains": 2, "target_accept": 0.95}
WINDOWS = {"1 year": 52, "5 years": 260}
TRUTH_LAGS = 60  # geometric truth truncated here; 0.6**60 is negligible
AWARENESS = 1
RESULTS = Path(__file__).with_name("adstock_check_results.csv")


def true_b_lags(truth) -> np.ndarray:
    """True lag coefficients, shape ``(K, m * (TRUTH_LAGS + 1))``, lag-major."""
    k, m = truth.b.shape
    blocks = np.zeros((k, TRUTH_LAGS + 1, m))
    blocks[:, 0] = truth.b
    blocks[:, 1] += truth.b_lag
    w = truth.adstock
    carry = w.any()
    if carry:
        lags = np.arange(TRUTH_LAGS + 1)
        weights = (1 - w)[None, :] * w[None, :] ** lags[:, None]  # (L + 1, m)
        has = w > 0
        blocks[:, :, has] = truth.b_total[:, None, has] * weights[None, :, has]
    return blocks.reshape(k, -1)


def sales_returns(multipliers: xr.DataArray) -> dict[str, np.ndarray]:
    """Summed extra weekly sales per unit of each channel, one value per draw."""
    out = {}
    for ch in CHANNELS:
        sales = multipliers.sel(exog_variable=ch).values[..., 0]  # (C, D, H)
        path = np.expm1(sales.reshape(-1, sales.shape[-1]))
        for label, weeks in WINDOWS.items():
            out[f"return {label}: {ch}"] = path[:, :weeks].sum(axis=1)
    return out


def truth_quantities(truth, names) -> dict[str, float]:
    one = lambda a: a[None, None]  # noqa: E731
    posterior = xr.Dataset(
        {
            "alpha": (("chain", "draw", "k", "r"), one(truth.alpha)),
            "beta": (("chain", "draw", "k", "r"), one(truth.beta)),
            "Gamma": (("chain", "draw", "k", "g"), one(truth.gamma)),
            "Sigma": (("chain", "draw", "k", "k2"), one(np.eye(3))),
            "B": (("chain", "draw", "k", "c"), one(true_b_lags(truth))),
        },
        coords={"chain": [0], "draw": [0]},
    )
    mult = compute_exog_irf(
        SimpleNamespace(posterior=posterior),
        1,
        max(WINDOWS.values()),
        exog_names=names,
        exog_lags=TRUTH_LAGS,
    )
    out = {key: float(v[0]) for key, v in sales_returns(mult).items()}
    for j, ch in enumerate(CHANNELS):
        out[f"total: {ch}"] = float(truth.b_total[AWARENESS, j])
        out[f"decay: {ch}"] = float(truth.adstock[j])
    return out


def fit_quantities(model, names) -> dict[str, np.ndarray]:
    post = model.idata_.posterior
    window = model.exog_window_
    m = len(names)
    b = post["B"].values  # (C, D, K, m * (window + 1))
    b = b.reshape(-1, b.shape[2], window + 1, m)
    out = {}
    for j, ch in enumerate(CHANNELS):
        out[f"total: {ch}"] = b[:, AWARENESS, :, j].sum(axis=1)
        if "adstock_alpha" in post:
            pos = model.adstock_idx_.index(j)
            out[f"decay: {ch}"] = post["adstock_alpha"].values[..., pos].ravel()
    out |= sales_returns(model.exog_irf(max(WINDOWS.values())))
    return out


def one_setting(setting: str, seed: int, sim_kwargs: dict, fits: list[str]) -> list[dict]:
    data = simulate_brand_data(seed=seed, **sim_kwargs)
    names = list(data.exog.columns)
    true = truth_quantities(data.truth, names)
    rows = []
    for fit_name in fits:
        fit_kwargs = FITS[fit_name]
        start = time.perf_counter()
        model = BayesianVECM(**SPEC, **fit_kwargs).fit(
            data.endog, exog=data.exog, random_seed=seed, progressbar=False, **SAMPLING
        )
        seconds = time.perf_counter() - start
        diverging = int(model.idata_.sample_stats["diverging"].sum())
        for name, draws in fit_quantities(model, names).items():
            draws = draws[np.isfinite(draws)]
            q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95])
            rows.append(
                {
                    "setting": setting,
                    "seed": seed,
                    "fit": fit_name,
                    "quantity": name,
                    "truth": true[name],
                    "median": q50,
                    "mean": draws.mean(),
                    "lower": q05,
                    "upper": q95,
                    "covered": q05 <= true[name] <= q95,
                    "diverging": diverging,
                    "seconds": seconds,
                }
            )
    return rows


def main() -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev["setting"], prev["seed"], prev["fit"], strict=True))
    for setting, (seeds, sim_kwargs) in SETTINGS.items():
        for seed in seeds:
            todo = [f for f in FITS if (setting, seed, f) not in done]
            if not todo:
                continue
            start = time.perf_counter()
            rows = pd.DataFrame(one_setting(setting, seed, sim_kwargs, todo))
            rows.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
            print(f"{setting} seed {seed}: {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
