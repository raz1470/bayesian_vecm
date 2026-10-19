"""Does the saturation option recover diminishing returns on simulated brand data?

Data settings from ``simulate_brand_data``, 260 weeks each, all with
geometric adstock in the truth (decays brand spend 0.5, PR 0.3, sponsorship
0.6) and the three channels as raw spend with mean 1:

* ``saturating`` (18 seeds): logistic saturation, lambda brand spend 5,
  PR 3, sponsorship 5. At mean spend one more unit is worth about half the
  average return.
* ``strong`` (5 seeds): lambda 8 on every channel.
* ``linear`` (5 seeds): lambda 0.01, so the effect is linear in raw spend.
  Measures what the option costs when there is no saturation.
* ``wide`` (5 seeds): as ``saturating`` with ``spend_log_sd=0.6``, so spend
  covers more of the curve.

Fits, all rank 2, one lag of differences, ``"co"``, adstock on the three
channels with ``adstock_max_lag=8``:

* ``linear``: the channels demeaned, as the guide prepares them.
* ``saturation``: raw channels, ``saturation`` on all three, default
  ``Gamma(2, 0.3)`` prior on lambda.
* ``saturation Gamma(3,1)``: the same with the pymc-marketing default
  ``Gamma(3, 1)`` prior on lambda. First 8 ``saturating`` seeds only.

Each runs 2 chains of 1000 draws after 1000 tuning steps with
``target_accept=0.95``.

Recorded per channel, all on awareness (the only series the channels act
on; the rest of the system is linear, so errors carry through to sales in
proportion):

* ``marginal``: total effect of one more unit at in-sample mean spend.
* ``step x<k>``: change in awareness growth from holding spend at ``k``
  times its mean instead of at its mean, for ``k`` in 0 (switched off),
  0.5, 1.5 and 2.
* ``lambda`` (saturation fits only).

Posterior median and 90% equal-tailed interval for each. Takes about 70
minutes on two cores. Results go to
``tools/saturation_check_results.csv``; finished (setting, seed, fit)
triples are skipped, so an interrupted run resumes.

    uv run python tools/saturation_check.py [fit ...]
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_vecm import BayesianVECM, simulate_brand_data

warnings.filterwarnings("ignore")

CHANNELS = ("brand_spend", "pr_reach", "sponsorship_reach")
DECAYS = {"brand_spend": 0.5, "pr_reach": 0.3, "sponsorship_reach": 0.6}
LAMBDAS = {"brand_spend": 5.0, "pr_reach": 3.0, "sponsorship_reach": 5.0}
SETTINGS = {
    "saturating": (range(18), {"exog_saturation": LAMBDAS}),
    "strong": (range(5), {"exog_saturation": dict.fromkeys(CHANNELS, 8.0)}),
    "linear": (range(5), {"exog_saturation": dict.fromkeys(CHANNELS, 0.01)}),
    "wide": (range(5), {"exog_saturation": LAMBDAS, "spend_log_sd": 0.6}),
}
ADSTOCK = {"adstock": list(CHANNELS), "adstock_max_lag": 8}
FITS = {
    "linear": {**ADSTOCK, "demean": True},
    "saturation": {**ADSTOCK, "saturation": list(CHANNELS)},
    "saturation Gamma(3,1)": {
        **ADSTOCK,
        "saturation": list(CHANNELS),
        "priors": {"saturation_lam": {"dist": "Gamma", "alpha": 3.0, "beta": 1.0}},
    },
}
SPEC = {"k_ar_diff": 1, "coint_rank": 2, "deterministic": "co"}
SAMPLING = {"draws": 1000, "tune": 1000, "chains": 2, "target_accept": 0.95}
STEPS = (0.0, 0.5, 1.5, 2.0)
AWARENESS = 1
RESULTS = Path(__file__).with_name("saturation_check_results.csv")


def truth_quantities(truth, exog: pd.DataFrame) -> dict[str, float]:
    out = {}
    for j, ch in enumerate(CHANNELS):
        mean = float(exog[ch].mean())
        out[f"marginal: {ch}"] = float(truth.marginal_effect(ch, mean)[AWARENESS])
        base = truth.response(ch, mean)[AWARENESS]
        for k in STEPS:
            out[f"step x{k:g}: {ch}"] = float(truth.response(ch, k * mean)[AWARENESS] - base)
        out[f"lambda: {ch}"] = float(truth.saturation[j])
    return out


def fit_quantities(model, exog: pd.DataFrame) -> dict[str, np.ndarray]:
    post = model.idata_.posterior
    out = {}
    if "saturation_lam" not in post:
        total = post["B_total"].values.reshape(-1, *post["B_total"].shape[2:])
        for ch in CHANNELS:
            b = total[:, AWARENESS, model.adstock_idx_.index(list(exog.columns).index(ch))]
            mean = float(exog[ch].mean())
            out[f"marginal: {ch}"] = b
            for k in STEPS:
                out[f"step x{k:g}: {ch}"] = b * (k - 1.0) * mean
        return out
    for ch in CHANNELS:
        mean = float(exog[ch].mean())
        curve = model.saturation_curve(ch, [mean * k for k in (1.0, *STEPS)])
        draws = curve.sel(response_variable="brand_awareness").values  # (C, D, L)
        draws = draws.reshape(-1, draws.shape[-1])
        for i, k in enumerate(STEPS, start=1):
            out[f"step x{k:g}: {ch}"] = draws[:, i] - draws[:, 0]
        marg = model.saturation_curve(ch, [mean], marginal=True)
        out[f"marginal: {ch}"] = marg.sel(response_variable="brand_awareness").values.ravel()
        pos = model.saturation_idx_.index(list(exog.columns).index(ch))
        out[f"lambda: {ch}"] = post["saturation_lam"].values[..., pos].ravel()
    return out


def one_setting(setting: str, seed: int, sim_kwargs: dict, fits: list[str]) -> list[dict]:
    data = simulate_brand_data(seed=seed, exog_adstock=DECAYS, **sim_kwargs)
    true = truth_quantities(data.truth, data.exog)
    rows = []
    for fit_name in fits:
        fit_kwargs = dict(FITS[fit_name])
        exog = data.exog
        if fit_kwargs.pop("demean", False):
            exog = exog.copy()
            exog[list(CHANNELS)] -= exog[list(CHANNELS)].mean()
        start = time.perf_counter()
        model = BayesianVECM(**SPEC, **fit_kwargs).fit(
            data.endog, exog=exog, random_seed=seed, progressbar=False, **SAMPLING
        )
        seconds = time.perf_counter() - start
        diverging = int(model.idata_.sample_stats["diverging"].sum())
        for name, draws in fit_quantities(model, data.exog).items():
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


def main(only: list[str]) -> None:
    fits = only or list(FITS)
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev["setting"], prev["seed"], prev["fit"], strict=True))
    for setting, (seeds, sim_kwargs) in SETTINGS.items():
        for seed in seeds:
            todo = [
                f
                for f in fits
                if (setting, seed, f) not in done
                and (f != "saturation Gamma(3,1)" or (setting == "saturating" and seed < 8))
            ]
            if not todo:
                continue
            start = time.perf_counter()
            rows = pd.DataFrame(one_setting(setting, seed, sim_kwargs, todo))
            rows.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
            print(f"{setting} seed {seed}: {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
