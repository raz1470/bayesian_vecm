"""Do adstock, saturation and same-week links work together?

Data from ``simulate_brand_data``, 260 weeks, 10 seeds, with all three in
the truth:

* geometric adstock, decays brand spend 0.5, PR 0.3, sponsorship 0.6;
* logistic saturation, lambda brand spend 5, PR 3, sponsorship 5;
* a same-week link from awareness to consideration of 0.2.

Fits, all rank 2, one lag of differences, ``"co"``:

* ``linear``: the channels demeaned and in the same week, no links. The
  model ``statsmodels`` can fit.
* ``full``: ``adstock``, ``saturation``, ``contemporaneous`` and
  ``excluded`` (the channels excluded from consideration).

Each runs 2 chains of 1000 draws after 1000 tuning steps with
``target_accept=0.95``.

Recorded: the response of sales to holding brand spend at ``k`` times its
mean for 52 weeks from the end of the sample, against holding it at the
mean, for ``k`` in 0, 0.5, 1.5 and 2. Two numbers per level: the uplift in
sales at week 52 and the uplift summed over the 52 weeks (both in % of
sales). Forecasts use common random numbers, so the noise cancels. Also the
link, the brand spend decay and lambda. Posterior median and 90%
equal-tailed interval. Takes about 25 minutes on two cores. Results go to
``tools/combined_check_results.csv``; finished (seed, fit) pairs are skipped.

    uv run python tools/combined_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._saturation import logistic_saturation

warnings.filterwarnings("ignore")

SALES, AWARE, CONS = "organic_sales", "brand_awareness", "brand_consideration"
CHANNELS = ["brand_spend", "pr_reach", "sponsorship_reach"]
DECAYS = {"brand_spend": 0.5, "pr_reach": 0.3, "sponsorship_reach": 0.6}
LAMBDAS = {"brand_spend": 5.0, "pr_reach": 3.0, "sponsorship_reach": 5.0}
LINK = 0.2
SEEDS = range(10)
LEVELS = (0.0, 0.5, 1.5, 2.0)
WEEKS = 52
SPEC = {"k_ar_diff": 1, "coint_rank": 2, "deterministic": "co"}
SAMPLING = {"draws": 1000, "tune": 1000, "chains": 2, "target_accept": 0.95}
FITS = {
    "linear": {},
    "full": {
        "adstock": CHANNELS,
        "saturation": CHANNELS,
        "contemporaneous": {CONS: [AWARE]},
        "excluded": {CONS: CHANNELS},
    },
}
RESULTS = Path(__file__).with_name("combined_check_results.csv")


def true_uplift(truth, exog: pd.DataFrame, level: float) -> np.ndarray:
    """Sales log uplift per week from holding brand spend at ``level``."""
    j = list(exog.columns).index("brand_spend")
    w, lam, scale = truth.adstock[j], truth.saturation[j], truth.spend_scale[j]
    # Adstock state at the end of the sample, then held at level or the mean.
    stock = float(exog["brand_spend"].iloc[0])
    for x in exog["brand_spend"].to_numpy()[1:]:
        stock = (1 - w) * x + w * stock
    mean = float(exog["brand_spend"].mean())

    def growth_shifts(x: float) -> np.ndarray:
        a, out = stock, []
        for _ in range(WEEKS):
            a = (1 - w) * x + w * a
            out.append(truth.b_saturation[:, j] * logistic_saturation(a / scale, lam))
        return np.array(out)

    shift = growth_shifts(level) - growth_shifts(mean)
    pi = truth.alpha @ truth.beta.T
    y, dy, path = np.zeros(3), np.zeros(3), []
    for t in range(WEEKS):
        dy = pi @ y + truth.gamma @ dy + shift[t]
        y = y + dy
        path.append(y[0])
    return np.array(path)


def fit_uplift(model, exog: pd.DataFrame, level: float, demean: bool) -> np.ndarray:
    """Posterior sales log uplift per week, shape (draws, WEEKS)."""
    mean = exog.mean()
    # Every driver at its mean: zero for the demeaned fit, raw spend otherwise.
    base = np.zeros((WEEKS, exog.shape[1])) if demean else np.tile(mean.to_numpy(), (WEEKS, 1))
    scen = base.copy()
    j = list(exog.columns).index("brand_spend")
    scen[:, j] = level - (mean.iloc[j] if demean else 0.0)
    ys = []
    for x in (base, scen):
        fc = model.sample_posterior_predictive(WEEKS, exog_future=x, random_seed=0)
        y = fc.posterior_predictive["y"].sel(variable=SALES).values
        ys.append(y.reshape(-1, WEEKS))
    return ys[1] - ys[0]


def summarise(rows, seed, fit, name, truth_value, draws, extra) -> None:
    q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95])
    rows.append(
        {
            "seed": seed,
            "fit": fit,
            "quantity": name,
            "truth": truth_value,
            "median": q50,
            "lower": q05,
            "upper": q95,
            "covered": q05 <= truth_value <= q95,
            **extra,
        }
    )


def one(seed: int, fit: str) -> list[dict]:
    data = simulate_brand_data(
        seed=seed,
        exog_adstock=DECAYS,
        exog_saturation=LAMBDAS,
        contemporaneous={CONS: {AWARE: LINK}},
    )
    endog, exog, truth = data.endog, data.exog, data.truth
    demean = fit == "linear"
    fit_exog = exog.copy()
    if demean:
        fit_exog[CHANNELS] -= fit_exog[CHANNELS].mean()
    start = time.perf_counter()
    model = BayesianVECM(**SPEC, **FITS[fit]).fit(
        endog, exog=fit_exog, random_seed=seed, progressbar=False, **SAMPLING
    )
    extra = {
        "diverging": int(model.idata_.sample_stats["diverging"].sum()),
        "seconds": time.perf_counter() - start,
    }
    mean = float(exog["brand_spend"].mean())
    rows: list[dict] = []
    for k in LEVELS:
        t = 100 * true_uplift(truth, exog, k * mean)
        f = 100 * fit_uplift(model, exog, k * mean, demean)
        summarise(rows, seed, fit, f"sales week 52: x{k:g}", t[-1], f[:, -1], extra)
        summarise(rows, seed, fit, f"sales summed: x{k:g}", t.sum(), f.sum(axis=1), extra)
    if fit == "full":
        post = model.idata_.posterior
        draws = {
            "link": post["A0_links"].values.ravel(),
            "decay: brand_spend": post["adstock_alpha"].values[..., 0].ravel(),
            "lambda: brand_spend": post["saturation_lam"].values[..., 0].ravel(),
        }
        truths = {"link": LINK, "decay: brand_spend": 0.5, "lambda: brand_spend": 5.0}
        for name, d in draws.items():
            summarise(rows, seed, fit, name, truths[name], d, extra)
    return rows


def main() -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev["seed"], prev["fit"], strict=True))
    for seed in SEEDS:
        for fit in FITS:
            if (seed, fit) in done:
                continue
            start = time.perf_counter()
            rows = pd.DataFrame(one(seed, fit))
            if RESULTS.exists():
                rows = pd.concat([pd.read_csv(RESULTS), rows], ignore_index=True)
            rows.to_csv(RESULTS, index=False)
            print(f"seed {seed} {fit}: {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
