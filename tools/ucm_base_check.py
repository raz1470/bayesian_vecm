"""Which base sales should the VECM get when the analyst only has total sales?

Data from ``simulate_brand_data(total_sales=True)``, 260 weeks, 10 seeds.
Total sales in £ = base (the brand system) + a same-week hit from brand
spend + paid search + paid social + promotions + annual seasonality + weekly
noise outside the brand system. Every short-term effect is linear and in the
same week, so a UCM with regressors is correctly specified. The noise is
run at 0%, 2% and 5% of the reference level.

Arms, each a ``BayesianVECM`` with rank 2, one lag of differences and
``"co"``, the five brand drivers in levels as in the other guides:

* ``true_base``: the true base, ``exp(organic_sales)``. The oracle. It does not
  depend on the noise, so it runs once per seed.
* ``ucm_minus``: a ``statsmodels`` UCM on total sales in £
  (``level="lldtrend"``, two fixed annual harmonics, brand spend, paid
  search, paid social and promo as regressors). Base = total sales minus
  the regressor contributions and the seasonal component. The noise stays.
* ``ucm_level``: the same UCM. Base = its smoothed level (Cain's approach).
* ``ucm_keep_season``: the same UCM. Base = total sales minus the
  regressor contributions only. The seasonality stays in the base and the
  VECM gets the seasonal harmonics, as weekly changes, in the sales
  equation. This separates errors in the UCM's seasonal estimate from the
  rest.
* ``total``: log total sales straight into the VECM. The short-term drivers
  and the seasonal harmonics enter the sales equation only, as weekly
  changes, because they move the level of sales for one week, not for good.

The first week is dropped in every arm so all fit the same weeks.

Recorded, posterior median and 90% interval against the truth:

* the permanent lift in weekly sales, in £, from a one-unit (£1,000) brand
  spend pulse: the log response at week 260 times the average weekly sales
  of the modelled series over the last 52 weeks. The truth uses the true
  base.
* the lift at week 4, same units: how fast the effect builds.
* ``alpha`` for sales (truth -0.24 in the base arms) and the sales
  innovation standard deviation.
* Granger evidence that sales lead awareness or consideration (truth: no
  link), short run and long run.
* the autocorrelation and standard deviation of weekly changes in log base
  (true base: about 0.35).

Each fit runs 2 chains of 1000 draws after 1000 tuning steps. Results go to
``tools/ucm_base_check_results.csv``; finished (seed, noise, arm) rows are
skipped.

    uv run python tools/ucm_base_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_vecm import BayesianVECM, simulate_brand_data

warnings.filterwarnings("ignore")

SALES, AWARE, CONS = "organic_sales", "brand_awareness", "brand_consideration"
SHORT_TERM = ["brand_spend", "paid_search", "paid_social", "promo"]
SEEDS = range(10)
NOISES = (0.0, 0.02, 0.05)
ARMS = ("true_base", "ucm_minus", "ucm_level", "total", "ucm_keep_season")
HORIZON = 260
UNIT_COST = 1000.0  # £ per unit of brand spend
SPEC = {"k_ar_diff": 1, "coint_rank": 2, "deterministic": "co"}
SAMPLING = {"draws": 1000, "tune": 1000, "chains": 2, "target_accept": 0.9}
PERIOD = 365.25 / 7.0
RESULTS = Path(__file__).with_name("ucm_base_check_results.csv")


def harmonics(index: pd.DatetimeIndex, n: int = 2) -> pd.DataFrame:
    """Annual sine and cosine terms, week 0 at the first date."""
    angle = 2 * np.pi * np.arange(len(index)) / PERIOD
    cols = {}
    for h in range(1, n + 1):
        cols[f"sin{h}"] = np.sin(h * angle)
        cols[f"cos{h}"] = np.cos(h * angle)
    return pd.DataFrame(cols, index=index)


def ucm_bases(
    total: pd.Series,
    regressors: pd.DataFrame,
    n_harmonics: int = 2,
    stochastic_seasonal: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    """Three UCM bases: minus contributions and seasonality, smoothed level, minus contributions."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    res = UnobservedComponents(
        total,
        level="lldtrend",
        freq_seasonal=[{"period": PERIOD, "harmonics": n_harmonics}],
        stochastic_freq_seasonal=[stochastic_seasonal],
        exog=regressors,
    ).fit(disp=False, maxiter=1000)
    coef = res.params[[f"beta.{c}" for c in regressors.columns]].to_numpy()
    seasonal = res.freq_seasonal[0]["smoothed"]
    keep_season = total - regressors.to_numpy() @ coef
    minus = keep_season - seasonal
    level = pd.Series(res.level["smoothed"], index=total.index)
    info = {
        "ucm_irregular_sd": float(np.sqrt(res.params["sigma2.irregular"])),
        "ucm_level_sd": float(np.sqrt(res.params["sigma2.level"])),
        **{f"ucm_coef_{c}": float(b) for c, b in zip(regressors.columns, coef, strict=True)},
    }
    return minus, level, keep_season, info


def true_response(truth, steps: int) -> np.ndarray:
    """Log response of each series to a one-unit brand spend pulse, (steps + 1, K)."""
    pi = truth.alpha @ truth.beta.T
    impact = truth.b[:, 0]
    y, dy, out = np.zeros(3), np.zeros(3), []
    for h in range(steps + 1):
        dy = pi @ y + truth.gamma @ dy + (impact if h == 0 else 0.0)
        y = y + dy
        out.append(y.copy())
    return np.array(out)


def smoothness(log_base: pd.Series) -> dict:
    changes = log_base.diff().dropna()
    return {"change_autocorr": float(changes.autocorr()), "change_sd": float(changes.std())}


def interval(draws: np.ndarray) -> tuple[float, float, float]:
    q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95])
    return float(q05), float(q50), float(q95)


def one(seed: int, noise: float, arm: str) -> list[dict]:
    data = simulate_brand_data(seed=seed, total_sales=True, sales_noise=noise)
    return evaluate(data, arm, seed, {"noise": noise})


def coint_ranks(fit_endog: pd.DataFrame) -> dict:
    """Johansen rank of all three series (truth 2) and of sales with consideration (truth 1)."""
    from bayesian_vecm import select_coint_rank

    return {
        "rank_all": select_coint_rank(fit_endog, k_ar_diff=1).rank,
        "rank_sales_cons": select_coint_rank(fit_endog[[SALES, CONS]], k_ar_diff=1).rank,
    }


def in_sales_equation(exog: pd.DataFrame, short_run: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add weekly changes of ``short_run`` to ``exog``, kept out of the brand metric equations."""
    changes = short_run.diff()
    changes.columns = [f"d_{c}" for c in changes.columns]
    excluded = {AWARE: list(changes.columns), CONS: list(changes.columns)}
    return pd.concat([exog, changes], axis=1), excluded


def evaluate(
    data,
    arm: str,
    seed: int,
    tags: dict,
    n_harmonics: int = 2,
    stochastic_seasonal: bool = False,
    vecm_harmonics: int = 2,
    total_adstock: list[str] | None = None,
) -> list[dict]:
    """Build the base for ``arm``, fit the VECM and score it against the truth.

    ``total_adstock`` names short-term drivers that get the VECM's own
    adstock in the ``total`` arm (applied to their weekly changes).
    """
    exog = data.exog
    total = data.sales["total_sales"]
    regressors = pd.concat([exog[["brand_spend"]], data.sales_drivers], axis=1)[SHORT_TERM]

    info: dict = {}
    fit_exog = exog.copy()
    kwargs: dict = {}
    if arm == "true_base":
        base = data.truth.sales_parts["base"]
    elif arm in ("ucm_minus", "ucm_level", "ucm_keep_season"):
        minus, level, keep_season, info = ucm_bases(
            total, regressors, n_harmonics, stochastic_seasonal
        )
        base = {"ucm_minus": minus, "ucm_level": level, "ucm_keep_season": keep_season}[arm]
        if arm == "ucm_keep_season":
            fit_exog, kwargs["excluded"] = in_sales_equation(
                exog, harmonics(total.index, vecm_harmonics)
            )
    else:
        base = total
        fit_exog, kwargs["excluded"] = in_sales_equation(
            exog, pd.concat([regressors, harmonics(total.index, vecm_harmonics)], axis=1)
        )
        if total_adstock:
            kwargs["adstock"] = [f"d_{c}" for c in total_adstock]
    return score(data, base, fit_exog, kwargs, arm, seed, tags, info)


def score(data, base, fit_exog, kwargs, arm, seed, tags, info) -> list[dict]:
    """Fit the VECM with ``base`` as sales and score it against the truth."""
    endog, truth = data.endog, data.truth
    true_base = truth.sales_parts["base"]
    if (base <= 0).any():
        raise ValueError(f"non-positive base in seed {seed}, {tags}, arm {arm}")

    fit_endog = endog.copy()
    fit_endog[SALES] = np.log(base)
    fit_endog, fit_exog = fit_endog.iloc[1:], fit_exog.iloc[1:]
    ranks = coint_ranks(fit_endog)

    start = time.perf_counter()
    model = BayesianVECM(**SPEC, **kwargs).fit(
        fit_endog, exog=fit_exog, random_seed=seed, progressbar=False, **SAMPLING
    )
    seconds = time.perf_counter() - start

    irf = model.exog_irf(HORIZON).sel(response_variable=SALES, exog_variable="brand_spend")
    irf = irf.stack(sample=("chain", "draw")).transpose("sample", ...).values
    level_fit = float(base.iloc[-52:].mean())
    level_true = float(true_base.iloc[-52:].mean())
    path = true_response(truth, HORIZON)[:, 0]

    post = model.idata_.posterior
    # Sales is the first endogenous variable.
    alpha = post["alpha"].values[..., 0, 0].ravel()
    sigma = np.sqrt(post["Sigma"].values[..., 0, 0].ravel())
    causality = model.granger_causality()

    def evidence(table: pd.DataFrame, effect: str) -> float:
        row = table[(table["cause"] == SALES) & (table["effect"] == effect)]
        return float(row["evidence"].iloc[0])

    common = {
        "seed": seed,
        **tags,
        "arm": arm,
        "diverging": int(model.idata_.sample_stats["diverging"].sum()),
        "seconds": seconds,
        "evidence_sales_to_aware_short": evidence(causality.short_run, AWARE),
        "evidence_sales_to_cons_short": evidence(causality.short_run, CONS),
        "evidence_sales_to_aware_long": evidence(causality.long_run, AWARE),
        "evidence_sales_to_cons_long": evidence(causality.long_run, CONS),
        **smoothness(np.log(base)),
        **ranks,
        **info,
    }
    quantities = {
        "lift_permanent": (path[-1] * level_true, irf[:, -1] * level_fit),
        "lift_week4": (path[4] * level_true, irf[:, 4] * level_fit),
        "alpha_sales": (truth.alpha[0, 0], alpha),
        "sigma_sales": (np.sqrt(truth.sigma[0, 0]), sigma),
    }
    rows = []
    for name, (t, draws) in quantities.items():
        lower, median, upper = interval(draws)
        rows.append(
            {
                **common,
                "quantity": name,
                "truth": float(t),
                "median": median,
                "lower": lower,
                "upper": upper,
                "covered": lower <= t <= upper,
            }
        )
    return rows


def main() -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS, dtype={"arm": str})
        done = set(zip(prev["seed"], prev["noise"], prev["arm"], strict=True))
    for seed in SEEDS:
        for noise in NOISES:
            for arm in ARMS:
                if arm == "true_base" and noise != NOISES[0]:
                    continue
                if (seed, noise, arm) in done:
                    continue
                start = time.perf_counter()
                rows = pd.DataFrame(one(seed, noise, arm))
                if RESULTS.exists():
                    rows = pd.concat(
                        [pd.read_csv(RESULTS, dtype={"arm": str}), rows], ignore_index=True
                    )
                rows.to_csv(RESULTS, index=False)
                print(f"seed {seed} noise {noise} {arm}: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
