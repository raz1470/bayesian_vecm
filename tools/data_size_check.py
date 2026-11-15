"""How much data does the VECM need? Weekly or monthly, three years or five.

Simulates five years of weekly data with ``simulate_brand_data`` (the
default brand: every lift lasts, rank 2, one lag of differences) on 20
seeds. Each seed gives four data sets:

* ``weekly 5y``: all 261 weeks
* ``weekly 3y``: the first 157 weeks
* ``monthly 5y`` and ``monthly 3y``: the same weeks as calendar months.
  Sales are summed, awareness and consideration averaged, drivers averaged.
  Partial months are dropped, which leaves 60 and 36 months.

For each data set the script records:

* the rank chosen by the Johansen trace test (truth 2), with one lag of
  differences and a constant;
* the 1-year and 5-year return on brand spend from a Bayesian fit with the
  true rank and one lag of differences.

The return is the extra sales from one more unit of brand spend held for one
month (52 / 12 weeks), summed over 1 or 5 years, in months of sales. Weekly
fits get the month as a pulse of four weeks and a third, built from their
weekly multipliers. Monthly fits get one period. The truth comes from the
weekly parameters in the same way. Intervals are 89% and equal-tailed.

Results go to ``tools/data_size_check_results.csv``, one row per seed, data
set and window. Finished rows are skipped, so an interrupted run resumes.

    uv run python tools/data_size_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from bayesian_vecm import BayesianVECM, select_coint_rank, simulate_brand_data
from bayesian_vecm._irf import compute_exog_irf

warnings.filterwarnings("ignore")

SEEDS = range(20)
WEEKS_5Y, WEEKS_3Y = 261, 157
MONTH = 52 / 12
WINDOWS = {"1 year": 1, "5 years": 5}
SPEC = {"k_ar_diff": 1, "coint_rank": 2, "deterministic": "co"}
SAMPLING = {"draws": 1000, "tune": 1000, "chains": 2, "target_accept": 0.95}
TAIL = (5.5, 94.5)
SALES = "organic_sales"
RESULTS = Path(__file__).with_name("data_size_check_results.csv")


def monthly(endog: pd.DataFrame, exog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar months: sales summed, brand metrics and drivers averaged. Partial months dropped."""
    levels = np.exp(endog)
    month = levels.index.to_period("M")
    out = pd.DataFrame(
        {
            SALES: levels[SALES].groupby(month).sum(),
            "brand_awareness": levels["brand_awareness"].groupby(month).mean(),
            "brand_consideration": levels["brand_consideration"].groupby(month).mean(),
        }
    )
    full = levels[SALES].groupby(month).count() >= 4
    x = exog.groupby(month).mean()
    out, x = out[full], x[full]
    stamp = out.index.to_timestamp()
    return np.log(out).set_index(stamp), x.set_index(stamp)


def multipliers(alpha, beta, gamma, b, names, steps) -> np.ndarray:
    """Log-sales response to a one-period unit pulse of brand spend, per draw."""
    posterior = xr.Dataset(
        {
            "alpha": (("chain", "draw", "k", "r"), alpha[None]),
            "beta": (("chain", "draw", "k", "r"), beta[None]),
            "Gamma": (("chain", "draw", "k", "g"), gamma[None]),
            "Sigma": (("chain", "draw", "k", "k2"), np.broadcast_to(np.eye(3), (1, len(b), 3, 3))),
            "B": (("chain", "draw", "k", "m"), b[None]),
        },
        coords={"chain": [0], "draw": np.arange(len(b))},
    )
    mult = compute_exog_irf(SimpleNamespace(posterior=posterior), 1, steps, exog_names=names)
    return mult.sel(exog_variable="brand_spend").values[0, :, :, 0]


def weekly_returns(path: np.ndarray) -> dict[str, np.ndarray]:
    """Months of sales from one month of spend, given weekly pulse multipliers (draw, week)."""
    weights = np.array([1.0, 1.0, 1.0, 1.0, MONTH - 4])
    month_path = sum(
        w * np.pad(path, ((0, 0), (i, 0)))[:, : path.shape[1]] for i, w in enumerate(weights)
    )
    extra = np.expm1(month_path)
    return {lab: extra[:, : 52 * yrs].sum(axis=1) / MONTH for lab, yrs in WINDOWS.items()}


def monthly_returns(path: np.ndarray) -> dict[str, np.ndarray]:
    """Months of sales from one month of spend, given monthly pulse multipliers (draw, month)."""
    extra = np.expm1(path)
    return {lab: extra[:, : 12 * yrs].sum(axis=1) for lab, yrs in WINDOWS.items()}


def one(seed: int, label: str, endog: pd.DataFrame, exog: pd.DataFrame, truth: dict) -> list[dict]:
    rank = select_coint_rank(endog.to_numpy(), k_ar_diff=1, det_order=0).rank
    start = time.perf_counter()
    model = BayesianVECM(**SPEC).fit(
        endog, exog=exog, random_seed=seed, progressbar=False, **SAMPLING
    )
    seconds = time.perf_counter() - start
    post = model.idata_.posterior.to_dataset().stack(sample=("chain", "draw"))
    draws = lambda name: np.moveaxis(post[name].values, -1, 0)  # noqa: E731
    names = list(exog.columns)
    is_monthly = label.startswith("monthly")
    steps = 60 if is_monthly else 260
    path = multipliers(draws("alpha"), draws("beta"), draws("Gamma"), draws("B"), names, steps)
    fit = monthly_returns(path) if is_monthly else weekly_returns(path)
    diverging = int(model.idata_.sample_stats["diverging"].sum())
    rows = []
    for window, values in fit.items():
        finite = values[np.isfinite(values)]
        lo, med, hi = np.percentile(finite, [TAIL[0], 50, TAIL[1]])
        rows.append(
            {
                "seed": seed,
                "data": label,
                "n_periods": len(endog),
                "rank": rank,
                "window": window,
                "truth": truth[window],
                "median": med,
                "lower": lo,
                "upper": hi,
                "covered": lo <= truth[window] <= hi,
                "explosive_share": 1 - len(finite) / len(values),
                "diverging": diverging,
                "seconds": seconds,
            }
        )
    return rows


def main() -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev["seed"], prev["data"], strict=True))
    for seed in SEEDS:
        data = simulate_brand_data(n_obs=WEEKS_5Y, seed=seed)
        t = data.truth
        names = list(data.exog.columns)
        true_path = multipliers(t.alpha[None], t.beta[None], t.gamma[None], t.b[None], names, 260)
        truth = {k: v[0] for k, v in weekly_returns(true_path).items()}
        sets = {
            "weekly 5y": (data.endog, data.exog),
            "weekly 3y": (data.endog.iloc[:WEEKS_3Y], data.exog.iloc[:WEEKS_3Y]),
        }
        sets["monthly 5y"] = monthly(*sets["weekly 5y"])
        sets["monthly 3y"] = monthly(*sets["weekly 3y"])
        for label, (endog, exog) in sets.items():
            if (seed, label) in done:
                continue
            rows = one(seed, label, endog, exog, truth)
            out = pd.DataFrame(rows)
            out.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
            r = rows[1]
            print(
                f"seed {seed} {label} n={r['n_periods']} rank {r['rank']} "
                f"5y {r['median']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}] truth {r['truth']:.3f} "
                f"{r['seconds']:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
