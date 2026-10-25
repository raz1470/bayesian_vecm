"""How robust is "keep the seasonality in the base and model it in the VECM"?

``tools/ucm_base_check.py`` found that subtracting the UCM's seasonal
estimate makes the lift build too slowly, and that keeping the seasonal
swing in the base and giving the VECM's sales equation the harmonics fixes
it. That was one setting: a fixed annual pattern with two harmonics, five
years of data and a UCM with the right number of harmonics. This checks the
two routes away from it.

Data from ``simulate_brand_data(total_sales=True, sales_noise=0.02)``,
10 seeds. Every short-term effect is linear and same-week. Three
experiments:

* ``harmonics``: the UCM (and, for the kept route, the VECM) uses 1, 2 or 3
  harmonics against a truth of 2. 260 weeks.
* ``length``: 156, 260 and 416 weeks (3, 5 and 8 years), 2 harmonics. The
  level and an annual cycle should be easier to tell apart with more years.
* ``drift``: the seasonal pattern drifts (``seasonal_drift`` 0.02 and 0.05;
  at 0.05 the pattern in year 5 correlates about 0.77 with year 1). The UCM
  seasonal is fixed or stochastic. The kept route's VECM harmonics are fixed,
  so they are misspecified here.

Arms: ``ucm_minus`` (seasonal estimate removed), ``ucm_keep_season``
(seasonality kept, harmonics in the VECM) and, in the length experiment,
``true_base``. Scored as in ``tools/ucm_base_check.py``. Results go to
``tools/seasonality_check_results.csv``; finished rows are skipped.

    uv run python tools/seasonality_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd
from ucm_base_check import evaluate

from bayesian_vecm import simulate_brand_data

warnings.filterwarnings("ignore")

SEEDS = range(10)
NOISE = 0.02
ROUTES = ("ucm_minus", "ucm_keep_season")
RESULTS = Path(__file__).with_name("seasonality_check_results.csv")
KEYS = ["experiment", "seed", "n_harmonics", "n_obs", "drift", "stochastic", "arm"]


def settings():
    """Yield (experiment, n_harmonics, n_obs, drift, stochastic, arm)."""
    for h in (1, 2, 3):
        for arm in ROUTES:
            yield "harmonics", h, 260, 0.0, False, arm
    for n in (156, 260, 416):
        for arm in ("true_base", *ROUTES):
            yield "length", 2, n, 0.0, False, arm
    for drift in (0.02, 0.05):
        for stochastic in (False, True):
            for arm in ROUTES:
                yield "drift", 2, 260, drift, stochastic, arm


def one(seed, experiment, n_harmonics, n_obs, drift, stochastic, arm) -> list[dict]:
    data = simulate_brand_data(
        n_obs, seed=seed, total_sales=True, sales_noise=NOISE, seasonal_drift=drift
    )
    tags = {
        "experiment": experiment,
        "n_harmonics": n_harmonics,
        "n_obs": n_obs,
        "drift": drift,
        "stochastic": stochastic,
    }
    return evaluate(
        data,
        arm,
        seed,
        tags,
        n_harmonics=n_harmonics,
        stochastic_seasonal=stochastic,
        vecm_harmonics=n_harmonics,
    )


def main() -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS, dtype={"arm": str})
        done = set(map(tuple, prev[KEYS].drop_duplicates().to_numpy().tolist()))
    for experiment, h, n, drift, stochastic, arm in settings():
        for seed in SEEDS:
            key = (experiment, seed, h, n, drift, stochastic, arm)
            if key in done:
                continue
            start = time.perf_counter()
            rows = pd.DataFrame(one(seed, experiment, h, n, drift, stochastic, arm))
            if RESULTS.exists():
                rows = pd.concat(
                    [pd.read_csv(RESULTS, dtype={"arm": str}), rows], ignore_index=True
                )
            rows.to_csv(RESULTS, index=False)
            print(f"{key}: {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
