"""Does base sales from an MMM survive performance spend that follows demand?

Performance budgets often chase demand: when more people are in the market,
search volume rises and so does search spend. An MMM fitted on total sales
then sees search spend move with sales that the brand produced. It can credit
search with some of that demand and take it out of base sales, so the base
the VECM receives is missing part of the brand's own growth.

Data as in ``tools/mmm_trend_check.py`` (260 weeks, 2% weekly noise, carryover
and saturation on the short-term channels), plus
``paid_search_follows=s``: the log of paid search spend gets ``s`` times the
standardised four-week growth in consideration, which base sales follow.
``s`` is 0, 0.25 and 0.5. The true search effect and the base do not depend
on ``s``. 10 seeds.

Arms:

* ``true_base``: the oracle, once per seed (the base does not depend on ``s``).
* ``mmm78_minus``: a PyMC-Marketing MMM with a time-varying intercept
  (lengthscale prior mean 78 weeks), base = total sales minus the MMM's
  channel, control and seasonal contributions. The route the article
  recommends.

Also recorded per seed and strength: the correlation of log search spend with
the demand signal; the MMM's search contribution against the truth; and a
Granger tests of whether the last four weeks' changes in log total sales, or
in log consideration, help predict log search spend (p-values). Both need
only data an analyst has.

Results go to ``tools/perf_feedback_check_results.csv``. MMM bases are cached
in ``tools/perf_feedback_check_bases/``. Needs ``pymc-marketing`` (tested
with 1.3.0).

    uv run --with pymc-marketing==1.3.0 python tools/perf_feedback_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import mmm_trend_check as mtc
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from ucm_base_check import evaluate

from bayesian_vecm import simulate_brand_data

warnings.filterwarnings("ignore")

SEEDS = range(10)
STRENGTHS = (0.0, 0.25, 0.5)
LS = 78
HERE = Path(__file__).parent
RESULTS = HERE / "perf_feedback_check_results.csv"
CACHE = HERE / "perf_feedback_check_bases"


def simulate(seed: int, strength: float):
    return simulate_brand_data(
        seed=seed,
        total_sales=True,
        sales_noise=mtc.NOISE,
        sales_adstock=mtc.SALES_ADSTOCK,
        sales_saturation=mtc.SALES_SATURATION,
        paid_search_follows=strength,
    )


def feedback_info(data) -> dict:
    """What an analyst could see, and how strong the feedback really is."""
    search = np.log(data.sales_drivers["paid_search"].to_numpy())
    sales = np.log(data.sales["total_sales"].to_numpy())
    cons = data.endog["brand_consideration"].to_numpy()
    demand = np.zeros_like(cons)
    demand[4:] = cons[4:] - cons[:-4]

    def granger(cause: np.ndarray) -> float:
        frame = np.column_stack([search[1:], np.diff(cause)])
        return float(grangercausalitytests(frame, maxlag=4, verbose=False)[4][0]["ssr_ftest"][1])

    return {
        "corr_search_demand": float(np.corrcoef(search[4:], demand[4:])[0, 1]),
        "granger_p_sales": granger(sales),
        "granger_p_consideration": granger(cons),
    }


def mmm_rows(data, seed: int, strength: float) -> list[dict]:
    mtc.CACHE = CACHE / f"s{strength}"
    mtc.CACHE.mkdir(parents=True, exist_ok=True)
    bases = mtc.mmm_bases(data, seed, LS)
    parts = data.truth.sales_parts
    info = {
        "mmm_seconds": float(bases["mmm_seconds"].iloc[0]),
        "mmm_diverging": int(bases["mmm_diverging"].iloc[0]),
        **{
            f"mmm_contrib_error_{c}": float(bases[f"contrib_{c}"].mean() / parts[c].mean() - 1)
            for c in ("paid_search", "paid_social")
        },
        "base_error_mean": float(bases["minus"].mean() / parts["base"].mean() - 1),
        "base_growth_ratio": float(
            np.log(bases["minus"].iloc[-52:].mean() / bases["minus"].iloc[:52].mean())
            / np.log(parts["base"].iloc[-52:].mean() / parts["base"].iloc[:52].mean())
        ),
        **feedback_info(data),
    }
    return mtc.score(data, bases["minus"], data.exog.copy(), {}, f"mmm{LS}_minus", seed, {}, info)


def main() -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev["seed"], prev["strength"], prev["arm"], strict=True))
    for seed in SEEDS:
        for strength in STRENGTHS:
            data = simulate(seed, strength)
            todo = []
            if strength == 0.0 and (seed, 0.0, "true_base") not in done:
                todo.append("true_base")
            if (seed, strength, f"mmm{LS}_minus") not in done:
                todo.append(f"mmm{LS}_minus")
            for arm in todo:
                start = time.perf_counter()
                if arm == "true_base":
                    rows = evaluate(data, "true_base", seed, {})
                else:
                    rows = mmm_rows(data, seed, strength)
                out = pd.DataFrame(rows).assign(strength=strength, arm=arm)
                if RESULTS.exists():
                    out = pd.concat([pd.read_csv(RESULTS), out], ignore_index=True)
                out.to_csv(RESULTS, index=False)
                print(
                    f"seed {seed} s={strength} {arm}: {time.perf_counter() - start:.0f} s",
                    flush=True,
                )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
