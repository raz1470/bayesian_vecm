"""Does base sales still move with consideration when search spend follows demand?

Paid search budgets often rise and fall with demand: more people search
when interest in the brand grows, and spend goes up with the searches. If
the UCM then credits search with sales that the brand's demand produced, it
strips that demand out of base sales. The base may stop moving with
consideration, and the long-run return on brand spend would be understated.

Data from ``simulate_brand_data(total_sales=True, sales_noise=0.02,
paid_search_follows=s)``, 260 weeks, 10 seeds. ``s`` is 0, 0.1 and 0.25:
the log of paid search spend gets ``s`` times the standardised four-week
growth in consideration. At 0.1 the correlation between log search spend and
that growth is about 0.4, at 0.25 about 0.7. The true search effect on sales
is unchanged and the base does not depend on ``s``.

Arms, as in ``tools/ucm_base_check.py``: ``true_base`` (once per seed),
``ucm_minus``, ``ucm_keep_season`` and ``total``. Also recorded: the UCM's
search coefficient against the truth, and the Johansen rank of the three
series (truth 2) and of sales with consideration (truth 1).

Results go to ``tools/demand_feedback_check_results.csv``; finished rows are
skipped.

    uv run python tools/demand_feedback_check.py
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
STRENGTHS = (0.0, 0.1, 0.25)
ARMS = ("true_base", "ucm_minus", "ucm_keep_season", "total")
NOISE = 0.02
RESULTS = Path(__file__).with_name("demand_feedback_check_results.csv")


def one(seed: int, strength: float, arm: str) -> list[dict]:
    data = simulate_brand_data(
        seed=seed, total_sales=True, sales_noise=NOISE, paid_search_follows=strength
    )
    rows = evaluate(data, arm, seed, {"strength": strength, "noise": NOISE})
    true_coef = data.truth.sales_short_term["coef"]["paid_search"]
    for row in rows:
        row["true_coef_paid_search"] = true_coef
    return rows


def main() -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS, dtype={"arm": str})
        done = set(zip(prev["seed"], prev["strength"], prev["arm"], strict=True))
    for seed in SEEDS:
        for strength in STRENGTHS:
            for arm in ARMS:
                if arm == "true_base" and strength != STRENGTHS[0]:
                    continue
                if (seed, strength, arm) in done:
                    continue
                start = time.perf_counter()
                rows = pd.DataFrame(one(seed, strength, arm))
                if RESULTS.exists():
                    rows = pd.concat(
                        [pd.read_csv(RESULTS, dtype={"arm": str}), rows], ignore_index=True
                    )
                rows.to_csv(RESULTS, index=False)
                elapsed = time.perf_counter() - start
                print(f"seed {seed} strength {strength} {arm}: {elapsed:.0f} s", flush=True)


if __name__ == "__main__":
    main()
