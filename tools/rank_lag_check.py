"""How often the usual tests pick the true lag order and rank on simulated brand data.

Runs the lag-order criteria of a levels VAR and the Johansen trace test on
200 seeds of ``simulate_brand_data``, at 3 and 5 years of weekly data. The
true model has one lag of differences (a VAR(2) in levels) and rank 2.
Takes about a minute. The numbers feed ``docs/choosing_rank_and_lags.md``.

    uv run python tools/rank_lag_check.py
"""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.api import VAR

from bayesian_vecm import select_coint_rank, simulate_brand_data

warnings.filterwarnings("ignore")

SEEDS = range(200)
SAMPLE_SIZES = (156, 260)
MAX_LAGS = 8


def run() -> pd.DataFrame:
    rows = []
    for n_obs in SAMPLE_SIZES:
        for seed in SEEDS:
            data = simulate_brand_data(n_obs=n_obs, seed=seed)
            y = data.endog.to_numpy()
            x = data.exog.to_numpy()
            row = {"n_obs": n_obs, "seed": seed}
            for label, exog in (("without_exog", None), ("with_exog", x)):
                orders = VAR(y, exog=exog).select_order(maxlags=MAX_LAGS, trend="c")
                for crit in ("aic", "bic", "hqic"):
                    row[f"p_{crit}_{label}"] = int(orders.selected_orders[crit])
            for k_ar_diff in (1, 2, 3, 4):
                for det_order in (0, 1):
                    result = select_coint_rank(y, k_ar_diff=k_ar_diff, det_order=det_order)
                    row[f"rank_k{k_ar_diff}_det{det_order}"] = result.rank
            rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    for n_obs, group in df.groupby("n_obs"):
        print(f"\n{n_obs} weeks, {len(group)} seeds")
        print("Share of seeds choosing each lag order p of the levels VAR (true p = 2)")
        for col in [c for c in df if c.startswith("p_")]:
            shares = group[col].value_counts(normalize=True).sort_index().round(2)
            print(f"  {col:18s} {shares.to_dict()}")
        print("Share of seeds choosing each rank (true rank = 2)")
        for col in [c for c in df if c.startswith("rank_")]:
            shares = group[col].value_counts(normalize=True).sort_index().round(2)
            print(f"  {col:16s} {shares.to_dict()}")


if __name__ == "__main__":
    report(run())
