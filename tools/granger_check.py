"""How often the Granger pretest finds the true links on simulated brand data.

Runs ``granger_pretest`` on 200 seeds of ``simulate_brand_data`` at two
sample shapes:

* 260 weeks, as simulated
* 60 months: 261 weeks summed (sales) or averaged (awareness, consideration
  and the drivers) into calendar months

The drivers are brand spend and the two macro series. The truth is the chain
spend -> awareness -> consideration -> sales with the macro series acting on
consideration. Nothing feeds back into the drivers. Takes about a minute. The
numbers feed ``docs/granger_pretest.md``.

    uv run python tools/granger_check.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bayesian_vecm import granger_pretest, simulate_brand_data

SEEDS = range(200)
DRIVERS = ["brand_spend", "interest_rate", "consumer_confidence"]


def monthly(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = simulate_brand_data(n_obs=261, seed=seed)
    levels = np.exp(data.endog)
    month = levels.index.to_period("M")
    endog = pd.DataFrame(
        {
            "organic_sales": levels["organic_sales"].groupby(month).sum(),
            "brand_awareness": levels["brand_awareness"].groupby(month).mean(),
            "brand_consideration": levels["brand_consideration"].groupby(month).mean(),
        }
    )
    return np.log(endog), data.exog[DRIVERS].groupby(month).mean()


def weekly(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = simulate_brand_data(seed=seed)
    return data.endog, data.exog[DRIVERS]


def run() -> None:
    for label, make in (("260 weeks", weekly), ("60 months", monthly)):
        tests, same, lags = [], [], []
        for seed in SEEDS:
            res = granger_pretest(*make(seed))
            lags.append(res.lags)
            both = pd.concat([res.tests, res.delay])
            tests.append(both.assign(hit=both["p_value"] < 0.05))
            same.append(res.same_period.assign(hit=res.same_period["p_value"] < 0.05))
        chosen = pd.Series(lags).value_counts().to_dict()
        print(f"\n{label}, {len(SEEDS)} seeds. Lags chosen: {chosen}")
        print("Share of seeds with p < 0.05")
        table = pd.concat(tests).groupby(["group", "cause", "effect"])["hit"].mean()
        print(table.round(2).to_string())
        print("Same period: share of seeds with p < 0.05")
        table = pd.concat(same).groupby(["variable_1", "variable_2"])["hit"].mean()
        print(table.round(2).to_string())


if __name__ == "__main__":
    run()
