"""How often the same-week IV test finds a link on simulated brand data.

Runs ``granger_pretest(contemporaneous=..., excluded=...)`` on 200 seeds of
``simulate_brand_data`` with no same-week link and with awareness ->
consideration at 0.2. Two sample shapes:

* 261 weeks, as simulated
* 58 months: weeks summed (sales) or averaged (awareness, consideration and
  the drivers) into calendar months

Links tested: awareness -> consideration with the three channels excluded
from consideration. Consideration and awareness -> sales with all five
drivers excluded from sales. Takes under a minute. Results go to
``tools/same_week_pretest_check_results.csv``. The numbers feed
``docs/granger_pretest.md``.

    uv run python tools/same_week_pretest_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_vecm import granger_pretest, simulate_brand_data

SEEDS = range(200)
SALES, AWARE, CONS = "organic_sales", "brand_awareness", "brand_consideration"
CHANNELS = ["brand_spend", "pr_reach", "sponsorship_reach"]
MACRO = ["interest_rate", "consumer_confidence"]
LINKS = {CONS: [AWARE], SALES: [CONS, AWARE]}
EXCLUDED = {CONS: CHANNELS, SALES: CHANNELS + MACRO}
RESULTS = Path(__file__).with_name("same_week_pretest_check_results.csv")


def monthly(endog: pd.DataFrame, exog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    levels = np.exp(endog)
    months = pd.DataFrame(
        {
            SALES: levels[SALES].resample("MS").sum(),
            AWARE: levels[AWARE].resample("MS").mean(),
            CONS: levels[CONS].resample("MS").mean(),
        }
    )
    full = levels[SALES].resample("MS").count() >= 4
    return np.log(months[full]), exog.resample("MS").mean()[full]


def main() -> None:
    tables = []
    for value in (0.0, 0.2):
        links = {CONS: {AWARE: value}} if value else None
        for seed in SEEDS:
            data = simulate_brand_data(261, seed=seed, contemporaneous=links)
            shapes = {"weekly": (data.endog, data.exog), "monthly": monthly(data.endog, data.exog)}
            for freq, (endog, exog) in shapes.items():
                result = granger_pretest(endog, exog, contemporaneous=LINKS, excluded=EXCLUDED)
                tables.append(
                    result.same_week.assign(truth=value, seed=seed, freq=freq, lags=result.lags)
                )
    out = pd.concat(tables, ignore_index=True)
    out.to_csv(RESULTS, index=False)
    for (truth, freq, cause, effect), g in out.groupby(
        ["truth", "freq", "cause", "effect"], sort=False
    ):
        share = g["verdict"].value_counts(normalize=True).round(2).to_dict()
        print(
            f"truth {truth}  {freq:<8}{cause} -> {effect}: OLS {g.ols.mean():.2f}  "
            f"2SLS {g.estimate.mean():.2f}  F {g.first_stage_f.mean():.0f}  "
            f"Sargan p<0.05 {np.mean(g.sargan_p < 0.05):.0%}\n    {share}"
        )


if __name__ == "__main__":
    main()
