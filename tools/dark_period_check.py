"""Does a dark spell in the data tell a lasting brand effect from a fading one?

``tools/fade_check.py`` found that five years of ordinary weekly data pin the
lasting share of a brand lift down only loosely. This check asks whether data
that include a period with brand spend switched off do better.

Data from ``simulate_brand_data`` (260 weeks, linear drivers), two truths:

* ``lasts``: every lift in the brand metrics lasts.
* ``fades``: the whole lift fades with a half-life of 52 weeks
  (``exog_fade``, all three channels).

Three versions of each: brand spend never off; off for 26 weeks from week 130;
off for 52 weeks from week 130 (``exog_dark``).

One fit on each: ``fading`` on the three channels, rank 2, one lag of
differences, ``"co"``, 2 chains of 1000 draws.

Recorded for brand spend: the fade share (posterior median, 90% interval and
its width), and the extra sales from one more unit for one week summed over 1
and 5 years, against the truth. Five seeds per cell.

Results go to ``tools/dark_period_check_results.csv``; finished cells are
skipped, so an interrupted run resumes.

    uv run python tools/dark_period_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fade_check import CHANNELS, summary, truth_sales_path

from bayesian_vecm import BayesianVECM, simulate_brand_data

warnings.filterwarnings("ignore")

SEEDS = range(5)
TRUTHS = {
    "lasts": {},
    "fades": {"exog_fade": dict.fromkeys(CHANNELS, 52.0)},
}
DARK = {
    "never off": None,
    "off 26 weeks": {"brand_spend": (130, 26)},
    "off 52 weeks": {"brand_spend": (130, 52)},
}
HORIZON = 260
OUT = Path(__file__).with_name("dark_period_check_results.csv")


def run_one(truth_name: str, dark_name: str, seed: int) -> list[dict]:
    data = simulate_brand_data(seed=seed, exog_dark=DARK[dark_name], **TRUTHS[truth_name])
    truth = data.truth
    model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co", fading=list(CHANNELS))
    t0 = time.time()
    model.fit(
        data.endog,
        exog=data.exog,
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=seed,
        progressbar=False,
    )
    base = {
        "truth": truth_name,
        "dark": dark_name,
        "seed": seed,
        "seconds": time.time() - t0,
        "div": int(model.idata_.sample_stats["diverging"].sum()),
    }
    j = list(data.exog.columns).index("brand_spend")
    rows = []
    share = model.idata_.posterior["fade_share"].values[..., model.fading_idx_.index(j)].ravel()
    true_share = truth.fade_share if np.isfinite(truth.fade_half_life[j]) else 0.0
    rows.append({**base, "quantity": "fade_share", **summary(share), "true": true_share})
    pulse = model.exog_irf(steps=HORIZON - 1).sel(
        response_variable="organic_sales", exog_variable="brand_spend"
    )
    pulse = pulse.stack(sample=("chain", "draw")).transpose("sample", "horizon").values
    u = np.zeros(HORIZON)
    u[0] = 1.0
    true_pulse = truth_sales_path(truth, j, u)
    for label, w in (("1 year", 52), ("5 years", 260)):
        rows.append(
            {
                **base,
                "quantity": f"pulse sales, {label}",
                **summary(np.expm1(pulse[:, :w]).sum(axis=1)),
                "true": float(np.expm1(true_pulse[:w]).sum()),
            }
        )
    return rows


def main() -> None:
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    finished = (
        set(zip(done["truth"], done["dark"], done["seed"], strict=True)) if len(done) else set()
    )
    for truth_name in TRUTHS:
        for dark_name in DARK:
            for seed in SEEDS:
                if (truth_name, dark_name, seed) in finished:
                    continue
                rows = run_one(truth_name, dark_name, seed)
                pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                print(truth_name, dark_name, seed, f"{rows[0]['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
