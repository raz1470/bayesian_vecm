"""Can the model tell a brand effect that lasts from one that fades?

Data from ``simulate_brand_data`` (260 weeks, linear drivers, no links), in
four settings that differ only in what happens to a lift in the brand metrics:

* ``permanent``: the default. A week of brand spend moves the metrics for good.
* ``fade 26``, ``fade 52``, ``fade 104``: ``exog_fade`` on all three channels
  with that half-life in weeks. The whole lift drains away.

Two fits on each data set, rank 2, one lag of differences, ``"co"``:

* ``permanent model``: the package's default model, where every effect lasts.
* ``fading model``: ``fading=`` on the three channels. Each channel gets a
  share of its effect that fades (``fade_share``, flat prior) and a half-life
  for that share (``fade_half_life``).

Recorded for brand spend: the extra sales from one more unit for one week,
summed over 1 and 5 years, in units of weekly sales; the sales level 52, 156
and 260 weeks after brand spend drops by one unit for good (the "go dark"
path, as a % change); and, for the fading model, the fade share and
half-life. Posterior median and 90% equal-tailed interval, and the truth.

Results go to ``tools/fade_check_results.csv``; finished (setting, seed, fit)
triples are skipped, so an interrupted run resumes.

    uv run python tools/fade_check.py [fits...]
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_vecm import BayesianVECM, simulate_brand_data

warnings.filterwarnings("ignore")

CHANNELS = ("brand_spend", "pr_reach", "sponsorship_reach")
SEEDS = range(5)
SETTINGS = {
    "permanent": {},
    "fade 26": {"exog_fade": dict.fromkeys(CHANNELS, 26.0)},
    "fade 52": {"exog_fade": dict.fromkeys(CHANNELS, 52.0)},
    "fade 104": {"exog_fade": dict.fromkeys(CHANNELS, 104.0)},
}
FITS = {
    "permanent model": {},
    "fading model": {"fading": list(CHANNELS)},
}
HORIZON = 260
DARK_WEEKS = (52, 156, 260)
OUT = Path(__file__).with_name("fade_check_results.csv")


def truth_sales_path(truth, driver: int, u: np.ndarray) -> np.ndarray:
    """Log sales deviation when driver ``driver`` moves by ``u`` (one value per week)."""
    k = truth.alpha.shape[0]
    pi = truth.alpha @ truth.beta.T
    b = truth.b[:, driver]
    delta = truth.fade_decay[driver]
    phi = truth.fade_share if np.isfinite(truth.fade_half_life[driver]) else 0.0
    y, dy, state, out = np.zeros(k), np.zeros(k), 0.0, []
    for t in range(len(u)):
        if t > 0:
            state = delta * state + u[t - 1]
        dy = pi @ y + truth.gamma @ dy + b * u[t] - phi * (1.0 - delta) * b * state
        y = y + dy
        out.append(y[0])
    return np.array(out)


def summary(x: np.ndarray) -> dict[str, float]:
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return {"median": med, "lo90": lo, "hi90": hi}


def run_one(setting: str, seed: int, fit: str) -> list[dict]:
    data = simulate_brand_data(seed=seed, **SETTINGS[setting])
    truth = data.truth
    model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co", **FITS[fit])
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
    seconds = time.time() - t0
    j = list(data.exog.columns).index("brand_spend")
    divergences = int(model.idata_.sample_stats["diverging"].sum())

    pulse = model.exog_irf(steps=HORIZON - 1).sel(
        response_variable="organic_sales", exog_variable="brand_spend"
    )
    pulse = pulse.stack(sample=("chain", "draw")).transpose("sample", "horizon").values
    u = np.zeros(HORIZON)
    u[0] = 1.0
    true_pulse = truth_sales_path(truth, j, u)
    step = np.cumsum(pulse, axis=1)  # a unit rise held for good
    true_step = truth_sales_path(truth, j, np.ones(HORIZON))

    rows = []
    base = {"setting": setting, "seed": seed, "fit": fit, "seconds": seconds, "div": divergences}
    for label, w in (("1 year", 52), ("5 years", 260)):
        rows.append(
            {
                **base,
                "quantity": f"pulse sales, {label}",
                **summary(np.expm1(pulse[:, :w]).sum(axis=1)),
                "truth": float(np.expm1(true_pulse[:w]).sum()),
            }
        )
    for w in DARK_WEEKS:
        # Dropping spend by one unit for good is minus the held rise.
        rows.append(
            {
                **base,
                "quantity": f"go dark, sales % at week {w}",
                **summary(100 * np.expm1(-step[:, w - 1])),
                "truth": float(100 * np.expm1(-true_step[w - 1])),
            }
        )
    post = model.idata_.posterior
    if "fade_share" in post:
        names = list(model.fading_idx_)
        pos = names.index(j)
        for name, true_value in (
            ("fade_share", truth.fade_share),
            ("fade_half_life", truth.fade_half_life[j]),
        ):
            values = post[name].values[..., pos].ravel()
            rows.append({**base, "quantity": name, **summary(values), "truth": float(true_value)})
    return rows


def main() -> None:
    fits = sys.argv[1:] or list(FITS)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    finished = (
        set(zip(done["setting"], done["seed"], done["fit"], strict=True)) if len(done) else set()
    )
    for setting in SETTINGS:
        for seed in SEEDS:
            for fit in fits:
                if (setting, seed, fit) in finished:
                    continue
                rows = run_one(setting, seed, fit)
                frame = pd.DataFrame(rows)
                frame.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
                print(setting, seed, fit, f"{rows[0]['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
