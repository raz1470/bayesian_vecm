"""Export the numbers behind ``docs/walkthrough.html``.

Fits the model exactly as ``notebooks/guides/01_brand_vecm_walkthrough.ipynb``
does and writes every chart series and table the walkthrough needs to
``results/walkthrough.json``. The page's chart constants are copied from that
file by hand.

Run from the repo root::

    uv run python tools/walkthrough_data/export.py

Paths are summarised as a posterior mean and an 89% equal-tailed interval.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian_vecm import (
    BayesianVECM,
    check_stationarity,
    select_coint_rank,
    simulate_brand_data,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Same settings as the notebook's fast run, so the numbers match its committed outputs.
DRAWS, TUNE, CHAINS = 1000, 1000, 2
SEED = 42
HOLDOUT = 52  # weeks held back for the out-of-sample check
SPURIOUS_SEED = 7  # one pair of unrelated random walks for the illustration
SPURIOUS_PAIRS = 1000
CI_LO, CI_HI = 5.5, 94.5
OUT = Path(__file__).parent / "results" / "walkthrough.json"

CHANNEL_UNITS = {
    "brand_spend": {"unit": "£1,000 of spend", "cost_gbp": 1_000.0},
    "pr_reach": {"unit": "100,000 people reached", "cost_gbp": None},
    "sponsorship_reach": {"unit": "100,000 people reached", "cost_gbp": None},
}


def band(draws: np.ndarray, decimals: int = 5) -> dict:
    """Mean and 89% interval over the first axis, rounded for the page."""
    lo, hi = np.percentile(draws, [CI_LO, CI_HI], axis=0)
    return {
        "mean": np.round(draws.mean(axis=0), decimals).tolist(),
        "lo": np.round(lo, decimals).tolist(),
        "hi": np.round(hi, decimals).tolist(),
    }


def records(df: pd.DataFrame, decimals: int = 4) -> list[dict]:
    return json.loads(df.round(decimals).to_json(orient="records"))


def main() -> None:
    start = time.time()
    out: dict = {}

    # --- 1 · The data ------------------------------------------------------------
    data = simulate_brand_data()
    endog, exog, truth = data.endog, data.exog, data.truth
    names = list(endog.columns)
    levels = np.exp(endog)
    out["meta"] = {
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "seed": SEED,
        "interval": "89% equal-tailed",
        "n_weeks": len(endog),
        "start": str(endog.index[0].date()),
        "end": str(endog.index[-1].date()),
    }
    out["series"] = {
        "dates": [str(d.date()) for d in endog.index],
        "levels": {c: np.round(levels[c].values, 2).tolist() for c in names},
        "exog": {c: np.round(exog[c].values, 4).tolist() for c in exog.columns},
        "growth_pct": {
            c: round(float(levels[c].iloc[-1] / levels[c].iloc[0] - 1) * 100, 1) for c in names
        },
    }

    # --- 2 · The problem (illustration, not an estimate) ---------------------------
    weeks = np.arange(53)
    out["illustration"] = {
        "weeks": weeks.tolist(),
        "mmm": np.round(0.7**weeks, 4).tolist(),
        "vecm": np.round(1.0 - (1.0 - 0.12) ** weeks, 4).tolist(),
    }

    # --- What a VECM is: spurious regression against cointegration ----------------
    from statsmodels.tsa.stattools import coint

    def ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        """Slope, its t-statistic, R squared and residuals of y on x with a constant."""
        design = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        s2 = resid @ resid / (len(y) - 2)
        se = np.sqrt(s2 * np.linalg.inv(design.T @ design)[1, 1])
        r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
        return float(coef[1]), float(coef[1] / se), float(r2), resid

    rng = np.random.default_rng(SPURIOUS_SEED)
    walks = rng.normal(size=(len(endog), 2)).cumsum(axis=0)
    slope, t_stat, r2, _ = ols(walks[:, 0], walks[:, 1])
    significant = 0
    pair_rng = np.random.default_rng(0)
    for _ in range(SPURIOUS_PAIRS):
        pair = pair_rng.normal(size=(len(endog), 2)).cumsum(axis=0)
        significant += abs(ols(pair[:, 0], pair[:, 1])[1]) > 1.96
    _, _, eg_r2, _ = ols(endog["organic_sales"].values, endog["brand_consideration"].values)
    out["spurious"] = {
        "walk_a": np.round(walks[:, 0], 3).tolist(),
        "walk_b": np.round(walks[:, 1], 3).tolist(),
        "slope": round(slope, 3),
        "t_stat": round(t_stat, 1),
        "r2": round(r2, 2),
        "coint_p": round(float(coint(walks[:, 0], walks[:, 1])[1]), 3),
        "pairs": SPURIOUS_PAIRS,
        "share_significant_pct": round(100 * significant / SPURIOUS_PAIRS, 1),
        "sales_consideration_r2": round(eg_r2, 3),
        "sales_consideration_coint_p": round(
            float(coint(endog["organic_sales"], endog["brand_consideration"])[1]), 4
        ),
    }

    # --- 3 · The long run ----------------------------------------------------------
    stationarity = check_stationarity(pd.concat([endog, exog], axis=1))
    out["stationarity"] = {
        "endog": stationarity.endog,
        "exog": stationarity.exog,
        "table": records(stationarity.table),
    }
    coint_rank = select_coint_rank(endog, det_order=0, k_ar_diff=1)
    out["rank"] = {
        "recommended": int(coint_rank.rank),
        "trace": np.round(coint_rank.test_stats, 2).tolist(),
        "crit_5pct": np.round(coint_rank.crit_vals, 2).tolist(),
    }

    model = BayesianVECM(k_ar_diff=1, coint_rank=max(coint_rank.rank, 1), deterministic="co")
    fit_start = time.time()
    model.fit(
        endog,
        exog=exog,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        target_accept=0.90,
        random_seed=SEED,
        progressbar=False,
    )
    fit_start_elapsed = time.time() - fit_start
    summary = model.summary().apply(pd.to_numeric, errors="coerce")
    out["sampler"] = {
        "divergences": int(model.idata.sample_stats["diverging"].sum()),
        "max_rhat": round(float(summary["r_hat"].max()), 3),
        "min_ess_bulk": int(summary["ess_bulk"].min()),
        "min_ess_tail": int(summary["ess_tail"].min()),
        "fit_s": round(fit_start_elapsed, 0),
    }

    # In-sample fit of the weekly changes and residual checks.
    fitted = model.fittedvalues.mean(("chain", "draw")).values
    resid = model.resid.mean(("chain", "draw")).values
    actual = fitted + resid
    whiteness = model.test_whiteness(lags=10)
    normality = model.test_normality()
    out["fit"] = [
        {
            "variable": name,
            "r2": round(float(1 - resid[:, i].var() / actual[:, i].var()), 3),
            "lb_p": round(float(whiteness["p_value"].iloc[i]), 3),
            "jb_p": round(float(normality["p_value"].iloc[i]), 3),
        }
        for i, name in enumerate(names)
    ]

    post = model.idata.posterior
    r = post["alpha"].shape[-1]
    k = len(names)

    def param_table(var: str, true_values: np.ndarray, rows: list[int]) -> list[dict]:
        draws = post[var].values.reshape(-1, *post[var].shape[2:])[:, :k, :]
        lo, hi = np.percentile(draws, [CI_LO, CI_HI], axis=0)
        mean = draws.mean(axis=0)
        return [
            {
                "variable": names[i],
                "relation": j + 1,
                "mean": round(float(mean[i, j]), 4),
                "lo": round(float(lo[i, j]), 4),
                "hi": round(float(hi[i, j]), 4),
                "truth": round(float(true_values[i, j]), 4),
            }
            for i in rows
            for j in range(r)
        ]

    out["beta"] = param_table("beta", truth.beta, rows=[2])
    out["alpha"] = param_table("alpha", truth.alpha, rows=list(range(k)))
    beta_mean = post["beta"].mean(dim=["chain", "draw"]).values[:k]
    gaps = endog.values @ beta_mean
    gaps = gaps - gaps.mean(axis=0)
    out["relations"] = {f"relation_{j + 1}": np.round(gaps[:, j], 4).tolist() for j in range(r)}

    # --- 4 · The awareness effect --------------------------------------------------
    irf = model.irf(steps=52, method="girf")
    horizons = irf["horizon"].values

    def irf_draws(result, response: str, shock: str) -> np.ndarray:
        values = result.sel(response_variable=response, shock_variable=shock).values
        return values.reshape(-1, len(horizons))

    out["irf_awareness"] = {
        "horizon": horizons.tolist(),
        **{resp: band(irf_draws(irf, resp, "brand_awareness") * 100, 4) for resp in names},
        "units": "% response (log units x 100)",
    }
    sales = irf_draws(irf, "organic_sales", "brand_awareness")
    cum = sales.cumsum(axis=1)
    cum_mean = cum.mean(axis=0)
    adstock = (0.7**horizons).cumsum()
    mmm_cum = adstock * cum_mean[4] / adstock[4]
    out["irf_cumulative"] = {
        "horizon": horizons.tolist(),
        "vecm": band(cum, 4),
        "adstock_matched_week4": np.round(mmm_cum, 4).tolist(),
        "ratio_week52": round(float(cum_mean[52] / mmm_cum[52]), 1),
    }

    funnel = ["brand_awareness", "brand_consideration", "organic_sales"]
    compare = {}
    for label, result in [
        ("girf", irf),
        ("cholesky_awareness_first", model.irf(steps=52, method="cholesky", order=funnel)),
        ("cholesky_sales_first", model.irf(steps=52, method="cholesky")),
    ]:
        wk52 = irf_draws(result, "organic_sales", "brand_awareness")[:, -1] * 100
        lo, hi = np.percentile(wk52, [CI_LO, CI_HI])
        compare[label] = {"mean": round(wk52.mean(), 2), "lo": round(lo, 2), "hi": round(hi, 2)}
    out["irf_week52_sales"] = compare

    # --- 5 · What drives what ------------------------------------------------------
    causality = model.granger_causality()
    channels = list(CHANNEL_UNITS)
    exog_table = causality.exog.copy()
    exog_table["truth"] = [
        float(truth.b[names.index(e), list(exog.columns).index(c)])
        for c, e in zip(exog_table["cause"], exog_table["effect"], strict=True)
    ]
    out["causality"] = {
        "short_run": records(causality.short_run),
        "long_run": records(causality.long_run),
        "weak_exogeneity": records(causality.weak_exogeneity),
        "exog": records(exog_table),
    }

    fevd = model.fevd(steps=52, method="girf")
    fevd_mean = fevd.mean(dim=["chain", "draw"])
    out["fevd"] = {
        "horizon": fevd["horizon"].values.tolist(),
        "shares": {
            resp: {
                shock: np.round(
                    fevd_mean.sel(response_variable=resp, shock_variable=shock).values, 4
                ).tolist()
                for shock in names
            }
            for resp in names
        },
    }

    # --- 6 · A campaign ------------------------------------------------------------
    steps, uplift_weeks = 52, 26
    spend_sd = float(exog["brand_spend"].std())
    spend_col = exog.columns.get_loc("brand_spend")
    base_x = np.zeros((steps, exog.shape[1]))
    up_x = base_x.copy()
    up_x[:uplift_weeks, spend_col] = spend_sd
    y_base = model.sample_posterior_predictive(steps, exog_future=base_x, random_seed=0)
    y_base = y_base.posterior_predictive["y"].values
    y_up = model.sample_posterior_predictive(steps, exog_future=up_x, random_seed=0)
    y_up = y_up.posterior_predictive["y"].values
    uplift = np.expm1(y_up - y_base).reshape(-1, steps, k)
    recent_sales = float(levels["organic_sales"].iloc[-52:].mean())
    sales_up = uplift[..., names.index("organic_sales")]
    revenue = sales_up.sum(axis=-1) * recent_sales
    rev_lo, rev_hi = np.percentile(revenue, [CI_LO, CI_HI])
    out["campaign"] = {
        "week": list(range(1, steps + 1)),
        "uplift_weeks": uplift_weeks,
        "spend_sd": round(spend_sd, 4),
        "spend_uplift_path": np.round(up_x[:, spend_col], 4).tolist(),
        "awareness_pct": band(uplift[..., names.index("brand_awareness")] * 100, 3),
        "sales_pct": band(sales_up * 100, 3),
        "recent_weekly_sales": round(recent_sales, 0),
        "revenue_52w": {
            "mean": round(float(revenue.mean()), 0),
            "lo": round(float(rev_lo), 0),
            "hi": round(float(rev_hi), 0),
        },
    }

    levels_sd = np.linspace(-2, 2, 13)
    curve = []
    sales_idx = names.index("organic_sales")
    for z in levels_sd:
        x = base_x.copy()
        x[:, spend_col] = z * spend_sd
        fc = model.sample_posterior_predictive(steps, exog_future=x, random_seed=0)
        y = fc.posterior_predictive["y"].values[..., sales_idx]
        year = np.expm1(y - y_base[..., sales_idx]).mean(axis=-1).ravel() * 100
        lo, hi = np.percentile(year, [CI_LO, CI_HI])
        curve.append(
            {
                "spend_sd": round(float(z), 3),
                "mean": round(year.mean(), 3),
                "lo": round(lo, 3),
                "hi": round(hi, 3),
            }
        )
    out["response_curve"] = curve

    # --- 7 · Return per channel ----------------------------------------------------
    windows = {"1 year": 52, "3 years": 156, "5 years": 260}
    mult = model.exog_irf(steps=max(windows.values()))
    sales_mult = (
        mult.sel(response_variable="organic_sales", exog_variable=channels)
        .stack(sample=("chain", "draw"))
        .transpose("sample", "horizon", "exog_variable")
        .values
    )
    cum_rev = (np.expm1(sales_mult) * recent_sales).cumsum(axis=1)

    def true_cum_revenue(channel: str, n_steps: int) -> np.ndarray:
        x = np.zeros(truth.b.shape[1])
        x[exog.columns.get_loc(channel)] = 1.0
        level, growth, path = np.zeros(k), np.zeros(k), []
        for h in range(n_steps + 1):
            growth = truth.alpha @ truth.beta.T @ level + truth.gamma @ growth
            growth = growth + (truth.b @ x if h == 0 else 0.0)
            level = level + growth
            path.append(level[sales_idx])
        return (np.expm1(np.array(path)) * recent_sales).cumsum()

    n_h = cum_rev.shape[1]
    per_channel = {}
    for j, ch in enumerate(channels):
        cost = CHANNEL_UNITS[ch]["cost_gbp"]
        scale = cost or 1.0
        draws = cum_rev[..., j] / scale
        table = {}
        for label, w in windows.items():
            total = cum_rev[:, w - 1, j]
            lo, hi = np.percentile(total, [CI_LO, CI_HI])
            row = {
                "per_unit": {"mean": round(total.mean(), 0), "lo": round(lo, 0), "hi": round(hi, 0)}
            }
            if cost:
                row["per_gbp"] = {
                    "mean": round(total.mean() / cost, 2),
                    "lo": round(lo / cost, 2),
                    "hi": round(hi / cost, 2),
                }
            table[label] = row
        per_channel[ch] = {
            "unit": CHANNEL_UNITS[ch]["unit"],
            "cost_gbp": cost,
            "path_units": "£ per £1" if cost else f"£ per {CHANNEL_UNITS[ch]['unit']}",
            "path": band(draws, 3),
            "truth": np.round(true_cum_revenue(ch, n_h - 1) / scale, 3).tolist(),
            "windows": table,
        }
    share_y1 = cum_rev[:, 51, 0] / cum_rev[:, 259, 0]
    out["return_per_channel"] = {
        "week": list(range(1, n_h + 1)),
        "windows": windows,
        "channels": per_channel,
        "brand_share_year1_pct": round(float(share_y1.mean()) * 100, 1),
    }

    # --- Hold-out forecast: fit on the first four years, forecast the fifth ------------
    train = len(endog) - HOLDOUT
    holdout_model = BayesianVECM(k_ar_diff=1, coint_rank=model.coint_rank, deterministic="co")
    holdout_model.fit(
        endog.iloc[:train],
        exog=exog.iloc[:train],
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        target_accept=0.90,
        random_seed=SEED,
        progressbar=False,
    )
    fc = holdout_model.sample_posterior_predictive(
        HOLDOUT, exog_future=exog.iloc[train:].values, random_seed=0
    )
    fc_y = fc.posterior_predictive["y"].values.reshape(-1, HOLDOUT, k)
    test = endog.iloc[train:].values
    hist = endog.iloc[:train].values
    drift = (hist[-1] - hist[0]) / (train - 1)
    naive = hist[-1] + drift * np.arange(1, HOLDOUT + 1)[:, None]
    fc_levels = np.exp(fc_y)
    fc_mean = fc_levels.mean(axis=0)
    fc_lo, fc_hi = np.percentile(fc_levels, [CI_LO, CI_HI], axis=0)
    actual_levels = np.exp(test)
    naive_levels = np.exp(naive)
    holdout = {
        "train_end": str(endog.index[train - 1].date()),
        "weeks": HOLDOUT,
        "dates": [str(d.date()) for d in endog.index[train - 26 :]],
        "history_sales": np.round(np.exp(hist[-26:, sales_idx]), 1).tolist(),
        "series": {},
    }
    for i, name in enumerate(names):
        inside = (actual_levels[:, i] >= fc_lo[:, i]) & (actual_levels[:, i] <= fc_hi[:, i])
        holdout["series"][name] = {
            "actual": np.round(actual_levels[:, i], 2).tolist(),
            "mean": np.round(fc_mean[:, i], 2).tolist(),
            "lo": np.round(fc_lo[:, i], 2).tolist(),
            "hi": np.round(fc_hi[:, i], 2).tolist(),
            "naive": np.round(naive_levels[:, i], 2).tolist(),
            "mape_vecm": round(
                float(np.mean(np.abs(fc_mean[:, i] / actual_levels[:, i] - 1)) * 100), 1
            ),
            "mape_naive": round(
                float(np.mean(np.abs(naive_levels[:, i] / actual_levels[:, i] - 1)) * 100), 1
            ),
            "coverage_pct": round(float(inside.mean()) * 100, 0),
        }
    out["holdout"] = holdout

    out["meta"]["runtime_s"] = round(time.time() - start, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"Wrote {OUT} in {out['meta']['runtime_s']} s")


if __name__ == "__main__":
    main()
