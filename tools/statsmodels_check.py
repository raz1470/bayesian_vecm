"""BayesianVECM against statsmodels' VECM on simulated brand data.

Fits both models to the same data on 30 seeds of ``simulate_brand_data`` at
two sample sizes:

* 260 weeks, the default
* 60 periods of the same weekly process, the sample size of five years of
  monthly data. The truth stays known, which it would not after summing
  weeks into months.

Both fits use rank 2, one lag of differences, ``deterministic="co"`` and all
five drivers. The Bayesian fit runs 2 chains of 1000 draws after 1000
tuning steps with ``target_accept=0.95``.

For each parameter and seed the script records the truth, both point
estimates and both 90% intervals. statsmodels gives asymptotic standard
errors for alpha, beta, Gamma, the constant and B. It gives no interval for
Sigma or for anything derived from the parameters. For the derived returns
the statsmodels interval comes from a residual bootstrap (499 refits).

The derived quantities are the returns used in the practitioner guide: the
extra sales, in units of weekly sales, from one more unit of a channel for
one week, summed over 1 year (52 weeks) and 5 years (260 weeks). A few
posterior draws at 60 periods are explosive and give an infinite 5-year
return, so the posterior median is the point estimate for the returns.

Takes about 40 minutes on two cores. Results go to
``tools/statsmodels_check_results.csv``, one row per seed, sample size and
quantity. Seeds already in the file are skipped, so an interrupted run
resumes. The numbers feed ``notebooks/guides/02_statsmodels_comparison.ipynb``.

    uv run python tools/statsmodels_check.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import norm
from statsmodels.tsa.vector_ar.vecm import VECM

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._irf import compute_exog_irf

warnings.filterwarnings("ignore")

SEEDS = range(30)
SAMPLE_SIZES = {"260 weeks": 260, "60 periods": 60}
CHANNELS = ("brand_spend", "pr_reach", "sponsorship_reach")
WINDOWS = {"1 year": 52, "5 years": 260}
LEVEL = 0.90
N_BOOT = 499
SPEC = {"k_ar_diff": 1, "coint_rank": 2, "deterministic": "co"}
SAMPLING = {"draws": 1000, "tune": 1000, "chains": 2, "target_accept": 0.95}
RESULTS = Path(__file__).with_name("statsmodels_check_results.csv")


def returns(alpha, beta, gamma, b, names) -> dict[str, np.ndarray]:
    """Summed sales returns per unit of each channel, for a stack of parameter sets.

    Inputs carry a leading draw dimension. Returns one array per channel and
    window, of length ``n_draws``.
    """
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
    mult = compute_exog_irf(
        SimpleNamespace(posterior=posterior), 1, max(WINDOWS.values()), exog_names=names
    )
    out = {}
    for ch in CHANNELS:
        path = np.expm1(mult.sel(exog_variable=ch).values[0, :, :, 0])  # sales: (draw, horizon)
        for label, weeks in WINDOWS.items():
            out[f"return {label}: {ch}"] = path[:, :weeks].sum(axis=1)
    return out


def bootstrap(res, y, x, names, rng) -> dict[str, np.ndarray]:
    """Residual bootstrap of the statsmodels fit: resample residual rows, rebuild, refit."""
    n_obs = len(y)
    pi = res.alpha @ res.beta.T
    det = np.column_stack([np.ones(n_obs), x])
    resid = res.resid - res.resid.mean(axis=0)
    paths = np.repeat(y[None], N_BOOT, axis=0)
    shocks = resid[rng.integers(0, len(resid), size=(N_BOOT, len(resid)))]
    for t in range(2, n_obs):
        prev, prev2 = paths[:, t - 1], paths[:, t - 2]
        paths[:, t] = (
            prev + prev @ pi.T + (prev - prev2) @ res.gamma.T + det[t] @ res.det_coef.T
        ) + shocks[:, t - 2]
    fits = [VECM(p, exog=x, **SPEC).fit() for p in paths]
    stack = {
        "alpha": np.stack([f.alpha for f in fits]),
        "beta": np.stack([f.beta for f in fits]),
        "gamma": np.stack([f.gamma for f in fits]),
        "b": np.stack([f.det_coef[:, 1:] for f in fits]),
    }
    return returns(**stack, names=names)


def entries(block: str, values, rows, cols) -> dict[str, float]:
    """Flatten a matrix into ``block[row, col]`` labels. Sigma keeps its lower triangle."""
    values = np.asarray(values)
    return {
        f"{block}[{r}, {c}]": values[i, j]
        for i, r in enumerate(rows)
        for j, c in enumerate(cols)
        if block != "Sigma" or j <= i
    }


def one_fit(seed: int, label: str, n_obs: int) -> list[dict]:
    data = simulate_brand_data(n_obs=n_obs, seed=seed)
    endog, exog, truth = data.endog, data.exog, data.truth
    y, x = endog.to_numpy(), exog.to_numpy()
    k, m = list(endog.columns), list(exog.columns)
    lags = [f"d.{n}" for n in k]
    ec = ["ec1", "ec2"]
    labels = {
        "alpha": (k, ec),
        "beta": (k[2:], ec),
        "Gamma": (k, lags),
        "const": (k, ["c"]),
        "B": (k, m),
        "Sigma": (k, k),
    }
    z = norm.ppf(0.5 + LEVEL / 2)
    tail = (100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2)

    # statsmodels: ML estimates and asymptotic intervals
    res = VECM(y, exog=x, **SPEC).fit()
    sm_est, sm_se = {}, {}
    for out, part in ((sm_est, "est"), (sm_se, "se")):
        mats = {
            "alpha": (res.alpha, res.stderr_alpha),
            "beta": (res.beta[2:], res.stderr_beta[2:]),
            "Gamma": (res.gamma, res.stderr_gamma),
            "const": (res.det_coef[:, :1], res.stderr_det_coef[:, :1]),
            "B": (res.det_coef[:, 1:], res.stderr_det_coef[:, 1:]),
        }
        for block, (est, se) in mats.items():
            out |= entries(block, se if part == "se" else est, *labels[block])
    sm_est |= entries("Sigma", res.sigma_u, k, k)
    sm_ret = returns(res.alpha[None], res.beta[None], res.gamma[None], res.det_coef[None, :, 1:], m)
    boot = bootstrap(res, y, x, m, np.random.default_rng(seed))

    # BayesianVECM: posterior means and equal-tailed intervals
    start = time.perf_counter()
    model = BayesianVECM(**SPEC).fit(
        endog, exog=exog, random_seed=seed, progressbar=False, **SAMPLING
    )
    seconds = time.perf_counter() - start
    post = model.idata_.posterior.to_dataset().stack(sample=("chain", "draw"))
    draws = lambda name: np.moveaxis(post[name].values, -1, 0)  # noqa: E731
    gamma = draws("Gamma")
    by = {}
    by |= entries("alpha", np.moveaxis(draws("alpha"), 0, -1), k, ["ec1", "ec2"])
    by |= entries("beta", np.moveaxis(draws("beta")[:, 2:], 0, -1), k[2:], ["ec1", "ec2"])
    by |= entries("Gamma", np.moveaxis(gamma[:, :, :3], 0, -1), k, lags)
    by |= entries("const", np.moveaxis(gamma[:, :, 3:], 0, -1), k, ["c"])
    by |= entries("B", np.moveaxis(draws("B"), 0, -1), k, m)
    by |= entries("Sigma", np.moveaxis(draws("Sigma"), 0, -1), k, k)
    by |= returns(draws("alpha"), draws("beta"), gamma, draws("B"), m)

    true = {}
    true |= entries("alpha", truth.alpha, k, ["ec1", "ec2"])
    true |= entries("beta", truth.beta[2:], k[2:], ["ec1", "ec2"])
    true |= entries("Gamma", truth.gamma, k, lags)
    true |= entries("const", truth.const[:, None], k, ["c"])
    true |= entries("B", truth.b, k, m)
    true |= entries("Sigma", truth.sigma, k, k)
    true |= {
        key: val[0]
        for key, val in returns(
            truth.alpha[None], truth.beta[None], truth.gamma[None], truth.b[None], m
        ).items()
    }

    diverging = int(model.idata_.sample_stats["diverging"].sum())
    rows = []
    for name, value in true.items():
        row = {
            "sample": label,
            "seed": seed,
            "quantity": name,
            "block": "return" if name.startswith("return") else name.split("[")[0],
            "truth": value,
            "bayes_mean": by[name].mean(),
            "bayes_median": np.median(by[name]),
            "bayes_lo": np.percentile(by[name], tail[0]),
            "bayes_hi": np.percentile(by[name], tail[1]),
            "divergences": diverging,
            "seconds": seconds,
        }
        if name.startswith("return"):
            row |= {
                "sm_est": sm_ret[name][0],
                "sm_lo": np.percentile(boot[name], tail[0]),
                "sm_hi": np.percentile(boot[name], tail[1]),
            }
        else:
            row["sm_est"] = sm_est[name]
            if name in sm_se:
                row |= {
                    "sm_lo": sm_est[name] - z * sm_se[name],
                    "sm_hi": sm_est[name] + z * sm_se[name],
                }
        rows.append(row)
    return rows


def run() -> pd.DataFrame:
    done = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(columns=["sample", "seed"])
    finished = set(zip(done["sample"], done["seed"], strict=True))
    for label, n_obs in SAMPLE_SIZES.items():
        for seed in SEEDS:
            if (label, seed) in finished:
                continue
            rows = pd.DataFrame(one_fit(seed, label, n_obs))
            rows.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
            print(f"{label} seed {seed}: {rows['seconds'].iloc[0]:.0f} s", flush=True)
    return pd.read_csv(RESULTS)


def report(df: pd.DataFrame) -> None:
    df = df.assign(
        bayes_in=(df.bayes_lo <= df.truth) & (df.truth <= df.bayes_hi),
        sm_in=(df.sm_lo <= df.truth) & (df.truth <= df.sm_hi),
        bayes_width=df.bayes_hi - df.bayes_lo,
        sm_width=df.sm_hi - df.sm_lo,
        gap=(df.bayes_mean.where(df.block != "return", df.bayes_median) - df.sm_est).abs(),
    )
    for label, group in df.groupby("sample", sort=False):
        n_seeds = group.seed.nunique()
        print(f"\n{label}, {n_seeds} seeds. 90% interval coverage and mean width by block")
        table = group.groupby("block", sort=False).agg(
            bayes_cover=("bayes_in", "mean"),
            sm_cover=("sm_in", "mean"),
            bayes_width=("bayes_width", "mean"),
            sm_width=("sm_width", "mean"),
            max_gap=("gap", "max"),
        )
        print(table.round(4).to_string())
        ret = group[group.block == "return"]
        print("Returns: truth, mean estimate, coverage")
        print(
            ret.groupby("quantity")
            .agg(
                truth=("truth", "first"),
                bayes_median=("bayes_median", "mean"),
                sm_mean=("sm_est", "mean"),
                bayes_cover=("bayes_in", "mean"),
                sm_cover=("sm_in", "mean"),
            )
            .round(3)
            .to_string()
        )
        print(f"Divergences per fit: {group.groupby('seed').divergences.first().mean():.1f}")


if __name__ == "__main__":
    report(run())
