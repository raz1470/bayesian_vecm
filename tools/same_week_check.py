"""Does ``contemporaneous=`` recover same-week links on simulated brand data?

Settings from ``simulate_brand_data``, 260 weeks each:

* ``weekly link 0`` (20 seeds): no same-week link.
* ``weekly link 0.2`` (20 seeds): awareness -> consideration at 0.2.
* ``monthly link 0`` and ``monthly link 0.2`` (20 seeds each): the same data
  summed (sales) or averaged (awareness, consideration and the drivers) into
  calendar months. The truth is the same-month ratio implied by the weekly
  parameters (``monthly_truth``), which includes the lagged weekly link and
  error correction that fall inside the month.
* ``wrong exclusion`` (10 seeds, weekly, link 0.2): brand spend also acts on
  consideration directly, at a third of its effect on awareness. The fit
  still excludes it.
* ``weak`` (20 seeds, monthly, no link): consideration and awareness ->
  sales with all five drivers excluded from sales. Instruments are weak on
  monthly data.

Fits, all rank 2, one lag of differences, ``"co"``:

* ``default``: no links (weekly settings only).
* ``links``: ``contemporaneous`` and ``excluded`` as in the setting.

Each runs 2 chains of 1000 draws after 1000 tuning steps with
``target_accept=0.95``. Recorded per link: posterior median, mean, 90%
equal-tailed interval, evidence score and the 2SLS estimate, first-stage F
and Sargan p-value from ``granger_pretest`` on the same data. Weekly settings also record the
reduced-form same-week effect of brand spend on consideration. Takes about
four hours on two cores. Results go to ``tools/same_week_check_results.csv``;
finished (setting, seed, fit) triples are skipped, so an interrupted run
resumes. A fit with divergences is retried once with a new seed.

    uv run python tools/same_week_check.py [setting ...]
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

import bayesian_vecm._simulate as sim
from bayesian_vecm import BayesianVECM, granger_pretest, simulate_brand_data

warnings.filterwarnings("ignore")

SALES, AWARE, CONS = "organic_sales", "brand_awareness", "brand_consideration"
ENDOG = [SALES, AWARE, CONS]
CHANNELS = ["brand_spend", "pr_reach", "sponsorship_reach"]
MACRO = ["interest_rate", "consumer_confidence"]
CONS_LINK = ({CONS: [AWARE]}, {CONS: CHANNELS})
SALES_LINKS = ({SALES: [CONS, AWARE]}, {SALES: CHANNELS + MACRO})
SETTINGS = {
    "weekly link 0": {"seeds": range(20), "value": 0.0, "freq": "weekly"},
    "weekly link 0.2": {"seeds": range(20), "value": 0.2, "freq": "weekly"},
    "monthly link 0": {"seeds": range(20), "value": 0.0, "freq": "monthly"},
    "monthly link 0.2": {"seeds": range(20), "value": 0.2, "freq": "monthly"},
    "wrong exclusion": {"seeds": range(10), "value": 0.2, "freq": "weekly", "direct": 1 / 3},
    "weak": {"seeds": range(20), "value": 0.0, "freq": "monthly", "links": SALES_LINKS},
}
SPEC = {"k_ar_diff": 1, "coint_rank": 2, "deterministic": "co"}
SAMPLING = {"draws": 1000, "tune": 1000, "chains": 2, "target_accept": 0.95}
RESULTS = Path(__file__).with_name("same_week_check_results.csv")


def monthly(endog: pd.DataFrame, exog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calendar months: sales summed, the rest averaged. Partial months dropped."""
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


def b_matrix(direct: float) -> np.ndarray:
    """Structural driver effects, with an optional direct brand spend -> consideration."""
    b = sim._B.copy()
    b[2, 0] = direct * b[1, 0]
    return b


def _month_response(b: np.ndarray, a0_inv: np.ndarray, driver: int, weeks: int) -> np.ndarray:
    """Same-month change in month-average log levels from one month of a driver."""
    pi = sim._ALPHA @ sim._BETA.T
    y = np.zeros(3)
    dy = np.zeros(3)
    path = []
    x = np.zeros(b.shape[1])
    x[driver] = 1.0
    impact = a0_inv @ b @ x
    for _ in range(weeks):
        dy = pi @ y + sim._GAMMA @ dy + impact
        y = y + dy
        path.append(y.copy())
    path = np.array(path)
    # Log of a month sum or mean is the mean of weekly logs to first order.
    return path.mean(axis=0)


def monthly_truth(value: float, causes: tuple[str, ...] = (AWARE,), effect: str = CONS) -> dict:
    """Same-month links implied by the weekly parameters, months of 4 and 5 weeks."""
    a0 = np.eye(3)
    a0[2, 1] = -value
    a0_inv = np.linalg.inv(a0)
    b = sim._B
    out = []
    for weeks, share in ((4, 0.65), (5, 0.35)):
        resp = {d: _month_response(b, a0_inv, d, weeks) for d in range(b.shape[1])}
        e = ENDOG.index(effect)
        if causes == (AWARE,):
            r = resp[0]
            est = {AWARE: r[e] / r[1]}
        else:
            # Two causes: channels move awareness (and so consideration);
            # macro moves consideration only.
            r_ch, r_mac = resp[0], resp[3]
            a_cons = r_mac[e] / r_mac[2]
            a_aware = (r_ch[e] - a_cons * r_ch[2]) / r_ch[1]
            est = {CONS: a_cons, AWARE: a_aware}
        out.append((share, est))
    return {c: sum(s * est[c] for s, est in out) for c in causes}


def simulate(setting: dict, seed: int):
    value = setting["value"]
    links = {CONS: {AWARE: value}} if value else None
    direct = setting.get("direct", 0.0)
    if not direct:
        return simulate_brand_data(seed=seed, contemporaneous=links)
    saved = sim._B
    sim._B = b_matrix(direct)
    try:
        return simulate_brand_data(seed=seed, contemporaneous=links)
    finally:
        sim._B = saved


def truths(setting: dict, data) -> dict[str, float]:
    links, _ = setting.get("links", CONS_LINK)
    out = {}
    for effect, causes in links.items():
        if setting["freq"] == "weekly":
            for c in causes:
                out[f"link: {c} -> {effect}"] = -float(
                    data.truth.a0[ENDOG.index(effect), ENDOG.index(c)]
                )
        else:
            for c, v in monthly_truth(setting["value"], tuple(causes), effect).items():
                out[f"link: {c} -> {effect}"] = float(v)
    if setting["freq"] == "weekly":
        out[f"B: brand_spend -> {CONS}"] = float(data.truth.b[2, 0])
    return out


def two_sls(setting: dict, endog, exog) -> dict[str, tuple[float, float, float]]:
    links, excl = setting.get("links", CONS_LINK)
    table = granger_pretest(endog, exog, contemporaneous=links, excluded=excl).same_week
    return {
        f"link: {r.cause} -> {r.effect}": (
            float(r.estimate),
            float(r.first_stage_f),
            float(r.sargan_p),
        )
        for r in table.itertuples()
    }


def fit_draws(model) -> dict[str, np.ndarray]:
    post = model.idata_.posterior
    out = {}
    names = model.variable_names_
    if "A0_links" in post:
        draws = post["A0_links"].values.reshape(-1, len(model.links_))
        for i, (effect, cause) in enumerate(model.links_):
            out[f"link: {names[cause]} -> {names[effect]}"] = draws[:, i]
    b = post["B"].values
    b = b.reshape(-1, *b.shape[2:])
    out[f"B: brand_spend -> {CONS}"] = b[:, 2, 0]
    return out


def run_fit(setting: dict, fit: str, endog, exog, seed: int):
    links, excl = setting.get("links", CONS_LINK)
    kwargs = {"contemporaneous": links, "excluded": excl} if fit == "links" else {}
    for attempt in range(2):
        start = time.perf_counter()
        model = BayesianVECM(**SPEC, **kwargs).fit(
            endog, exog=exog, random_seed=seed + 1000 * attempt, progressbar=False, **SAMPLING
        )
        seconds = time.perf_counter() - start
        diverging = int(model.idata_.sample_stats["diverging"].sum())
        if diverging == 0:
            break
    return model, seconds, diverging, attempt


def one(name: str, seed: int, fits: list[str]) -> list[dict]:
    setting = SETTINGS[name]
    data = simulate(setting, seed)
    endog, exog = data.endog, data.exog
    if setting["freq"] == "monthly":
        endog, exog = monthly(endog, exog)
    true = truths(setting, data)
    iv = two_sls(setting, endog, exog)
    rows = []
    for fit in fits:
        model, seconds, diverging, retries = run_fit(setting, fit, endog, exog, seed)
        summ = az.summary(model.idata_, var_names=["alpha", "Gamma", "B"], kind="diagnostics")
        if fit == "links":
            summ = pd.concat(
                [summ, az.summary(model.idata_, var_names=["A0_links"], kind="diagnostics")]
            )
        for q, draws in fit_draws(model).items():
            if q not in true:
                continue
            q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95])
            evidence = 1.0 - 2.0 * min((draws > 0).mean(), (draws < 0).mean())
            est, f, sargan = iv.get(q, (np.nan, np.nan, np.nan))
            rows.append(
                {
                    "setting": name,
                    "seed": seed,
                    "fit": fit,
                    "quantity": q,
                    "truth": true[q],
                    "median": q50,
                    "mean": draws.mean(),
                    "sd": draws.std(),
                    "lower": q05,
                    "upper": q95,
                    "covered": q05 <= true[q] <= q95,
                    "evidence": evidence,
                    "two_sls": est,
                    "first_stage_f": f,
                    "sargan_p": sargan,
                    "diverging": diverging,
                    "retries": retries,
                    "max_rhat": float(summ["r_hat"].max()),
                    "min_ess": float(summ["ess_bulk"].min()),
                    "seconds": seconds,
                    "n_obs": len(endog),
                }
            )
    return rows


def main(only: list[str]) -> None:
    done = set()
    if RESULTS.exists():
        prev = pd.read_csv(RESULTS)
        done = set(zip(prev["setting"], prev["seed"], prev["fit"], strict=True))
    for name in only or list(SETTINGS):
        setting = SETTINGS[name]
        fits = ["default", "links"] if setting["freq"] == "weekly" else ["links"]
        for seed in setting["seeds"]:
            todo = [f for f in fits if (name, seed, f) not in done]
            if not todo:
                continue
            start = time.perf_counter()
            rows = pd.DataFrame(one(name, seed, todo))
            if RESULTS.exists():
                rows = pd.concat([pd.read_csv(RESULTS), rows], ignore_index=True)
            rows.to_csv(RESULTS, index=False)
            print(f"{name} seed {seed}: {time.perf_counter() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
