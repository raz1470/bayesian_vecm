"""Granger pretest: what helps predict what, before the rank is known.

Run it on the raw levels before fitting. It answers four questions:

* **Drivers.** Does past spend (or a macro series) help predict a brand
  metric beyond the metric's own past?
* **Delay.** Does past spend still help once this period's drivers are in
  the equation? If it does, part of the effect arrives late and same-period
  ``exog`` misses it.
* **Links.** Does past awareness help predict consideration? Past
  consideration help predict sales? And the reverse directions.
* **Feedback.** Do past brand metrics help predict spend? If they do,
  budgets follow performance. Spend is then not exogenous and the
  coefficients on it in ``B`` are biased.

.. code-block:: python

    from bayesian_vecm import granger_pretest

    result = granger_pretest(data.endog, data.exog[["brand_spend", "interest_rate"]])
    print(result)

The method
----------
Toda and Yamamoto (1995). Fit a VAR in levels with ``lags + extra_lags``
lags and test only the first ``lags``. The extra lags absorb the unit roots.
The test is then valid whether the series are stationary or cointegrated.
This is why it can run before the cointegration rank is chosen.

Each test is an F test that the ``lags`` coefficients of the cause in the
effect's equation are all zero. Drivers and brand metrics enter the VAR on
equal terms, so feedback from metrics to drivers is tested too.

The delay test adds the current drivers to each brand metric's equation.
This treats the drivers as exogenous within the period.

What it cannot see
------------------
A Granger test looks at **past** values only. An effect inside the same
period is invisible to it. With monthly data most of a brand effect may land
in the month of the spend. The ``same_period`` table shows the correlation
of the VAR residuals for each pair. It says the two series move together
within the period. It cannot say which way.

A driver can matter in the model without passing the Granger test. ``B`` in
:class:`~bayesian_vecm.BayesianVECM` estimates the same-period effect.

The reverse also holds. Spend that persists from one period to the next
passes the driver test through its same-period effect alone. Past spend
stands in for current spend. Read the delay table before concluding that
spend acts with a lag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from bayesian_vecm._data import validate_endog

if TYPE_CHECKING:
    import pandas as pd

_GROUP_ORDER = (
    ("driver -> metric", "Drivers: does past X help predict the brand metric?"),
    ("delay", "Delay: does past X still help once this period's drivers are known?"),
    ("metric -> metric", "Links: does past X help predict the other brand metric?"),
    ("metric -> driver", "Feedback: does the past brand metric help predict the driver?"),
)


@dataclass(frozen=True)
class GrangerPretestResult:
    """Toda-Yamamoto Granger tests and same-period correlations.

    Printing the result gives the tests grouped into drivers, links and
    feedback. Each attribute below is a ``pandas.DataFrame`` for further work.

    Attributes
    ----------
    tests : pandas.DataFrame
        One row per ordered pair. Columns ``cause``, ``effect``, ``group``,
        ``f_stat``, ``df_num``, ``df_den`` and ``p_value``. ``group`` is one of
        ``"driver -> metric"``, ``"metric -> metric"``, ``"metric -> driver"``
        or ``"driver -> driver"``.
    delay : pandas.DataFrame or None
        Same columns as ``tests`` for each driver and brand metric pair, with
        ``group`` set to ``"delay"``. The metric's equation also holds the
        current values of every driver. ``None`` without ``exog``.
    same_period : pandas.DataFrame
        One row per unordered pair. Columns ``variable_1``, ``variable_2``,
        ``group``, ``corr`` and ``p_value``. The correlation is between the
        two VAR residuals.
    lags : int
        Lags tested. Chosen by the Hannan-Quinn criterion unless given.
    extra_lags : int
        Lags added and left untested.
    n_obs : int
        Observations used in the regression after lags are dropped.
    alpha : float
        Tests with ``p_value`` below this are reported as links.
    seasonal : int or None
        Period of the seasonal dummies, if any.
    """

    tests: pd.DataFrame
    delay: pd.DataFrame | None
    same_period: pd.DataFrame
    lags: int
    extra_lags: int
    n_obs: int
    alpha: float
    seasonal: int | None

    def __repr__(self) -> str:
        return (
            f"GrangerPretestResult(lags={self.lags}, extra_lags={self.extra_lags}, "
            f"n_obs={self.n_obs})"
        )

    def __str__(self) -> str:
        import pandas as pd

        season = f"   seasonal = {self.seasonal}" if self.seasonal else ""
        lines = [
            "Granger pretest (Toda-Yamamoto)",
            f"  lags tested = {self.lags}   extra lags = {self.extra_lags}"
            f"   n_obs = {self.n_obs}   alpha = {self.alpha:g}{season}",
        ]
        tables = self.tests if self.delay is None else pd.concat([self.tests, self.delay])
        for group, title in _GROUP_ORDER:
            table = tables[tables["group"] == group]
            if len(table) == 0:
                continue
            lines += ["", title]
            lines.append(f"  {'cause -> effect':<46}{'F':>8}{'p-value':>10}  verdict")
            lines.append("  " + "-" * 80)
            for row in table.itertuples(index=False):
                pair = f"{row.cause} -> {row.effect}"
                verdict = "helps predict" if row.p_value < self.alpha else "no clear evidence"
                lines.append(f"  {pair:<46}{row.f_stat:>8.2f}{row.p_value:>10.3f}  {verdict}")
        same = self.same_period[self.same_period["group"] != "driver - driver"]
        if len(same):
            lines += ["", "Same period: correlation of the VAR residuals (direction unknown)"]
            lines.append(f"  {'pair':<46}{'corr':>8}{'p-value':>10}  verdict")
            lines.append("  " + "-" * 80)
            for row in same.itertuples(index=False):
                pair = f"{row.variable_1} ~ {row.variable_2}"
                verdict = "move together" if row.p_value < self.alpha else "no clear evidence"
                lines.append(f"  {pair:<46}{row.corr:>8.2f}{row.p_value:>10.3f}  {verdict}")
        return "\n".join(lines)


def granger_pretest(
    endog: Any,
    exog: Any = None,
    *,
    lags: int | None = None,
    max_lags: int = 8,
    extra_lags: int = 1,
    seasonal: int | None = None,
    alpha: float = 0.05,
) -> GrangerPretestResult:
    """Test which series help predict which, before choosing the rank.

    Parameters
    ----------
    endog
        Brand metrics in levels (logs), shape ``(T, K)``. A ``DataFrame``
        gives the variable names.
    exog
        Drivers in levels, shape ``(T, m)``. They enter the VAR as equals of
        the brand metrics so that feedback can be tested. ``None`` tests the
        brand metrics only.
    lags
        Lags to test. ``None`` picks them with the Hannan-Quinn criterion on
        a levels VAR, at least 1.
    max_lags
        Largest lag order the criterion considers. It is capped so that the
        test regression keeps at least half its rows as residual degrees of
        freedom. Near that limit the criterion picks long lags by chance.
    extra_lags
        Lags added and not tested. 1 for series that are at most I(1).
    seasonal
        Period for centred seasonal dummies, e.g. 12 for monthly data.
        ``None`` for no dummies.
    alpha
        Significance level used in the printed verdicts.

    Returns
    -------
    GrangerPretestResult

    Raises
    ------
    ValueError
        If the inputs fail validation or the VAR has too few degrees of
        freedom.
    """
    import pandas as pd

    y = validate_endog(endog, min_obs=10)
    k = y.shape[1]
    names = _names(endog, k, "y")
    if exog is not None:
        x = validate_endog(exog, min_obs=10)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"exog must have the same number of rows as endog; got {x.shape[0]} "
                f"and {y.shape[0]}"
            )
        names += _names(exog, x.shape[1], "x")
        data = np.column_stack([y, x])
    else:
        data = y
    if len(set(names)) != len(names):
        raise ValueError(f"variable names must be unique; got {names}")
    if extra_lags < 0:
        raise ValueError(f"extra_lags must be >= 0; got {extra_lags}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1; got {alpha}")
    if seasonal is not None and seasonal < 2:
        raise ValueError(f"seasonal must be at least 2; got {seasonal}")

    det = _deterministic(data.shape[0], seasonal)
    if lags is None:
        lags = select_lag_order(data, det, max_lags=max_lags, extra_lags=extra_lags)
    elif lags < 1:
        raise ValueError(f"lags must be >= 1; got {lags}")

    n_lags = lags + extra_lags
    target, design = _var_design(data, det, n_lags)
    n_obs, n_coef = design.shape
    df_den = n_obs - n_coef
    if df_den < 1:
        raise ValueError(
            f"Too few observations: a VAR with {n_lags} lags of {data.shape[1]} series "
            f"needs more than {n_coef} rows after lags; got {n_obs}"
        )

    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    xtx_inv = np.linalg.pinv(design.T @ design)
    n_det = det.shape[1]
    n_var = data.shape[1]
    is_metric = [i < k for i in range(n_var)]

    rows = []
    for effect in range(n_var):
        s2 = resid[:, effect] @ resid[:, effect] / df_den
        for cause in range(n_var):
            if cause == effect:
                continue
            idx = [n_det + lag * n_var + cause for lag in range(lags)]
            b = coef[idx, effect]
            v = s2 * xtx_inv[np.ix_(idx, idx)]
            wald = float(b @ np.linalg.solve(v, b))
            f_stat = wald / lags
            rows.append(
                {
                    "cause": names[cause],
                    "effect": names[effect],
                    "group": _group(is_metric[cause], is_metric[effect], "->"),
                    "f_stat": f_stat,
                    "df_num": lags,
                    "df_den": df_den,
                    "p_value": float(stats.f.sf(f_stat, lags, df_den)),
                }
            )

    delay = None
    if n_var > k:
        delay = _delay_tests(target, design, data[n_lags:, k:], names, k, lags, n_det)

    corr = np.corrcoef(resid, rowvar=False)
    same = []
    for i in range(n_var):
        for j in range(i + 1, n_var):
            r = float(np.clip(corr[i, j], -0.999999, 0.999999))
            t_stat = r * np.sqrt((n_obs - 2) / (1 - r**2))
            same.append(
                {
                    "variable_1": names[i],
                    "variable_2": names[j],
                    "group": _group(is_metric[i], is_metric[j], "-"),
                    "corr": r,
                    "p_value": float(2 * stats.t.sf(abs(t_stat), n_obs - 2)),
                }
            )

    return GrangerPretestResult(
        tests=pd.DataFrame(rows),
        delay=delay,
        same_period=pd.DataFrame(same),
        lags=int(lags),
        extra_lags=int(extra_lags),
        n_obs=int(n_obs),
        alpha=alpha,
        seasonal=seasonal,
    )


def _f_test(
    target: NDArray[np.float64], design: NDArray[np.float64], idx: list[int]
) -> tuple[float, int, float]:
    """F test that ``design[:, idx]`` has zero coefficients for one equation."""
    df_den = design.shape[0] - design.shape[1]
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ coef
    s2 = resid @ resid / df_den
    v = s2 * np.linalg.pinv(design.T @ design)[np.ix_(idx, idx)]
    b = coef[idx]
    f_stat = float(b @ np.linalg.solve(v, b)) / len(idx)
    return f_stat, df_den, float(stats.f.sf(f_stat, len(idx), df_den))


def _delay_tests(
    target: NDArray[np.float64],
    design: NDArray[np.float64],
    current: NDArray[np.float64],
    names: list[str],
    k: int,
    lags: int,
    n_det: int,
) -> pd.DataFrame | None:
    """Driver lag tests in each metric's equation with current drivers added."""
    import pandas as pd

    augmented = np.column_stack([design, current])
    if augmented.shape[0] - augmented.shape[1] < 1:
        return None
    n_var = len(names)
    rows = []
    for effect in range(k):
        for cause in range(k, n_var):
            idx = [n_det + lag * n_var + cause for lag in range(lags)]
            f_stat, df_den, p_value = _f_test(target[:, effect], augmented, idx)
            rows.append(
                {
                    "cause": names[cause],
                    "effect": names[effect],
                    "group": "delay",
                    "f_stat": f_stat,
                    "df_num": lags,
                    "df_den": df_den,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def select_lag_order(
    data: NDArray[np.float64],
    det: NDArray[np.float64],
    *,
    max_lags: int = 8,
    extra_lags: int = 1,
) -> int:
    """Hannan-Quinn lag order of a levels VAR, at least 1.

    Every candidate is fitted on the same rows so the criteria compare. The
    penalty counts the ``K * K * p`` lag coefficients. The deterministic
    terms are the same for every candidate and cancel.
    """
    n_total, n_var = data.shape
    max_lags = min(max_lags, _largest_lag(n_total, n_var, det.shape[1], extra_lags))
    if max_lags < 1:
        raise ValueError(
            f"Too few observations to fit even one lag of {n_var} series; got T={n_total}"
        )
    best_p, best_hq = 1, np.inf
    for p in range(1, max_lags + 1):
        target, design = _var_design(data, det, p, start=max_lags)
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ coef
        n_obs = target.shape[0]
        sigma = resid.T @ resid / n_obs
        _, logdet = np.linalg.slogdet(sigma)
        hq = logdet + 2 * np.log(np.log(n_obs)) * n_var**2 * p / n_obs
        if hq < best_hq:
            best_p, best_hq = p, hq
    return best_p


def _largest_lag(n_total: int, n_var: int, n_det: int, extra_lags: int) -> int:
    """Largest p whose test regression keeps half its rows as residual dof.

    The test regression has ``p + extra_lags`` lags. The floor is also at
    least 10 degrees of freedom.
    """

    def ok(p: int) -> bool:
        n_lags = p + extra_lags
        rows = n_total - n_lags
        return rows - (n_det + n_var * n_lags) >= max(10, rows / 2)

    p = 0
    while ok(p + 1):
        p += 1
    return p


def _var_design(
    data: NDArray[np.float64], det: NDArray[np.float64], n_lags: int, *, start: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Target rows and regressors ``[det, y_{t-1}, ..., y_{t-n_lags}]``."""
    start = n_lags if start is None else start
    n_total = data.shape[0]
    blocks = [det[start:]]
    for lag in range(1, n_lags + 1):
        blocks.append(data[start - lag : n_total - lag])
    return data[start:], np.column_stack(blocks)


def _deterministic(n_total: int, seasonal: int | None) -> NDArray[np.float64]:
    """Constant plus ``seasonal - 1`` centred seasonal dummies."""
    cols = [np.ones(n_total)]
    if seasonal:
        pos = np.arange(n_total) % seasonal
        for s in range(1, seasonal):
            cols.append((pos == s) - 1.0 / seasonal)
    return np.column_stack(cols)


def _group(first_is_metric: bool, second_is_metric: bool, sep: str) -> str:
    first = "metric" if first_is_metric else "driver"
    second = "metric" if second_is_metric else "driver"
    return f"{first} {sep} {second}"


def _names(data: Any, n: int, prefix: str) -> list[str]:
    if hasattr(data, "columns"):
        return [str(c) for c in data.columns]
    return [f"{prefix}{i}" for i in range(n)]
