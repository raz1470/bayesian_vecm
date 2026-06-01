"""Classical diagnostic tests for ``BayesianVECM`` residuals.

Both tests operate on the **posterior-mean residuals** — a single
:math:`(T_{\\text{eff}}, K)` array obtained by averaging the full posterior
residual distribution over chains and draws.  This is intentional: the tests
are classical frequentist checks (Jarque-Bera, Ljung-Box) that expect a single
residual series, not a posterior distribution.  For a richer Bayesian
residual-adequacy check, use posterior predictive checks with ArviZ.

Tests provided
--------------
``normality_test``
    Jarque-Bera test of normality applied variable-by-variable to the
    posterior-mean residuals.  H0: residuals are normally distributed.

``whiteness_test``
    Ljung-Box portmanteau test for residual autocorrelation, applied
    variable-by-variable.  H0: no autocorrelation up to ``lags`` periods.
    The Q-statistic is

    .. math::

        Q(m) = T(T + 2) \\sum_{k=1}^{m} \\frac{\\hat{\\rho}_k^2}{T - k},

    where :math:`\\hat{\\rho}_k` is the sample autocorrelation at lag
    :math:`k` and :math:`T = T_{\\text{eff}}` is the effective sample size.
    Under H0, :math:`Q(m) \\sim \\chi^2(m)`.

Scipy is the only dependency (for chi-squared p-values and the Jarque-Bera
statistic).  Scipy is already a transitive dependency of PyMC, but is
declared explicitly in ``pyproject.toml`` to document the runtime requirement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import chi2

from bayesian_vecm._output import compute_resid


def normality_test(
    idata: xr.DataTree,
    k_ar_diff: int,
    variable_names: list[str] | None = None,
) -> pd.DataFrame:
    """Jarque-Bera normality test on posterior-mean residuals.

    Tests each endogenous variable independently using
    :func:`scipy.stats.jarque_bera`.  H0: residuals are normally distributed.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``.
    k_ar_diff
        Number of lagged-difference blocks in the model.
    variable_names
        Optional variable labels used as the DataFrame index.

    Returns
    -------
    pandas.DataFrame
        One row per endogenous variable, columns:

        * ``jb_stat`` — Jarque-Bera test statistic.
        * ``p_value`` — two-sided p-value under :math:`\\chi^2(2)`.

        A small ``p_value`` (e.g. < 0.05) is evidence against normality.
    """
    resid_da = compute_resid(idata, k_ar_diff, variable_names=variable_names)
    # Posterior-mean residuals: (T_eff, K)
    resid_mean: NDArray[np.floating] = resid_da.mean(("chain", "draw")).values

    n_vars = resid_mean.shape[1]
    rows = []
    for j in range(n_vars):
        jb_stat, p_val = stats.jarque_bera(resid_mean[:, j])
        rows.append({"jb_stat": float(jb_stat), "p_value": float(p_val)})

    index = variable_names if variable_names is not None else [f"var_{j}" for j in range(n_vars)]
    return pd.DataFrame(rows, index=index)


def whiteness_test(
    idata: xr.DataTree,
    k_ar_diff: int,
    *,
    lags: int = 10,
    variable_names: list[str] | None = None,
) -> pd.DataFrame:
    """Ljung-Box whiteness (no-autocorrelation) test on posterior-mean residuals.

    Tests each endogenous variable independently using the Ljung-Box
    Q-statistic.  H0: no autocorrelation at any lag up to ``lags``.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``.
    k_ar_diff
        Number of lagged-difference blocks in the model.
    lags
        Maximum lag order for the portmanteau test.  Defaults to ``10``.
        The Q-statistic is compared to :math:`\\chi^2(\\text{lags})`.
    variable_names
        Optional variable labels used as the DataFrame index.

    Returns
    -------
    pandas.DataFrame
        One row per endogenous variable, columns:

        * ``lb_stat`` — Ljung-Box Q-statistic at ``lags`` periods.
        * ``p_value`` — p-value under :math:`\\chi^2(\\text{lags})`.
        * ``lags`` — the lag order used.

        A small ``p_value`` (e.g. < 0.05) is evidence against white noise.

    Raises
    ------
    ValueError
        If ``lags < 1``.
    """
    if lags < 1:
        raise ValueError(f"lags must be at least 1; got lags={lags}")

    resid_da = compute_resid(idata, k_ar_diff, variable_names=variable_names)
    resid_mean: NDArray[np.floating] = resid_da.mean(("chain", "draw")).values

    t_eff, n_vars = resid_mean.shape
    rows = []

    for j in range(n_vars):
        r = resid_mean[:, j]
        r_centered = r - r.mean()
        c0 = np.dot(r_centered, r_centered) / t_eff

        if c0 == 0.0:
            # Degenerate case: constant residuals — no autocorrelation by definition.
            rows.append({"lb_stat": 0.0, "p_value": 1.0, "lags": lags})
            continue

        # Ljung-Box Q = T(T+2) * sum_{m=1}^{lags} rho_m^2 / (T - m)
        q = 0.0
        for m in range(1, lags + 1):
            rho_m = np.dot(r_centered[m:], r_centered[:-m]) / (t_eff * c0)
            q += rho_m**2 / (t_eff - m)
        q *= t_eff * (t_eff + 2)

        p_val = float(1.0 - chi2.cdf(q, df=lags))
        rows.append({"lb_stat": float(q), "p_value": p_val, "lags": lags})

    index = variable_names if variable_names is not None else [f"var_{j}" for j in range(n_vars)]
    return pd.DataFrame(rows, index=index)
