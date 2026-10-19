"""Posterior helpers for saturated drivers.

A saturated driver enters the mean as :math:`b \\, [f(a_t / s) - \\bar f]`
rather than through the lag blocks of ``B``. The fit stores what is needed to
rebuild that term from the posterior:

* ``saturation_lam``, ``saturation_weights`` (lag weights, adstock or same
  week), ``saturation_mean`` (:math:`\\bar f`) and ``B_saturation``
  (:math:`b`) in the posterior;
* ``saturation_idx`` and ``saturation_scale`` (:math:`s`) in
  ``constant_data``.

``B`` holds, for saturated drivers, the slope per unit of spend at the
in-sample mean. In-sample fitted values and forecasts zero those columns and
add the exact term instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bayesian_vecm._transform import logistic_saturation, logistic_saturation_slope


def has_saturation(idata: Any) -> bool:
    """Whether the fit has saturated drivers."""
    return "saturation_lam" in idata.posterior


def saturation_info(idata: Any) -> tuple[list[int], NDArray[np.float64]]:
    """Saturated driver positions and their spend scales."""
    const = idata.constant_data
    idx = [int(i) for i in np.asarray(const["saturation_idx"].values)]
    return idx, np.asarray(const["saturation_scale"].values, dtype=np.float64)


def _draws(posterior: Any, name: str) -> NDArray[np.float64]:
    values = np.asarray(posterior[name].values, dtype=np.float64)
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def zero_saturated_columns(b: NDArray[np.floating], idx: list[int], n_exog: int) -> NDArray:
    """Copy of lag-major ``b`` (``..., K, m * (L + 1)``) with saturated columns zeroed."""
    out = np.array(b, dtype=np.float64, copy=True)
    for j in idx:
        out[..., j::n_exog] = 0.0
    return out


def saturation_term(idata: Any, x_lagged: NDArray[np.floating]) -> NDArray[np.float64]:
    """Saturated contribution to the mean, shape ``(D, T, K)``.

    ``x_lagged`` is the lag-major exog design, ``(T, m * (L + 1))``, in raw
    units for the saturated drivers.
    """
    posterior = idata.posterior
    idx, scale = saturation_info(idata)
    lam = _draws(posterior, "saturation_lam")  # (D, m_sat)
    weights = _draws(posterior, "saturation_weights")  # (D, m_sat, n_lags)
    f_mean = _draws(posterior, "saturation_mean")  # (D, m_sat)
    b_sat = _draws(posterior, "B_saturation")  # (D, K, m_sat)
    n_lags = weights.shape[-1]
    x = np.asarray(x_lagged, dtype=np.float64)
    n_exog = x.shape[1] // n_lags
    lagged = x.reshape(x.shape[0], n_lags, n_exog)[:, :, idx]  # (T, n_lags, m_sat)
    stock = np.einsum("tlj,djl->dtj", lagged, weights) / scale  # (D, T, m_sat)
    f = logistic_saturation(stock, lam[:, None, :]) - f_mean[:, None, :]
    return np.einsum("dtj,dkj->dtk", f, b_sat)


def linearised_b(idata: Any, levels: NDArray[np.floating]) -> NDArray[np.float64]:
    """``B`` draws with saturated drivers linearised at ``levels``.

    ``levels`` holds one spend level per saturated driver, in raw units.
    Returns ``(C, D, K, m * (L + 1))``, the shape of the posterior ``B``.
    """
    posterior = idata.posterior
    idx, scale = saturation_info(idata)
    b = np.asarray(posterior["B"].values, dtype=np.float64)
    n_chains, n_draws = b.shape[:2]
    lam = _draws(posterior, "saturation_lam")  # (D, m_sat)
    weights = _draws(posterior, "saturation_weights")  # (D, m_sat, n_lags)
    b_sat = _draws(posterior, "B_saturation")  # (D, K, m_sat)
    n_lags = weights.shape[-1]
    n_exog = b.shape[-1] // n_lags
    slope = logistic_saturation_slope(np.asarray(levels) / scale, lam) / scale  # (D, m_sat)
    out = b.reshape(n_chains * n_draws, *b.shape[2:]).copy()
    for pos, j in enumerate(idx):
        # (D, K, n_lags) = b * slope * weight_l, into columns l * m + j.
        blocks = b_sat[:, :, pos, None] * (slope[:, pos, None] * weights[:, pos, :])[:, None, :]
        out[:, :, j::n_exog] = blocks
    return out.reshape(b.shape)


def response_curve(
    idata: Any, driver: int, levels: NDArray[np.floating], *, marginal: bool = False
) -> NDArray[np.float64]:
    """Total effect on each series' growth of holding a driver at each level.

    Returns ``(C, D, n_levels, K)``. Relative to the driver's in-sample
    average contribution. With ``marginal=True``, the slope per unit at each
    level instead.
    """
    posterior = idata.posterior
    idx, scale = saturation_info(idata)
    pos = idx.index(driver)
    lam_all = np.asarray(posterior["saturation_lam"].values, dtype=np.float64)
    n_chains, n_draws = lam_all.shape[:2]
    lam = _draws(posterior, "saturation_lam")[:, pos, None]  # (D, 1)
    b_sat = _draws(posterior, "B_saturation")[:, :, pos]  # (D, K)
    u = np.asarray(levels, dtype=np.float64)[None, :] / scale[pos]  # (1, n)
    if marginal:
        g = logistic_saturation_slope(u, lam) / scale[pos]
    else:
        g = logistic_saturation(u, lam) - _draws(posterior, "saturation_mean")[:, pos, None]
    out = g[:, :, None] * b_sat[:, None, :]  # (D, n, K)
    return out.reshape(n_chains, n_draws, *out.shape[1:])
