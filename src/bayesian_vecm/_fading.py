"""Posterior helpers for drivers whose effect fades.

A fading driver :math:`j` adds :math:`c_t` to the growth of the endogenous
series in the week it acts (its lag blocks of ``B`` plus any saturated term),
and then takes back the share :math:`\\phi_j` (``fade_share``) of the lift at
weekly rate :math:`\\delta_j = 0.5^{1/h_j}` (``fade_half_life``):

.. math::

    -\\phi_j (1 - \\delta_j) S_t, \\qquad
    S_t = \\sum_{k \\ge 1} \\delta_j^{k-1} c_{t-k},

with :math:`c` before the sample at its in-sample mean. A one-week pulse then
lifts the levels by :math:`C c` at first, and only :math:`(1 - \\phi_j) C c`
stays for good.

The fit stores ``fade_share`` and ``fade_half_life`` in the posterior and
``fading_idx`` in ``constant_data``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from bayesian_vecm._saturation import has_saturation, saturation_info, saturation_parts


def has_fading(idata: Any) -> bool:
    """Whether the fit has fading drivers."""
    return "fade_share" in idata.posterior


def fading_info(idata: Any) -> tuple[list[int], NDArray[np.float64], NDArray[np.float64]]:
    """Fading driver positions, and ``fade_share`` and weekly decay draws ``(D, n_fade)``."""
    idx = [int(i) for i in np.asarray(idata.constant_data["fading_idx"].values)]
    post = idata.posterior
    share = np.asarray(post["fade_share"].values, dtype=np.float64)
    half_life = np.asarray(post["fade_half_life"].values, dtype=np.float64)
    n = share.shape[0] * share.shape[1]
    share = share.reshape(n, -1)
    delta = 0.5 ** (1.0 / half_life.reshape(n, -1))
    return idx, share, delta


def driver_contributions(
    idata: Any, x_lagged: NDArray[np.floating], drivers: list[int]
) -> NDArray[np.float64]:
    """Each driver's contribution to growth, ``(D, T, n_drivers, K)``.

    ``x_lagged`` is the lag-major exog design ``(T, m * (L + 1))``. Uses the
    reduced-form ``B`` (and ``B_saturation`` for saturated drivers), so the
    same-week links are included.
    """
    post = idata.posterior
    b = np.asarray(post["B"].values, dtype=np.float64)
    b = b.reshape(b.shape[0] * b.shape[1], *b.shape[2:])  # (D, K, m * (L + 1))
    x = np.asarray(x_lagged, dtype=np.float64)
    n_lags = _n_lags(idata, b.shape[-1], x.shape[1])
    n_exog = b.shape[-1] // n_lags
    sat_idx: list[int] = []
    sat = None
    if has_saturation(idata):
        sat_idx, _ = saturation_info(idata)
        sat = saturation_parts(idata, x)  # (D, T, m_sat, K)
    out = np.zeros((b.shape[0], x.shape[0], len(drivers), b.shape[1]))
    for pos, j in enumerate(drivers):
        if j in sat_idx:
            out[:, :, pos, :] = sat[:, :, sat_idx.index(j), :]
        else:
            out[:, :, pos, :] = np.einsum("tl,dkl->dtk", x[:, j::n_exog], b[:, :, j::n_exog])
    return out


def fade_term(idata: Any, x_lagged: NDArray[np.floating], n_fit: int) -> NDArray[np.float64]:
    """Fading part of the mean for the rows of ``x_lagged``, ``(D, T, K)``.

    The first ``n_fit`` rows must be the fitted design's rows, which set the
    in-sample mean that stands in for the weeks before the sample. Rows
    after them (a forecast) continue the same recursion.
    """
    idx, share, delta = fading_info(idata)
    contrib = driver_contributions(idata, x_lagged, idx)  # (D, T, n_fade, K)
    n_draws, n_rows = contrib.shape[:2]
    mean = contrib[:, :n_fit].mean(axis=1)  # (D, n_fade, K)
    state = mean / (1.0 - delta)[:, :, None]  # S_0
    out = np.zeros((n_draws, n_rows, contrib.shape[-1]))
    for t in range(n_rows):
        if t > 0:
            state = delta[:, :, None] * state + contrib[:, t - 1]
        out[:, t] = -(share[:, :, None] * (1.0 - delta)[:, :, None] * state).sum(axis=1)
    return out


def fade_kernel(idata: Any, steps: int) -> tuple[list[int], NDArray[np.float64]]:
    """Weight of a one-period rise at each horizon, per fading driver, ``(D, n_fade, steps + 1)``.

    ``1`` at horizon 0 and :math:`-\\phi (1 - \\delta) \\delta^{h-1}` after it,
    so the weights sum to :math:`1 - \\phi`.
    """
    idx, share, delta = fading_info(idata)
    h = np.arange(1, steps + 1, dtype=np.float64)
    tail = -(share * (1.0 - delta))[:, :, None] * delta[:, :, None] ** (h - 1.0)
    head = np.ones((*share.shape, 1))
    return idx, np.concatenate([head, tail], axis=2)


def faded_b(b: NDArray[np.floating], idata: Any, steps: int, n_lags: int) -> NDArray[np.float64]:
    """Lag-major ``B`` draws extended to ``steps + 1`` lags with the fade applied.

    ``b`` has shape ``(C, D, K, m * n_lags)``. Returns
    ``(C, D, K, m * (steps + 1))``: each fading driver's lag coefficients are
    convolved with :func:`fade_kernel`; the other drivers keep theirs.
    """
    b = np.asarray(b, dtype=np.float64)
    n_chains, n_draws, n_vars, n_cols = b.shape
    n_exog = n_cols // n_lags
    flat = b.reshape(n_chains * n_draws, n_vars, n_lags, n_exog)
    out = np.zeros((n_chains * n_draws, n_vars, steps + 1, n_exog))
    keep = min(n_lags, steps + 1)
    out[:, :, :keep, :] = flat[:, :, :keep, :]
    idx, kernel = fade_kernel(idata, steps)
    for pos, j in enumerate(idx):
        series = np.zeros((n_chains * n_draws, n_vars, steps + 1))
        for lag in range(keep):
            series[:, :, lag:] += flat[:, :, lag, j, None] * kernel[:, None, pos, : steps + 1 - lag]
        out[:, :, :, j] = series
    return out.reshape(n_chains, n_draws, n_vars, (steps + 1) * n_exog)


def _n_lags(idata: Any, b_cols: int, x_cols: int) -> int:
    """Number of lag blocks in ``B``, from the window stored at fit time."""
    if b_cols != x_cols:
        raise ValueError(f"exog design has {x_cols} columns but B has {b_cols}")
    return int(np.asarray(idata.constant_data["exog_window"].values)) + 1
