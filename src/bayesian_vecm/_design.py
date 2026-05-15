"""Design matrices for VECM estimation.

This module exposes a single helper, :func:`cointegration_design`, that turns a
raw ``(T, K)`` observation matrix into the three aligned matrices a VECM
regression equation actually consumes.

The VECM in error-correction form is

.. math::

    \\Delta y_t = \\alpha \\beta' y_{t-1}
                 + \\sum_{i=1}^{k} \\Gamma_i \\, \\Delta y_{t-i}
                 + \\varepsilon_t.

For each usable time :math:`t` the regression row needs three things:

* :math:`\\Delta y_t` — the left-hand side.
* :math:`y_{t-1}` — feeds the cointegration relation :math:`\\beta' y_{t-1}`. It
  is the level lag, **not** a difference; it is what makes this a VECM and not
  a plain VAR-in-differences.
* :math:`[\\Delta y_{t-1}, \\dots, \\Delta y_{t-k}]` — the regressors that pick
  up short-run dynamics through the :math:`\\Gamma_i` matrices.

The smallest :math:`t` for which all three are well defined is
:math:`t = k + 1` (0-based), so the effective sample size is

.. math::

    T_{\\text{eff}} = T - k - 1.

Deterministic terms (constants and trends inside or outside the cointegration
relation) are deferred to a follow-up; this module ships only the "no
constants, no trends" case.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from bayesian_vecm._data import difference, lag_matrix, validate_endog


class CointegrationDesign(NamedTuple):
    """Aligned design matrices for a VECM regression.

    All three arrays have the same number of rows, ``T_eff = T - k_ar_diff - 1``;
    row :math:`r` of each refers to the same underlying time index.

    Attributes
    ----------
    delta_y
        Left-hand side :math:`\\Delta y_t`, shape ``(T_eff, K)``.
    delta_x
        Stacked lagged differences :math:`[\\Delta y_{t-1}, \\dots, \\Delta
        y_{t-k}]`, shape ``(T_eff, K * k_ar_diff)``. Columns are lag-major and
        most-recent-first (the convention used by
        :mod:`statsmodels.tsa.vector_ar`).
    y_lag1
        Level lag :math:`y_{t-1}`, shape ``(T_eff, K)``. This is the term the
        cointegration relation :math:`\\beta' y_{t-1}` is built from.
    """

    delta_y: NDArray[np.float64]
    delta_x: NDArray[np.float64]
    y_lag1: NDArray[np.float64]


def cointegration_design(data: Any, k_ar_diff: int) -> CointegrationDesign:
    """Build the aligned ``(delta_y, delta_x, y_lag1)`` triple for a VECM.

    Parameters
    ----------
    data
        Endogenous time series as a 2-D array of shape ``(T, K)``. Accepts
        anything :func:`bayesian_vecm._data.validate_endog` accepts — including
        objects with a ``.to_numpy()`` method, such as a ``pandas.DataFrame``.
    k_ar_diff
        Number of lagged-difference blocks ``Δy_{t-1}, ..., Δy_{t-k}`` to
        include in ``delta_x``. ``k_ar_diff = 0`` means no short-run dynamics
        — the regression reduces to a pure error-correction equation — and in
        that case ``delta_x`` has zero columns but the other two matrices are
        still well-defined.

    Returns
    -------
    CointegrationDesign
        ``(delta_y, delta_x, y_lag1)`` with a common ``T_eff = T - k_ar_diff - 1``
        rows.

    Raises
    ------
    ValueError
        If ``k_ar_diff`` is negative, if ``k_ar_diff >= T - 1`` (no usable
        rows), or if ``data`` fails any of ``validate_endog``'s checks.
    """
    if k_ar_diff < 0:
        raise ValueError(f"k_ar_diff must be non-negative; got k_ar_diff={k_ar_diff}")

    # Clean and shape-check the data first with the default min_obs=2. The
    # tighter "do we have enough rows for this k_ar_diff?" question is asked
    # below so we can give a dedicated error message instead of having the
    # generic min_obs check fire.
    arr = validate_endog(data)

    n_obs = arr.shape[0]
    if k_ar_diff >= n_obs - 1:
        raise ValueError(
            "k_ar_diff must be fewer than T - 1 so the effective sample is non-empty; "
            f"got k_ar_diff={k_ar_diff}, T={n_obs}"
        )

    # First differences once; both delta_y and delta_x are slices/lag-stacks of it.
    delta = difference(arr, d=1)  # shape (T - 1, K)

    # delta_y: drop the first k_ar_diff rows so that delta_y[0] corresponds to
    # the earliest t for which all lagged-difference regressors exist.
    delta_y = np.ascontiguousarray(delta[k_ar_diff:])  # shape (T_eff, K)

    # delta_x: stack k_ar_diff lags of the differenced series. lag_matrix
    # naturally produces shape (T - 1 - k_ar_diff, K * k_ar_diff) = (T_eff, K * k).
    # For k_ar_diff == 0 we want a (T_eff, 0) array, which lag_matrix already
    # returns — but with T_eff rows derived from `delta`, not from `arr`, so we
    # special-case it to be explicit about shape.
    if k_ar_diff == 0:
        delta_x = np.empty((delta_y.shape[0], 0), dtype=delta.dtype)
    else:
        delta_x = lag_matrix(delta, n_lags=k_ar_diff)

    # y_lag1: y_{t-1} for each usable t. The earliest usable t is k_ar_diff + 1
    # (0-based), so y_{t-1} starts at index k_ar_diff and runs through T - 2.
    y_lag1 = np.ascontiguousarray(arr[k_ar_diff : n_obs - 1])  # shape (T_eff, K)

    return CointegrationDesign(delta_y=delta_y, delta_x=delta_x, y_lag1=y_lag1)
