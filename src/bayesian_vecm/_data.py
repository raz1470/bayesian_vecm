"""Data preprocessing utilities for Bayesian VECM models.

These are pure numpy helpers shared by every estimation path in the package.
They are deliberately small, pure functions so they can be unit-tested in
isolation before any PyMC machinery is wired in.

Econometric notation used throughout this module
------------------------------------------------
Let :math:`y_t \\in \\mathbb{R}^K` be a vector of observations at time
:math:`t = 1, \\dots, T` collected as rows of a matrix
:math:`Y \\in \\mathbb{R}^{T \\times K}`. The VECM in error-correction form is

.. math::

    \\Delta y_t = \\alpha \\beta' y_{t-1}
                 + \\sum_{i=1}^{k} \\Gamma_i \\, \\Delta y_{t-i}
                 + \\varepsilon_t,

so estimation requires (a) the differenced series :math:`\\Delta y` and
(b) a stack of its lags. That is exactly what :func:`difference` and
:func:`lag_matrix` produce, and :func:`validate_endog` is the input
sanitiser that runs before either of them touches user data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_endog(data: Any, *, min_obs: int = 2) -> NDArray[np.float64]:
    """Coerce ``data`` into a clean ``(T, K)`` float64 array.

    Accepts a 2-D ``numpy`` array or any object exposing a ``to_numpy()``
    method (e.g. a ``pandas.DataFrame``). The returned array is always a
    fresh ``float64`` copy, so downstream code can mutate it freely.

    Parameters
    ----------
    data
        The endogenous time series. Must be 2-D with shape ``(T, K)`` where
        ``T`` is the number of observations and ``K`` the number of variables.
    min_obs
        Minimum number of time points required. Defaults to ``2``.

    Returns
    -------
    numpy.ndarray
        A contiguous ``float64`` array of shape ``(T, K)``.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, contains NaN, has fewer than ``min_obs``
        rows, has zero columns, or cannot be cast to floating point.
    """
    # Duck-type the pandas path: anything with ``to_numpy`` is fair game.
    if hasattr(data, "to_numpy"):
        array: ArrayLike = data.to_numpy()
    else:
        array = data

    try:
        arr = np.array(array, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("endog must be convertible to a numeric (float) array") from exc

    if arr.ndim != 2:
        raise ValueError(f"endog must be 2-dimensional with shape (T, K); got ndim={arr.ndim}")

    if arr.shape[1] < 1:
        raise ValueError("endog must have at least one variable (K >= 1)")

    if arr.shape[0] < min_obs:
        raise ValueError(f"endog must have at least {min_obs} observations; got T={arr.shape[0]}")

    if np.isnan(arr).any():
        raise ValueError("endog must not contain NaN values")

    return np.ascontiguousarray(arr)


def difference(data: NDArray[np.floating], d: int = 1) -> NDArray[np.floating]:
    """Return the ``d``-th order differences of a multivariate series.

    For ``d = 1`` this is :math:`\\Delta y_t = y_t - y_{t-1}`. For higher
    ``d``, the operator is applied recursively, so the output has shape
    ``(T - d, K)``.

    Parameters
    ----------
    data
        Array of shape ``(T, K)``.
    d
        Order of differencing. ``d = 0`` returns a copy of ``data``.

    Returns
    -------
    numpy.ndarray
        Differenced series, shape ``(T - d, K)``.

    Raises
    ------
    ValueError
        If ``d`` is negative, or ``d`` is greater than or equal to ``T``.
    """
    if d < 0:
        raise ValueError(f"d must be non-negative; got d={d}")
    if d >= data.shape[0]:
        raise ValueError(
            f"d must be fewer than the number of observations T={data.shape[0]}; got d={d}"
        )

    result = np.array(data, copy=True)
    for _ in range(d):
        result = result[1:] - result[:-1]
    return result


def lag_matrix(data: NDArray[np.floating], n_lags: int) -> NDArray[np.floating]:
    """Stack lagged observations into a regressor matrix.

    Given ``data`` of shape ``(T, K)``, returns an array of shape
    ``(T - n_lags, K * n_lags)`` where row ``r`` holds

    .. math::

        [\\, y_{r + n\\_lags - 1}, \\; y_{r + n\\_lags - 2}, \\; \\dots, \\; y_{r} \\,],

    i.e. lags are ordered from most recent to most distant, with the ``K``
    entries of each lag laid out contiguously. This convention matches
    :mod:`statsmodels.tsa.vector_ar`.

    Parameters
    ----------
    data
        Array of shape ``(T, K)``.
    n_lags
        Number of lags to stack. ``n_lags = 0`` returns an array of shape
        ``(T, 0)``.

    Returns
    -------
    numpy.ndarray
        Lagged design matrix, shape ``(T - n_lags, K * n_lags)``.

    Raises
    ------
    ValueError
        If ``n_lags`` is negative, or ``n_lags`` is greater than or equal
        to ``T``.
    """
    if n_lags < 0:
        raise ValueError(f"n_lags must be non-negative; got n_lags={n_lags}")
    n_obs, n_vars = data.shape
    if n_lags >= n_obs:
        raise ValueError(
            f"n_lags must be fewer than the number of observations T={n_obs}; got n_lags={n_lags}"
        )

    if n_lags == 0:
        return np.empty((n_obs, 0), dtype=data.dtype)

    rows = n_obs - n_lags
    out = np.empty((rows, n_vars * n_lags), dtype=data.dtype)
    for lag in range(1, n_lags + 1):
        col_start = (lag - 1) * n_vars
        col_stop = lag * n_vars
        out[:, col_start:col_stop] = data[n_lags - lag : n_obs - lag]
    return out
