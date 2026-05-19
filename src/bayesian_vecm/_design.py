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

Deterministic terms (constants and linear trends, either inside or outside
the cointegration relation) are supported via the ``deterministic`` argument.
Single-character codes only in v0; compound cases — Johansen cases 4 and 5,
e.g. ``"colo"``, ``"cili"`` — are a planned follow-up.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from bayesian_vecm._data import difference, lag_matrix, validate_endog

#: Single-character ``deterministic`` codes accepted in v0.
#:
#: * ``"n"``  — no deterministic terms (default).
#: * ``"co"`` — constant *outside* the cointegration relation; appended as a
#:   column to ``delta_x``.
#: * ``"ci"`` — constant *inside* the cointegration relation; appended as a
#:   column to ``y_lag1``.
#: * ``"lo"`` — linear trend *outside* the cointegration relation; appended as
#:   a column to ``delta_x``.
#: * ``"li"`` — linear trend *inside* the cointegration relation; appended as
#:   a column to ``y_lag1``.
VALID_DETERMINISTIC: tuple[str, ...] = ("n", "co", "ci", "lo", "li")


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


def cointegration_design(
    data: Any,
    k_ar_diff: int,
    deterministic: str = "n",
) -> CointegrationDesign:
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
    deterministic
        Which deterministic terms to add to the design. One of
        :data:`VALID_DETERMINISTIC`. Defaults to ``"n"`` (no deterministic
        terms). Outside terms (``"co"``, ``"lo"``) append a trailing column
        to ``delta_x``; inside terms (``"ci"``, ``"li"``) append a trailing
        column to ``y_lag1``. Trend columns are 1-indexed:
        ``[1, 2, ..., T_eff]`` (the origin is arbitrary — only the slope
        coefficient is invariant under shifts — so we choose the convention
        that keeps the column strictly positive). Compound cases such as
        ``"colo"`` and ``"cili"`` (Johansen cases 4 and 5) are a planned
        follow-up and are rejected in v0.

    Returns
    -------
    CointegrationDesign
        ``(delta_y, delta_x, y_lag1)`` with a common ``T_eff = T - k_ar_diff - 1``
        rows. The shapes of ``delta_x`` and ``y_lag1`` reflect any
        deterministic columns appended at the end.

    Raises
    ------
    ValueError
        If ``k_ar_diff`` is negative, if ``k_ar_diff >= T - 1`` (no usable
        rows), if ``deterministic`` is not in :data:`VALID_DETERMINISTIC`, or
        if ``data`` fails any of ``validate_endog``'s checks.
    """
    if k_ar_diff < 0:
        raise ValueError(f"k_ar_diff must be non-negative; got k_ar_diff={k_ar_diff}")

    if deterministic not in VALID_DETERMINISTIC:
        # Pick a message tailored to the failure mode so users know whether
        # they've mistyped a code or hit the (intentional) v0 gap on compound
        # cases like "colo" / "cili" — Johansen cases 4 and 5 — which are a
        # planned follow-up.
        if _looks_like_compound_code(deterministic):
            raise ValueError(
                f"deterministic={deterministic!r} looks like a compound code. "
                "Compound deterministic cases (Johansen cases 4 and 5, e.g. "
                "'colo', 'cili') are a planned follow-up; v0 supports only "
                f"single codes {VALID_DETERMINISTIC}."
            )
        raise ValueError(
            f"deterministic must be one of {VALID_DETERMINISTIC}; "
            f"got deterministic={deterministic!r}."
        )

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

    # Append deterministic-term columns. Outside terms ("co", "lo") attach to
    # delta_x; inside terms ("ci", "li") attach to y_lag1 (so that, downstream,
    # the cointegration relation β' y_{t-1} naturally absorbs them).
    n_eff = delta_y.shape[0]
    if deterministic in ("co", "lo"):
        column = _deterministic_column(deterministic, n_eff, dtype=delta_x.dtype)
        delta_x = np.column_stack([delta_x, column])
    elif deterministic in ("ci", "li"):
        column = _deterministic_column(deterministic, n_eff, dtype=y_lag1.dtype)
        y_lag1 = np.column_stack([y_lag1, column])
    # deterministic == "n" → no-op; handled by the membership check above.

    return CointegrationDesign(delta_y=delta_y, delta_x=delta_x, y_lag1=y_lag1)


def _deterministic_column(code: str, n_eff: int, dtype: np.dtype) -> NDArray[np.float64]:
    """Build the single deterministic column corresponding to ``code``.

    ``"co"`` and ``"ci"`` yield a column of ones; ``"lo"`` and ``"li"`` yield
    the 1-indexed trend ``[1, 2, ..., n_eff]``. The caller is responsible for
    deciding which output matrix the column attaches to.
    """
    if code in ("co", "ci"):
        return np.ones(n_eff, dtype=dtype)
    if code in ("lo", "li"):
        return np.arange(1, n_eff + 1, dtype=dtype)
    # Unreachable: validated against VALID_DETERMINISTIC in the caller.
    raise AssertionError(f"unhandled deterministic code: {code!r}")  # pragma: no cover


def _looks_like_compound_code(code: str) -> bool:
    """True if ``code`` parses as two valid single codes glued together.

    Used purely to give a more helpful error message — the actual rejection
    happens via the membership check against :data:`VALID_DETERMINISTIC`.
    """
    if len(code) != 4:
        return False
    return code[:2] in VALID_DETERMINISTIC and code[2:] in VALID_DETERMINISTIC
