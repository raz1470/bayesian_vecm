"""Tests for ``bayesian_vecm._design.cointegration_design``.

The helper turns a raw ``(T, K)`` series into the three matrices a VECM
regression actually consumes::

    delta_y  : Δy_t           — LHS                          shape (T_eff, K)
    delta_x  : [Δy_{t-1}, …, Δy_{t-k}] stacked, lag-major    shape (T_eff, K * k)
    y_lag1   : y_{t-1}        — feeds β' y_{t-1}             shape (T_eff, K)

with ``T_eff = T - k - 1``.

The tests are written TDD-style — they fully specify the expected output of a
hand-built tiny example before any implementation existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_vecm._design import CointegrationDesign, cointegration_design


# ---------------------------------------------------------------------------
# Minimal pandas.DataFrame stand-in — same trick as test_data.py. We don't
# pull pandas in as a test dependency; we just need *something* with a
# ``.to_numpy()`` method for the duck-typed input path.
# ---------------------------------------------------------------------------
class _FakeFrame:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def to_numpy(self, dtype: object = None) -> np.ndarray:
        return self._array if dtype is None else self._array.astype(dtype)


# ---------------------------------------------------------------------------
# Hand-built tiny example: T=5, K=2, k_ar_diff=1.
#
# y_0 = [1, 10]
# y_1 = [2, 20]      Δy_1 = [1, 10]
# y_2 = [4, 40]      Δy_2 = [2, 20]
# y_3 = [7, 70]      Δy_3 = [3, 30]
# y_4 = [11, 110]    Δy_4 = [4, 40]
#
# T_eff = T - k - 1 = 3. Output rows r ∈ {0, 1, 2} correspond to the
# regression equation evaluated at t ∈ {2, 3, 4} (0-based), i.e. the
# three rows where Δy_t, y_{t-1} *and* Δy_{t-1} are all defined.
# ---------------------------------------------------------------------------
TINY_Y = np.array(
    [
        [1.0, 10.0],
        [2.0, 20.0],
        [4.0, 40.0],
        [7.0, 70.0],
        [11.0, 110.0],
    ]
)

EXPECTED_DELTA_Y = np.array(
    [
        [2.0, 20.0],  # Δy_2
        [3.0, 30.0],  # Δy_3
        [4.0, 40.0],  # Δy_4
    ]
)
EXPECTED_DELTA_X = np.array(
    [
        [1.0, 10.0],  # Δy_1
        [2.0, 20.0],  # Δy_2
        [3.0, 30.0],  # Δy_3
    ]
)
EXPECTED_Y_LAG1 = np.array(
    [
        [2.0, 20.0],  # y_1
        [4.0, 40.0],  # y_2
        [7.0, 70.0],  # y_3
    ]
)


# ---------------------------------------------------------------------------
# Row-by-row spec: this is the test that *defines* the function.
# ---------------------------------------------------------------------------
class TestHandBuiltExample:
    def test_returns_a_cointegration_design_named_tuple(self) -> None:
        result = cointegration_design(TINY_Y, k_ar_diff=1)
        assert isinstance(result, CointegrationDesign)
        # Unpacking ordering: delta_y, delta_x, y_lag1.
        delta_y, delta_x, y_lag1 = result
        assert delta_y is result.delta_y
        assert delta_x is result.delta_x
        assert y_lag1 is result.y_lag1

    def test_delta_y_row_by_row(self) -> None:
        result = cointegration_design(TINY_Y, k_ar_diff=1)
        np.testing.assert_array_equal(result.delta_y, EXPECTED_DELTA_Y)

    def test_delta_x_row_by_row(self) -> None:
        result = cointegration_design(TINY_Y, k_ar_diff=1)
        np.testing.assert_array_equal(result.delta_x, EXPECTED_DELTA_X)

    def test_y_lag1_row_by_row(self) -> None:
        result = cointegration_design(TINY_Y, k_ar_diff=1)
        np.testing.assert_array_equal(result.y_lag1, EXPECTED_Y_LAG1)


# ---------------------------------------------------------------------------
# Shape contract for arbitrary T, K, k_ar_diff.
# ---------------------------------------------------------------------------
class TestShapes:
    @pytest.mark.parametrize("k_ar_diff", [0, 1, 2, 3])
    def test_shapes_match_spec(self, k_ar_diff: int) -> None:
        n_obs, n_vars = 10, 3
        data = np.arange(n_obs * n_vars, dtype=float).reshape(n_obs, n_vars)

        result = cointegration_design(data, k_ar_diff=k_ar_diff)

        n_eff = n_obs - k_ar_diff - 1
        assert result.delta_y.shape == (n_eff, n_vars)
        assert result.delta_x.shape == (n_eff, n_vars * k_ar_diff)
        assert result.y_lag1.shape == (n_eff, n_vars)


# ---------------------------------------------------------------------------
# k_ar_diff = 0 edge case: no Γ block, but delta_y and y_lag1 still well-defined.
# This is the "VAR(1) in levels with cointegration term" special case.
# ---------------------------------------------------------------------------
class TestZeroLagDifferences:
    def test_delta_x_has_zero_columns(self) -> None:
        result = cointegration_design(TINY_Y, k_ar_diff=0)
        assert result.delta_x.shape == (TINY_Y.shape[0] - 1, 0)

    def test_delta_y_is_just_first_differences(self) -> None:
        result = cointegration_design(TINY_Y, k_ar_diff=0)
        # With k=0, T_eff = T-1, so every Δy_t for t=1..T-1 is included.
        expected = np.diff(TINY_Y, axis=0)
        np.testing.assert_array_equal(result.delta_y, expected)

    def test_y_lag1_is_all_but_the_last_row(self) -> None:
        result = cointegration_design(TINY_Y, k_ar_diff=0)
        np.testing.assert_array_equal(result.y_lag1, TINY_Y[:-1])


# ---------------------------------------------------------------------------
# Input validation.
# ---------------------------------------------------------------------------
class TestValidation:
    def test_rejects_negative_k_ar_diff(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            cointegration_design(TINY_Y, k_ar_diff=-1)

    def test_rejects_k_ar_diff_equal_to_t_minus_1(self) -> None:
        # T=5, so k=4 leaves T_eff=0 (no usable rows). Must reject.
        with pytest.raises(ValueError, match="fewer than"):
            cointegration_design(TINY_Y, k_ar_diff=4)

    def test_rejects_k_ar_diff_greater_than_t_minus_1(self) -> None:
        with pytest.raises(ValueError, match="fewer than"):
            cointegration_design(TINY_Y, k_ar_diff=99)

    def test_propagates_validate_endog_errors(self) -> None:
        # 1-D input is caught inside validate_endog.
        with pytest.raises(ValueError, match="2-dimensional"):
            cointegration_design(np.arange(10, dtype=float), k_ar_diff=1)

    def test_propagates_nan_rejection(self) -> None:
        bad = TINY_Y.copy()
        bad[2, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            cointegration_design(bad, k_ar_diff=1)


# ---------------------------------------------------------------------------
# DataFrame-like input.
# ---------------------------------------------------------------------------
class TestDataFrameLikeInput:
    def test_accepts_object_with_to_numpy(self) -> None:
        df = _FakeFrame(TINY_Y)
        result = cointegration_design(df, k_ar_diff=1)
        np.testing.assert_array_equal(result.delta_y, EXPECTED_DELTA_Y)
        np.testing.assert_array_equal(result.delta_x, EXPECTED_DELTA_X)
        np.testing.assert_array_equal(result.y_lag1, EXPECTED_Y_LAG1)


# ---------------------------------------------------------------------------
# Alignment: row r of every output must reference the same underlying t.
#
# We pick a series whose first column is just the time index, so we can
# read off the time the row corresponds to directly from the value.
# ---------------------------------------------------------------------------
class TestAlignment:
    @pytest.mark.parametrize("k_ar_diff", [1, 2, 3])
    def test_rows_correspond_to_the_same_time_index(self, k_ar_diff: int) -> None:
        n_obs = 8
        # Column 0 is the time index t (0-based). Column 1 is the same * 10.
        y = np.column_stack([np.arange(n_obs, dtype=float), np.arange(n_obs, dtype=float) * 10.0])

        result = cointegration_design(y, k_ar_diff=k_ar_diff)

        # For row r, the regression is evaluated at time t = k_ar_diff + 1 + r.
        # So:
        #   delta_y[r] should equal y[t] - y[t-1]   (== [1, 10] for this series).
        #   y_lag1[r]  should equal y[t-1].
        #   delta_x[r] should have, in its first K columns, y[t-1] - y[t-2] == [1, 10].
        n_eff = n_obs - k_ar_diff - 1
        for r in range(n_eff):
            t = k_ar_diff + 1 + r
            np.testing.assert_array_equal(result.delta_y[r], y[t] - y[t - 1])
            np.testing.assert_array_equal(result.y_lag1[r], y[t - 1])
            # First lag of delta_x is Δy_{t-1} = y[t-1] - y[t-2].
            np.testing.assert_array_equal(result.delta_x[r, :2], y[t - 1] - y[t - 2])


# ---------------------------------------------------------------------------
# k_ar_diff = 2: explicit check that the lag-major / most-recent-first
# ordering carries over correctly from lag_matrix into the design matrix.
# ---------------------------------------------------------------------------
class TestLagMajorOrdering:
    def test_two_lags_orders_most_recent_first(self) -> None:
        result = cointegration_design(TINY_Y, k_ar_diff=2)

        # T=5, k=2 → T_eff = 2. Rows r ∈ {0, 1} ↔ t ∈ {3, 4}.
        # delta_x[r] = [Δy_{t-1}, Δy_{t-2}], each a K=2 vector, concatenated.
        # For r=0 (t=3): [Δy_2, Δy_1] = [2, 20, 1, 10].
        # For r=1 (t=4): [Δy_3, Δy_2] = [3, 30, 2, 20].
        expected_delta_x = np.array(
            [
                [2.0, 20.0, 1.0, 10.0],
                [3.0, 30.0, 2.0, 20.0],
            ]
        )
        np.testing.assert_array_equal(result.delta_x, expected_delta_x)
