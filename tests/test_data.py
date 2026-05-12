"""Tests for the data utilities in ``bayesian_vecm._data``.

These cover the three preprocessing primitives every VAR/VECM pipeline needs:

* :func:`validate_endog` — coerce user input to a clean ``(T, K)`` float array.
* :func:`difference`     — apply ``(1 - L)^d`` to a multivariate series.
* :func:`lag_matrix`     — stack lagged observations into a design matrix.

The tests are intentionally numpy-only; we duck-type the pandas path with a tiny
stand-in object so the suite has no pandas dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_vecm._data import difference, lag_matrix, validate_endog


# ---------------------------------------------------------------------------
# A minimal stand-in for ``pandas.DataFrame``.
# ``validate_endog`` only needs ``.to_numpy()``, so we duck-type rather than
# pulling pandas in as a test dependency.
# ---------------------------------------------------------------------------
class _FakeFrame:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def to_numpy(self, dtype: object = None) -> np.ndarray:
        arr = self._array
        return arr.astype(dtype) if dtype is not None else arr


# ---------------------------------------------------------------------------
# validate_endog
# ---------------------------------------------------------------------------
class TestValidateEndog:
    def test_accepts_2d_ndarray(self) -> None:
        data = np.arange(12, dtype=float).reshape(6, 2)
        result = validate_endog(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (6, 2)
        assert result.dtype == np.float64

    def test_converts_integer_dtype_to_float(self) -> None:
        data = np.arange(12).reshape(6, 2)  # int dtype
        result = validate_endog(data)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, data.astype(float))

    def test_accepts_dataframe_like(self) -> None:
        underlying = np.arange(12, dtype=float).reshape(6, 2)
        df = _FakeFrame(underlying)
        result = validate_endog(df)
        assert result.shape == (6, 2)
        np.testing.assert_array_equal(result, underlying)

    def test_returns_independent_copy(self) -> None:
        """Mutating the returned array must not touch the caller's input."""
        data = np.arange(12, dtype=float).reshape(6, 2)
        result = validate_endog(data)
        result[0, 0] = 999.0
        assert data[0, 0] == 0.0

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2-dimensional"):
            validate_endog(np.arange(10, dtype=float))

    def test_rejects_3d_input(self) -> None:
        with pytest.raises(ValueError, match="2-dimensional"):
            validate_endog(np.zeros((4, 2, 2)))

    def test_rejects_nan(self) -> None:
        data = np.arange(12, dtype=float).reshape(6, 2)
        data[2, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_endog(data)

    def test_rejects_too_few_observations(self) -> None:
        data = np.arange(4, dtype=float).reshape(2, 2)
        with pytest.raises(ValueError, match="at least 5"):
            validate_endog(data, min_obs=5)

    def test_rejects_zero_variables(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            validate_endog(np.empty((10, 0)))

    def test_rejects_non_numeric_dtype(self) -> None:
        data = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises((ValueError, TypeError)):
            validate_endog(data)


# ---------------------------------------------------------------------------
# difference
# ---------------------------------------------------------------------------
class TestDifference:
    def test_first_difference_matches_manual(self) -> None:
        data = np.array([[1.0, 10.0], [2.0, 12.0], [4.0, 15.0], [7.0, 19.0]])
        expected = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        np.testing.assert_array_equal(difference(data, d=1), expected)

    def test_second_difference_matches_manual(self) -> None:
        data = np.array([[1.0], [2.0], [4.0], [7.0], [11.0]])
        # 1st diff: 1, 2, 3, 4 ; 2nd diff: 1, 1, 1
        expected = np.array([[1.0], [1.0], [1.0]])
        np.testing.assert_array_equal(difference(data, d=2), expected)

    def test_shape_after_d_differences(self) -> None:
        data = np.arange(20, dtype=float).reshape(10, 2)
        assert difference(data, d=1).shape == (9, 2)
        assert difference(data, d=3).shape == (7, 2)

    def test_d_zero_returns_copy(self) -> None:
        data = np.arange(12, dtype=float).reshape(6, 2)
        result = difference(data, d=0)
        np.testing.assert_array_equal(result, data)
        assert result is not data  # should be a copy, not an alias

    def test_rejects_negative_d(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            difference(np.zeros((5, 2)), d=-1)

    def test_rejects_d_geq_t(self) -> None:
        with pytest.raises(ValueError, match="fewer than"):
            difference(np.zeros((3, 2)), d=3)


# ---------------------------------------------------------------------------
# lag_matrix
# ---------------------------------------------------------------------------
class TestLagMatrix:
    def test_one_lag_shape_and_content(self) -> None:
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        result = lag_matrix(data, n_lags=1)
        # Output row t corresponds to input time t+1, holding y_{t}.
        expected = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, expected)

    def test_two_lags_orders_by_lag_then_variable(self) -> None:
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        result = lag_matrix(data, n_lags=2)
        # Row t holds [y_{t+1}, y_{t}] for t=0,1 in the output.
        # i.e. first row = [y_1, y_0] = [2, 20, 1, 10].
        expected = np.array(
            [
                [2.0, 20.0, 1.0, 10.0],
                [3.0, 30.0, 2.0, 20.0],
            ]
        )
        assert result.shape == (2, 4)
        np.testing.assert_array_equal(result, expected)

    def test_zero_lags_returns_empty_columns(self) -> None:
        data = np.arange(8, dtype=float).reshape(4, 2)
        result = lag_matrix(data, n_lags=0)
        assert result.shape == (4, 0)

    def test_rejects_negative_lags(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            lag_matrix(np.zeros((5, 2)), n_lags=-1)

    def test_rejects_lags_geq_t(self) -> None:
        with pytest.raises(ValueError, match="fewer than"):
            lag_matrix(np.zeros((3, 2)), n_lags=3)

    def test_output_is_independent_of_input(self) -> None:
        data = np.arange(8, dtype=float).reshape(4, 2)
        result = lag_matrix(data, n_lags=1)
        result[0, 0] = 999.0
        assert data[0, 0] == 0.0
