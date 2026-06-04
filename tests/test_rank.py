"""Tests for select_coint_rank and CointRankResult.

These tests require statsmodels, which is a dev dependency.  The test suite
skips gracefully if it is not installed (though CI should always have it).
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_vecm import CointRankResult, select_coint_rank

pytest.importorskip("statsmodels", reason="statsmodels not installed")

# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------


def _cointegrated_bivariate(n_obs: int = 300, seed: int = 42) -> np.ndarray:
    """Bivariate cointegrated series with rank 1.

    Built from one common I(1) trend so the Johansen test reliably returns
    rank 1: y1 ~ trend, y2 ~ 0.5 * trend.  The cointegrating vector is
    [1, -2].  Strong signal (large noise relative to no EC dynamics) ensures
    the test-statistic margin over the 5% critical value is comfortable.
    """
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(scale=1.0, size=n_obs))
    y = np.column_stack(
        [
            trend + rng.normal(scale=0.5, size=n_obs),
            0.5 * trend + rng.normal(scale=0.5, size=n_obs),
        ]
    )
    return y


def _cointegrated_trivariate_r2(n_obs: int = 300, seed: int = 42) -> np.ndarray:
    """Trivariate series with cointegration rank 2.

    Built from one common I(1) trend (rank = K - n_trends = 3 - 1 = 2):
    y1 ~ trend, y2 ~ 0.5 * trend, y3 ~ 0.7 * trend.
    """
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(scale=1.0, size=n_obs))
    y = np.column_stack(
        [
            1.0 * trend + rng.normal(scale=0.3, size=n_obs),
            0.5 * trend + rng.normal(scale=0.3, size=n_obs),
            0.7 * trend + rng.normal(scale=0.3, size=n_obs),
        ]
    )
    return y


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_coint_rank_result(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        assert isinstance(result, CointRankResult)

    def test_rank_is_int(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        assert isinstance(result.rank, int)

    def test_rank_in_valid_range(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        n_vars = data.shape[1]
        assert 0 <= result.rank <= n_vars

    def test_test_stats_shape(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        n_vars = data.shape[1]
        assert result.test_stats.shape == (n_vars,)

    def test_crit_vals_shape(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        n_vars = data.shape[1]
        assert result.crit_vals.shape == (n_vars,)

    def test_test_stats_are_finite(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        assert np.all(np.isfinite(result.test_stats))

    def test_crit_vals_are_positive(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        assert np.all(result.crit_vals > 0)

    def test_k_ar_diff_stored(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data, k_ar_diff=2)
        assert result.k_ar_diff == 2

    def test_det_order_stored(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data, det_order=1)
        assert result.det_order == 1


# ---------------------------------------------------------------------------
# Rank detection
# ---------------------------------------------------------------------------


class TestRankDetection:
    def test_bivariate_rank1_detected(self):
        """Strong cointegration signal at T=200 should resolve to rank 1."""
        data = _cointegrated_bivariate()
        result = select_coint_rank(data, k_ar_diff=1)
        assert result.rank == 1

    def test_trivariate_rank2_detected(self):
        """Trivariate r=2 DGP should resolve to rank 2 at T=200."""
        data = _cointegrated_trivariate_r2()
        result = select_coint_rank(data, k_ar_diff=1)
        assert result.rank == 2


# ---------------------------------------------------------------------------
# Deterministic-term and k_ar_diff variants
# ---------------------------------------------------------------------------


class TestVariants:
    @pytest.mark.parametrize("det_order", [-1, 0, 1])
    def test_det_order_variants(self, det_order):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data, det_order=det_order)
        assert isinstance(result, CointRankResult)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_k_ar_diff_variants(self, k):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data, k_ar_diff=k)
        assert isinstance(result, CointRankResult)


# ---------------------------------------------------------------------------
# Pandas DataFrame input
# ---------------------------------------------------------------------------


class TestDataFrameInput:
    def test_accepts_dataframe(self):
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(_cointegrated_bivariate(), columns=["y1", "y2"])
        result = select_coint_rank(data)
        assert isinstance(result, CointRankResult)

    def test_variable_names_extracted(self):
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(_cointegrated_bivariate(), columns=["sales", "awareness"])
        result = select_coint_rank(data)
        assert result.variable_names == ["sales", "awareness"]

    def test_numpy_array_variable_names_none(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        assert result.variable_names is None


# ---------------------------------------------------------------------------
# String representations
# ---------------------------------------------------------------------------


class TestStringReprs:
    def test_repr_contains_rank(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        assert f"rank={result.rank}" in repr(result)

    def test_str_contains_header(self):
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        summary = str(result)
        assert "Johansen" in summary
        assert "Recommended rank" in summary

    def test_str_contains_all_rows(self):
        """One row per null hypothesis H0: r <= i."""
        data = _cointegrated_bivariate()
        result = select_coint_rank(data)
        summary = str(result)
        n_vars = data.shape[1]
        for i in range(n_vars):
            assert f"r <= {i}" in summary


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_det_order_raises(self):
        data = _cointegrated_bivariate()
        with pytest.raises(ValueError, match="det_order must be"):
            select_coint_rank(data, det_order=2)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            select_coint_rank(np.ones(50))

    def test_single_variable_raises(self):
        with pytest.raises(ValueError):
            select_coint_rank(np.ones((50, 1)))
