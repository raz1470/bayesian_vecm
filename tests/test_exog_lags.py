"""Tests for lagged exogenous drivers (``exog_lags``).

Coverage
--------
* Design: lag-major stacking, row alignment, trimming when the lags reach
  further back than the lagged differences, validation.
* PyMC graph: shape of ``B`` and the scaled default prior.
* Exog IRF on the brand DGP with a lagged effect, no sampling.
* Simulator: ``exog_lag_share``.
* Model integration: fit, exog IRF, evidence table, forecast, fitted values.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._design import cointegration_design
from bayesian_vecm._pymc import build_pymc_model

T = 40
K = 2
M = 2


def _endog(seed: int = 0, t: int = T) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal((t, K)), axis=0)


def _exog(t: int = T, m: int = M) -> np.ndarray:
    # Column j holds 100 * j + row index, so every value names its own row.
    return np.arange(t, dtype=np.float64)[:, None] + 100.0 * np.arange(m)[None, :]


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------


class TestDesign:
    def test_zero_lags_matches_same_period_design(self):
        y, x = _endog(), _exog()
        old = cointegration_design(y, 1, exog=x)
        new = cointegration_design(y, 1, exog=x, exog_lags=0)
        np.testing.assert_array_equal(old.exog, new.exog)
        np.testing.assert_array_equal(old.delta_y, new.delta_y)

    def test_columns_are_lag_major(self):
        d = cointegration_design(_endog(), 1, exog=_exog(), exog_lags=1)
        assert d.exog.shape == (T - 2, 2 * M)
        # First usable row is t = 2: [x_2, x_1] for each column.
        np.testing.assert_array_equal(d.exog[0], [2.0, 102.0, 1.0, 101.0])
        np.testing.assert_array_equal(d.exog[-1], [T - 1, 100 + T - 1, T - 2, 100 + T - 2])

    @pytest.mark.parametrize(("k", "lags"), [(0, 1), (1, 2), (2, 1)])
    def test_no_trimming_when_differences_reach_as_far(self, k, lags):
        d = cointegration_design(_endog(), k, exog=_exog(), exog_lags=lags)
        assert d.delta_y.shape[0] == T - k - 1

    def test_trims_rows_when_lags_reach_further(self):
        y, x = _endog(), _exog()
        d = cointegration_design(y, 1, exog=x, exog_lags=4)
        assert d.delta_y.shape[0] == T - 4
        # Row 0 is t = 4: y_lag1 is y_3 and the exog lags run x_4 .. x_0.
        np.testing.assert_array_equal(d.y_lag1[0], y[3])
        np.testing.assert_array_equal(d.exog[0, ::M], [4.0, 3.0, 2.0, 1.0, 0.0])
        np.testing.assert_allclose(d.delta_y[0], y[4] - y[3])

    def test_trimming_keeps_exog_coint_aligned(self):
        y, x = _endog(), _exog()
        xc = np.arange(T, dtype=np.float64)[:, None]
        d = cointegration_design(y, 0, exog=x, exog_coint=xc, exog_lags=3)
        # Row 0 is t = 3, so y_lag1 holds y_2 and exog_coint_2.
        np.testing.assert_array_equal(d.y_lag1[0], [*y[2], 3.0])

    def test_ignored_without_exog(self):
        d = cointegration_design(_endog(), 1, exog_lags=5)
        assert d.exog is None
        assert d.delta_y.shape[0] == T - 2

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="exog_lags must be non-negative"):
            cointegration_design(_endog(), 1, exog=_exog(), exog_lags=-1)

    def test_no_usable_rows_raises(self):
        with pytest.raises(ValueError, match="exog_lags leaves no usable rows"):
            cointegration_design(_endog(t=5), 0, exog=_exog(t=5), exog_lags=5)


# ---------------------------------------------------------------------------
# PyMC graph
# ---------------------------------------------------------------------------


class TestGraph:
    def _model(self, lags: int, priors=None) -> pm.Model:
        d = cointegration_design(_endog(), 1, exog=_exog(), exog_lags=lags)
        return build_pymc_model(
            d, k_ar_diff=1, coint_rank=1, deterministic="n", priors=priors, exog_lags=lags
        )

    def test_b_has_one_block_per_lag(self):
        assert self._model(2).initial_point()["B"].shape == (K, 3 * M)

    @pytest.mark.parametrize("lags", [0, 1, 3])
    def test_default_prior_keeps_the_summed_effect_at_unit_scale(self, lags):
        draws = pm.draw(self._model(lags)["B"], draws=4000, random_seed=1)
        assert draws.std() == pytest.approx(1.0 / np.sqrt(lags + 1), rel=0.05)

    def test_user_prior_is_not_rescaled(self):
        priors = {"B": {"dist": "Normal", "mu": 0.0, "sigma": 2.0}}
        draws = pm.draw(self._model(1, priors)["B"], draws=4000, random_seed=1)
        assert draws.std() == pytest.approx(2.0, rel=0.05)


# ---------------------------------------------------------------------------
# Exog IRF on the brand DGP with half the effect a week late, no sampling
# ---------------------------------------------------------------------------


def _lagged_truth_posterior(*, lagged: bool = True) -> xr.DataTree:
    """One draw at the generating parameters, half the B effect a week late."""
    truth = simulate_brand_data(exog_lag_share=0.5).truth

    def one(value: np.ndarray) -> np.ndarray:
        return value[np.newaxis, np.newaxis]

    b = np.concatenate([truth.b, truth.b_lag], axis=1) if lagged else truth.b + truth.b_lag
    data = {
        "alpha": (("chain", "draw", "variable", "relation"), one(truth.alpha)),
        "beta": (("chain", "draw", "variable", "relation"), one(truth.beta)),
        "Gamma": (("chain", "draw", "variable", "regressor"), one(truth.gamma)),
        "Sigma": (("chain", "draw", "variable", "variable_bis"), one(truth.sigma)),
        "B": (("chain", "draw", "variable", "exog"), one(b)),
    }
    coords = {"chain": [0], "draw": [0]}
    return xr.DataTree.from_dict({"posterior": xr.Dataset(data, coords=coords)})


class TestExogIrf:
    def test_impact_is_the_same_period_block(self):
        from bayesian_vecm._irf import compute_exog_irf

        truth = simulate_brand_data(exog_lag_share=0.5).truth
        result = compute_exog_irf(_lagged_truth_posterior(), 1, 5, exog_lags=1)
        assert result.sizes["exog_variable"] == truth.b.shape[1]
        np.testing.assert_allclose(result.values[0, 0, 0], truth.b)

    def test_second_period_adds_the_lag_block(self):
        from bayesian_vecm._irf import compute_exog_irf, ma_coefficients

        truth = simulate_brand_data(exog_lag_share=0.5).truth
        posterior = _lagged_truth_posterior()
        phi, _ = ma_coefficients(posterior.posterior, 1, 2)
        result = compute_exog_irf(posterior, 1, 5, exog_lags=1)
        expected = phi[0, 1] @ truth.b + truth.b_lag
        np.testing.assert_allclose(result.values[0, 0, 1], expected, atol=1e-12)

    def test_settles_at_the_same_long_run_effect_as_no_lag(self):
        from bayesian_vecm._irf import compute_exog_irf

        lagged = compute_exog_irf(_lagged_truth_posterior(), 1, 400, exog_lags=1)
        no_lag = compute_exog_irf(_lagged_truth_posterior(lagged=False), 1, 400)
        np.testing.assert_allclose(lagged.values[0, 0, -1], no_lag.values[0, 0, -1], atol=1e-10)

    def test_steps_shorter_than_lags(self):
        from bayesian_vecm._irf import compute_exog_irf

        result = compute_exog_irf(_lagged_truth_posterior(), 1, 1, exog_lags=1)
        assert result.sizes["horizon"] == 2

    def test_columns_that_do_not_split_raise(self):
        from bayesian_vecm._irf import compute_exog_irf

        with pytest.raises(ValueError, match="does not split"):
            compute_exog_irf(_lagged_truth_posterior(), 1, 5, exog_lags=2)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class TestSimulator:
    def test_default_has_no_lag(self):
        truth = simulate_brand_data().truth
        assert not truth.b_lag.any()

    def test_share_splits_b_and_keeps_the_total(self):
        base = simulate_brand_data().truth.b
        truth = simulate_brand_data(exog_lag_share=0.3).truth
        np.testing.assert_allclose(truth.b + truth.b_lag, base)
        np.testing.assert_allclose(truth.b_lag, 0.3 * base)

    def test_share_changes_the_data(self):
        a = simulate_brand_data(seed=2).endog
        b = simulate_brand_data(seed=2, exog_lag_share=0.5).endog
        assert not np.allclose(a, b)

    def test_data_follow_the_lagged_equation(self):
        data = simulate_brand_data(exog_lag_share=0.5)
        truth = data.truth
        y, x = data.endog.to_numpy(), data.exog.to_numpy()
        dy = np.diff(y, axis=0)
        resid = (
            dy[1:]
            - truth.const
            - y[1:-1] @ (truth.alpha @ truth.beta.T).T
            - dy[:-1] @ truth.gamma.T
            - x[2:] @ truth.b.T
            - x[1:-1] @ truth.b_lag.T
        )
        assert np.abs(resid.mean(axis=0)).max() < 0.002
        np.testing.assert_allclose(np.cov(resid.T), truth.sigma, atol=4e-5)

    @pytest.mark.parametrize("share", [-0.1, 1.1])
    def test_share_out_of_range_raises(self, share):
        with pytest.raises(ValueError, match="exog_lag_share"):
            simulate_brand_data(exog_lag_share=share)


# ---------------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    import pandas as pd

    endog = pd.DataFrame(_endog(), columns=["a", "b"])
    exog = pd.DataFrame(np.random.default_rng(3).standard_normal((T, M)), columns=["x", "z"])
    model = BayesianVECM(k_ar_diff=1, coint_rank=1, deterministic="n", exog_lags=1)
    model.fit(
        endog, exog=exog, draws=20, tune=20, chains=1, cores=1, progressbar=False, random_seed=0
    )
    return model, endog, exog


class TestModel:
    def test_negative_raises(self):
        with pytest.raises(ValueError, match="exog_lags must be non-negative"):
            BayesianVECM(exog_lags=-1)

    def test_default_is_zero(self):
        assert BayesianVECM().exog_lags == 0

    def test_b_posterior_shape(self, fitted):
        model, _, _ = fitted
        assert model.idata_.posterior["B"].shape[2:] == (K, 2 * M)

    def test_exog_irf_has_one_column_per_driver(self, fitted):
        model, _, _ = fitted
        result = model.exog_irf(4)
        assert list(result.coords["exog_variable"].values) == ["x", "z"]
        b = model.idata_.posterior["B"].values
        np.testing.assert_allclose(result.values[:, :, 0], b[..., :M])

    def test_evidence_table_sums_over_lags(self, fitted):
        model, _, _ = fitted
        table = model.granger_causality().exog
        assert len(table) == M * K
        b = model.idata_.posterior["B"].values
        row = table[(table.cause == "z") & (table.effect == "a")].iloc[0]
        assert row["mean"] == pytest.approx((b[..., 0, 1] + b[..., 0, 3]).mean())

    def test_fittedvalues_length(self, fitted):
        model, _, _ = fitted
        assert model.fittedvalues.sizes["time"] == T - 2

    def test_forecast_uses_the_last_fit_rows_for_the_lags(self, fitted):
        from bayesian_vecm._forecast import forecast_posterior

        model, _, exog = fitted
        future = np.random.default_rng(4).standard_normal((3, M))
        got = model.sample_posterior_predictive(3, exog_future=future, random_seed=5)
        stacked = np.column_stack([future, np.vstack([exog.to_numpy()[-1:], future[:-1]])])
        expected = forecast_posterior(
            idata=model.idata_,
            endog=model.endog_,
            k_ar_diff=1,
            steps=3,
            variable_names=model.variable_names_,
            exog_future=stacked,
            rng=np.random.default_rng(5),
            deterministic="n",
        )
        np.testing.assert_allclose(
            got.posterior_predictive["y"].values, expected.posterior_predictive["y"].values
        )

    def test_forecast_checks_driver_count_not_column_count(self, fitted):
        model, _, _ = fitted
        with pytest.raises(ValueError, match="fitted with 2 exog column"):
            model.sample_posterior_predictive(3, exog_future=np.zeros((3, 2 * M)))
