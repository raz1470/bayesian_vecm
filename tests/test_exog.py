"""Tests for exogenous regressor support (feat/exog).

Coverage
--------
* Design-layer: exog/exog_coint alignment, shape, validation errors.
* PyMC graph: B RV present/absent, correct shape.
* Model integration: fit + B in posterior, fittedvalues includes exog term,
  forecast with exog_future, error on missing exog_future.
"""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_vecm._design import CointegrationDesign, cointegration_design
from bayesian_vecm._pymc import build_pymc_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)

T = 30
K = 2
M = 1  # exog columns


def _make_endog(t: int = T, k: int = K) -> np.ndarray:
    y = np.cumsum(RNG.standard_normal((t, k)), axis=0)
    return y.astype(np.float64)


def _make_exog(t: int = T, m: int = M) -> np.ndarray:
    return RNG.standard_normal((t, m)).astype(np.float64)


# Module-scoped fitted model for integration tests (shared across tests
# that only read from idata_ to avoid re-running pm.sample repeatedly).
@pytest.fixture(scope="module")
def fitted_model_with_exog():
    from bayesian_vecm import BayesianVECM

    endog = _make_endog()
    exog = _make_exog()
    model = BayesianVECM(k_ar_diff=1, coint_rank=1, deterministic="n")
    model.fit(
        endog,
        exog=exog,
        draws=20,
        tune=20,
        chains=1,
        cores=1,
        progressbar=False,
        random_seed=0,
    )
    return model, endog, exog


@pytest.fixture(scope="module")
def fitted_model_no_exog():
    from bayesian_vecm import BayesianVECM

    endog = _make_endog()
    model = BayesianVECM(k_ar_diff=1, coint_rank=1, deterministic="n")
    model.fit(
        endog,
        draws=20,
        tune=20,
        chains=1,
        cores=1,
        progressbar=False,
        random_seed=0,
    )
    return model, endog


# ---------------------------------------------------------------------------
# 1. Design layer
# ---------------------------------------------------------------------------


class TestDesignExog:
    def test_exog_none_returns_none_field(self):
        y = _make_endog()
        d = cointegration_design(y, k_ar_diff=1)
        assert d.exog is None

    def test_exog_shape(self):
        y = _make_endog()
        x = _make_exog()
        d = cointegration_design(y, k_ar_diff=1, exog=x)
        t_eff = T - 1 - 1  # T - k_ar_diff - 1
        assert d.exog is not None
        assert d.exog.shape == (t_eff, M)

    def test_exog_k0_shape(self):
        """k_ar_diff=0 → T_eff = T - 1."""
        y = _make_endog()
        x = _make_exog()
        d = cointegration_design(y, k_ar_diff=0, exog=x)
        assert d.exog.shape == (T - 1, M)

    def test_exog_multicolumn(self):
        y = _make_endog()
        x = _make_exog(m=3)
        d = cointegration_design(y, k_ar_diff=1, exog=x)
        assert d.exog.shape == (T - 2, 3)

    def test_exog_alignment(self):
        """exog[k+1:] should match design.exog row-for-row."""
        y = _make_endog()
        x = _make_exog()
        k = 1
        d = cointegration_design(y, k_ar_diff=k, exog=x)
        expected = x[k + 1 :]
        np.testing.assert_array_equal(d.exog, expected)

    def test_exog_coint_widens_y_lag1(self):
        """exog_coint columns are appended to y_lag1."""
        y = _make_endog()
        x_c = _make_exog(m=2)
        d_base = cointegration_design(y, k_ar_diff=1)
        d_coint = cointegration_design(y, k_ar_diff=1, exog_coint=x_c)
        # y_lag1 should gain 2 extra columns
        assert d_coint.y_lag1.shape[1] == d_base.y_lag1.shape[1] + 2
        assert d_coint.exog is None  # exog_coint does NOT go into design.exog

    def test_exog_coint_not_stored_in_exog_field(self):
        y = _make_endog()
        x_c = _make_exog()
        d = cointegration_design(y, k_ar_diff=1, exog_coint=x_c)
        assert d.exog is None

    def test_exog_and_exog_coint_together(self):
        y = _make_endog()
        x = _make_exog(m=1)
        x_c = _make_exog(m=2)
        d = cointegration_design(y, k_ar_diff=1, exog=x, exog_coint=x_c)
        assert d.exog is not None
        assert d.exog.shape == (T - 2, 1)
        assert d.y_lag1.shape[1] == K + 2  # K base + 2 from exog_coint

    # --- validation errors ---

    def test_exog_wrong_rows_raises(self):
        y = _make_endog()
        x = _make_exog(t=T + 1)
        with pytest.raises(ValueError, match="same number of rows"):
            cointegration_design(y, k_ar_diff=1, exog=x)

    def test_exog_1d_raises(self):
        y = _make_endog()
        x = np.ones(T)  # 1-D
        with pytest.raises(ValueError, match="2-D"):
            cointegration_design(y, k_ar_diff=1, exog=x)

    def test_exog_coint_wrong_rows_raises(self):
        y = _make_endog()
        x_c = _make_exog(t=T - 1)
        with pytest.raises(ValueError, match="same number of rows"):
            cointegration_design(y, k_ar_diff=1, exog_coint=x_c)

    def test_cointegration_design_is_named_tuple(self):
        y = _make_endog()
        x = _make_exog()
        d = cointegration_design(y, k_ar_diff=1, exog=x)
        assert isinstance(d, CointegrationDesign)
        assert hasattr(d, "exog")


# ---------------------------------------------------------------------------
# 2. PyMC graph
# ---------------------------------------------------------------------------


class TestPyMCExog:
    def _design(self, with_exog: bool = True):
        y = _make_endog()
        x = _make_exog() if with_exog else None
        return cointegration_design(y, k_ar_diff=1, exog=x)

    def test_b_rv_absent_without_exog(self):
        design = self._design(with_exog=False)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic="n")
        assert "B" not in model.named_vars

    def test_b_rv_present_with_exog(self):
        design = self._design(with_exog=True)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic="n")
        assert "B" in model.named_vars

    def test_b_shape(self):
        design = self._design(with_exog=True)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic="n")
        b = model.named_vars["B"]
        assert b.type.shape == (K, M)

    def test_b_multicolumn_shape(self):
        y = _make_endog()
        x = _make_exog(m=3)
        design = cointegration_design(y, k_ar_diff=1, exog=x)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic="n")
        b = model.named_vars["B"]
        assert b.type.shape == (K, 3)

    def test_exog_data_node_stored(self):
        design = self._design(with_exog=True)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic="n")
        assert "exog" in model.named_vars

    def test_b_prior_override(self):
        design = self._design(with_exog=True)
        model = build_pymc_model(
            design,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"B": {"dist": "Laplace", "mu": 0.0, "b": 1.0}},
        )
        assert "B" in model.named_vars


# ---------------------------------------------------------------------------
# 3. Model integration — fit
# ---------------------------------------------------------------------------


class TestModelFitExog:
    def test_b_in_posterior(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        assert "B" in model.idata_.posterior

    def test_b_posterior_shape(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        b = model.idata_.posterior["B"]
        # dims: (chain, draw, B_dim_0, B_dim_1) = (1, 20, K, M)
        assert b.shape[-2] == K
        assert b.shape[-1] == M

    def test_exog_stored_in_constant_data(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        assert "exog" in model.idata_.constant_data

    def test_exog_stored_shape(self, fitted_model_with_exog):
        model, endog, _ = fitted_model_with_exog
        t_eff = len(endog) - 1 - 1  # T - k_ar_diff - 1
        exog_cd = model.idata_.constant_data["exog"]
        assert exog_cd.shape == (t_eff, M)

    def test_exog_attr_set(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        assert model.exog_ is not None
        assert model.exog_.shape[1] == M

    def test_no_exog_attr_is_none(self, fitted_model_no_exog):
        model, _ = fitted_model_no_exog
        assert model.exog_ is None

    def test_b_absent_without_exog(self, fitted_model_no_exog):
        model, _ = fitted_model_no_exog
        assert "B" not in model.idata_.posterior

    def test_summary_includes_b(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        summary = model.summary()
        assert any("B" in str(idx) for idx in summary.index)

    def test_summary_excludes_b_without_exog(self, fitted_model_no_exog):
        model, _ = fitted_model_no_exog
        summary = model.summary()
        assert not any("B[" in str(idx) for idx in summary.index)


# ---------------------------------------------------------------------------
# 4. fittedvalues includes exog term
# ---------------------------------------------------------------------------


class TestFittedValuesExog:
    def test_fittedvalues_shape_unchanged(self, fitted_model_with_exog):
        model, endog, _ = fitted_model_with_exog
        fv = model.fittedvalues
        t_eff = len(endog) - 1 - 1
        assert fv.shape == (1, 20, t_eff, K)

    def test_fittedvalues_differ_with_exog(self, fitted_model_with_exog, fitted_model_no_exog):
        """fittedvalues from a model with exog should differ from one without."""
        model_x, endog, _ = fitted_model_with_exog
        model_0, _ = fitted_model_no_exog
        # Both models fit the same endog; with exog the fit will differ.
        fv_x = model_x.fittedvalues.mean(("chain", "draw")).values
        fv_0 = model_0.fittedvalues.mean(("chain", "draw")).values
        # They won't be identical because the models are different.
        assert fv_x.shape == fv_0.shape
        # Just check they're not bitwise identical (different models).
        assert not np.allclose(fv_x, fv_0)


# ---------------------------------------------------------------------------
# 5. Forecasting with exog_future
# ---------------------------------------------------------------------------


class TestForecastExog:
    STEPS = 5

    def test_forecast_with_exog_future(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        x_fut = np.ones((self.STEPS, M))
        result = model.sample_posterior_predictive(self.STEPS, exog_future=x_fut, random_seed=0)
        y_fc = result.posterior_predictive["y"]
        assert y_fc.shape[-2] == self.STEPS
        assert y_fc.shape[-1] == K

    def test_forecast_missing_exog_future_raises(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        with pytest.raises(ValueError, match="exog_future must be provided"):
            model.sample_posterior_predictive(self.STEPS)

    def test_forecast_exog_future_wrong_steps_raises(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        x_fut = np.ones((self.STEPS + 1, M))  # wrong number of rows
        with pytest.raises(ValueError, match=f"{self.STEPS} rows"):
            model.sample_posterior_predictive(self.STEPS, exog_future=x_fut)

    def test_forecast_exog_future_wrong_cols_raises(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        x_fut = np.ones((self.STEPS, M + 1))  # wrong number of columns
        with pytest.raises(ValueError, match="column"):
            model.sample_posterior_predictive(self.STEPS, exog_future=x_fut)

    def test_forecast_no_exog_ignores_exog_future(self, fitted_model_no_exog):
        """exog_future is silently ignored when the model has no exog."""
        model, _ = fitted_model_no_exog
        x_fut = np.ones((self.STEPS, M))
        # Should not raise — exog_future is irrelevant when model has no exog.
        result = model.sample_posterior_predictive(self.STEPS, exog_future=x_fut, random_seed=0)
        assert result.posterior_predictive["y"].shape[-2] == self.STEPS

    def test_forecast_reproducible_with_seed(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        x_fut = RNG.standard_normal((self.STEPS, M))
        r1 = model.sample_posterior_predictive(self.STEPS, exog_future=x_fut, random_seed=7)
        r2 = model.sample_posterior_predictive(self.STEPS, exog_future=x_fut, random_seed=7)
        np.testing.assert_array_equal(
            r1.posterior_predictive["y"].values,
            r2.posterior_predictive["y"].values,
        )

    def test_forecast_finite_values(self, fitted_model_with_exog):
        model, _, _ = fitted_model_with_exog
        x_fut = np.zeros((self.STEPS, M))
        result = model.sample_posterior_predictive(self.STEPS, exog_future=x_fut, random_seed=0)
        assert np.isfinite(result.posterior_predictive["y"].values).all()
