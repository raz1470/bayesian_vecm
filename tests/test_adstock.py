"""Tests for geometric adstock on exogenous drivers (``adstock``).

Coverage
--------
* PyMC graph: ``B`` is a deterministic with the usual layout, lag weights
  follow normalised geometric decay, lags sum to ``B_total``, drivers not
  adstocked keep free lags up to ``exog_lags`` and zeros beyond, priors.
* Simulator: ``exog_adstock``.
* Model: validation, name and position lookup, window and row trimming,
  summary, evidence table on the total, exog IRF, forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._design import cointegration_design
from bayesian_vecm._pymc import build_pymc_model

T = 60
K = 2
M = 3


def _endog(seed: int = 0, t: int = T) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal((t, K)), axis=0)


def _exog(seed: int = 3, t: int = T, m: int = M) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((t, m))


def _model(*, adstock=(0, 2), max_lag: int = 3, exog_lags: int = 0, priors=None) -> pm.Model:
    window = max(max_lag, exog_lags)
    d = cointegration_design(_endog(), 1, exog=_exog(), exog_lags=window)
    return build_pymc_model(
        d,
        k_ar_diff=1,
        coint_rank=1,
        deterministic="n",
        priors=priors,
        exog_lags=exog_lags,
        adstock=adstock,
        adstock_max_lag=max_lag,
    )


def _b_at(model: pm.Model, **values) -> np.ndarray:
    """Draw the deterministic ``B`` with the named free variables fixed."""
    fixed = {name: np.asarray(v, dtype=np.float64) for name, v in values.items()}
    return pm.draw(pm.do(model, fixed)["B"], random_seed=0)


# ---------------------------------------------------------------------------
# PyMC graph
# ---------------------------------------------------------------------------


class TestGraph:
    def test_b_keeps_the_lag_major_shape(self):
        model = _model()
        assert {v.name for v in model.free_RVs} >= {"adstock_alpha", "B_total", "B_free"}
        assert "B" in {v.name for v in model.deterministics}
        assert _b_at(model).shape == (K, 4 * M)

    def test_weights_are_normalised_geometric(self):
        model = _model(adstock=(0, 2), max_lag=3)
        b_total = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = _b_at(model, adstock_alpha=[0.5, 0.2], B_total=b_total).reshape(K, 4, M)
        w = 0.5 ** np.arange(4)
        np.testing.assert_allclose(b[:, :, 0], b_total[:, [0]] * w / w.sum())
        w = 0.2 ** np.arange(4)
        np.testing.assert_allclose(b[:, :, 2], b_total[:, [1]] * w / w.sum())

    def test_lags_sum_to_the_total(self):
        model = _model()
        b_total = np.array([[0.3, -0.7], [1.1, 0.05]])
        b = _b_at(model, adstock_alpha=[0.8, 0.4], B_total=b_total).reshape(K, 4, M)
        np.testing.assert_allclose(b[:, :, [0, 2]].sum(axis=1), b_total)

    def test_other_driver_is_same_week_only_by_default(self):
        b = _b_at(_model(exog_lags=0)).reshape(K, 4, M)
        assert not b[:, 1:, 1].any()

    def test_other_driver_keeps_free_lags_up_to_exog_lags(self):
        model = _model(max_lag=3, exog_lags=1)
        assert model["B_free"].type.shape == (K, 1, 2)
        free = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        b = _b_at(model, B_free=free).reshape(K, 4, M)
        np.testing.assert_allclose(b[:, :2, 1], free[:, 0, :])
        assert not b[:, 2:, 1].any()

    def test_window_longer_than_adstock_pads_zeros(self):
        model = _model(max_lag=2, exog_lags=4)
        b = _b_at(model, adstock_alpha=[0.5, 0.5]).reshape(K, 5, M)
        assert not b[:, 3:, [0, 2]].any()

    def test_no_free_block_when_every_driver_is_adstocked(self):
        model = _model(adstock=(0, 1, 2))
        assert "B_free" not in {v.name for v in model.free_RVs}

    def test_default_decay_prior_is_flat(self):
        draws = pm.draw(_model()["adstock_alpha"], draws=4000, random_seed=1)
        assert draws.mean() == pytest.approx(0.5, abs=0.02)
        assert draws.std() == pytest.approx(np.sqrt(1 / 12), abs=0.01)

    def test_decay_prior_can_be_set(self):
        priors = {"adstock_alpha": {"dist": "Beta", "alpha": 1.0, "beta": 3.0}}
        draws = pm.draw(_model(priors=priors)["adstock_alpha"], draws=4000, random_seed=1)
        assert draws.mean() == pytest.approx(0.25, abs=0.02)

    def test_total_has_unit_prior_and_b_key_sets_it(self):
        draws = pm.draw(_model()["B_total"], draws=4000, random_seed=1)
        assert draws.std() == pytest.approx(1.0, rel=0.05)
        priors = {"B": {"dist": "Normal", "mu": 0.0, "sigma": 0.1}}
        draws = pm.draw(_model(priors=priors)["B_total"], draws=4000, random_seed=1)
        assert draws.std() == pytest.approx(0.1, rel=0.05)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class TestSimulator:
    def test_default_has_no_adstock(self):
        truth = simulate_brand_data().truth
        assert not truth.adstock.any()
        np.testing.assert_allclose(truth.b_total, truth.b)

    def test_total_is_unchanged_and_same_week_share_is_scaled(self):
        base = simulate_brand_data().truth.b
        truth = simulate_brand_data(exog_adstock={"brand_spend": 0.5}).truth
        np.testing.assert_allclose(truth.b_total, base)
        np.testing.assert_allclose(truth.b[:, 0], 0.5 * base[:, 0])
        np.testing.assert_allclose(truth.b[:, 1:], base[:, 1:])
        assert truth.adstock[0] == 0.5

    def test_adstock_changes_endog_not_exog(self):
        a = simulate_brand_data(seed=2)
        b = simulate_brand_data(seed=2, exog_adstock={"pr_reach": 0.6})
        assert not np.allclose(a.endog, b.endog)
        pd.testing.assert_frame_equal(a.exog, b.exog)

    def test_data_follow_the_adstock_equation(self):
        w = 0.6
        data = simulate_brand_data(n_obs=2000, exog_adstock={"brand_spend": w})
        truth = data.truth
        y, x = data.endog.to_numpy(), data.exog.to_numpy()
        stock = x.copy()
        for t in range(1, len(x)):
            stock[t, 0] = (1 - w) * x[t, 0] + w * stock[t - 1, 0]
        dy = np.diff(y, axis=0)
        resid = (
            dy[1:]
            - truth.const
            - y[1:-1] @ (truth.alpha @ truth.beta.T).T
            - dy[:-1] @ truth.gamma.T
            - stock[2:] @ truth.b_total.T
        )
        np.testing.assert_allclose(np.cov(resid.T), truth.sigma, atol=4e-5)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown exog_adstock"):
            simulate_brand_data(exog_adstock={"tv": 0.5})

    @pytest.mark.parametrize("w", [-0.1, 1.0])
    def test_decay_out_of_range_raises(self, w):
        with pytest.raises(ValueError, match="must be in"):
            simulate_brand_data(exog_adstock={"brand_spend": w})

    def test_cannot_combine_with_lag_share(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            simulate_brand_data(exog_adstock={"brand_spend": 0.5}, exog_lag_share=0.2)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    endog = pd.DataFrame(_endog(), columns=["a", "b"])
    exog = pd.DataFrame(_exog(), columns=["x", "z", "v"])
    model = BayesianVECM(
        k_ar_diff=1, coint_rank=1, deterministic="n", adstock=["v", "x"], adstock_max_lag=3
    )
    model.fit(
        endog, exog=exog, draws=30, tune=30, chains=1, cores=1, progressbar=False, random_seed=0
    )
    return model, endog, exog


class TestModelValidation:
    def test_defaults(self):
        model = BayesianVECM()
        assert model.adstock is None
        assert model.adstock_max_lag == 8

    def test_single_name_becomes_a_list(self):
        assert BayesianVECM(adstock="x").adstock == ["x"]

    def test_max_lag_below_one_raises(self):
        with pytest.raises(ValueError, match="adstock_max_lag must be at least 1"):
            BayesianVECM(adstock_max_lag=0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one driver"):
            BayesianVECM(adstock=[])

    def test_duplicates_raise(self):
        with pytest.raises(ValueError, match="duplicate"):
            BayesianVECM(adstock=["x", "x"])

    def test_needs_exog(self):
        with pytest.raises(ValueError, match="adstock needs exog"):
            BayesianVECM(adstock=["x"]).fit(_endog(), draws=5, tune=5, chains=1)

    def test_unknown_name_raises(self):
        exog = pd.DataFrame(_exog(), columns=["x", "z", "v"])
        with pytest.raises(ValueError, match="not an exog column"):
            BayesianVECM(adstock=["tv"]).fit(_endog(), exog=exog, draws=5, tune=5, chains=1)

    def test_name_needs_columns(self):
        with pytest.raises(ValueError, match="not an exog column"):
            BayesianVECM(adstock=["x"]).fit(_endog(), exog=_exog(), draws=5, tune=5, chains=1)

    def test_position_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            BayesianVECM(adstock=[3]).fit(_endog(), exog=_exog(), draws=5, tune=5, chains=1)

    def test_name_and_position_of_same_driver_raise(self):
        exog = pd.DataFrame(_exog(), columns=["x", "z", "v"])
        with pytest.raises(ValueError, match="same driver twice"):
            BayesianVECM(adstock=["x", 0]).fit(_endog(), exog=exog, draws=5, tune=5, chains=1)


class TestModelFit:
    def test_names_map_to_positions(self, fitted):
        model, _, _ = fitted
        assert model.adstock_idx_ == (2, 0)
        assert model.exog_window_ == 3

    def test_rows_dropped_for_the_window(self, fitted):
        model, _, _ = fitted
        # max(k_ar_diff + 1, window) = 3 rows lost to lags.
        assert model.exog_.shape == (T - 3, 4 * M)
        assert model.fittedvalues.sizes["time"] == T - 3

    def test_posterior_has_total_and_decay(self, fitted):
        model, _, _ = fitted
        post = model.idata_.posterior
        assert post["B"].shape[2:] == (K, 4 * M)
        assert post["B_total"].shape[2:] == (K, 2)
        assert post["adstock_alpha"].shape[2:] == (2,)
        b = post["B"].values.reshape(*post["B"].shape[:3], 4, M)
        np.testing.assert_allclose(b[..., 2].sum(axis=-1), post["B_total"].values[..., 0])
        np.testing.assert_allclose(b[..., 0].sum(axis=-1), post["B_total"].values[..., 1])

    def test_summary_lists_total_and_decay(self, fitted):
        model, _, _ = fitted
        index = model.summary().index
        assert any(i.startswith("B_total") for i in index)
        assert any(i.startswith("adstock_alpha") for i in index)

    def test_evidence_table_reads_the_total(self, fitted):
        model, _, _ = fitted
        table = model.granger_causality().exog
        total = model.idata_.posterior["B_total"].values[..., 1, 0]  # v -> b
        row = table[(table.cause == "v") & (table.effect == "b")].iloc[0]
        assert row["mean"] == pytest.approx(total.mean())
        from bayesian_vecm._causality import evidence_against_zero

        assert row["evidence"] == pytest.approx(evidence_against_zero(total.ravel()))

    def test_exog_irf_impact_is_the_same_week_block(self, fitted):
        model, _, _ = fitted
        result = model.exog_irf(5)
        assert list(result.coords["exog_variable"].values) == ["x", "z", "v"]
        b = model.idata_.posterior["B"].values
        np.testing.assert_allclose(result.values[:, :, 0], b[..., :M])

    def test_forecast_seeds_lags_from_the_window(self, fitted):
        model, _, exog = fitted
        assert model.exog_tail_.shape == (3, M)
        np.testing.assert_allclose(model.exog_tail_, exog.to_numpy()[-3:])
        fc = model.sample_posterior_predictive(4, exog_future=np.zeros((4, M)), random_seed=1)
        assert np.isfinite(fc.posterior_predictive["y"].values).all()
