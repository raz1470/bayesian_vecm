"""Tests for logistic saturation on exogenous drivers (``saturation``).

Coverage
--------
* Transforms: logistic saturation matches the pymc-marketing formula, its
  slope, normalised adstock weights.
* PyMC graph: free and derived variables, ``B`` spreads ``B_total`` over the
  lags, ``B_saturation`` is ``B_total`` over the slope at the reference
  level, and the likelihood matches the NumPy saturated term used by fitted
  values and forecasts.
* Simulator: ``exog_saturation`` and ``spend_log_sd``.
* Model: validation, summary, ``saturation_curve``, ``exog_irf(at=...)``,
  evidence table, fitted values and forecasts.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
import xarray as xr
from scipy import stats

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._design import cointegration_design
from bayesian_vecm._pymc import build_pymc_model
from bayesian_vecm._saturation import saturation_term, zero_saturated_columns
from bayesian_vecm._transform import (
    geometric_adstock_weights,
    logistic_saturation,
    logistic_saturation_slope,
)

T = 60
K = 2
M = 3


def _endog(seed: int = 0, t: int = T) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal((t, K)), axis=0)


def _exog(seed: int = 3, t: int = T, m: int = M) -> np.ndarray:
    """Driver 0 and 2 are raw spend (positive), driver 1 a deviation."""
    x = np.random.default_rng(seed).standard_normal((t, m))
    x[:, [0, 2]] = np.exp(0.4 * x[:, [0, 2]])
    return x


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_matches_pymc_marketing_formula(self):
        u = np.linspace(0, 3, 13)
        lam = 2.7
        expected = (1 - np.exp(-lam * u)) / (1 + np.exp(-lam * u))
        np.testing.assert_allclose(logistic_saturation(u, lam), expected)

    def test_slope_is_the_derivative(self):
        u = np.linspace(0.05, 2, 9)
        lam = 4.0
        h = 1e-6
        numeric = (logistic_saturation(u + h, lam) - logistic_saturation(u - h, lam)) / (2 * h)
        np.testing.assert_allclose(logistic_saturation_slope(u, lam), numeric, rtol=1e-6)

    def test_adstock_weights_are_normalised_geometric(self):
        w = geometric_adstock_weights([0.5, 0.0], 3)
        np.testing.assert_allclose(w.sum(axis=-1), 1.0)
        np.testing.assert_allclose(w[0], np.array([1, 0.5, 0.25, 0.125]) / 1.875)
        np.testing.assert_allclose(w[1], [1, 0, 0, 0])


# ---------------------------------------------------------------------------
# PyMC graph
# ---------------------------------------------------------------------------


def _graph(*, adstock=(0,), saturation=(0, 2), max_lag: int = 3, priors=None):
    window = max_lag if adstock else 0
    x = _exog()
    d = cointegration_design(_endog(), 1, exog=x, exog_lags=window)
    scale = x[:, list(saturation)].max(axis=0)
    level = x[:, list(saturation)].mean(axis=0)
    model = build_pymc_model(
        d,
        k_ar_diff=1,
        coint_rank=1,
        deterministic="n",
        priors=priors,
        adstock=adstock,
        adstock_max_lag=max_lag,
        saturation=saturation,
        saturation_scale=scale,
        saturation_level=level,
    )
    return model, d, scale, level


def _point(model: pm.Model, seed: int = 0) -> dict:
    """Initial point with the free variables jittered away from defaults."""
    rng = np.random.default_rng(seed)
    point = model.initial_point()
    return {k: v + 0.3 * rng.standard_normal(np.shape(v)) for k, v in point.items()}


def _values(model: pm.Model, point: dict, names: list[str]) -> dict:
    """Evaluate named variables at a point given on the value variables."""
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([model[n] for n in names]),
        on_unused_input="ignore",
    )
    out = fn(*[point[v.name] for v in model.value_vars])
    return dict(zip(names, out, strict=True))


class TestGraph:
    def test_variables(self):
        model, _, _, _ = _graph()
        free = {v.name for v in model.free_RVs}
        assert {"B_total", "saturation_lam", "adstock_alpha", "B_free"} <= free
        det = {v.name for v in model.deterministics}
        assert {"B", "B_saturation", "saturation_mean", "saturation_weights"} <= det

    def test_default_lambda_prior(self):
        draws = pm.draw(_graph()[0]["saturation_lam"], draws=4000, random_seed=1)
        # Gamma(2, rate 0.3): mean 6.67, sd 4.71.
        assert draws.mean() == pytest.approx(2 / 0.3, rel=0.05)
        assert draws.std() == pytest.approx(np.sqrt(2) / 0.3, rel=0.08)

    def test_lambda_prior_can_be_set(self):
        priors = {"saturation_lam": {"dist": "Gamma", "alpha": 3.0, "beta": 1.0}}
        draws = pm.draw(_graph(priors=priors)[0]["saturation_lam"], draws=4000, random_seed=1)
        assert draws.mean() == pytest.approx(3.0, rel=0.05)

    def test_b_spreads_the_total_over_the_lags(self):
        model, _, _, _ = _graph()
        point = _point(model)
        v = _values(model, point, ["B", "B_total", "adstock_alpha", "saturation_weights"])
        b = v["B"].reshape(K, 4, M)
        # Driver 0: adstocked and saturated; driver 2: saturated only.
        expected0 = v["B_total"][:, 0, None] * geometric_adstock_weights(v["adstock_alpha"][0], 3)
        np.testing.assert_allclose(b[:, :, 0], expected0)
        np.testing.assert_allclose(b[:, 0, 2], v["B_total"][:, 1])
        np.testing.assert_allclose(b[:, 1:, 2], 0.0)
        np.testing.assert_allclose(v["saturation_weights"][1], [1, 0, 0, 0])

    def test_saturation_effect_is_total_over_slope(self):
        model, _, scale, level = _graph()
        point = _point(model)
        v = _values(model, point, ["B_total", "B_saturation", "saturation_lam"])
        slope = logistic_saturation_slope(level / scale, v["saturation_lam"]) / scale
        np.testing.assert_allclose(v["B_saturation"], v["B_total"] / slope)

    @pytest.mark.parametrize("adstock", [(0,), ()])
    def test_likelihood_matches_numpy_term(self, adstock):
        model, d, scale, _ = _graph(adstock=adstock)
        point = _point(model, seed=1)
        names = [
            "alpha",
            "beta",
            "Gamma",
            "Sigma",
            "B",
            "saturation_lam",
            "saturation_weights",
            "saturation_mean",
            "B_saturation",
        ]
        v = _values(model, point, names)
        post = xr.Dataset(
            {
                n: (("chain", "draw", *[f"{n}_{i}" for i in range(np.ndim(a))]), a[None, None])
                for n, a in v.items()
            }
        )
        const = xr.Dataset(
            {
                "saturation_idx": ("s", np.array([0, 2])),
                "saturation_scale": ("s", scale),
            }
        )
        idata = SimpleNamespace(posterior=post, constant_data=const)
        b_lin = zero_saturated_columns(v["B"], [0, 2], M)
        mu = (
            d.y_lag1 @ v["beta"] @ v["alpha"].T
            + d.delta_x @ v["Gamma"].T
            + d.exog @ b_lin.T
            + saturation_term(idata, d.exog)[0]
        )
        expected = stats.multivariate_normal(np.zeros(K), v["Sigma"]).logpdf(d.delta_y - mu).sum()
        logp = model.compile_logp(vars=model.observed_RVs, jacobian=False)
        got = float(logp(point))
        assert got == pytest.approx(expected, rel=1e-8), (got, expected)

    def test_saturation_mean_centres_the_term(self):
        model, d, scale, _ = _graph()
        point = _point(model, seed=2)
        v = _values(model, point, ["saturation_lam", "saturation_weights", "saturation_mean"])
        lagged = d.exog.reshape(len(d.exog), 4, M)[:, :, [0, 2]]
        stock = np.einsum("tlj,jl->tj", lagged, v["saturation_weights"]) / scale
        f = logistic_saturation(stock, v["saturation_lam"])
        np.testing.assert_allclose(v["saturation_mean"], f.mean(axis=0))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

LAMBDAS = {"brand_spend": 5.0, "pr_reach": 3.0}


class TestSimulator:
    def test_default_is_linear(self):
        truth = simulate_brand_data().truth
        assert not truth.saturation.any()
        np.testing.assert_allclose(truth.b_saturation, 0.0)

    def test_saturated_channels_are_raw_spend(self):
        base = simulate_brand_data(seed=4)
        data = simulate_brand_data(seed=4, exog_saturation=LAMBDAS)
        spend = data.exog[["brand_spend", "pr_reach"]]
        assert (spend > 0).all().all()
        np.testing.assert_allclose(spend.mean(), 1.0, atol=0.1)
        # Same draws: log spend is the scaled deviation.
        np.testing.assert_allclose(
            np.log(spend["brand_spend"]) + 0.045, base.exog["brand_spend"], atol=1e-12
        )
        pd.testing.assert_frame_equal(data.exog.iloc[:, 2:], base.exog.iloc[:, 2:])

    def test_spend_log_sd_widens_spend(self):
        narrow = simulate_brand_data(exog_saturation=LAMBDAS).exog["brand_spend"]
        wide = simulate_brand_data(exog_saturation=LAMBDAS, spend_log_sd=0.6).exog["brand_spend"]
        assert np.log(wide).std() == pytest.approx(2 * np.log(narrow).std())

    def test_marginal_effect_at_mean_spend_is_the_linear_b(self):
        base = simulate_brand_data().truth
        truth = simulate_brand_data(exog_saturation=LAMBDAS).truth
        for j, name in enumerate(["brand_spend", "pr_reach"]):
            np.testing.assert_allclose(truth.marginal_effect(name, 1.0), base.b[:, j])
        np.testing.assert_allclose(truth.b_total, base.b_total)

    def test_marginal_effect_is_the_slope_of_the_response(self):
        truth = simulate_brand_data(exog_saturation=LAMBDAS).truth
        levels = np.array([0.3, 1.0, 1.8])
        h = 1e-6
        numeric = (
            truth.response("brand_spend", levels + h) - truth.response("brand_spend", levels - h)
        ) / (2 * h)
        np.testing.assert_allclose(truth.marginal_effect("brand_spend", levels), numeric, atol=1e-8)

    def test_returns_diminish(self):
        truth = simulate_brand_data(exog_saturation=LAMBDAS).truth
        marg = truth.marginal_effect("brand_spend", [0.5, 1.0, 2.0])[:, 1]
        assert marg[0] > marg[1] > marg[2] > 0

    def test_linear_driver_response_is_linear(self):
        truth = simulate_brand_data(exog_saturation=LAMBDAS).truth
        np.testing.assert_allclose(
            truth.response("sponsorship_reach", 2.0), 2.0 * truth.b_total[:, 2]
        )

    def test_data_follow_the_saturation_equation(self):
        data = simulate_brand_data(
            n_obs=2000, exog_saturation=LAMBDAS, exog_adstock={"brand_spend": 0.5}
        )
        truth = data.truth
        y, x = data.endog.to_numpy(), data.exog.to_numpy()
        stock = x.copy()
        for t in range(1, len(x)):
            stock[t, 0] = 0.5 * x[t, 0] + 0.5 * stock[t - 1, 0]
        reg = x.copy()
        sat = truth.saturation > 0
        reg[:, sat] = (
            logistic_saturation(stock[:, sat] / truth.spend_scale[sat], truth.saturation[sat])
            - truth.saturation_mean[sat]
        )
        coef = truth.b_total.copy()
        coef[:, sat] = truth.b_saturation[:, sat]
        dy = np.diff(y, axis=0)
        resid = (
            dy[1:]
            - truth.const
            - y[1:-1] @ (truth.alpha @ truth.beta.T).T
            - dy[:-1] @ truth.gamma.T
            - reg[2:] @ coef.T
        )
        np.testing.assert_allclose(np.cov(resid.T), truth.sigma, atol=4e-5)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"exog_saturation": {"interest_rate": 2.0}}, "only name channels"),
            ({"exog_saturation": {"brand_spend": 0.0}}, "must be positive"),
            ({"exog_saturation": LAMBDAS, "exog_lag_share": 0.2}, "cannot be combined"),
            (
                {"exog_saturation": LAMBDAS, "exog_unit_root": ("pr_reach",)},
                "cannot be in exog_unit_root",
            ),
            ({"exog_saturation": LAMBDAS, "spend_log_sd": 0.0}, "spend_log_sd"),
        ],
    )
    def test_invalid_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            simulate_brand_data(**kwargs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    endog = pd.DataFrame(_endog(), columns=["a", "b"])
    exog = pd.DataFrame(_exog(), columns=["x", "z", "v"])
    model = BayesianVECM(
        k_ar_diff=1,
        coint_rank=1,
        deterministic="n",
        adstock=["x"],
        adstock_max_lag=3,
        saturation=["v", "x"],
    )
    model.fit(
        endog, exog=exog, draws=30, tune=30, chains=1, cores=1, progressbar=False, random_seed=0
    )
    return model, endog, exog


class TestModelValidation:
    def test_default_is_none(self):
        assert BayesianVECM().saturation is None

    def test_single_name_becomes_a_list(self):
        assert BayesianVECM(saturation="x").saturation == ["x"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one driver"):
            BayesianVECM(saturation=[])

    def test_duplicates_raise(self):
        with pytest.raises(ValueError, match="duplicate"):
            BayesianVECM(saturation=["x", "x"])

    def test_needs_exog(self):
        with pytest.raises(ValueError, match="saturation needs exog"):
            BayesianVECM(saturation=["x"]).fit(_endog(), draws=5, tune=5, chains=1)

    def test_negative_spend_raises(self):
        exog = pd.DataFrame(_exog(), columns=["x", "z", "v"])
        with pytest.raises(ValueError, match="non-negative"):
            BayesianVECM(saturation=["z"]).fit(_endog(), exog=exog, draws=5, tune=5, chains=1)

    def test_all_zero_spend_raises(self):
        x = _exog()
        x[:, 0] = 0.0
        with pytest.raises(ValueError, match="some positive spend"):
            BayesianVECM(saturation=[0]).fit(_endog(), exog=x, draws=5, tune=5, chains=1)


class TestModelFit:
    def test_attributes(self, fitted):
        model, _, exog = fitted
        assert model.saturation_idx_ == (2, 0)
        assert model.total_idx_ == (0, 2)
        np.testing.assert_allclose(model.saturation_scale_, exog[["v", "x"]].max())
        np.testing.assert_allclose(model.saturation_level_, exog[["v", "x"]].mean())

    def test_summary_lists_lambda(self, fitted):
        model, _, _ = fitted
        index = model.summary().index
        assert any(i.startswith("saturation_lam") for i in index)
        assert any(i.startswith("B_total") for i in index)

    def test_marginal_curve_at_mean_is_the_total(self, fitted):
        model, _, exog = fitted
        marg = model.saturation_curve("v", [exog["v"].mean()], marginal=True)
        total = model.idata_.posterior["B_total"].values[..., 1]  # (C, D, K)
        np.testing.assert_allclose(marg.values[:, :, 0, :], total)

    def test_curve_shape_and_coords(self, fitted):
        model, _, _ = fitted
        curve = model.saturation_curve("x", [0.0, 1.0, 2.0])
        assert curve.dims == ("chain", "draw", "level", "response_variable")
        assert list(curve.coords["response_variable"].values) == ["a", "b"]

    def test_curve_rejects_linear_driver_and_negative_level(self, fitted):
        model, _, _ = fitted
        with pytest.raises(ValueError, match="not saturated"):
            model.saturation_curve("z", [1.0])
        with pytest.raises(ValueError, match="non-negative"):
            model.saturation_curve("x", [-1.0])

    def test_exog_irf_at_mean_reads_b(self, fitted):
        model, _, _ = fitted
        np.testing.assert_allclose(
            model.exog_irf(4).values, model.exog_irf(4, at={"v": model.saturation_level_[0]}).values
        )

    def test_exog_irf_at_higher_spend_is_smaller(self, fitted):
        model, _, exog = fitted
        lam = model.idata_.posterior["saturation_lam"].values[..., 0]
        base = model.exog_irf(4).sel(exog_variable="v").values[:, :, 0]
        high = model.exog_irf(4, at={"v": 3 * exog["v"].max()}).sel(exog_variable="v")
        ratio = high.values[:, :, 0] / base
        assert (np.abs(ratio) < 1).all() and lam.min() > 0
        # Linear drivers are unchanged.
        np.testing.assert_allclose(
            model.exog_irf(4, at="recent").sel(exog_variable="z").values,
            model.exog_irf(4).sel(exog_variable="z").values,
        )

    def test_exog_irf_bad_at_raises(self, fitted):
        model, _, _ = fitted
        with pytest.raises(ValueError, match="at must be"):
            model.exog_irf(4, at="max")
        with pytest.raises(ValueError, match="not saturated"):
            model.exog_irf(4, at={"z": 1.0})

    def test_evidence_table_reads_the_total(self, fitted):
        model, _, _ = fitted
        table = model.granger_causality().exog
        total = model.idata_.posterior["B_total"].values[..., 0, 1]  # v -> a
        row = table[(table.cause == "v") & (table.effect == "a")].iloc[0]
        from bayesian_vecm._causality import evidence_against_zero

        assert row["evidence"] == pytest.approx(evidence_against_zero(total.ravel()))

    def test_fitted_values_use_the_curve(self, fitted):
        model, _, _ = fitted
        idata = model.idata_
        post = idata.posterior
        n = post.sizes["chain"] * post.sizes["draw"]
        b = post["B"].values.reshape(n, K, -1)
        b_lin = zero_saturated_columns(b, [2, 0], M)
        alpha = post["alpha"].values.reshape(n, K, 1)
        beta = post["beta"].values.reshape(n, K, 1)
        gamma = post["Gamma"].values.reshape(n, K, -1)
        d = idata.constant_data
        mu = (
            np.einsum("ti,dij,dkj->dtk", d["y_lag1"].values, beta, alpha)
            + np.einsum("ti,dki->dtk", d["delta_x"].values, gamma)
            + np.einsum("ti,dki->dtk", model.exog_, b_lin)
            + saturation_term(idata, model.exog_)
        )
        np.testing.assert_allclose(model.fittedvalues.values.reshape(mu.shape), mu)

    def test_forecast_holds_spend_and_runs(self, fitted):
        model, _, exog = fitted
        future = np.tile(exog.mean().to_numpy(), (5, 1))
        fc = model.sample_posterior_predictive(5, exog_future=future, random_seed=1)
        assert np.isfinite(fc.posterior_predictive["y"].values).all()

    def test_forecast_rejects_negative_spend(self, fitted):
        model, _, _ = fitted
        future = np.zeros((3, M))
        future[0, 2] = -1.0
        with pytest.raises(ValueError, match="negative"):
            model.sample_posterior_predictive(3, exog_future=future)

    def test_forecast_mean_follows_the_curve(self, fitted):
        model, _, exog = fitted
        # Same innovations, two spend levels: the gap in growth at step 1 is
        # the saturated term at the new level, held from the start.
        base = np.tile(exog.mean().to_numpy(), (1, 1))
        high = base.copy()
        high[0, 2] = 2 * exog["v"].max()
        fc_base = model.sample_posterior_predictive(1, exog_future=base, random_seed=7)
        fc_high = model.sample_posterior_predictive(1, exog_future=high, random_seed=7)
        gap = (
            fc_high.posterior_predictive["delta_y"].values[:, :, 0, :]
            - fc_base.posterior_predictive["delta_y"].values[:, :, 0, :]
        )
        post = model.idata_.posterior
        lam = post["saturation_lam"].values[..., 0]
        b_sat = post["B_saturation"].values[..., 0]  # (C, D, K)
        scale = model.saturation_scale_[0]
        f = logistic_saturation(high[0, 2] / scale, lam) - logistic_saturation(
            base[0, 2] / scale, lam
        )
        np.testing.assert_allclose(gap, f[..., None] * b_sat, atol=1e-12)
