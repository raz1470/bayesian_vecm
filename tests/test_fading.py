"""Tests for drivers whose effect fades (``fading``).

Coverage
--------
* Simulator: ``exog_fade`` validation, no change without it, a pulse's
  effect drains away at the half-life.
* PyMC graph: the likelihood's mean equals the NumPy fade recursion used by
  fitted values and forecasts, with and without saturation and links.
* Helpers: the fade kernel sums to ``1 - fade_share``; ``faded_b`` extends
  the lags.
* Model: validation, a short fit, exog IRF settling at ``1 - fade_share``
  of the impact path, fitted values and forecasts.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.stats import multivariate_normal

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._design import cointegration_design
from bayesian_vecm._fading import fade_kernel, fade_term, faded_b
from bayesian_vecm._pymc import build_pymc_model

T = 60
K = 2
M = 3


def _endog(seed: int = 0, t: int = T) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal((t, K)), axis=0)


def _exog(seed: int = 3, t: int = T, m: int = M) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((t, m)) + 0.5


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class TestSimulator:
    def test_no_change_without_fade(self):
        a = simulate_brand_data(seed=2)
        b = simulate_brand_data(seed=2, exog_fade=None)
        np.testing.assert_array_equal(a.endog.values, b.endog.values)
        assert np.isinf(a.truth.fade_half_life).all()
        assert a.truth.fade_share == 0.0
        np.testing.assert_array_equal(a.truth.fade_decay, 1.0)

    def test_fade_changes_the_data(self):
        a = simulate_brand_data(seed=2)
        b = simulate_brand_data(seed=2, exog_fade={"brand_spend": 26.0})
        assert not np.allclose(a.endog.values, b.endog.values)
        assert b.truth.fade_half_life[0] == 26.0
        assert b.truth.fade_share == 1.0
        assert b.truth.fade_decay[0] == pytest.approx(0.5 ** (1 / 26))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"exog_fade": {"tv": 26.0}}, "unknown exog_fade"),
            ({"exog_fade": {"brand_spend": 0.0}}, "must be positive"),
            ({"exog_fade": {"brand_spend": 26.0}, "exog_fade_share": 0.0}, r"\(0, 1\]"),
            ({"exog_fade": {"brand_spend": 26.0}, "exog_lag_share": 0.5}, "exog_lag_share"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            simulate_brand_data(**kwargs)


# ---------------------------------------------------------------------------
# PyMC graph against the NumPy recursion
# ---------------------------------------------------------------------------
def _graph_and_numpy_logp(*, saturation=(), links=(), fading=(0, 2)):
    exog = np.abs(_exog()) if saturation else _exog()
    design = cointegration_design(_endog(), 1, deterministic="n", exog=exog)
    kwargs = {}
    if saturation:
        kwargs = {
            "saturation": saturation,
            "saturation_scale": exog[:, list(saturation)].max(axis=0),
            "saturation_level": exog[:, list(saturation)].mean(axis=0),
        }
    model = build_pymc_model(
        design,
        k_ar_diff=1,
        coint_rank=1,
        deterministic="n",
        fading=fading,
        links=links,
        b_mask=np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) if links else None,
        **kwargs,
    )
    point = model.initial_point()
    rng = np.random.default_rng(5)
    point = {k: v + 0.3 * rng.standard_normal(np.shape(v)) for k, v in point.items()}
    names = ["alpha", "beta", "Gamma", "Sigma", "B", "fade_share", "fade_half_life"]
    if saturation:
        names += ["saturation_lam", "saturation_weights", "saturation_mean", "B_saturation"]
    outs = model.replace_rvs_by_values([model[n] for n in names])
    fn = model.compile_fn(outs, inputs=model.value_vars, on_unused_input="ignore")
    values = dict(zip(names, fn(point), strict=True))
    graph_logp = float(model.compile_logp(vars=[model["delta_y_obs"]])(point))

    def da(name, x):
        x = np.asarray(x)[None, None]
        dims = ["chain", "draw", *[f"{name}_{i}" for i in range(x.ndim - 2)]]
        return xr.DataArray(x, dims=dims)

    constant = {
        "fading_idx": ("fading_variable", np.array(fading)),
        "exog_window": ((), np.array(0)),
    }
    if saturation:
        constant["saturation_idx"] = ("saturation_variable", np.array(saturation))
        constant["saturation_scale"] = ("saturation_variable", kwargs["saturation_scale"])
    idata = SimpleNamespace(
        posterior=xr.Dataset({n: da(n, v) for n, v in values.items()}),
        constant_data=xr.Dataset(constant),
    )
    b = values["B"].copy()
    for j in saturation:
        b[:, j] = 0.0
    mu = (
        design.y_lag1 @ values["beta"] @ values["alpha"].T
        + design.delta_x @ values["Gamma"].T
        + design.exog @ b.T
    )
    if saturation:
        from bayesian_vecm._saturation import saturation_term

        mu = mu + saturation_term(idata, design.exog)[0]
    mu = mu + fade_term(idata, design.exog, design.exog.shape[0])[0]
    numpy_logp = multivariate_normal(np.zeros(K), values["Sigma"]).logpdf(design.delta_y - mu).sum()
    return graph_logp, numpy_logp


class TestGraph:
    def test_free_variables(self):
        design = cointegration_design(_endog(), 1, exog=_exog())
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic="n", fading=(1,))
        assert model["fade_share"].type.shape == (1,)
        assert model["fade_half_life"].type.shape == (1,)

    def test_matches_numpy_recursion(self):
        graph, numpy = _graph_and_numpy_logp()
        assert graph == pytest.approx(numpy, rel=1e-8)

    def test_matches_numpy_recursion_with_saturation(self):
        graph, numpy = _graph_and_numpy_logp(saturation=(0,), fading=(0, 1))
        assert graph == pytest.approx(numpy, rel=1e-8)

    def test_matches_numpy_recursion_with_links(self):
        graph, numpy = _graph_and_numpy_logp(links=((0, 1),))
        assert graph == pytest.approx(numpy, rel=1e-8)

    def test_prior_override(self):
        design = cointegration_design(_endog(), 1, exog=_exog())
        model = build_pymc_model(
            design,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            fading=(0,),
            priors={"fade_share": {"dist": "Beta", "alpha": 2.0, "beta": 2.0}},
        )
        assert "fade_share" in model.named_vars


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_idata(share, half_life, fading=(0,)):
    share = np.asarray(share, dtype=np.float64).reshape(1, 1, -1)
    half_life = np.asarray(half_life, dtype=np.float64).reshape(1, 1, -1)
    return SimpleNamespace(
        posterior=xr.Dataset(
            {
                "fade_share": (("chain", "draw", "f"), share),
                "fade_half_life": (("chain", "draw", "f"), half_life),
            }
        ),
        constant_data=xr.Dataset({"fading_idx": ("f", np.array(fading))}),
    )


class TestHelpers:
    @pytest.mark.parametrize("share", [0.0, 0.4, 1.0])
    def test_kernel_sums_to_lasting_share(self, share):
        _, kernel = fade_kernel(_fake_idata(share, 10.0), steps=2000)
        assert kernel[0, 0, 0] == 1.0
        assert kernel.sum() == pytest.approx(1.0 - share, abs=1e-6)

    def test_faded_b_extends_lags(self):
        b = np.arange(1.0, 1.0 + K * M * 2).reshape(1, 1, K, M * 2)  # two lag blocks
        out = faded_b(b, _fake_idata(1.0, 5.0, fading=(1,)), steps=10, n_lags=2)
        assert out.shape == (1, 1, K, M * 11)
        flat = out.reshape(K, 11, M)
        lagged = b.reshape(K, 2, M)
        # Drivers that do not fade keep their two lags and zeros after.
        np.testing.assert_array_equal(flat[:, :2, 0], lagged[:, :, 0])
        np.testing.assert_array_equal(flat[:, 2:, 0], 0.0)
        # The faded driver starts with its own lag 0.
        np.testing.assert_array_equal(flat[:, 0, 1], lagged[:, 0, 1])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TestModel:
    def test_needs_exog(self):
        model = BayesianVECM(coint_rank=1, fading=[0])
        with pytest.raises(ValueError, match="fading needs exog"):
            model.fit(_endog(), draws=10, tune=10, chains=1)

    def test_rejects_duplicates(self):
        with pytest.raises(ValueError, match="fading"):
            BayesianVECM(fading=[0, 0])

    @pytest.fixture(scope="class")
    def fitted(self):
        data = simulate_brand_data(n_obs=120, seed=4, exog_fade={"brand_spend": 13.0})
        model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co", fading=["brand_spend"])
        model.fit(
            data.endog,
            exog=data.exog,
            draws=40,
            tune=40,
            chains=1,
            random_seed=0,
            progressbar=False,
        )
        return model, data

    def test_stores_fading(self, fitted):
        model, _ = fitted
        assert model.fading_idx_ == (0,)
        assert "fade_share" in model.idata_.posterior
        assert int(model.idata_.constant_data["fading_idx"].values[0]) == 0

    def test_exog_irf_settles_at_lasting_share(self, fitted):
        model, _ = fitted
        steps = 1500
        faded = model.exog_irf(steps=steps).sel(exog_variable="brand_spend")
        share = model.idata_.posterior["fade_share"].values[..., 0]
        # The same model without the fade: the plain dynamic multipliers.
        from bayesian_vecm._irf import compute_exog_irf

        plain = compute_exog_irf(
            model.idata_, k_ar_diff=1, steps=steps, exog_names=model.exog_names_
        ).sel(exog_variable="brand_spend")
        # Draws whose fade has run its course by the last horizon.
        done = model.idata_.posterior["fade_half_life"].values[..., 0] < steps / 15
        assert done.any()
        end_faded = faded.isel(horizon=-1).values[done]
        end_plain = plain.isel(horizon=-1).values[done]
        expected = (1.0 - share[done])[:, None] * end_plain
        np.testing.assert_allclose(end_faded, expected, atol=1e-6)
        np.testing.assert_allclose(
            faded.isel(horizon=0).values, plain.isel(horizon=0).values, atol=1e-12
        )

    def test_fittedvalues_include_fade(self, fitted):
        model, _ = fitted
        assert np.isfinite(model.fittedvalues.values).all()

    def test_forecast_runs(self, fitted):
        model, data = fitted
        future = np.zeros((20, data.exog.shape[1]))
        fc = model.sample_posterior_predictive(20, exog_future=future, random_seed=1)
        assert fc.posterior_predictive["y"].shape[2] == 20

    def test_forecast_mean_drains_after_campaign(self, fitted):
        model, data = fitted
        steps = 200
        base = np.zeros((steps, data.exog.shape[1]))
        plan = base.copy()
        plan[:13, 0] = 1.0
        lo = model.sample_posterior_predictive(steps, exog_future=base, random_seed=3)
        hi = model.sample_posterior_predictive(steps, exog_future=plan, random_seed=3)
        gap = (hi.posterior_predictive["y"] - lo.posterior_predictive["y"]).values
        share = model.idata_.posterior["fade_share"].values.ravel()
        peak = np.abs(gap[:, :, :30, 1]).max(axis=2).ravel()
        end = np.abs(gap[:, :, -1, 1]).ravel()
        # Once the fade has run, what is left is the lasting share of the lift.
        fades = share > 0.8
        assert fades.any()
        assert (end[fades] < peak[fades]).all()
