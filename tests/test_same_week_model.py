"""Same-week links in BayesianVECM: graph, likelihood and fitted model."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._design import cointegration_design
from bayesian_vecm._pymc import build_pymc_model

T = 60
K = 3
M = 3
LINKS = ((1, 0), (2, 1))  # y1 <- y0, y2 <- y1: a chain
MASK = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])


def _endog(seed: int = 0, t: int = T) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).standard_normal((t, K)), axis=0)


def _exog(seed: int = 3, t: int = T) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((t, M))
    x[:, 2] = np.exp(0.4 * x[:, 2])  # raw spend for saturation
    return x


def _graph(*, links=LINKS, mask=MASK, adstock=(), saturation=(), max_lag=2):
    window = max_lag if adstock else 0
    x = _exog()
    d = cointegration_design(_endog(), 1, exog=x, exog_lags=window)
    sat = list(saturation)
    model = build_pymc_model(
        d,
        k_ar_diff=1,
        coint_rank=1,
        deterministic="n",
        adstock=adstock,
        adstock_max_lag=max_lag,
        saturation=saturation,
        saturation_scale=x[:, sat].max(axis=0) if sat else None,
        saturation_level=x[:, sat].mean(axis=0) if sat else None,
        links=links,
        b_mask=mask,
    )
    return model, d


def _point(model: pm.Model, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    point = model.initial_point()
    return {k: v + 0.3 * rng.standard_normal(np.shape(v)) for k, v in point.items()}


def _values(model: pm.Model, point: dict, names: list[str]) -> dict:
    fn = pytensor.function(
        model.value_vars,
        model.replace_rvs_by_values([model[n] for n in names]),
        on_unused_input="ignore",
    )
    out = fn(*[point[v.name] for v in model.value_vars])
    return dict(zip(names, out, strict=True))


def _loglik(model: pm.Model, point: dict) -> float:
    return float(model.compile_logp(vars=model.observed_RVs, jacobian=False)(point))


class TestGraph:
    def test_variables(self):
        model, _ = _graph()
        assert {v.name for v in model.free_RVs} >= {"A0_links", "B_raw"}
        assert {v.name for v in model.deterministics} >= {"A0", "B", "B_structural"}

    def test_a0_and_reduced_b(self):
        model, _ = _graph()
        v = _values(model, _point(model), ["A0_links", "A0", "B", "B_structural"])
        a0 = np.eye(K)
        a0[1, 0], a0[2, 1] = -v["A0_links"]
        np.testing.assert_allclose(v["A0"], a0)
        np.testing.assert_allclose(v["B"], np.linalg.inv(a0) @ v["B_structural"], atol=1e-12)
        np.testing.assert_array_equal(v["B_structural"][MASK == 0], 0.0)

    def test_likelihood_is_the_reduced_form(self):
        """Same likelihood as a plain model whose B is A0^-1 B_s."""
        model, _ = _graph()
        point = _point(model, seed=1)
        plain, _ = _graph(links=(), mask=None)
        v = _values(model, point, ["B"])
        plain_point = {k: point[k] for k in plain.initial_point() if k != "B"}
        plain_point["B"] = v["B"]
        assert _loglik(model, point) == pytest.approx(_loglik(plain, plain_point), rel=1e-10)

    def test_likelihood_with_adstock_and_saturation(self):
        kw = {"adstock": (0,), "saturation": (2,)}
        model, _ = _graph(**kw)
        point = _point(model, seed=2)
        plain, _ = _graph(links=(), mask=None, **kw)
        v = _values(model, point, ["B", "B_total", "B_saturation", "B_saturation_structural"])
        a0_inv = np.linalg.inv(_values(model, point, ["A0"])["A0"])
        np.testing.assert_allclose(v["B_saturation"], a0_inv @ v["B_saturation_structural"])
        plain_point = {k: point[k] for k in plain.initial_point() if k not in ("B_total", "B_free")}
        plain_point["B_total"] = v["B_total"]
        plain_point["B_free"] = v["B"][:, [1]][:, :, None]  # driver 1, lag 0
        assert _loglik(model, point) == pytest.approx(_loglik(plain, plain_point), rel=1e-10)

    def test_mask_reaches_every_lag(self):
        x = _exog()
        d = cointegration_design(_endog(), 1, exog=x, exog_lags=2)
        model = build_pymc_model(
            d, k_ar_diff=1, coint_rank=1, deterministic="n", exog_lags=2, links=LINKS, b_mask=MASK
        )
        b_s = _values(model, _point(model), ["B_structural"])["B_structural"]
        for lag in range(3):
            np.testing.assert_array_equal(b_s[:, lag * M : (lag + 1) * M][MASK == 0], 0.0)

    def test_exclusions_without_links(self):
        model, _ = _graph(links=())
        assert "A0_links" not in model.named_vars
        v = _values(model, _point(model), ["B", "B_structural"])
        np.testing.assert_array_equal(v["B"], v["B_structural"])
        np.testing.assert_array_equal(v["B"][MASK == 0], 0.0)

    def test_a0_prior_override(self):
        x = _exog()
        d = cointegration_design(_endog(), 1, exog=x)
        model = build_pymc_model(
            d,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"A0": {"dist": "Normal", "mu": 0.0, "sigma": 0.1}},
            links=LINKS,
            b_mask=MASK,
        )
        assert model["A0_links"].owner.inputs[-1].eval() == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# BayesianVECM
# ---------------------------------------------------------------------------

NAMES = ["sales", "awareness", "consideration"]
DRIVERS = ["spend", "rate", "pr"]
CONTEMP = {"consideration": ["awareness"]}
EXCLUDED = {"consideration": ["spend", "pr"]}


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(_endog(), columns=NAMES), pd.DataFrame(_exog(), columns=DRIVERS)


@pytest.fixture(scope="module")
def fitted():
    endog, exog = _frames()
    model = BayesianVECM(k_ar_diff=1, coint_rank=1, contemporaneous=CONTEMP, excluded=EXCLUDED)
    # Random drivers do not move awareness, so the instruments are weak.
    with pytest.warns(UserWarning, match="weak instruments"):
        model.fit(
            endog,
            exog=exog,
            draws=30,
            tune=30,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=0,
        )
    return model


class TestModel:
    def test_defaults(self):
        model = BayesianVECM()
        assert model.contemporaneous is None
        assert model.excluded is None

    def test_links_table(self, fitted):
        table = fitted.same_week_links()
        assert list(table.columns) == [
            "cause",
            "effect",
            "mean",
            "lower",
            "upper",
            "evidence",
            "link",
            "first_stage_f",
            "sargan_p",
        ]
        row = table.iloc[0]
        assert (row.cause, row.effect) == ("awareness", "consideration")
        assert row["lower"] <= row["mean"] <= row["upper"]

    def test_posterior_layout(self, fitted):
        post = fitted.idata.posterior
        b = post["B"].values
        b_s = post["B_structural"].values
        link = post["A0_links"].values[..., 0]
        np.testing.assert_array_equal(b_s[..., 2, [0, 2]], 0.0)
        np.testing.assert_allclose(b[..., 2, :], b_s[..., 2, :] + link[..., None] * b_s[..., 1, :])

    def test_summary_lists_the_links(self, fitted):
        assert any(i.startswith("A0_links") for i in fitted.summary().index)

    def test_other_methods_still_run(self, fitted):
        _, exog = _frames()
        assert fitted.exog_irf(4).shape[-3:] == (5, K, M)
        assert fitted.fittedvalues.shape[-1] == K
        fc = fitted.sample_posterior_predictive(3, exog_future=exog.iloc[:3])
        assert np.isfinite(fc.posterior_predictive["y"].values).all()

    def test_links_table_reports_the_instrument_check(self, fitted):
        check = fitted.instrument_check_
        table = fitted.same_week_links()
        assert table.first_stage_f.iloc[0] == pytest.approx(check.first_stage_f.iloc[0])
        assert table.sargan_p.iloc[0] == pytest.approx(check.sargan_p.iloc[0])

    def test_drivers_table_splits_direct_and_via_links(self, fitted):
        table = fitted.granger_causality().exog
        assert {"direct", "via_links"} <= set(table.columns)
        np.testing.assert_allclose(table.direct + table.via_links, table["mean"])
        cons = table[table.effect == "consideration"].set_index("cause")
        # Excluded drivers reach consideration only through awareness.
        assert cons.loc["spend", "direct"] == 0.0
        assert cons.loc["pr", "direct"] == 0.0
        assert cons.loc["spend", "via_links"] != 0.0
        # Sales have no links, so everything is direct.
        sales = table[table.effect == "sales"]
        np.testing.assert_allclose(sales.via_links, 0.0, atol=1e-12)
        assert "Same-week effects split" in str(fitted.granger_causality())

    def test_drivers_table_without_links_has_no_split(self):
        endog, exog = _frames()
        model = BayesianVECM(coint_rank=1, excluded=EXCLUDED)
        model.fit(endog, exog=exog, draws=5, tune=5, chains=1, cores=1, progressbar=False)
        result = model.granger_causality()
        assert "direct" not in result.exog.columns
        assert "Same-week effects split" not in str(result)

    def test_no_links_raises(self):
        endog, exog = _frames()
        model = BayesianVECM(coint_rank=1)
        model.fit(endog, exog=exog, draws=5, tune=5, chains=1, cores=1, progressbar=False)
        with pytest.raises(ValueError, match="no same-week links"):
            model.same_week_links()

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            BayesianVECM(contemporaneous=CONTEMP).same_week_links()


class TestValidation:
    def _fit(self, **kwargs):
        endog, exog = _frames()
        exog = kwargs.pop("exog", exog)
        BayesianVECM(coint_rank=1, **kwargs).fit(
            endog, exog=exog, draws=5, tune=5, chains=1, cores=1, progressbar=False
        )

    def test_not_identified(self):
        with pytest.raises(ValueError, match="not identified"):
            self._fit(contemporaneous=CONTEMP)

    def test_two_causes_need_two_exclusions(self):
        with pytest.raises(ValueError, match="2 cause"):
            self._fit(
                contemporaneous={"consideration": ["awareness", "sales"]},
                excluded={"consideration": ["spend"]},
            )

    def test_cycle(self):
        with pytest.raises(ValueError, match="cycle"):
            self._fit(
                contemporaneous={"consideration": ["awareness"], "awareness": ["consideration"]},
                excluded={"consideration": ["spend"], "awareness": ["rate"]},
            )

    def test_unknown_names(self):
        with pytest.raises(ValueError, match="unknown brand metric"):
            self._fit(contemporaneous={"trust": ["awareness"]}, excluded={"trust": ["spend"]})
        with pytest.raises(ValueError, match="unknown driver"):
            self._fit(contemporaneous=CONTEMP, excluded={"consideration": ["tv"]})

    def test_needs_exog(self):
        with pytest.raises(ValueError, match="need exog"):
            self._fit(contemporaneous=CONTEMP, excluded=EXCLUDED, exog=None)


SIM_LINK = {"brand_consideration": ["brand_awareness"]}
SIM_CHANNELS = ["brand_spend", "pr_reach", "sponsorship_reach"]


class TestInstrumentCheck:
    @pytest.fixture(scope="class")
    def data(self):
        return simulate_brand_data(
            contemporaneous={"brand_consideration": {"brand_awareness": 0.2}}
        )

    def _check(self, data, excluded):
        model = BayesianVECM(k_ar_diff=1, contemporaneous=SIM_LINK, excluded=excluded)
        return model._check_instruments(data.endog, data.exog)

    def test_good_instruments_do_not_warn(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            table = self._check(data, {"brand_consideration": SIM_CHANNELS})
        assert table.first_stage_f.iloc[0] > 10
        assert table.estimate.iloc[0] == pytest.approx(0.2, abs=0.1)

    def test_wrong_exclusion_warns(self, data):
        # Consumer confidence acts on consideration directly.
        excluded = {"brand_consideration": [*SIM_CHANNELS, "consumer_confidence"]}
        with pytest.warns(UserWarning, match="disagree"):
            self._check(data, excluded)

    def test_weak_instruments_warn(self, data):
        short = data.endog.iloc[:40], data.exog.iloc[:40]
        model = BayesianVECM(
            k_ar_diff=1, contemporaneous=SIM_LINK, excluded={"brand_consideration": ["pr_reach"]}
        )
        with pytest.warns(UserWarning, match="weak instruments"):
            model._check_instruments(*short)

    def test_check_that_cannot_run_returns_none(self, data):
        model = BayesianVECM(
            k_ar_diff=1, contemporaneous=SIM_LINK, excluded={"brand_consideration": SIM_CHANNELS}
        )
        assert model._check_instruments(data.endog.iloc[:8], data.exog.iloc[:8]) is None
