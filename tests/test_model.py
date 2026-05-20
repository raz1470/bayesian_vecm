"""Tests for the ``BayesianVECM`` class.

Two layers of testing live here:

1. **Construction + validation** — fast tests covering defaults, every
   deterministic code, eager rejection of bad config. These predate any
   PyMC code and are unchanged by the first-PyMC-model slice.
2. **Estimation** — ``fit`` is now live for ``coint_rank=1`` +
   ``deterministic="n"``. The integration test at the bottom of the file
   actually samples (briefly) so we exercise the full validate → design
   → build → sample → store-state pipeline, plus the not-yet-fitted error
   paths on ``idata`` and ``summary``. ``sample_posterior_predictive``
   stays a stub — forecasting is its own follow-up slice.
"""

from __future__ import annotations

import numpy as np
import pytest

import bayesian_vecm
from bayesian_vecm import BayesianVECM
from bayesian_vecm._model import _VALID_DETERMINISTIC


# ---------------------------------------------------------------------------
# Re-export + import surface
# ---------------------------------------------------------------------------
def test_bayesian_vecm_reexported_from_package_root() -> None:
    """``from bayesian_vecm import BayesianVECM`` is the documented entry point."""
    assert hasattr(bayesian_vecm, "BayesianVECM")
    assert bayesian_vecm.BayesianVECM is BayesianVECM


def test_bayesian_vecm_listed_in_package_all() -> None:
    """``BayesianVECM`` should be advertised in ``__all__``."""
    assert "BayesianVECM" in bayesian_vecm.__all__


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_default_construction_stores_documented_defaults() -> None:
    """No-arg construction matches the defaults documented in the class docstring."""
    model = BayesianVECM()

    assert model.k_ar_diff == 1
    assert model.coint_rank == 1
    assert model.deterministic == "n"
    assert model.priors is None


# ---------------------------------------------------------------------------
# Custom construction — happy paths
# ---------------------------------------------------------------------------
def test_custom_construction_round_trips_attributes() -> None:
    """Every constructor argument is stored unchanged on ``self``."""
    priors = {"alpha": {"dist": "Normal", "mu": 0.0, "sigma": 1.0}}
    model = BayesianVECM(
        k_ar_diff=3,
        coint_rank=2,
        deterministic="ci",
        priors=priors,
    )

    assert model.k_ar_diff == 3
    assert model.coint_rank == 2
    assert model.deterministic == "ci"
    assert model.priors is priors  # store by reference, do not deep-copy


@pytest.mark.parametrize("code", sorted(_VALID_DETERMINISTIC))
def test_every_valid_deterministic_code_is_accepted(code: str) -> None:
    """Each v0 deterministic code constructs without error."""
    model = BayesianVECM(deterministic=code)
    assert model.deterministic == code


def test_k_ar_diff_zero_is_allowed() -> None:
    """``k_ar_diff = 0`` is the pure-error-correction case and must be allowed."""
    model = BayesianVECM(k_ar_diff=0)
    assert model.k_ar_diff == 0


# ---------------------------------------------------------------------------
# Validation — invalid inputs are rejected eagerly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_k", [-1, -2, -100])
def test_negative_k_ar_diff_rejected(bad_k: int) -> None:
    """Negative lag counts have no econometric meaning."""
    with pytest.raises(ValueError, match="k_ar_diff must be non-negative"):
        BayesianVECM(k_ar_diff=bad_k)


@pytest.mark.parametrize("bad_r", [0, -1, -5])
def test_non_positive_coint_rank_rejected(bad_r: int) -> None:
    """``r = 0`` means no cointegration (trivial VAR-in-differences); ``r < 0`` is nonsense."""
    with pytest.raises(ValueError, match="coint_rank must be at least 1"):
        BayesianVECM(coint_rank=bad_r)


@pytest.mark.parametrize(
    "bad_code",
    [
        "",  # empty string
        "N",  # uppercase — case matters
        "nc",  # nonsense
        "cili",  # compound (Johansen case 4) — explicitly deferred
        "trend",  # English word, not the code
    ],
)
def test_unknown_deterministic_code_rejected(bad_code: str) -> None:
    """Typos and not-yet-supported compound codes both error eagerly."""
    with pytest.raises(ValueError, match="deterministic must be one of"):
        BayesianVECM(deterministic=bad_code)


def test_deterministic_error_message_mentions_compound_codes() -> None:
    """The error should signpost the v0.x follow-up rather than just listing codes."""
    with pytest.raises(ValueError, match="follow-up"):
        BayesianVECM(deterministic="cili")


@pytest.mark.parametrize("bad_priors", [42, "alpha=Normal", [("alpha", {})], (1, 2)])
def test_non_dict_priors_rejected(bad_priors: object) -> None:
    """``priors`` must be ``None`` or a ``dict``; anything else is a type error."""
    with pytest.raises(TypeError, match="priors must be a dict or None"):
        BayesianVECM(priors=bad_priors)  # type: ignore[arg-type]


def test_empty_dict_priors_is_accepted() -> None:
    """``priors={}`` is a legitimate "use all defaults" spelling, distinct from ``None``."""
    model = BayesianVECM(priors={})
    assert model.priors == {}


# ---------------------------------------------------------------------------
# Pre-fit error paths
# ---------------------------------------------------------------------------
def test_idata_before_fit_raises_runtime_error() -> None:
    """Accessing ``idata`` before ``fit`` should fail loudly, not return ``None``."""
    model = BayesianVECM()
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        _ = model.idata


def test_summary_before_fit_raises_runtime_error() -> None:
    model = BayesianVECM()
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        model.summary()


def test_sample_posterior_predictive_raises_not_implemented() -> None:
    """Forecasting is deferred to its own follow-up slice."""
    model = BayesianVECM()
    with pytest.raises(NotImplementedError, match="sample_posterior_predictive"):
        model.sample_posterior_predictive(steps=12)


# ---------------------------------------------------------------------------
# v0 scope guards on ``fit``
# ---------------------------------------------------------------------------
def test_fit_with_coint_rank_above_one_raises_not_implemented() -> None:
    """The PyMC graph only supports r=1 in v0 — fit should fail loudly."""
    rng = np.random.default_rng(0)
    endog = rng.normal(size=(40, 2)).cumsum(axis=0)
    model = BayesianVECM(coint_rank=2)
    with pytest.raises(NotImplementedError, match="coint_rank=2"):
        model.fit(endog, draws=5, tune=5, chains=1, progressbar=False)


def test_fit_with_non_n_deterministic_raises_not_implemented() -> None:
    rng = np.random.default_rng(0)
    endog = rng.normal(size=(40, 2)).cumsum(axis=0)
    model = BayesianVECM(deterministic="ci")
    with pytest.raises(NotImplementedError, match="deterministic="):
        model.fit(endog, draws=5, tune=5, chains=1, progressbar=False)


# ---------------------------------------------------------------------------
# Integration test — actually sample
# ---------------------------------------------------------------------------
# These run pm.sample, so they're slower than the rest of the suite (PyTensor
# compilation dominates). Kept to a minimum and configured for speed: small
# T, K=2, 1 chain, tiny draws/tune. We're checking the plumbing end-to-end,
# not posterior quality.


def _tiny_cointegrated_series(n_obs: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    y = np.zeros((n_obs, 2))
    y[0] = rng.normal(size=2)
    for t in range(1, n_obs):
        ec = y[t - 1, 0] - 0.5 * y[t - 1, 1]
        y[t, 0] = y[t - 1, 0] - 0.4 * ec + rng.normal(scale=0.5)
        y[t, 1] = y[t - 1, 1] + 0.2 * ec + rng.normal(scale=0.5)
    return y


@pytest.fixture(scope="module")
def fitted_model() -> BayesianVECM:
    """A model fitted once and shared across the integration tests below.

    Module-scoped so we pay the PyTensor compile cost once. Tiny sample
    config: this is a smoke test of the plumbing, not a stats check.
    """
    model = BayesianVECM(k_ar_diff=1, coint_rank=1, deterministic="n")
    model.fit(
        _tiny_cointegrated_series(),
        draws=20,
        tune=20,
        chains=1,
        progressbar=False,
        random_seed=0,
        compute_convergence_checks=False,
    )
    return model


def test_fit_returns_self_for_chaining(fitted_model: BayesianVECM) -> None:
    """``fit`` returns ``self`` so ``model.fit(...).summary()`` works."""
    # The fixture has already fitted; we just check the type.
    assert isinstance(fitted_model, BayesianVECM)


def test_fit_sets_trailing_underscore_attributes(fitted_model: BayesianVECM) -> None:
    assert hasattr(fitted_model, "endog_")
    assert hasattr(fitted_model, "idata_")
    assert hasattr(fitted_model, "variable_names_")
    # ndarray-like attributes
    assert fitted_model.endog_.shape == (60, 2)
    # variable_names_ is None for raw ndarray input (no .columns)
    assert fitted_model.variable_names_ is None


def test_fit_stores_endog_in_idata_constant_data(fitted_model: BayesianVECM) -> None:
    """A serialised idata must be self-contained — endog stashed inside it."""
    assert "endog" in fitted_model.idata_.constant_data
    np.testing.assert_array_equal(
        fitted_model.idata_.constant_data["endog"].values,
        fitted_model.endog_,
    )


def test_idata_property_returns_inference_data(fitted_model: BayesianVECM) -> None:
    import arviz as az

    assert isinstance(fitted_model.idata, az.InferenceData)
    # Headline groups must be present.
    assert "posterior" in fitted_model.idata
    assert "constant_data" in fitted_model.idata


def test_posterior_has_expected_variable_names(fitted_model: BayesianVECM) -> None:
    posterior_vars = set(fitted_model.idata.posterior.data_vars)
    # Free parameters
    assert "alpha" in posterior_vars
    assert "beta_free" in posterior_vars
    assert "Gamma" in posterior_vars
    # Deterministics
    assert "beta" in posterior_vars
    assert "Sigma" in posterior_vars


def test_beta_first_entry_is_one_in_every_posterior_draw(fitted_model: BayesianVECM) -> None:
    """The identification normalisation: β[0, 0] = 1 in every draw, every chain."""
    beta = fitted_model.idata.posterior["beta"].values
    # Shape: (chain, draw, K, r). For our config: (1, 20, 2, 1).
    assert beta.shape == (1, 20, 2, 1)
    np.testing.assert_array_equal(beta[..., 0, 0], 1.0)


def test_summary_returns_dataframe_with_expected_rows(fitted_model: BayesianVECM) -> None:
    summary = fitted_model.summary()
    # Each row is a scalar parameter; for K=2, r=1, k_ar_diff=1 we expect:
    #   alpha (K*r = 2), beta (K*r = 2), Gamma (K*K*k = 4), Sigma (K*K = 4)
    # = 12 rows total. We test the row labels rather than counts to avoid
    # tying the test to ArviZ's exact indexing format.
    index_strs = [str(i) for i in summary.index]
    assert any(s.startswith("alpha") for s in index_strs)
    assert any(s.startswith("beta") for s in index_strs)
    assert any(s.startswith("Gamma") for s in index_strs)
    assert any(s.startswith("Sigma") for s in index_strs)


def test_summary_accepts_var_names_override(fitted_model: BayesianVECM) -> None:
    """User can request a subset of parameters."""
    summary = fitted_model.summary(var_names=["alpha"])
    index_strs = [str(i) for i in summary.index]
    assert all(s.startswith("alpha") for s in index_strs)


def test_dataframe_input_captures_variable_names() -> None:
    """Column labels round-trip into ``variable_names_`` when endog is a DataFrame-like."""
    # Use a minimal DataFrame-like duck type — avoids forcing pandas as a test dep.
    class _FakeDF:
        def __init__(self, arr: np.ndarray, columns: list[str]) -> None:
            self._arr = arr
            self.columns = columns

        def to_numpy(self) -> np.ndarray:
            return self._arr

    df = _FakeDF(_tiny_cointegrated_series(), columns=["gdp", "cpi"])
    model = BayesianVECM().fit(
        df,
        draws=5,
        tune=5,
        chains=1,
        progressbar=False,
        random_seed=0,
        compute_convergence_checks=False,
    )
    assert model.variable_names_ == ["gdp", "cpi"]
    # And the names land in idata.constant_data["endog"]'s "variable" coord.
    coord = model.idata.constant_data["endog"].coords["variable"].values.tolist()
    assert coord == ["gdp", "cpi"]
