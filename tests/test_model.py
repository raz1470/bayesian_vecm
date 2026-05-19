"""Tests for the ``BayesianVECM`` skeleton.

This is a stub class — ``fit``, ``idata``, ``summary``, and
``sample_posterior_predictive`` all raise :class:`NotImplementedError` in v0.
What we *can* test, and do, is:

1. The configuration stored in ``__init__`` round-trips correctly for defaults
   and for every supported deterministic code.
2. Invalid configuration is rejected at construction time, not silently
   deferred — fail fast on typos and shape mistakes.
3. The estimation methods do in fact raise ``NotImplementedError`` (so the
   class is honest about what's not implemented yet, and so coverage's
   ``raise NotImplementedError`` exclusion stays accurate).
4. ``BayesianVECM`` is re-exported from the package root, since the target
   API in ``NOTES.md`` is ``from bayesian_vecm import BayesianVECM``.
"""

from __future__ import annotations

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
# Estimation methods are honest stubs
# ---------------------------------------------------------------------------
def test_fit_raises_not_implemented() -> None:
    model = BayesianVECM()
    with pytest.raises(NotImplementedError, match="fit is not implemented"):
        model.fit(endog=None)


def test_idata_property_raises_not_implemented() -> None:
    model = BayesianVECM()
    with pytest.raises(NotImplementedError, match="idata is not implemented"):
        _ = model.idata


def test_summary_raises_not_implemented() -> None:
    model = BayesianVECM()
    with pytest.raises(NotImplementedError, match="summary is not implemented"):
        model.summary()


def test_sample_posterior_predictive_raises_not_implemented() -> None:
    model = BayesianVECM()
    with pytest.raises(NotImplementedError, match="sample_posterior_predictive is not implemented"):
        model.sample_posterior_predictive(steps=12)
