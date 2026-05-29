"""Tests for the v0 PyMC model graph.

These tests build models but do not run any sampler — sampling is exercised
by the integration tests in ``test_model.py``. Keeping graph-construction
tests sampler-free makes them fast and lets the test file focus on the API
contract (shapes, scope guards, prior plumbing) rather than estimation
behaviour.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from bayesian_vecm._design import cointegration_design
from bayesian_vecm._pymc import build_pymc_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_cointegrated(n_obs: int = 80, seed: int = 42) -> np.ndarray:
    """Tiny bivariate cointegrated series with vector [1, -0.5]."""
    rng = np.random.default_rng(seed=seed)
    y = np.zeros((n_obs, 2))
    y[0] = rng.normal(size=2)
    for t in range(1, n_obs):
        ec = y[t - 1, 0] - 0.5 * y[t - 1, 1]
        y[t, 0] = y[t - 1, 0] - 0.4 * ec + rng.normal(scale=0.5)
        y[t, 1] = y[t - 1, 1] + 0.2 * ec + rng.normal(scale=0.5)
    return y


def _synthetic_trivariate_r2(n_obs: int = 100, seed: int = 7) -> np.ndarray:
    """Trivariate series with cointegration rank 2.

    Two cointegrating relations: [1, 0, -0.5] and [0, 1, -0.3].
    The third variable is a random walk that both relations load on.
    """
    rng = np.random.default_rng(seed=seed)
    y = np.zeros((n_obs, 3))
    y[0] = rng.normal(size=3)
    for t in range(1, n_obs):
        ec1 = y[t - 1, 0] - 0.5 * y[t - 1, 2]
        ec2 = y[t - 1, 1] - 0.3 * y[t - 1, 2]
        y[t, 0] = y[t - 1, 0] - 0.3 * ec1 + rng.normal(scale=0.5)
        y[t, 1] = y[t - 1, 1] - 0.3 * ec2 + rng.normal(scale=0.5)
        y[t, 2] = y[t - 1, 2] + 0.1 * ec1 + 0.1 * ec2 + rng.normal(scale=0.5)
    return y


@pytest.fixture
def design_k1():
    """Design with k_ar_diff=1 from a small bivariate cointegrated series."""
    return cointegration_design(_synthetic_cointegrated(), k_ar_diff=1)


@pytest.fixture
def design_k0():
    """Design with k_ar_diff=0 — pure error-correction case."""
    return cointegration_design(_synthetic_cointegrated(), k_ar_diff=0)


@pytest.fixture
def design_trivariate_r2():
    """Trivariate design, k_ar_diff=1, for r=2 tests."""
    return cointegration_design(_synthetic_trivariate_r2(), k_ar_diff=1)


def _build_default(design, *, k_ar_diff=1):
    return build_pymc_model(
        design,
        k_ar_diff=k_ar_diff,
        coint_rank=1,
        deterministic="n",
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestBuildHappyPath:
    def test_returns_pm_model(self, design_k1):
        assert isinstance(_build_default(design_k1), pm.Model)

    def test_named_vars_present(self, design_k1):
        model = _build_default(design_k1)
        names = set(model.named_vars)
        # Free parameters
        assert "alpha" in names
        assert "beta_free" in names
        assert "Gamma" in names
        assert "Sigma_chol" in names
        # Deterministics
        assert "beta" in names
        assert "Sigma" in names
        # Constant data
        assert "delta_y" in names
        assert "y_lag1" in names
        assert "delta_x" in names

    def test_observed_variable(self, design_k1):
        model = _build_default(design_k1)
        observed_names = {v.name for v in model.observed_RVs}
        assert observed_names == {"delta_y_obs"}

    def test_k0_drops_gamma_and_delta_x(self, design_k0):
        model = _build_default(design_k0, k_ar_diff=0)
        names = set(model.named_vars)
        # No short-run dynamics → no Gamma RV, no delta_x constant data.
        assert "Gamma" not in names
        assert "delta_x" not in names
        # The error-correction block still exists.
        assert "alpha" in names
        assert "beta" in names

    def test_beta_first_entry_pinned_to_one(self, design_k1):
        """The β-identification normalisation: β[0, 0] is fixed at 1.

        This is the headline econometric property of the v0 graph — if the
        normalisation ever breaks silently, every downstream β interpretation
        is wrong.
        """
        model = _build_default(design_k1)
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        # value has shape (K, r) = (2, 1).
        assert value.shape == (2, 1)
        np.testing.assert_array_equal(value[0, 0], 1.0)


# ---------------------------------------------------------------------------
# Scope guards
# ---------------------------------------------------------------------------


class TestScopeGuards:
    @pytest.mark.parametrize("code", ["co", "ci", "lo", "li"])
    def test_deterministic_other_than_n_raises(self, design_k1, code):
        with pytest.raises(NotImplementedError, match="deterministic="):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic=code,
            )


# ---------------------------------------------------------------------------
# r > 1: higher cointegration rank
# ---------------------------------------------------------------------------


class TestHigherRank:
    def test_r2_returns_pm_model(self, design_trivariate_r2):
        model = build_pymc_model(design_trivariate_r2, k_ar_diff=1, coint_rank=2, deterministic="n")
        assert isinstance(model, pm.Model)

    def test_r2_named_vars_present(self, design_trivariate_r2):
        model = build_pymc_model(design_trivariate_r2, k_ar_diff=1, coint_rank=2, deterministic="n")
        names = set(model.named_vars)
        assert {"alpha", "beta_free", "beta", "Gamma", "Sigma_chol", "Sigma"} <= names

    def test_r2_beta_shape(self, design_trivariate_r2):
        """beta should be (K, r) = (3, 2) for a trivariate r=2 model."""
        model = build_pymc_model(design_trivariate_r2, k_ar_diff=1, coint_rank=2, deterministic="n")
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        assert value.shape == (3, 2)

    def test_r2_beta_pin_is_identity(self, design_trivariate_r2):
        """Top r x r block of beta must be I_r in every draw."""
        model = build_pymc_model(design_trivariate_r2, k_ar_diff=1, coint_rank=2, deterministic="n")
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        # value shape: (3, 2); top 2x2 block must equal I_2
        np.testing.assert_array_equal(value[:2, :], np.eye(2))

    def test_r2_beta_free_rows_are_not_pinned(self, design_trivariate_r2):
        """The free row(s) of beta should vary across draws."""
        model = build_pymc_model(design_trivariate_r2, k_ar_diff=1, coint_rank=2, deterministic="n")
        with model:
            draws = pm.draw(model.named_vars["beta"], draws=50, random_seed=0)
        # draws shape: (50, 3, 2); row index 2 (the free row) should vary
        free_row = draws[:, 2, :]
        assert free_row.std() > 0.0

    def test_r2_alpha_shape(self, design_trivariate_r2):
        """alpha should be (K, r) = (3, 2)."""
        model = build_pymc_model(design_trivariate_r2, k_ar_diff=1, coint_rank=2, deterministic="n")
        with model:
            value = pm.draw(model.named_vars["alpha"], draws=1, random_seed=0)
        assert value.shape == (3, 2)

    def test_r1_beta_shape_unchanged(self, design_k1):
        """Existing r=1 bivariate model: beta still (2, 1)."""
        model = _build_default(design_k1)
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        assert value.shape == (2, 1)


# ---------------------------------------------------------------------------
# Priors API — accepted overrides
# ---------------------------------------------------------------------------


class TestPriors:
    def test_none_uses_defaults(self, design_k1):
        model = build_pymc_model(
            design_k1, k_ar_diff=1, coint_rank=1, deterministic="n", priors=None
        )
        assert isinstance(model, pm.Model)

    def test_empty_dict_uses_defaults(self, design_k1):
        model = build_pymc_model(design_k1, k_ar_diff=1, coint_rank=1, deterministic="n", priors={})
        assert isinstance(model, pm.Model)

    def test_user_alpha_override(self, design_k1):
        priors = {"alpha": {"dist": "Normal", "mu": 0.0, "sigma": 0.3}}
        model = build_pymc_model(
            design_k1, k_ar_diff=1, coint_rank=1, deterministic="n", priors=priors
        )
        assert "alpha" in set(model.named_vars)

    def test_user_beta_override_applies_to_beta_free(self, design_k1):
        priors = {"beta": {"dist": "Normal", "mu": 0.0, "sigma": 10.0}}
        model = build_pymc_model(
            design_k1, k_ar_diff=1, coint_rank=1, deterministic="n", priors=priors
        )
        # The user 'beta' key controls the free part of β; the pinned first
        # entry is unchanged.
        assert "beta_free" in set(model.named_vars)

    def test_user_gamma_override(self, design_k1):
        priors = {"Gamma": {"dist": "Normal", "mu": 0.0, "sigma": 0.1}}
        model = build_pymc_model(
            design_k1, k_ar_diff=1, coint_rank=1, deterministic="n", priors=priors
        )
        assert "Gamma" in set(model.named_vars)

    def test_user_sigma_eta_override(self, design_k1):
        priors = {"Sigma": {"eta": 5.0}}
        model = build_pymc_model(
            design_k1, k_ar_diff=1, coint_rank=1, deterministic="n", priors=priors
        )
        assert "Sigma_chol" in set(model.named_vars)

    def test_user_sigma_sd_sigma_override(self, design_k1):
        priors = {"Sigma": {"sd_sigma": 2.0}}
        model = build_pymc_model(
            design_k1, k_ar_diff=1, coint_rank=1, deterministic="n", priors=priors
        )
        assert "Sigma_chol" in set(model.named_vars)

    def test_alternative_dist_for_alpha(self, design_k1):
        """Non-Normal priors are accepted as long as the kwargs match."""
        priors = {"alpha": {"dist": "Laplace", "mu": 0.0, "b": 1.0}}
        model = build_pymc_model(
            design_k1, k_ar_diff=1, coint_rank=1, deterministic="n", priors=priors
        )
        assert "alpha" in set(model.named_vars)


# ---------------------------------------------------------------------------
# Priors API — validation
# ---------------------------------------------------------------------------


class TestPriorsValidation:
    def test_unknown_top_level_key_raises(self, design_k1):
        with pytest.raises(ValueError, match="unknown prior key"):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic="n",
                priors={"sigma": {}},  # lowercase typo
            )

    def test_unknown_dist_name_raises(self, design_k1):
        with pytest.raises(ValueError, match="unknown PyMC distribution"):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic="n",
                priors={"alpha": {"dist": "NotARealDist"}},
            )

    def test_missing_dist_key_raises(self, design_k1):
        with pytest.raises(ValueError, match="must include a 'dist' key"):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic="n",
                priors={"alpha": {"mu": 0.0, "sigma": 1.0}},
            )

    def test_non_dict_prior_value_raises(self, design_k1):
        with pytest.raises(TypeError, match="must be a dict"):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic="n",
                priors={"alpha": "Normal"},
            )

    def test_non_dict_sigma_value_raises(self, design_k1):
        with pytest.raises(TypeError, match="must be a dict"):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic="n",
                priors={"Sigma": 1.0},
            )

    def test_unknown_sigma_subkey_raises(self, design_k1):
        with pytest.raises(ValueError, match="unknown Sigma-prior key"):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic="n",
                priors={"Sigma": {"eta": 2.0, "alpha": 1.0}},
            )

    def test_non_dict_priors_argument_raises(self, design_k1):
        with pytest.raises(TypeError, match="priors must be a dict"):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic="n",
                priors="not a dict",  # type: ignore[arg-type]
            )
