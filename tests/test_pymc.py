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
# Deterministic terms
# ---------------------------------------------------------------------------


class TestDeterministicTerms:
    """Graph-construction tests for all five deterministic codes x r=1,2.

    No sampler is run — we use pm.draw to inspect parameter shapes.
    """

    # --- builds for all codes ------------------------------------------------

    @pytest.mark.parametrize("code", ["co", "ci", "lo", "li"])
    def test_builds_for_all_codes_r1(self, code):
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic=code)
        assert isinstance(model, pm.Model)

    @pytest.mark.parametrize("code", ["n", "co", "ci", "lo", "li"])
    def test_builds_for_all_codes_r2(self, code):
        data = _synthetic_trivariate_r2()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=2, deterministic=code)
        assert isinstance(model, pm.Model)

    # --- outside terms: Gamma widens, beta stays (K, r) ---------------------

    @pytest.mark.parametrize("code", ["co", "lo"])
    def test_outside_term_widens_gamma(self, code):
        """K=2, k=1: Gamma normally (2, 2), with outside term (2, 3)."""
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["Gamma"], draws=1, random_seed=0)
        assert value.shape == (2, 3)  # K=2, K*k + 1 = 3

    @pytest.mark.parametrize("code", ["co", "lo"])
    def test_outside_term_beta_shape_unchanged(self, code):
        """Outside terms only affect delta_x; beta shape stays (K, r)."""
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        assert value.shape == (2, 1)  # K=2, r=1

    # --- inside terms: beta widens, Gamma stays (K, K*k) --------------------

    @pytest.mark.parametrize("code", ["ci", "li"])
    def test_inside_term_widens_beta(self, code):
        """K=2, r=1: beta normally (2, 1), with inside term (3, 1)."""
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        assert value.shape == (3, 1)  # K + 1 = 3, r=1

    @pytest.mark.parametrize("code", ["ci", "li"])
    def test_inside_term_beta_pin_preserved(self, code):
        """The Johansen normalisation beta[0, 0] == 1 must hold for inside terms."""
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        np.testing.assert_array_equal(value[0, 0], 1.0)

    @pytest.mark.parametrize("code", ["ci", "li"])
    def test_inside_term_gamma_shape_unchanged(self, code):
        """Inside terms only affect y_lag1; Gamma shape stays (K, K*k)."""
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["Gamma"], draws=1, random_seed=0)
        assert value.shape == (2, 2)  # K=2, K*k=2

    # --- alpha shape is always (K, r) for all codes --------------------------

    @pytest.mark.parametrize("code", ["n", "co", "ci", "lo", "li"])
    def test_alpha_shape_always_k_r(self, code):
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=1, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["alpha"], draws=1, random_seed=0)
        assert value.shape == (2, 1)  # K=2, r=1

    # --- k=0 + outside term adds Gamma (edge case) ---------------------------

    def test_outside_term_k0_adds_gamma(self):
        """k_ar_diff=0 with 'co': delta_x gets one outside-term column.

        Without this case Gamma would be absent (pure EC model); the outside
        term changes that because delta_x_cols = 1 > 0.
        """
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=0, deterministic="co")
        model = build_pymc_model(design, k_ar_diff=0, coint_rank=1, deterministic="co")
        assert "Gamma" in set(model.named_vars)
        with model:
            value = pm.draw(model.named_vars["Gamma"], draws=1, random_seed=0)
        assert value.shape == (2, 1)  # K=2, delta_x_cols=1

    # --- r=2 x all codes: shapes consistent ----------------------------------

    @pytest.mark.parametrize("code", ["n", "co", "ci", "lo", "li"])
    def test_r2_alpha_shape(self, code):
        data = _synthetic_trivariate_r2()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=2, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["alpha"], draws=1, random_seed=0)
        assert value.shape == (3, 2)  # K=3, r=2

    @pytest.mark.parametrize("code", ["n", "co", "lo"])
    def test_r2_beta_shape_outside_or_none(self, code):
        """Outside/no-term codes: beta is (K, r) = (3, 2)."""
        data = _synthetic_trivariate_r2()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=2, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        assert value.shape == (3, 2)

    @pytest.mark.parametrize("code", ["ci", "li"])
    def test_r2_beta_shape_inside(self, code):
        """Inside-term codes: beta is (K+1, r) = (4, 2)."""
        data = _synthetic_trivariate_r2()
        design = cointegration_design(data, k_ar_diff=1, deterministic=code)
        model = build_pymc_model(design, k_ar_diff=1, coint_rank=2, deterministic=code)
        with model:
            value = pm.draw(model.named_vars["beta"], draws=1, random_seed=0)
        assert value.shape == (4, 2)


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


# ---------------------------------------------------------------------------
# Regularised horseshoe prior on Gamma
# ---------------------------------------------------------------------------


class TestHorseshoePrior:
    """Graph-construction tests for priors={"Gamma": {"dist": "Horseshoe"}}.

    No sampler is run — pm.draw is used to inspect shapes and finiteness.
    """

    def test_horseshoe_builds(self, design_k1):
        """Basic smoke test: model builds without error."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        assert isinstance(model, pm.Model)

    def test_horseshoe_auxiliary_vars_present(self, design_k1):
        """Gamma_tau, Gamma_lambda, Gamma_c2 must be in the model."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        names = set(model.named_vars)
        assert "Gamma_tau" in names
        assert "Gamma_lambda" in names
        assert "Gamma_c2" in names
        assert "Gamma" in names

    def test_horseshoe_gamma_shape(self, design_k1):
        """Gamma shape must be (K, K*k_ar_diff) = (2, 2) for the bivariate k=1 design."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        with model:
            value = pm.draw(model.named_vars["Gamma"], draws=1, random_seed=0)
        assert value.shape == (2, 2)

    def test_horseshoe_lambda_shape_matches_gamma(self, design_k1):
        """Gamma_lambda must have the same shape as Gamma."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        with model:
            gamma_val = pm.draw(model.named_vars["Gamma"], draws=1, random_seed=0)
            lam_val = pm.draw(model.named_vars["Gamma_lambda"], draws=1, random_seed=0)
        assert lam_val.shape == gamma_val.shape

    def test_horseshoe_tau_is_scalar(self, design_k1):
        """Gamma_tau is a scalar (global shrinkage — one value for all entries)."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        with model:
            tau_val = pm.draw(model.named_vars["Gamma_tau"], draws=1, random_seed=0)
        assert tau_val.shape == ()

    def test_horseshoe_draws_are_finite(self, design_k1):
        """Prior draws from Gamma under horseshoe must all be finite."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        with model:
            values = pm.draw(model.named_vars["Gamma"], draws=50, random_seed=0)
        assert np.all(np.isfinite(values))

    def test_horseshoe_tau_scale_override(self, design_k1):
        """Custom tau_scale is accepted and model builds without error."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe", "tau_scale": 0.1}},
        )
        assert isinstance(model, pm.Model)

    def test_horseshoe_slab_params_override(self, design_k1):
        """Custom slab_scale and slab_df are accepted."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe", "slab_scale": 3.0, "slab_df": 6.0}},
        )
        assert isinstance(model, pm.Model)

    def test_horseshoe_wider_gamma_outside_term(self):
        """Horseshoe applies to widened Gamma when deterministic='co'."""
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=1, deterministic="co")
        model = build_pymc_model(
            design,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="co",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        with model:
            value = pm.draw(model.named_vars["Gamma"], draws=1, random_seed=0)
        # K=2, delta_x_cols = K*k + 1 = 3
        assert value.shape == (2, 3)

    def test_horseshoe_k0_silently_no_gamma(self):
        """With k=0 and no outside term there is no Gamma block; horseshoe spec ignored."""
        data = _synthetic_cointegrated()
        design = cointegration_design(data, k_ar_diff=0)
        model = build_pymc_model(
            design,
            k_ar_diff=0,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        names = set(model.named_vars)
        assert "Gamma" not in names
        assert "Gamma_tau" not in names

    def test_horseshoe_unknown_kwarg_raises(self, design_k1):
        with pytest.raises(ValueError, match="unknown horseshoe prior key"):
            build_pymc_model(
                design_k1,
                k_ar_diff=1,
                coint_rank=1,
                deterministic="n",
                priors={"Gamma": {"dist": "Horseshoe", "bad_key": 1.0}},
            )

    def test_horseshoe_other_params_unchanged(self, design_k1):
        """When horseshoe is active, alpha and beta are still in the model."""
        model = build_pymc_model(
            design_k1,
            k_ar_diff=1,
            coint_rank=1,
            deterministic="n",
            priors={"Gamma": {"dist": "Horseshoe"}},
        )
        names = set(model.named_vars)
        assert "alpha" in names
        assert "beta" in names
        assert "Sigma" in names
