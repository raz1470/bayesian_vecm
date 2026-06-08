"""Tests for ``bayesian_vecm._irf`` and ``BayesianVECM.irf``.

Two layers:

1. **Guard tests** — pre-fit RuntimeError, bad-args ValueError; fast, no sampling.
2. **Estimation tests** — shape, dtype, coordinate layout, and mathematical
   identities at h=0 for both methods; reuse a module-scoped fitted model so
   PyTensor is compiled only once.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from bayesian_vecm import BayesianVECM

# ---------------------------------------------------------------------------
# Shared synthetic data (same DGP as test_model.py)
# ---------------------------------------------------------------------------


def _tiny_cointegrated_series(
    n_obs: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Bivariate I(1) cointegrated series with beta = (1, -0.5), alpha = (-0.4, 0.2)."""
    rng = np.random.default_rng(seed)
    n_vars = 2
    y = np.zeros((n_obs, n_vars))
    for t in range(1, n_obs):
        ec = y[t - 1, 0] - 0.5 * y[t - 1, 1]
        y[t, 0] = y[t - 1, 0] - 0.4 * ec + rng.standard_normal()
        y[t, 1] = y[t - 1, 1] + 0.2 * ec + rng.standard_normal()
    return y


# ---------------------------------------------------------------------------
# Module-scoped fixture — pay PyTensor compile cost once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model() -> BayesianVECM:
    model = BayesianVECM(k_ar_diff=1, coint_rank=1)
    model.fit(
        _tiny_cointegrated_series(),
        draws=20,
        tune=20,
        chains=1,
        cores=1,
        progressbar=False,
        random_seed=0,
        compute_convergence_checks=False,
    )
    return model


_IRF_STEPS = 10
_K = 2
_N_CHAINS = 1
_N_DRAWS = 20


@pytest.fixture(scope="module")
def girf(fitted_model: BayesianVECM) -> xr.DataArray:
    return fitted_model.irf(steps=_IRF_STEPS, method="girf")


@pytest.fixture(scope="module")
def chol_irf(fitted_model: BayesianVECM) -> xr.DataArray:
    return fitted_model.irf(steps=_IRF_STEPS, method="cholesky")


# ---------------------------------------------------------------------------
# Guard tests — no sampling required
# ---------------------------------------------------------------------------


def test_irf_raises_before_fit() -> None:
    model = BayesianVECM()
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.irf(steps=5)


def test_irf_raises_for_steps_zero() -> None:
    model = BayesianVECM()
    with pytest.raises((RuntimeError, ValueError)):
        model.irf(steps=0)


def test_compute_irf_raises_for_bad_method(fitted_model: BayesianVECM) -> None:
    with pytest.raises(ValueError, match="method"):
        fitted_model.irf(steps=5, method="invalid")


def test_compute_irf_raises_for_steps_zero_after_fit(fitted_model: BayesianVECM) -> None:
    with pytest.raises(ValueError, match="steps"):
        fitted_model.irf(steps=0)


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_girf_shape(girf: xr.DataArray) -> None:
    assert girf.shape == (_N_CHAINS, _N_DRAWS, _IRF_STEPS + 1, _K, _K)


def test_chol_irf_shape(chol_irf: xr.DataArray) -> None:
    assert chol_irf.shape == (_N_CHAINS, _N_DRAWS, _IRF_STEPS + 1, _K, _K)


def test_girf_dtype_is_float64(girf: xr.DataArray) -> None:
    assert girf.dtype == np.float64


def test_girf_is_xarray_dataarray(girf: xr.DataArray) -> None:
    assert isinstance(girf, xr.DataArray)


# ---------------------------------------------------------------------------
# Coordinate layout
# ---------------------------------------------------------------------------


def test_girf_dims(girf: xr.DataArray) -> None:
    assert girf.dims == ("chain", "draw", "horizon", "response_variable", "shock_variable")


def test_girf_horizon_coord_runs_from_zero_to_steps(girf: xr.DataArray) -> None:
    np.testing.assert_array_equal(girf.coords["horizon"].values, np.arange(_IRF_STEPS + 1))


def test_girf_no_variable_coords_when_names_not_provided(girf: xr.DataArray) -> None:
    # fitted_model was given a plain ndarray (no column names).
    assert "response_variable" not in girf.coords or len(girf.coords["response_variable"]) == _K


def test_girf_variable_coords_set_when_names_provided(fitted_model: BayesianVECM) -> None:
    """Variable names from a DataFrame are propagated to the IRF coordinates."""
    import pandas as pd

    names = ["awareness", "sales"]
    df = pd.DataFrame(_tiny_cointegrated_series(), columns=names)
    m = BayesianVECM(k_ar_diff=1, coint_rank=1)
    m.fit(
        df,
        draws=10,
        tune=10,
        chains=1,
        cores=1,
        progressbar=False,
        random_seed=1,
        compute_convergence_checks=False,
    )
    result = m.irf(steps=4)
    np.testing.assert_array_equal(result.coords["response_variable"].values, names)
    np.testing.assert_array_equal(result.coords["shock_variable"].values, names)


# ---------------------------------------------------------------------------
# Finite values
# ---------------------------------------------------------------------------


def test_girf_all_finite(girf: xr.DataArray) -> None:
    assert np.all(np.isfinite(girf.values))


def test_chol_irf_all_finite(chol_irf: xr.DataArray) -> None:
    assert np.all(np.isfinite(chol_irf.values))


# ---------------------------------------------------------------------------
# Mathematical identities
# ---------------------------------------------------------------------------


def test_girf_h0_equals_sigma_scaled_by_diag_sqrt(fitted_model: BayesianVECM) -> None:
    """At h=0, Phi_0 = I_K, so GIRF_0 = Sigma @ diag(Sigma)^{-1/2}."""
    girf = fitted_model.irf(steps=4, method="girf")
    sigma = fitted_model.idata_.posterior["Sigma"].values  # (C, D, K, K)

    sigma_diag_sqrt = np.sqrt(np.diagonal(sigma, axis1=2, axis2=3))  # (C, D, K)
    # expected[c, d, i, j] = Sigma[c,d,i,j] / sqrt(Sigma[c,d,j,j])
    expected = sigma / sigma_diag_sqrt[:, :, np.newaxis, :]  # (C, D, K, K)

    np.testing.assert_allclose(girf.values[:, :, 0, :, :], expected, rtol=1e-6)


def test_chol_irf_h0_equals_cholesky_of_sigma(fitted_model: BayesianVECM) -> None:
    """At h=0, Phi_0 = I_K, so OIR_0 = P (lower Cholesky of Sigma)."""
    chol_irf = fitted_model.irf(steps=4, method="cholesky")
    sigma = fitted_model.idata_.posterior["Sigma"].values  # (C, D, K, K)

    # np.linalg.cholesky broadcasts over batch dims.
    p_chol = np.linalg.cholesky(sigma)  # (C, D, K, K)

    np.testing.assert_allclose(chol_irf.values[:, :, 0, :, :], p_chol, rtol=1e-6)


def test_chol_irf_h0_is_lower_triangular(fitted_model: BayesianVECM) -> None:
    """Cholesky OIR at h=0 must have zeros above the main diagonal."""
    chol_irf = fitted_model.irf(steps=4, method="cholesky")
    h0 = chol_irf.values[:, :, 0, :, :]  # (C, D, K, K)
    # Upper triangle (above diagonal) should be zero for all draws.
    for i in range(_K):
        for j in range(i + 1, _K):
            np.testing.assert_allclose(h0[:, :, i, j], 0.0, atol=1e-10)


def test_girf_own_shock_at_h0_is_positive(fitted_model: BayesianVECM) -> None:
    """GIRF_0[i,i] = sqrt(Sigma[i,i]) > 0 — own-variable impact is positive."""
    girf = fitted_model.irf(steps=4, method="girf")
    sigma = fitted_model.idata_.posterior["Sigma"].values  # (C, D, K, K)
    sigma_diag_sqrt = np.sqrt(np.diagonal(sigma, axis1=2, axis2=3))  # (C, D, K)

    # GIRF_0[i, i] = sigma[i,i] / sqrt(sigma[i,i]) = sqrt(sigma[i,i])
    for i in range(_K):
        np.testing.assert_allclose(girf.values[:, :, 0, i, i], sigma_diag_sqrt[:, :, i], rtol=1e-6)
        assert np.all(girf.values[:, :, 0, i, i] > 0)


def test_irf_is_deterministic_given_same_idata(fitted_model: BayesianVECM) -> None:
    """IRFs have no stochastic component — two calls must return identical arrays."""
    result_a = fitted_model.irf(steps=6, method="girf")
    result_b = fitted_model.irf(steps=6, method="girf")
    np.testing.assert_array_equal(result_a.values, result_b.values)


# ---------------------------------------------------------------------------
# Long-run behaviour (cointegrated system)
# ---------------------------------------------------------------------------


def test_girf_long_run_response_is_nonzero(fitted_model: BayesianVECM) -> None:
    """In an I(1) cointegrated system the IRF does not decay to zero.

    Responses should still be non-trivially large at the final horizon for
    at least some draws — this distinguishes a VECM from a stationary VAR.
    """
    girf = fitted_model.irf(steps=20, method="girf")
    # Mean absolute response at the last horizon across all draws and pairs.
    last_horizon_mean_abs = np.mean(np.abs(girf.values[:, :, -1, :, :]))
    # Threshold is deliberately loose — just checking it's not numerical zero.
    assert last_horizon_mean_abs > 1e-4


def test_irf_deterministic_ci_does_not_raise() -> None:
    """irf() must not raise ValueError when fitted with deterministic='ci'.

    Regression test: beta in the posterior has shape (K+1, r) for inside
    deterministic terms; the IRF code must slice to (K, r) before reshaping.
    """
    model = BayesianVECM(k_ar_diff=1, coint_rank=1, deterministic="ci")
    model.fit(
        _tiny_cointegrated_series(),
        draws=20,
        tune=20,
        chains=1,
        cores=1,
        progressbar=False,
        random_seed=0,
        compute_convergence_checks=False,
    )
    result = model.irf(steps=5, method="girf")
    assert result.sizes["horizon"] == 6  # 0..5 inclusive
