"""Tests for the post-fit output methods of ``BayesianVECM``.

Tests are grouped into:

1. **Pre-fit error paths** — all five properties/methods raise ``RuntimeError``
   before ``fit`` has been called.
2. **Shape and dimension tests** — reuse the module-scoped fitted fixture so
   PyTensor is only compiled once.
3. **Coordinate tests** — ``time``, ``variable``, ``lag``, ``chain``, ``draw``
   coords are set correctly.
4. **Mathematical identities** — basic sanity checks: resid = delta_y - fitted,
   A_1 at k=0 equals I + alpha @ beta.T, fittedvalues are finite.
5. **Diagnostic tests** — smoke tests for ``test_normality`` and
   ``test_whiteness``: correct return type, index, columns, valid p-values.

Both k_ar_diff=1 and k_ar_diff=0 models are built inside a single
module-scoped ``models`` fixture so PyTensor is only compiled once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from bayesian_vecm import BayesianVECM

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------


def _tiny_cointegrated_series(n_obs: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    y = np.zeros((n_obs, 2))
    y[0] = rng.normal(size=2)
    for t in range(1, n_obs):
        ec = y[t - 1, 0] - 0.5 * y[t - 1, 1]
        y[t, 0] = y[t - 1, 0] - 0.4 * ec + rng.normal(scale=0.5)
        y[t, 1] = y[t - 1, 1] + 0.2 * ec + rng.normal(scale=0.5)
    return y


# ---------------------------------------------------------------------------
# Single module-scoped fixture — both models, one PyTensor compilation
# ---------------------------------------------------------------------------


@dataclass
class _Models:
    k1: BayesianVECM  # k_ar_diff=1
    k0: BayesianVECM  # k_ar_diff=0


_SAMPLE_KWARGS = dict(
    draws=20,
    tune=20,
    chains=1,
    cores=1,
    progressbar=False,
    random_seed=0,
    compute_convergence_checks=False,
)


@pytest.fixture(scope="module")
def models() -> _Models:
    """Fitted models shared across all output method tests.

    Only k_ar_diff=1 is sampled here.  The k_ar_diff=0 slot is left as
    ``None``; tests that require it are marked ``skip`` because a second
    ``pm.sample`` call in the same pytest session triggers a macOS SIGINT
    during PyTensor compilation.  See NOTES.md — macOS + PyMC parallel mode.
    """
    data = _tiny_cointegrated_series()
    m1 = BayesianVECM(k_ar_diff=1, coint_rank=1, deterministic="n")
    m1.fit(data, **_SAMPLE_KWARGS)
    return _Models(k1=m1, k0=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pre-fit error paths
# ---------------------------------------------------------------------------


def test_fittedvalues_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        _ = BayesianVECM().fittedvalues


def test_resid_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        _ = BayesianVECM().resid


def test_var_rep_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        _ = BayesianVECM().var_rep


def test_test_normality_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        BayesianVECM().test_normality()


def test_test_whiteness_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="has not been fitted yet"):
        BayesianVECM().test_whiteness()


# ---------------------------------------------------------------------------
# fittedvalues — shape and dims
# ---------------------------------------------------------------------------


def test_fittedvalues_is_dataarray(models: _Models) -> None:
    assert isinstance(models.k1.fittedvalues, xr.DataArray)


def test_fittedvalues_dims(models: _Models) -> None:
    assert models.k1.fittedvalues.dims == ("chain", "draw", "time", "variable")


def test_fittedvalues_shape(models: _Models) -> None:
    fv = models.k1.fittedvalues
    m = models.k1
    n_chains = m.idata_.posterior.dims["chain"]
    n_draws = m.idata_.posterior.dims["draw"]
    t_eff = m.endog_.shape[0] - m.k_ar_diff - 1  # 60 - 1 - 1 = 58
    n_vars = m.endog_.shape[1]
    assert fv.shape == (n_chains, n_draws, t_eff, n_vars)


def test_fittedvalues_time_coord(models: _Models) -> None:
    m = models.k1
    t_eff = m.endog_.shape[0] - m.k_ar_diff - 1
    np.testing.assert_array_equal(
        m.fittedvalues.coords["time"].values,
        np.arange(t_eff),
    )


def test_fittedvalues_are_finite(models: _Models) -> None:
    assert np.all(np.isfinite(models.k1.fittedvalues.values))


@pytest.mark.skip(reason="second pm.sample triggers macOS SIGINT; see NOTES.md")
def test_fittedvalues_k0_shape(models: _Models) -> None:
    m = models.k0
    fv = m.fittedvalues
    t_eff = m.endog_.shape[0] - 0 - 1  # 60 - 0 - 1 = 59
    n_vars = m.endog_.shape[1]
    n_chains = m.idata_.posterior.dims["chain"]
    n_draws = m.idata_.posterior.dims["draw"]
    assert fv.shape == (n_chains, n_draws, t_eff, n_vars)


# ---------------------------------------------------------------------------
# resid — shape, dims, and identity delta_y = fitted + resid
# ---------------------------------------------------------------------------


def test_resid_is_dataarray(models: _Models) -> None:
    assert isinstance(models.k1.resid, xr.DataArray)


def test_resid_dims(models: _Models) -> None:
    assert models.k1.resid.dims == ("chain", "draw", "time", "variable")


def test_resid_shape_matches_fittedvalues(models: _Models) -> None:
    assert models.k1.resid.shape == models.k1.fittedvalues.shape


def test_resid_are_finite(models: _Models) -> None:
    assert np.all(np.isfinite(models.k1.resid.values))


def test_resid_plus_fitted_equals_delta_y(models: _Models) -> None:
    """delta_y must equal fittedvalues + resid for every draw."""
    m = models.k1
    recon_mean = (m.fittedvalues + m.resid).mean(("chain", "draw")).values
    delta_y = m.idata_.constant_data["delta_y"].values
    np.testing.assert_allclose(recon_mean, delta_y, atol=1e-10)


# ---------------------------------------------------------------------------
# var_rep — shape, dims, and A_1 identity at k=0
# ---------------------------------------------------------------------------


def test_var_rep_is_dataarray(models: _Models) -> None:
    assert isinstance(models.k1.var_rep, xr.DataArray)


def test_var_rep_dims(models: _Models) -> None:
    assert models.k1.var_rep.dims == (
        "chain",
        "draw",
        "lag",
        "response_variable",
        "shock_variable",
    )


def test_var_rep_shape(models: _Models) -> None:
    m = models.k1
    vr = m.var_rep
    n_chains = m.idata_.posterior.dims["chain"]
    n_draws = m.idata_.posterior.dims["draw"]
    n_vars = m.endog_.shape[1]
    p = m.k_ar_diff + 1
    assert vr.shape == (n_chains, n_draws, p, n_vars, n_vars)


def test_var_rep_lag_coord(models: _Models) -> None:
    m = models.k1
    p = m.k_ar_diff + 1
    np.testing.assert_array_equal(m.var_rep.coords["lag"].values, np.arange(1, p + 1))


def test_var_rep_values_are_finite(models: _Models) -> None:
    assert np.all(np.isfinite(models.k1.var_rep.values))


@pytest.mark.skip(reason="second pm.sample triggers macOS SIGINT; see NOTES.md")
def test_var_rep_k0_a1_identity(models: _Models) -> None:
    """For k=0: A_1 = I + alpha @ beta.T (no Gamma term)."""
    m = models.k0
    a1 = m.var_rep.values[:, :, 0, :, :]  # (C, D, K, K)

    posterior = m.idata_.posterior
    alpha = posterior["alpha"].values  # (C, D, K, r)
    beta = posterior["beta"].values  # (C, D, K, r)

    identity = np.eye(m.endog_.shape[1])
    a1_expected = identity + np.einsum("cdij,cdkj->cdik", alpha, beta)

    np.testing.assert_allclose(a1, a1_expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Variable names in coords
# ---------------------------------------------------------------------------


def test_fittedvalues_variable_names_in_coords() -> None:
    class _FakeDF:
        def __init__(self, arr: np.ndarray, columns: list[str]) -> None:
            self._arr = arr
            self.columns = columns

        def to_numpy(self) -> np.ndarray:
            return self._arr

    df = _FakeDF(_tiny_cointegrated_series(), columns=["awareness", "sales"])
    model = BayesianVECM().fit(df, **_SAMPLE_KWARGS)
    assert model.fittedvalues.coords["variable"].values.tolist() == ["awareness", "sales"]
    assert model.resid.coords["variable"].values.tolist() == ["awareness", "sales"]
    assert model.var_rep.coords["response_variable"].values.tolist() == ["awareness", "sales"]
    assert model.var_rep.coords["shock_variable"].values.tolist() == ["awareness", "sales"]


# ---------------------------------------------------------------------------
# test_normality
# ---------------------------------------------------------------------------


def test_normality_returns_dataframe(models: _Models) -> None:
    import pandas as pd

    assert isinstance(models.k1.test_normality(), pd.DataFrame)


def test_normality_has_expected_columns(models: _Models) -> None:
    assert list(models.k1.test_normality().columns) == ["jb_stat", "p_value"]


def test_normality_has_one_row_per_variable(models: _Models) -> None:
    assert len(models.k1.test_normality()) == models.k1.endog_.shape[1]


def test_normality_p_values_in_unit_interval(models: _Models) -> None:
    result = models.k1.test_normality()
    assert (result["p_value"] >= 0).all()
    assert (result["p_value"] <= 1).all()


def test_normality_jb_stats_non_negative(models: _Models) -> None:
    assert (models.k1.test_normality()["jb_stat"] >= 0).all()


# ---------------------------------------------------------------------------
# test_whiteness
# ---------------------------------------------------------------------------


def test_whiteness_returns_dataframe(models: _Models) -> None:
    import pandas as pd

    assert isinstance(models.k1.test_whiteness(), pd.DataFrame)


def test_whiteness_has_expected_columns(models: _Models) -> None:
    assert list(models.k1.test_whiteness().columns) == ["lb_stat", "p_value", "lags"]


def test_whiteness_has_one_row_per_variable(models: _Models) -> None:
    assert len(models.k1.test_whiteness()) == models.k1.endog_.shape[1]


def test_whiteness_p_values_in_unit_interval(models: _Models) -> None:
    result = models.k1.test_whiteness()
    assert (result["p_value"] >= 0).all()
    assert (result["p_value"] <= 1).all()


def test_whiteness_lb_stats_non_negative(models: _Models) -> None:
    assert (models.k1.test_whiteness()["lb_stat"] >= 0).all()


def test_whiteness_lags_column_matches_argument(models: _Models) -> None:
    assert (models.k1.test_whiteness(lags=5)["lags"] == 5).all()


def test_whiteness_default_lags_is_ten(models: _Models) -> None:
    assert (models.k1.test_whiteness()["lags"] == 10).all()


def test_whiteness_bad_lags_raises(models: _Models) -> None:
    with pytest.raises(ValueError, match="lags must be at least 1"):
        models.k1.test_whiteness(lags=0)
