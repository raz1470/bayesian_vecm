"""Tests for deterministic terms in the forecast recursion.

Every test here builds the posterior by hand, with one draw of known
parameters and a near-zero ``Sigma``. No sampling is involved, so the forecast
can be compared with a path computed directly from the same parameters.

The checks, for each deterministic code and several lag orders:

1. The reference path satisfies the regression that ``cointegration_design``
   sets up, with zero residual. This ties the reference to the convention the
   model is fitted under, including the numbering of the trend.
2. A forecast started part-way along the reference path reproduces the rest
   of it.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._design import cointegration_design
from bayesian_vecm._forecast import forecast_posterior

_N_VARS = 2
_N_FIT = 40
_STEPS = 12
_CODES = ["n", "co", "ci", "lo", "li"]
_LAGS = [0, 1, 2]

_ALPHA = np.array([[-0.30], [0.10]])  # (K, r)
_BETA = np.array([[1.0], [-1.0]])  # (K, r)
_GAMMA_LAGS = [
    np.array([[0.20, 0.05], [-0.10, 0.15]]),
    np.array([[0.05, -0.02], [0.03, 0.04]]),
]
# One coefficient per code. The trend coefficients are small because the
# trend regressor grows with the sample.
_OUTSIDE = {"co": np.array([0.020, -0.010]), "lo": np.array([0.0004, -0.0002])}
_INSIDE = {"ci": np.array([0.50]), "li": np.array([0.01])}


def _det_value(code: str, row: int) -> float:
    """Deterministic regressor for design row ``row`` (0-based)."""
    return float(row + 1) if code in ("lo", "li") else 1.0


def _reference_path(code: str, k_ar_diff: int, n_obs: int) -> np.ndarray:
    """Noise-free path of the VECM with the module's parameters."""
    rng = np.random.default_rng(0)
    y = np.zeros((n_obs, _N_VARS))
    seed_rows = k_ar_diff + 1
    y[:seed_rows] = 1.0 + 0.05 * rng.standard_normal((seed_rows, _N_VARS)).cumsum(axis=0)
    for t in range(seed_rows, n_obs):
        det = _det_value(code, t - seed_rows)
        ec = _BETA.T @ y[t - 1]
        if code in _INSIDE:
            ec = ec + det * _INSIDE[code]
        dy = _ALPHA @ ec
        for lag in range(1, k_ar_diff + 1):
            dy = dy + _GAMMA_LAGS[lag - 1] @ (y[t - lag] - y[t - lag - 1])
        if code in _OUTSIDE:
            dy = dy + det * _OUTSIDE[code]
        y[t] = y[t - 1] + dy
    return y


def _gamma(code: str, k_ar_diff: int) -> np.ndarray | None:
    """Gamma as the model stores it: lag blocks, then any outside term."""
    blocks = [*_GAMMA_LAGS[:k_ar_diff]]
    if code in _OUTSIDE:
        blocks.append(_OUTSIDE[code][:, np.newaxis])
    return np.column_stack(blocks) if blocks else None


def _beta(code: str) -> np.ndarray:
    """beta as the model stores it: variable rows, then any inside term."""
    if code in _INSIDE:
        return np.vstack([_BETA, _INSIDE[code][np.newaxis, :]])
    return _BETA


def _posterior(code: str, k_ar_diff: int) -> xr.DataTree:
    """One chain, one draw, near-zero innovation covariance."""

    def draw(values: np.ndarray, *dims: str) -> xr.DataArray:
        return xr.DataArray(values[np.newaxis, np.newaxis], dims=("chain", "draw", *dims))

    data = {
        "alpha": draw(_ALPHA, "a0", "a1"),
        "beta": draw(_beta(code), "b0", "b1"),
        "Sigma": draw(1e-24 * np.eye(_N_VARS), "s0", "s1"),
    }
    gamma = _gamma(code, k_ar_diff)
    if gamma is not None:
        data["Gamma"] = draw(gamma, "g0", "g1")
    posterior = xr.Dataset(data, coords={"chain": [0], "draw": [0]})
    return xr.DataTree.from_dict({"posterior": posterior})


@pytest.mark.parametrize("k_ar_diff", _LAGS)
@pytest.mark.parametrize("code", _CODES)
def test_reference_path_matches_the_fitted_design(code: str, k_ar_diff: int) -> None:
    """The reference path leaves no residual in the regression the model fits."""
    y = _reference_path(code, k_ar_diff, _N_FIT + _STEPS)
    design = cointegration_design(y, k_ar_diff=k_ar_diff, deterministic=code)

    fitted = design.y_lag1 @ _beta(code) @ _ALPHA.T
    gamma = _gamma(code, k_ar_diff)
    if gamma is not None:
        fitted = fitted + design.delta_x @ gamma.T

    np.testing.assert_allclose(design.delta_y, fitted, atol=1e-12)


@pytest.mark.parametrize("k_ar_diff", _LAGS)
@pytest.mark.parametrize("code", _CODES)
def test_forecast_continues_the_reference_path(code: str, k_ar_diff: int) -> None:
    """A forecast from the first rows reproduces the rest of the path."""
    y = _reference_path(code, k_ar_diff, _N_FIT + _STEPS)

    result = forecast_posterior(
        idata=_posterior(code, k_ar_diff),
        endog=y[:_N_FIT],
        k_ar_diff=k_ar_diff,
        steps=_STEPS,
        rng=np.random.default_rng(0),
        deterministic=code,
    )
    forecast = result["posterior_predictive"]["y"].values[0, 0]

    np.testing.assert_allclose(forecast, y[_N_FIT:], atol=1e-8)


@pytest.mark.parametrize("code", ["co", "ci", "lo", "li"])
def test_deterministic_term_changes_the_forecast(code: str) -> None:
    """Leaving the term out gives a different path, so the tests above bite."""
    y = _reference_path(code, 1, _N_FIT + _STEPS)
    kwargs = {
        "idata": _posterior(code, 1),
        "endog": y[:_N_FIT],
        "k_ar_diff": 1,
        "steps": _STEPS,
    }
    with_term = forecast_posterior(**kwargs, rng=np.random.default_rng(0), deterministic=code)
    without = forecast_posterior(**kwargs, rng=np.random.default_rng(0), deterministic="n")

    gap = with_term["posterior_predictive"]["y"] - without["posterior_predictive"]["y"]
    assert float(np.abs(gap).max()) > 1e-3


def test_model_passes_its_deterministic_code_to_the_forecast() -> None:
    """``sample_posterior_predictive`` forecasts with the model's own code."""
    y = _reference_path("co", 1, _N_FIT + _STEPS)
    model = BayesianVECM(k_ar_diff=1, coint_rank=1, deterministic="co")
    # Stand in for fit(): attach a posterior and the data it came from.
    model.idata_ = _posterior("co", 1)
    model.endog_ = y[:_N_FIT]
    model.variable_names_ = None
    model.exog_ = None

    result = model.sample_posterior_predictive(_STEPS, random_seed=0)
    forecast = result["posterior_predictive"]["y"].values[0, 0]

    np.testing.assert_allclose(forecast, y[_N_FIT:], atol=1e-8)


def test_brand_data_forecast_keeps_growing_at_the_true_rate() -> None:
    """With the generating parameters, the forecast drifts at ``truth.growth``."""
    data = simulate_brand_data()
    truth = data.truth
    n_vars = truth.alpha.shape[0]

    def draw(values: np.ndarray, *dims: str) -> xr.DataArray:
        return xr.DataArray(values[np.newaxis, np.newaxis], dims=("chain", "draw", *dims))

    posterior = xr.Dataset(
        {
            "alpha": draw(truth.alpha, "a0", "a1"),
            "beta": draw(truth.beta, "b0", "b1"),
            "Gamma": draw(np.column_stack([truth.gamma, truth.const]), "g0", "g1"),
            "B": draw(truth.b, "x0", "x1"),
            "Sigma": draw(1e-24 * np.eye(n_vars), "s0", "s1"),
        },
        coords={"chain": [0], "draw": [0]},
    )
    steps = 300

    result = forecast_posterior(
        idata=xr.DataTree.from_dict({"posterior": posterior}),
        endog=data.endog.to_numpy(),
        k_ar_diff=1,
        steps=steps,
        exog_future=np.zeros((steps, truth.b.shape[1])),
        rng=np.random.default_rng(0),
        deterministic="co",
    )
    forecast = result["posterior_predictive"]["y"].values[0, 0]

    # Once the starting gaps have closed, each series grows at its true rate.
    late_growth = (forecast[-1] - forecast[-101]) / 100
    np.testing.assert_allclose(late_growth, truth.growth, rtol=1e-3)
    # The first step is a small move, not a jump away from the data.
    first_step = forecast[0] - data.endog.to_numpy()[-1]
    assert float(np.abs(first_step).max()) < 0.05
