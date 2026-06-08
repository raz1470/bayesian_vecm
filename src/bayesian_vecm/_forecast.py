"""Posterior predictive forecasting for ``BayesianVECM``.

Implements the VAR-level recursion:

.. math::

    \\Delta y_{T+h} = \\alpha \\beta' y_{T+h-1}
                     + \\sum_{i=1}^{k} \\Gamma_i \\, \\Delta y_{T+h-i}
                     + \\varepsilon_{T+h},
                     \\quad \\varepsilon_{T+h} \\sim \\mathcal{N}(0, \\Sigma)

    y_{T+h} = y_{T+h-1} + \\Delta y_{T+h}

for :math:`h = 1, \\dots, \\text{steps}`, seeded from the last
:math:`k_{\\text{ar\\_diff}} + 1` rows of the fitted ``endog``.

All posterior draws are processed simultaneously — the (chain x draw)
dimension is vectorised so only a single Python loop over forecast steps is
needed, keeping the inner body a handful of NumPy einsum calls.

Design choices
--------------
* **NumPy, not PyTensor.** The recursion has sequential data dependence
  (step h needs the output of step h-1), which rules out the standard
  PyMC ``pm.sample_posterior_predictive`` scan idiom. A NumPy loop over
  steps, fully vectorised over draws, is readable and fast enough for any
  reasonable ``steps`` count.
* **Levels and differences both returned.** ``y`` (levels) is the
  headline forecast; ``delta_y`` (differences) is free to compute and
  useful for diagnostic plots.
* **Innovations drawn from the posterior Cholesky.** The model stores
  ``Sigma`` (the full covariance) as a ``pm.Deterministic``; we
  re-Cholesky it once per fitted model, not once per step, so the
  per-step cost is just a matrix-vector product.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def forecast_posterior(
    idata: xr.DataTree,
    endog: NDArray[np.floating],
    k_ar_diff: int,
    steps: int,
    variable_names: list[str] | None = None,
    exog_future: NDArray[np.floating] | None = None,
    rng: np.random.Generator | None = None,
) -> xr.DataTree:
    """Roll the VECM recursion forward *steps* periods for every posterior draw.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``. Must contain a
        ``posterior`` group with variables ``alpha``, ``beta``, ``Sigma``,
        and — when ``k_ar_diff > 0`` — ``Gamma``, and — when fitted with
        exog — ``B``.
    endog
        Training data, shape ``(T, K)``. The last ``k_ar_diff + 1`` rows
        seed the recursion window.
    k_ar_diff
        Number of lagged-difference blocks in the model. ``0`` means a
        pure error-correction model with no short-run dynamics.
    steps
        Number of periods to forecast. Must be at least ``1``.
    variable_names
        Optional variable labels. When provided, added as the ``variable``
        coordinate on both output arrays.
    exog_future
        Future exogenous regressors, shape ``(steps, m)``, or ``None`` when
        the model was fitted without exog.  When provided, ``B @ X_{T+h}``
        is added to the mean at each forecast step.
    rng
        NumPy random generator. Pass a seeded generator for reproducibility;
        defaults to a fresh ``np.random.default_rng()`` if ``None``.

    Returns
    -------
    xarray.DataTree
        A DataTree whose ``posterior_predictive`` child node
        contains:

        * ``y`` — forecast levels, shape ``(chain, draw, steps, K)``.
        * ``delta_y`` — forecast first differences, same shape.

        Both arrays share dims ``("chain", "draw", "forecast_step",
        "variable")`` with a ``forecast_step`` coordinate running from
        ``1`` to ``steps``.

    Notes
    -----
    The lag-major column ordering of ``delta_x`` inside the recursion
    matches the ordering produced by
    :func:`bayesian_vecm._design.cointegration_design`: lag-1 differences
    occupy the first ``K`` columns, lag-2 the next ``K`` columns, and so
    on. Reversing ``np.diff(y_window, axis=1)`` along the lag axis produces
    that ordering directly.
    """
    if rng is None:
        rng = np.random.default_rng()

    posterior = idata.posterior

    # --- Extract posterior draws -------------------------------------------
    # ArviZ stores posteriors as (chain, draw, ...) xarray DataArrays;
    # .values gives us plain NumPy arrays.
    alpha_draws: NDArray[np.floating] = posterior["alpha"].values  # (C, D, K, r)
    beta_draws: NDArray[np.floating] = posterior["beta"].values  # (C, D, K, r)
    sigma_draws: NDArray[np.floating] = posterior["Sigma"].values  # (C, D, K, K)

    n_chains, n_draws, n_vars, r = alpha_draws.shape
    n_total = n_chains * n_draws  # D = total number of draws

    has_gamma = k_ar_diff > 0
    if has_gamma:
        gamma_draws: NDArray[np.floating] = posterior["Gamma"].values  # (C, D, K, Kk+?)

    has_exog = exog_future is not None and "B" in posterior
    if has_exog:
        b_draws: NDArray[np.floating] = posterior["B"].values  # (C, D, K, m)

    # --- Reshape to (D, ...) for fully-vectorised per-step ops -------------
    alpha = alpha_draws.reshape(n_total, n_vars, r)  # (D, K, r)
    beta = beta_draws[:, :, :n_vars, :].reshape(n_total, n_vars, r)  # (D, K, r)
    sigma = sigma_draws.reshape(n_total, n_vars, n_vars)  # (D, K, K)
    if has_gamma:
        # Slice to K*k_ar_diff dynamic columns — strips any outside
        # deterministic column appended by cointegration_design.
        gamma = gamma_draws.reshape(n_total, n_vars, -1)[:, :, : n_vars * k_ar_diff]  # (D, K, K*k)
    if has_exog:
        b_mat = b_draws.reshape(n_total, n_vars, -1)  # (D, K, m)

    # Pre-compute the Cholesky factor of Sigma once — reused at every step.
    # numpy.linalg.cholesky broadcasts over leading batch dims in NumPy >= 2.0;
    # for safety we compute it in a loop over draws (still O(K^3 * D), same
    # cost) so it works on older NumPy too.
    # Actually, np.linalg.cholesky does support batched input as of NumPy 1.14,
    # and our floor is numpy>=1.26, so the batched call is fine.
    chol = np.linalg.cholesky(sigma)  # (D, K, K), lower-triangular

    # --- Seed the recursion window -----------------------------------------
    # We need the last k_ar_diff + 1 rows of endog to compute:
    #   y_lag1 = y_{T-1}                     (the lagged level)
    #   dy_{T-1}, ..., dy_{T-k}              (the lagged differences)
    # For k_ar_diff = 0 the window is just the last row (size 1).
    seed_rows = k_ar_diff + 1
    seed = endog[-seed_rows:].copy()  # (k+1, K)

    # Replicate across draws — each draw starts from the same empirical window.
    # Shape: (D, k+1, K).
    y_window = np.tile(seed[np.newaxis, :, :], (n_total, 1, 1))

    # --- Storage -----------------------------------------------------------
    y_forecast = np.empty((n_total, steps, n_vars), dtype=np.float64)
    dy_forecast = np.empty((n_total, steps, n_vars), dtype=np.float64)

    # --- Forecast loop (sequential over h; vectorised over D) --------------
    for h in range(steps):
        # y_{T+h-1}: the most-recent level in the window.
        y_prev = y_window[:, -1, :]  # (D, K)

        # Error-correction term: alpha beta' y_{T+h-1}
        #   y_prev @ beta  -> (D, r)   [contract K]
        #   result @ alpha' -> (D, K)  [contract r; alpha' has shape (D, r, K)]
        ec = np.einsum("di,dij->dj", y_prev, beta)  # (D, r)
        mu = np.einsum("dj,dkj->dk", ec, alpha)  # (D, K)

        # Short-run dynamics: Gamma * delta_x
        # np.diff(y_window, axis=1) -> (D, k, K), row 0 = dy_{T+h-k}, last = dy_{T+h-1}
        # Reversed so row 0 = dy_{T+h-1} (lag 1), matching lag-major column order.
        if has_gamma:
            diffs = np.diff(y_window, axis=1)[:, ::-1, :]  # (D, k, K)
            delta_x_h = diffs.reshape(n_total, -1)  # (D, K*k), lag-major
            # delta_x @ Gamma':  Gamma shape (D, K, Kk) -> Gamma' shape (D, Kk, K)
            mu = mu + np.einsum("di,dji->dj", delta_x_h, gamma)  # (D, K)

        # Contemporaneous exog effect: B X_{T+h}
        # exog_future[h] shape (m,); b_mat shape (D, K, m)
        if has_exog:
            x_h = exog_future[h]  # (m,)
            mu = mu + np.einsum("dkm,m->dk", b_mat, x_h)  # (D, K)

        # Innovation: eps ~ MvNormal(0, Sigma) via pre-computed Cholesky
        z = rng.standard_normal((n_total, n_vars))  # (D, K)
        eps = np.einsum("dij,dj->di", chol, z)  # (D, K)

        delta_y_h = mu + eps  # (D, K)
        y_h = y_prev + delta_y_h  # (D, K)

        y_forecast[:, h, :] = y_h
        dy_forecast[:, h, :] = delta_y_h

        # Slide the window: drop the oldest row, append the new level.
        y_window = np.concatenate(
            [y_window[:, 1:, :], y_h[:, np.newaxis, :]], axis=1
        )  # (D, k+1, K)

    # --- Reshape back to (chain, draw, steps, K) ---------------------------
    y_out = y_forecast.reshape(n_chains, n_draws, steps, n_vars)
    dy_out = dy_forecast.reshape(n_chains, n_draws, steps, n_vars)

    # --- Build the xarray Dataset for ArviZ --------------------------------
    dims = ["chain", "draw", "forecast_step", "variable"]

    # Use the same chain/draw coords as the fitted posterior.
    chain_coords = posterior.coords["chain"].values
    draw_coords = posterior.coords["draw"].values
    forecast_step_coords = np.arange(1, steps + 1)

    coords: dict[str, Any] = {
        "chain": chain_coords,
        "draw": draw_coords,
        "forecast_step": forecast_step_coords,
    }
    if variable_names is not None:
        coords["variable"] = variable_names

    pp_ds = xr.Dataset(
        {
            "y": xr.DataArray(y_out, dims=dims, coords=coords),
            "delta_y": xr.DataArray(dy_out, dims=dims, coords=coords),
        }
    )

    # ArviZ 1.x dropped the InferenceData(**group_kwargs) constructor in favour
    # of xarray's DataTree. Build the result as a DataTree with a
    # "posterior_predictive" child node, which is what arviz.InferenceData is
    # now an alias for.
    return xr.DataTree.from_dict({"posterior_predictive": pp_ds})
