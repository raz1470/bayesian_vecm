"""In-sample output methods for ``BayesianVECM``.

Three quantities are provided, all as posterior distributions over draws:

``fittedvalues``
    In-sample fitted first-differences
    :math:`\\hat{\\Delta y}_t = \\alpha \\beta^\\top y_{t-1} + \\Gamma \\Delta x_t`.
    Shape ``(chain, draw, T_eff, K)``.

``resid``
    In-sample residuals :math:`\\hat{\\varepsilon}_t = \\Delta y_t - \\hat{\\Delta y}_t`.
    Same shape as ``fittedvalues``.

``var_rep``
    Levels VAR(:math:`p`) coefficient matrices :math:`A_1, \\dots, A_p`
    reconstructed from VECM parameters via the standard conversion:

    .. math::

        A_1 &= I_K + \\alpha\\beta^\\top + \\Gamma_1 \\\\
        A_j &= \\Gamma_j - \\Gamma_{j-1}, \\quad j = 2, \\dots, k \\\\
        A_{k+1} &= -\\Gamma_k

    Shape ``(chain, draw, lag, response_variable, shock_variable)`` where
    ``lag`` runs from ``1`` to ``p = k_{\\text{ar\\_diff}} + 1``.

Design notes
------------
* **Full posterior, not just the mean.** Returning
  ``(chain, draw, T_eff, K)`` DataArrays lets callers build uncertainty bands
  with ``da.mean(("chain", "draw"))`` for the point estimate, or
  ``az.hdi(da)`` for credible intervals.

* **Design matrices read from** ``idata.constant_data``.  ``build_pymc_model``
  stashes ``delta_y``, ``y_lag1``, and (when present) ``delta_x`` as
  ``pm.Data`` nodes, so no ``endog`` or ``CointegrationDesign`` re-derivation
  is needed here.

* **Inside deterministic terms and** ``var_rep``.  When ``deterministic="ci"``
  or ``"li"``, the fitted ``beta`` in ``idata.posterior`` has shape
  ``(chain, draw, K+1, r)`` — the extra row is the constant/trend loading
  *inside* the cointegrating relation.  For ``var_rep`` only the first ``K``
  rows enter the levels VAR conversion (the extra row becomes a deterministic
  intercept/trend in the VAR, not a lagged coefficient).  ``fittedvalues`` and
  ``resid`` use the full ``beta`` as fitted because that is the actual model mean.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def compute_fittedvalues(
    idata: xr.DataTree,
    k_ar_diff: int,
    variable_names: list[str] | None = None,
) -> xr.DataArray:
    """Compute posterior in-sample fitted values of :math:`\\Delta y`.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``.  Must have ``posterior``
        (``alpha``, ``beta``, optionally ``Gamma``) and ``constant_data``
        (``y_lag1``, optionally ``delta_x``).
    k_ar_diff
        Number of lagged-difference blocks in the model.
    variable_names
        Optional variable labels added as the ``variable`` coordinate.

    Returns
    -------
    xarray.DataArray
        Shape ``(chain, draw, time, variable)``.  ``time`` is zero-indexed
        over the :math:`T_{\\text{eff}}` in-sample observations.
    """
    posterior = idata.posterior
    const_data = idata.constant_data

    alpha_draws: NDArray[np.floating] = posterior["alpha"].values  # (C, D, K, r)
    beta_draws: NDArray[np.floating] = posterior["beta"].values  # (C, D, y_lag1_cols, r)
    n_chains, n_draws, n_vars, r = alpha_draws.shape
    n_total = n_chains * n_draws

    y_lag1: NDArray[np.floating] = const_data["y_lag1"].values  # (T_eff, y_lag1_cols)
    t_eff = y_lag1.shape[0]

    alpha = alpha_draws.reshape(n_total, n_vars, r)
    beta = beta_draws.reshape(n_total, -1, r)  # (D, y_lag1_cols, r)

    # EC term: mu_ec[d, t, k] = alpha[d,k,:] @ beta[d,:,:].T @ y_lag1[t,:]
    # Step 1: ec[d, t, j] = sum_i y_lag1[t,i] * beta[d,i,j]
    ec = np.einsum("ti,dij->dtj", y_lag1, beta)  # (D, T_eff, r)
    # Step 2: mu[d, t, k] = sum_j ec[d,t,j] * alpha[d,k,j]
    mu = np.einsum("dtj,dkj->dtk", ec, alpha)  # (D, T_eff, K)

    if "delta_x" in const_data:
        delta_x: NDArray[np.floating] = const_data["delta_x"].values  # (T_eff, delta_x_cols)
        gamma_draws: NDArray[np.floating] = posterior["Gamma"].values  # (C, D, K, delta_x_cols)
        gamma = gamma_draws.reshape(n_total, n_vars, -1)  # (D, K, delta_x_cols)
        # gamma_contrib[d, t, k] = sum_i delta_x[t,i] * gamma[d,k,i]
        mu = mu + np.einsum("ti,dki->dtk", delta_x, gamma)  # (D, T_eff, K)

    if "exog" in const_data and "B" in posterior:
        exog_vals: NDArray[np.floating] = const_data["exog"].values  # (T_eff, m)
        b_draws: NDArray[np.floating] = posterior["B"].values  # (C, D, K, m)
        b_mat = b_draws.reshape(n_total, n_vars, -1)  # (D, K, m)
        # exog_contrib[d, t, k] = sum_i exog[t,i] * B[d,k,i]
        mu = mu + np.einsum("ti,dki->dtk", exog_vals, b_mat)  # (D, T_eff, K)

    mu_out = mu.reshape(n_chains, n_draws, t_eff, n_vars)

    chain_c = posterior.coords["chain"].values
    draw_c = posterior.coords["draw"].values
    coords: dict = {
        "chain": chain_c,
        "draw": draw_c,
        "time": np.arange(t_eff),
    }
    if variable_names is not None:
        coords["variable"] = variable_names

    return xr.DataArray(
        mu_out,
        dims=["chain", "draw", "time", "variable"],
        coords=coords,
    )


def compute_resid(
    idata: xr.DataTree,
    k_ar_diff: int,
    variable_names: list[str] | None = None,
) -> xr.DataArray:
    """Compute posterior in-sample residuals :math:`\\Delta y - \\hat{\\Delta y}`.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``.
    k_ar_diff
        Number of lagged-difference blocks in the model.
    variable_names
        Optional variable labels added as the ``variable`` coordinate.

    Returns
    -------
    xarray.DataArray
        Shape ``(chain, draw, time, variable)``.
    """
    fitted = compute_fittedvalues(idata, k_ar_diff, variable_names=variable_names)

    delta_y: NDArray[np.floating] = idata.constant_data["delta_y"].values  # (T_eff, K)

    # Broadcast delta_y to (chain, draw, T_eff, K) and subtract fitted values.
    # xarray handles the broadcast automatically via matching dim names.
    delta_y_da = xr.DataArray(
        delta_y,
        dims=["time", "variable"],
        coords={
            "time": fitted.coords["time"],
            **({"variable": fitted.coords["variable"]} if variable_names is not None else {}),
        },
    )
    # Subtract in this order so dims follow fitted's ordering (chain, draw, time, variable).
    # delta_y_da - fitted would put (time, variable) first due to xarray's broadcast rules.
    return -(fitted - delta_y_da)


def compute_var_rep(
    idata: xr.DataTree,
    k_ar_diff: int,
    variable_names: list[str] | None = None,
) -> xr.DataArray:
    """Compute posterior levels VAR(:math:`p`) coefficient matrices.

    Reconstructs :math:`A_1, \\dots, A_p` from the VECM parameters using the
    standard VECM-to-VAR(p) conversion with :math:`p = k_{\\text{ar\\_diff}} + 1`.

    .. note::

        For inside deterministic terms (``"ci"``, ``"li"``), ``beta`` in the
        fitted posterior has shape ``(K+1, r)`` — the extra row is the
        constant/trend inside the cointegrating relation.  Only the first ``K``
        rows enter the levels VAR conversion; the constant/trend becomes a
        deterministic intercept/trend in the VAR representation.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``.
    k_ar_diff
        Number of lagged-difference blocks in the model.
    variable_names
        Optional variable labels added as ``response_variable`` and
        ``shock_variable`` coordinates.

    Returns
    -------
    xarray.DataArray
        Shape ``(chain, draw, lag, response_variable, shock_variable)``.
        ``lag`` coordinate runs from ``1`` to ``p = k_{\\text{ar\\_diff}} + 1``.
        Entry ``[..., j, i, k]`` is the ``(i, k)`` element of :math:`A_j`.
    """
    posterior = idata.posterior

    alpha_draws: NDArray[np.floating] = posterior["alpha"].values  # (C, D, K, r)
    beta_draws: NDArray[np.floating] = posterior["beta"].values  # (C, D, y_lag1_cols, r)
    n_chains, n_draws, n_vars, r = alpha_draws.shape
    n_total = n_chains * n_draws
    var_order = k_ar_diff + 1  # VAR(p) order

    alpha = alpha_draws.reshape(n_total, n_vars, r)  # (D, K, r)
    # Slice to the first K rows — strips constant/trend row for inside terms.
    beta = beta_draws.reshape(n_total, -1, r)[:, :n_vars, :]  # (D, K, r)

    has_gamma = k_ar_diff > 0
    if has_gamma:
        gamma_draws: NDArray[np.floating] = posterior["Gamma"].values  # (C, D, K, Kk+?)
        # Slice to the first K*k dynamic columns — strips outside deterministic columns.
        gamma_dyn = gamma_draws.reshape(n_total, n_vars, -1)[
            :, :, : n_vars * k_ar_diff
        ]  # (D, K, K*k)

    # Build A_1 .. A_p per draw.
    a_mats = np.zeros((n_total, var_order, n_vars, n_vars))

    # A_1 = I + alpha @ beta.T  [+ Gamma_1 if k > 0]
    a1 = np.eye(n_vars) + np.einsum("dij,dkj->dik", alpha, beta)  # (D, K, K)
    if has_gamma:
        a1 = a1 + gamma_dyn[:, :, :n_vars]  # add Gamma_1 block
    a_mats[:, 0, :, :] = a1

    if has_gamma:
        for j in range(1, k_ar_diff + 1):
            g_prev = gamma_dyn[:, :, (j - 1) * n_vars : j * n_vars]  # Gamma_j
            if j < k_ar_diff:
                g_curr = gamma_dyn[:, :, j * n_vars : (j + 1) * n_vars]  # Gamma_{j+1}
                a_mats[:, j, :, :] = g_curr - g_prev
            else:
                a_mats[:, j, :, :] = -g_prev  # A_{k+1} = -Gamma_k

    a_out = a_mats.reshape(n_chains, n_draws, var_order, n_vars, n_vars)

    chain_c = posterior.coords["chain"].values
    draw_c = posterior.coords["draw"].values
    coords: dict = {
        "chain": chain_c,
        "draw": draw_c,
        "lag": np.arange(1, var_order + 1),
    }
    if variable_names is not None:
        coords["response_variable"] = variable_names
        coords["shock_variable"] = variable_names

    return xr.DataArray(
        a_out,
        dims=["chain", "draw", "lag", "response_variable", "shock_variable"],
        coords=coords,
    )
