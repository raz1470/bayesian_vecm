"""Forecast error variance decomposition (FEVD) for ``BayesianVECM``.

The FEVD answers one question per variable: of the uncertainty in an
``h``-step-ahead forecast, what share is due to shocks in each variable?

It reuses the moving-average matrices :math:`\\Phi_s` from the IRF code. The
``h``-step forecast error covariance is

.. math::

    \\text{MSE}(h) = \\sum_{s=0}^{h-1} \\Phi_s \\, \\Sigma \\, \\Phi_s^\\top.

``"cholesky"`` (orthogonalised)
    With :math:`P` the lower Cholesky factor of :math:`\\Sigma`, the share of
    variable :math:`i`'s forecast error variance due to shock :math:`j` is

    .. math::

        \\frac{\\sum_{s=0}^{h-1} (\\Phi_s P)_{ij}^2}{\\text{MSE}(h)_{ii}}.

    Rows sum to 1 exactly. The result depends on the column order.

``"girf"`` (generalised, Pesaran & Shin 1998)
    The numerator becomes
    :math:`\\sigma_{jj}^{-1} \\sum_{s=0}^{h-1} (\\Phi_s \\Sigma)_{ij}^2`.
    This does not depend on the column order. The raw shares do not sum to 1
    when shocks are correlated, so each row is rescaled to sum to 1
    (Diebold & Yilmaz 2012).

In a cointegrated system the shares do not settle on a variance ratio of the
levels, because the forecast error variance grows without bound. They do
converge: at long horizons the permanent shocks account for everything.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from bayesian_vecm._irf import _VALID_METHODS, cholesky_impact, ma_coefficients, resolve_order


def compute_fevd(
    idata: xr.DataTree,
    k_ar_diff: int,
    steps: int,
    method: str = "girf",
    variable_names: list[str] | None = None,
    order: Any = None,
) -> xr.DataArray:
    """Compute the posterior FEVD for every draw in *idata*.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``.
    k_ar_diff
        Number of lagged-difference blocks in the model.
    steps
        Longest forecast horizon. Horizons ``1, ..., steps`` are returned.
    method
        ``"girf"`` (default) is order-invariant with rows rescaled to sum
        to 1. ``"cholesky"`` uses the column order, or ``order`` if given.
    variable_names
        Optional variable labels for the ``response_variable`` and
        ``shock_variable`` coordinates.
    order
        Cholesky ordering as variable names or column indices, listing each
        variable once. Only valid with ``method="cholesky"``.

    Returns
    -------
    xarray.DataArray
        Shape ``(chain, draw, horizon, response_variable, shock_variable)``.
        Entry ``[..., h, i, j]`` is the share of variable ``i``'s forecast
        error variance at horizon ``h`` that is due to shocks in variable
        ``j``. Each row sums to 1 over ``shock_variable``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"girf"`` or ``"cholesky"``, if ``steps < 1``,
        or if ``order`` is invalid.
    TypeError
        If ``order`` is a string or holds entries that are not names or
        integers.
    """
    if method not in _VALID_METHODS:
        valid = sorted(_VALID_METHODS)
        raise ValueError(f"method must be one of {valid}; got method={method!r}")
    if steps < 1:
        raise ValueError(f"steps must be at least 1; got steps={steps}")

    posterior = idata.posterior
    phi, sigma = ma_coefficients(posterior, k_ar_diff, steps)  # (D, H, K, K), (D, K, K)
    n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    n_vars = sigma.shape[-1]
    order_idx = resolve_order(order, method=method, variable_names=variable_names, n_vars=n_vars)

    if method == "cholesky":
        impact = cholesky_impact(sigma, order_idx)  # (D, K, K)
    else:
        sigma_diag_sqrt = np.sqrt(np.diagonal(sigma, axis1=1, axis2=2))  # (D, K)
        impact = sigma / sigma_diag_sqrt[:, np.newaxis, :]  # Sigma[:, j] / sqrt(sigma_jj)

    # theta[d, s, i, j]: response of i to shock j, s periods after the shock.
    theta = np.einsum("dsij,djk->dsik", phi, impact)
    contribution = np.cumsum(theta**2, axis=1)  # (D, H, K, K)

    if method == "cholesky":
        # sum_j theta_ij^2 equals the MSE diagonal, so this is the exact share.
        total = contribution.sum(axis=-1, keepdims=True)
    else:
        mse = np.cumsum(np.einsum("dsij,djk,dsik->dsi", phi, sigma, phi), axis=1)
        raw = contribution / mse[..., np.newaxis]
        contribution, total = raw, raw.sum(axis=-1, keepdims=True)

    fevd = (contribution / total).reshape(n_chains, n_draws, steps, n_vars, n_vars)

    coords: dict = {
        "chain": posterior.coords["chain"].values,
        "draw": posterior.coords["draw"].values,
        "horizon": np.arange(1, steps + 1),
    }
    if variable_names is not None:
        coords["response_variable"] = variable_names
        coords["shock_variable"] = variable_names

    return xr.DataArray(
        fevd,
        dims=["chain", "draw", "horizon", "response_variable", "shock_variable"],
        coords=coords,
    )
