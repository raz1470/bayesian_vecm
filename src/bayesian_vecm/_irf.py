"""Impulse Response Functions (IRFs) for ``BayesianVECM``.

Two identification schemes are supported:

``"girf"`` — Generalised IRFs (Pesaran & Shin 1998)
    Order-invariant: no column order is needed. A shock to variable
    :math:`j` moves the other variables by their expected same-period
    response under :math:`\\Sigma`. This equals the Cholesky IRF with
    :math:`j` ordered first. Each shock is treated as first in its own
    chain, so the responses to different shocks cannot all be structural
    at once (Kim 2009). Read them for a variable that leads the system.
    At horizon :math:`h`:

    .. math::

        \\text{GIRF}_h[:, j]
            = \\frac{1}{\\sqrt{\\sigma_{jj}}} \\, \\Phi_h \\, \\Sigma \\, e_j

    In matrix form: :math:`\\text{GIRF}_h = \\Phi_h \\, \\Sigma \\, D^{-1/2}`
    where :math:`D = \\mathrm{diag}(\\Sigma)`.

``"cholesky"`` — Orthogonalised IRFs (Sims 1980)
    Requires a strict recursive (triangular) causal ordering among the
    variables.  Appropriate only when you can defend a Wold causal chain
    with no contemporaneous feedback.  The shock is a Cholesky-orthogonalised
    unit vector:

    .. math::

        \\text{OIR}_h = \\Phi_h \\, P

    where :math:`P` is the lower-Cholesky factor of :math:`\\Sigma`.
    The order is the column order unless ``order`` is given. ``order``
    permutes :math:`\\Sigma` before the factorisation and maps the result
    back, so the output keeps the original variable order.

Both methods are computed via the **VAR companion form**: the VECM is first
converted to a levels VAR(:math:`p`) with :math:`p = k_{\\text{ar\\_diff}} + 1`,
then the companion matrix :math:`F` is iterated to produce the MA coefficient
matrices

.. math::

    \\Phi_h = J \\, F^h \\, J^\\top,
    \\quad J = [I_K \\;\\mid\\; 0_{K \\times K(p-1)}],

for horizons :math:`h = 0, 1, \\dots, \\text{steps}`.

VECM-to-VAR conversion
-----------------------
Given the VECM

.. math::

    \\Delta y_t = \\alpha \\beta^\\top y_{t-1}
                 + \\Gamma_1 \\Delta y_{t-1} + \\cdots + \\Gamma_k \\Delta y_{t-k}
                 + \\varepsilon_t,

the equivalent levels VAR(:math:`p`) with :math:`p = k + 1` has coefficient
matrices:

.. math::

    A_1 &= I_K + \\alpha \\beta^\\top + \\Gamma_1, \\\\
    A_j &= \\Gamma_j - \\Gamma_{j-1}, \\quad j = 2, \\dots, k, \\\\
    A_{k+1} &= -\\Gamma_k.

For :math:`k = 0` (pure error-correction, no short-run dynamics):
:math:`A_1 = I_K + \\alpha\\beta^\\top` and :math:`p = 1`.

Deterministic terms
-------------------
Outside deterministic terms (``"co"``, ``"lo"``) append an extra column to
the :math:`\\Gamma` matrix in the fitted posterior.  That column is a
constant or trend coefficient, not a VAR lag, so it must not be included in
the companion matrix.  ``compute_irf`` always takes only the first
:math:`K \\cdot k_{\\text{ar\\_diff}}` columns of ``Gamma``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

_VALID_METHODS: frozenset[str] = frozenset({"girf", "cholesky"})


def resolve_order(
    order: Any,
    *,
    method: str,
    variable_names: list[str] | None,
    n_vars: int,
) -> list[int] | None:
    """Turn a user ``order`` into column indices, or ``None`` for column order.

    ``order`` lists every variable once, by name or by integer index. It is
    only valid with ``method="cholesky"``.
    """
    if order is None:
        return None
    if method != "cholesky":
        raise ValueError(f"order only applies to method='cholesky'; got method={method!r}")
    if isinstance(order, str):
        raise TypeError("order must be a sequence of variable names or indices, not a string")
    indices: list[int] = []
    for item in order:
        if isinstance(item, str):
            if variable_names is None or item not in variable_names:
                raise ValueError(f"unknown variable {item!r} in order; known: {variable_names}")
            indices.append(variable_names.index(item))
        elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            indices.append(int(item))
        else:
            raise TypeError(f"order entries must be names or integers; got {item!r}")
    if sorted(indices) != list(range(n_vars)):
        raise ValueError(
            f"order must list each of the {n_vars} variables exactly once; got {list(order)}"
        )
    return indices


def cholesky_impact(
    sigma: NDArray[np.floating], order: list[int] | None = None
) -> NDArray[np.floating]:
    """Cholesky impact matrix for each draw, in the original variable order.

    With ``order`` the factorisation is of ``Sigma`` permuted to that order.
    Rows and columns are then mapped back, so entry ``[d, i, j]`` is still
    the impact on variable ``i`` of a shock to variable ``j``. The result
    ``P`` satisfies ``P @ P.T == Sigma`` for any order.
    """
    if order is None:
        return np.linalg.cholesky(sigma)
    idx = np.asarray(order)
    rows, cols = idx[:, np.newaxis], idx[np.newaxis, :]
    chol = np.linalg.cholesky(sigma[:, rows, cols])
    impact = np.empty_like(chol)
    impact[:, rows, cols] = chol
    return impact


def ma_coefficients(
    posterior: xr.Dataset | xr.DataTree,
    k_ar_diff: int,
    n_horizons: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Moving-average matrices and innovation covariance for every draw.

    Converts the VECM to its levels VAR and iterates the companion matrix.

    Parameters
    ----------
    posterior
        Posterior group holding ``alpha``, ``beta`` and ``Sigma``. ``Gamma``
        is also needed when ``k_ar_diff > 0``.
    k_ar_diff
        Number of lagged-difference blocks in the model.
    n_horizons
        Number of matrices to return: :math:`\\Phi_0, \\dots, \\Phi_{H-1}`.

    Returns
    -------
    phi : np.ndarray, shape (n_samples, n_horizons, K, K)
        ``phi[d, h]`` is :math:`\\Phi_h` for draw ``d``. Chains are stacked,
        chain-major, so ``n_samples = n_chains * n_draws``.
    sigma : np.ndarray, shape (n_samples, K, K)
        Innovation covariance for each draw.
    """
    # --- Extract posterior draws -------------------------------------------
    alpha_draws: NDArray[np.floating] = posterior["alpha"].values  # (C, D, K, r)
    beta_draws: NDArray[np.floating] = posterior["beta"].values  # (C, D, K, r)
    sigma_draws: NDArray[np.floating] = posterior["Sigma"].values  # (C, D, K, K)

    n_chains, n_draws, n_vars, r = alpha_draws.shape
    n_total = n_chains * n_draws
    var_order = k_ar_diff + 1  # levels VAR order p = k + 1
    kp = n_vars * var_order

    # Reshape to (n_total, ...) for vectorised ops.
    # beta may have shape (C, D, K+1, r) when deterministic="ci"/"li" appends
    # a trend row inside the cointegration space.  Only the first n_vars rows
    # correspond to the endogenous variables; slice before reshaping.
    alpha = alpha_draws.reshape(n_total, n_vars, r)  # (D, K, r)
    beta = beta_draws[:, :, :n_vars, :].reshape(n_total, n_vars, r)  # (D, K, r)
    sigma = sigma_draws.reshape(n_total, n_vars, n_vars)  # (D, K, K)

    has_gamma = k_ar_diff > 0
    if has_gamma:
        gamma_draws: NDArray[np.floating] = posterior["Gamma"].values  # (C, D, K, Kk+?)
        # Slice to the first K*k_ar_diff columns — VAR dynamics only.
        # Outside deterministic terms ("co", "lo") append an extra column to
        # Gamma in the fitted posterior; that column must not enter the
        # companion matrix.
        gamma_dyn = gamma_draws.reshape(n_total, n_vars, -1)[
            :, :, : n_vars * k_ar_diff
        ]  # (D, K, K*k)

    # --- Build VAR companion matrix F: (D, Kp, Kp) ------------------------
    #
    # Layout (block-column-major):
    #
    #   F = [ A_1  A_2  ...  A_p ]
    #       [ I_K   0   ...   0  ]
    #       [  0   I_K  ...   0  ]
    #       [  :    :   ...   :  ]
    #       [  0    0   ... I_K  0]
    #
    # where A_1 = I + alpha beta' + Gamma_1,
    #       A_j = Gamma_j - Gamma_{j-1}  for j = 2..k,
    #       A_{k+1} = -Gamma_k.
    #
    companion = np.zeros((n_total, kp, kp))

    # Bottom rows: identity shift blocks [I | 0 ... 0].
    if var_order > 1:
        companion[:, n_vars:, : n_vars * (var_order - 1)] = np.eye(n_vars * (var_order - 1))

    # a1 = I_K + alpha @ beta.T  [+ Gamma_1 if has_gamma]
    # einsum "dij,dkj->dik": contracts r — gives alpha @ beta.T per draw.
    a1 = np.eye(n_vars) + np.einsum("dij,dkj->dik", alpha, beta)  # (D, K, K)
    if has_gamma:
        a1 = a1 + gamma_dyn[:, :, :n_vars]  # add Gamma_1

    companion[:, :n_vars, :n_vars] = a1

    # A_{j+1} blocks for j = 1..k_ar_diff, placed in columns j*K..(j+1)*K.
    # A_{j+1} = Gamma_{j+1} - Gamma_j  (Gamma_{k+1} ≡ 0)
    if has_gamma:
        for j in range(1, k_ar_diff + 1):
            g_prev = gamma_dyn[:, :, (j - 1) * n_vars : j * n_vars]  # Gamma_j
            if j < k_ar_diff:
                g_curr = gamma_dyn[:, :, j * n_vars : (j + 1) * n_vars]  # Gamma_{j+1}
                a_next = g_curr - g_prev
            else:
                a_next = -g_prev  # A_{k+1} = -Gamma_k
            companion[:, :n_vars, j * n_vars : (j + 1) * n_vars] = a_next

    # --- Iterate Phi_h = top-left (K,K) block of F^h ----------------------
    irf_raw = np.empty((n_total, n_horizons, n_vars, n_vars), dtype=np.float64)

    current = np.broadcast_to(np.eye(kp), (n_total, kp, kp)).copy()  # companion^0 = I

    for h in range(n_horizons):
        irf_raw[:, h, :, :] = current[:, :n_vars, :n_vars]  # Phi_h
        if h < n_horizons - 1:
            current = np.einsum("dij,djk->dik", current, companion)

    return irf_raw, sigma


def compute_irf(
    idata: xr.DataTree,
    k_ar_diff: int,
    steps: int,
    method: str = "girf",
    variable_names: list[str] | None = None,
    order: Any = None,
) -> xr.DataArray:
    """Compute posterior IRFs for every draw in *idata*.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``.  Must contain a
        ``posterior`` group with variables ``alpha``, ``beta``, ``Sigma``,
        and — when ``k_ar_diff > 0`` — ``Gamma``.
    k_ar_diff
        Number of lagged-difference blocks in the model.
    steps
        Forecast horizon; IRFs are computed for :math:`h = 0, 1, \\dots,
        \\text{steps}` giving ``steps + 1`` horizons in total.
    method
        Identification scheme.  ``"girf"`` (default) — Generalised IRFs
        (Pesaran & Shin 1998), order-invariant.  ``"cholesky"`` —
        Orthogonalised IRFs (Sims 1980), requires a defensible recursive
        causal ordering.
    variable_names
        Optional variable labels.  When provided, added as
        ``response_variable`` and ``shock_variable`` coordinates.
    order
        Cholesky ordering as variable names or column indices, listing each
        variable once. ``None`` uses the column order. Only valid with
        ``method="cholesky"``.

    Returns
    -------
    xarray.DataArray
        Shape ``(chain, draw, horizon, response_variable, shock_variable)``.
        ``horizon`` runs from ``0`` to ``steps`` (inclusive).
        Entry ``[..., h, i, j]`` is the response of variable :math:`i` to a
        unit shock in variable :math:`j` at horizon :math:`h`.

    Raises
    ------
    ValueError
        If ``method`` is not ``"girf"`` or ``"cholesky"``, if ``steps``
        is less than ``1``, or if ``order`` is invalid.
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
    irf_raw, sigma = ma_coefficients(posterior, k_ar_diff, steps + 1)
    order_idx = resolve_order(
        order, method=method, variable_names=variable_names, n_vars=sigma.shape[-1]
    )
    n_chains, n_draws = posterior.sizes["chain"], posterior.sizes["draw"]
    n_horizons, n_vars = irf_raw.shape[1], irf_raw.shape[2]

    # --- Apply identification scheme ---------------------------------------
    if method == "cholesky":
        # OIR_h = Phi_h @ p_chol  where p_chol = chol(Sigma), lower triangular.
        p_chol = cholesky_impact(sigma, order_idx)  # (D, K, K)
        irf_id = np.einsum("dhij,djk->dhik", irf_raw, p_chol)  # (D, H, K, K)

    else:  # method == "girf"
        # GIRF_h = Phi_h @ Sigma @ D^{-1/2}
        # where D = diag(Sigma), so (Sigma @ D^{-1/2})[:,j] = Sigma[:,j] / sqrt(Sigma[j,j]).
        sigma_diag_sqrt = np.sqrt(np.diagonal(sigma, axis1=1, axis2=2))  # (D, K)
        # scaling[d, i, j] = Sigma[d, i, j] / sqrt(Sigma[d, j, j])
        scaling = sigma / sigma_diag_sqrt[:, np.newaxis, :]  # (D, K, K)
        irf_id = np.einsum("dhij,djk->dhik", irf_raw, scaling)  # (D, H, K, K)

    # --- Reshape and wrap in xr.DataArray ----------------------------------
    irf_out = irf_id.reshape(n_chains, n_draws, n_horizons, n_vars, n_vars)

    chain_coords = posterior.coords["chain"].values
    draw_coords = posterior.coords["draw"].values
    horizon_coords = np.arange(n_horizons)  # 0..steps

    coords: dict = {
        "chain": chain_coords,
        "draw": draw_coords,
        "horizon": horizon_coords,
    }
    if variable_names is not None:
        coords["response_variable"] = variable_names
        coords["shock_variable"] = variable_names

    return xr.DataArray(
        irf_out,
        dims=["chain", "draw", "horizon", "response_variable", "shock_variable"],
        coords=coords,
    )
