"""PyMC model graph for the ``BayesianVECM``.

This module is the home of the real econometrics in the package: turning the
three design matrices from
:func:`bayesian_vecm._design.cointegration_design` into a PyMC graph that
can sample.

Current scope
-------------
The model supports:

* Known cointegration rank :math:`r \\geq 1`.
* All five deterministic codes: ``"n"``, ``"co"``, ``"ci"``, ``"lo"``, ``"li"``.
* Weakly-informative defaults for :math:`\\alpha, \\beta, \\Gamma, \\Sigma`.

The graph reads shapes directly off the design matrices produced by
:func:`bayesian_vecm._design.cointegration_design`, so deterministic columns
are handled automatically:

* Outside terms (``"co"``, ``"lo"``) append a column to ``delta_x``, which
  widens :math:`\\Gamma` from :math:`(K, Kk)` to :math:`(K, Kk+1)`.
* Inside terms (``"ci"``, ``"li"``) append a column to ``y_{t-1}``, which
  widens :math:`\\beta` from :math:`(K, r)` to :math:`(K+1, r)`. The extra
  row is free (not pinned), so the constant/trend inside the cointegration
  relation is estimated from the data.

Identification of :math:`\\beta`
--------------------------------
The cointegration term :math:`\\alpha \\beta' y_{t-1}` is invariant under

.. math::

    (\\alpha, \\beta) \\;\\to\\; (\\alpha R^{-1}, \\, \\beta R^{\\top})

for any invertible :math:`r \\times r` matrix :math:`R`. Without a
normalisation the posterior is non-identified, the sampler wanders along the
orbit, and divergences pile up. The standard fix (Johansen, 1995) is to pin
:math:`\\beta[:r, :] = I_r`. The leading :math:`r \\times r` block is a
constant; the remaining :math:`(K - r, r)` free entries are sampled. For
:math:`r = 1` that means just the first entry is fixed at 1.

The fixed entries are *not* stored as free random variables; the full
:math:`(K, r)` :math:`\\beta` matrix is exposed as a :func:`pm.Deterministic`
so downstream consumers can read it directly from ``idata`` without having
to remember the normalisation.

Mean structure
--------------
For a row :math:`t` of the design,

.. math::

    \\mu_t \\;=\\; \\alpha\\,\\beta^{\\top} y_{t-1} + \\Gamma\\,\\Delta x_t,

where :math:`\\Delta x_t = [\\Delta y_{t-1}; \\dots; \\Delta y_{t-k}]` is the
stacked lagged-difference regressor. In matrix form

.. math::

    \\mu \\;=\\; y_{\\text{lag1}}\\,\\beta\\,\\alpha^{\\top}
                  \\;+\\; \\Delta x\\,\\Gamma^{\\top},

with shapes :math:`(T_{\\text{eff}}, K) = (T_{\\text{eff}}, K)(K, r)(r, K)
+ (T_{\\text{eff}}, Kk)(Kk, K)`. The :math:`\\Gamma` term is dropped
entirely when ``k_ar_diff == 0`` so the graph never multiplies a
zero-column matrix.

Default priors
--------------
All are weakly informative; any of the four can be overridden via the
``priors`` dict on ``BayesianVECM``.

* :math:`\\alpha`: ``Normal(0, 1.0)``, shape :math:`(K, r)`. EC loadings are
  typically small and can be negative.
* :math:`\\beta_{\\text{free}}`: ``Normal(0, 5.0)``, shape :math:`(K - r, r)`.
  Cointegrating-vector entries can have arbitrary scale; the wider prior is
  more permissive than :math:`\\alpha`'s. Not present when ``K == r``
  (fully cointegrated system — no free rows).
* :math:`\\Gamma`: ``Normal(0, 0.5)``, shape :math:`(K, Kk)`. Short-run
  dynamics are typically small and centred at zero.
* :math:`\\Sigma`: ``LKJCholeskyCov(eta=2.0, sd_dist=HalfNormal(1.0))``. The
  standard PyMC idiom; ``eta=2.0`` mildly favours weaker correlations.

The :math:`\\Sigma` prior is special-cased — the LKJCholeskyCov factory
doesn't fit the simple ``{"dist": ..., **kwargs}`` pattern — and v0 only
accepts ``eta`` and ``sd_sigma`` overrides for it. A richer covariance-prior
API is a follow-up.
"""

from __future__ import annotations

from typing import Any

import pymc as pm
import pytensor.tensor as pt

from bayesian_vecm._design import CointegrationDesign

#: Prior-dict keys recognised in v0. Sigma is handled specially — see
#: :func:`_build_sigma` — because its parameterisation doesn't fit the
#: ``{"dist": ..., **kwargs}`` pattern used for the other three.
#: ``B`` is the exogenous-regressor coefficient matrix; it is only added to
#: the graph when ``design.exog`` is not ``None``.
_VALID_PRIOR_KEYS: frozenset[str] = frozenset({"alpha", "beta", "Gamma", "Sigma", "B"})

#: Recognised override keys for the Sigma prior in v0.
_VALID_SIGMA_KEYS: frozenset[str] = frozenset({"eta", "sd_sigma"})


def build_pymc_model(
    design: CointegrationDesign,
    *,
    k_ar_diff: int,
    coint_rank: int,
    deterministic: str,
    priors: dict[str, Any] | None = None,
) -> pm.Model:
    """Build the PyMC model graph for a v0 BayesianVECM fit.

    Parameters
    ----------
    design
        Aligned design matrices from
        :func:`bayesian_vecm._design.cointegration_design`. Must have been
        built with the same ``k_ar_diff`` and ``deterministic`` settings as
        the model being built; the caller is responsible for that consistency.
    k_ar_diff
        Number of lagged-difference blocks. ``0`` means no short-run dynamics
        and the :math:`\\Gamma` block is omitted from the graph.
    coint_rank
        Cointegration rank :math:`r`. Must satisfy ``1 <= coint_rank < K``
        where ``K`` is the number of variables.
    deterministic
        Deterministic-term code. One of ``"n"``, ``"co"``, ``"ci"``,
        ``"lo"``, ``"li"``. The graph reads shapes off the design matrices,
        so no separate code path is needed per code.
    priors
        Optional mapping from parameter name to distribution spec. Recognised
        keys: ``"alpha"``, ``"beta"``, ``"Gamma"``, ``"Sigma"``. Any key not
        present falls back to the weakly-informative default. ``None`` and
        ``{}`` are both legal — both mean "use defaults".

    Returns
    -------
    pm.Model
        The compiled PyMC model with free parameters ``alpha``, ``beta_free``,
        ``Gamma`` (if ``k_ar_diff > 0``), and ``Sigma_chol``; deterministics
        ``beta`` and ``Sigma`` for convenient downstream access; and an
        observed ``delta_y_obs`` carrying the row-wise multivariate-Normal
        likelihood.

    Raises
    ------
    ValueError
        If ``priors`` contains unrecognised keys, an unknown distribution
        name, or unrecognised Sigma-override keys.
    TypeError
        If ``priors`` or any of its values is not a dict.
    """
    user_priors = _validate_priors_dict(priors)

    delta_y = design.delta_y
    delta_x = design.delta_x
    y_lag1 = design.y_lag1
    exog = design.exog  # (T_eff, m) or None

    n_eff, n_vars = delta_y.shape
    r = coint_rank  # short alias used heavily below

    # Read column counts off the actual design matrices rather than
    # reconstructing them from k_ar_diff and deterministic.  This is the key
    # move that makes deterministic-term support free: cointegration_design
    # already appended the right columns, so the graph just needs to match.
    #
    #   y_lag1_cols = K           for "n", "co", "lo"
    #               = K + 1       for "ci", "li"  (inside term appended)
    #
    #   delta_x_cols = K * k      for "n", "ci", "li"
    #                = K * k + 1  for "co", "lo"  (outside term appended)
    y_lag1_cols = y_lag1.shape[1]
    delta_x_cols = delta_x.shape[1]

    # Row-alignment sanity check — all three matrices must have T_eff rows.
    if y_lag1.shape[0] != n_eff or delta_x.shape[0] != n_eff:
        raise ValueError(
            f"design matrices are not row-aligned: delta_y has {n_eff} rows, "
            f"y_lag1 has {y_lag1.shape[0]}, delta_x has {delta_x.shape[0]}."
        )

    with pm.Model() as model:
        # Stash the design so a serialised idata is self-contained — anyone
        # with just the saved file can rebuild the predictive density without
        # re-deriving the design from raw endog.
        pm.Data("delta_y", delta_y)
        pm.Data("y_lag1", y_lag1)
        if delta_x_cols > 0:
            pm.Data("delta_x", delta_x)
        if exog is not None:
            pm.Data("exog", exog)

        # --- alpha: (K, r) loadings on the cointegration relation ------------
        # alpha is always (K, r) regardless of deterministic code — the
        # constant/trend rows in beta absorb inside terms, not alpha.
        alpha = _resolve_dist(
            name="alpha",
            user_spec=user_priors.get("alpha"),
            default_spec={"dist": "Normal", "mu": 0.0, "sigma": 1.0},
            shape=(n_vars, r),
        )

        # --- beta: (y_lag1_cols, r) with top r x r block pinned at I_r ------
        # For inside terms ("ci", "li") y_lag1_cols = K + 1, so beta gains an
        # extra free row for the constant/trend loading inside the cointegrating
        # relation.  Free part is (y_lag1_cols - r, r); the leading r x r
        # identity block is stacked on top as a constant.
        beta_free = _resolve_dist(
            name="beta_free",
            user_spec=user_priors.get("beta"),
            default_spec={"dist": "Normal", "mu": 0.0, "sigma": 5.0},
            shape=(y_lag1_cols - r, r),
        )
        beta = pm.Deterministic(
            "beta",
            pt.concatenate([pt.eye(r), beta_free], axis=0),
        )

        # --- Gamma: (K, delta_x_cols) short-run dynamics + outside terms -----
        # For outside terms ("co", "lo") delta_x_cols = K * k + 1, so Gamma
        # gains an extra column for the constant/trend outside the cointegrating
        # relation.  When delta_x_cols == 0 (k=0, no outside term) there are no
        # short-run regressors and Gamma is omitted entirely.
        if delta_x_cols > 0:
            gamma = _resolve_dist(
                name="Gamma",
                user_spec=user_priors.get("Gamma"),
                default_spec={"dist": "Normal", "mu": 0.0, "sigma": 0.5},
                shape=(n_vars, delta_x_cols),
            )

        # --- Sigma: K x K covariance via LKJCholeskyCov ----------------------
        chol = _build_sigma(n_vars=n_vars, user_spec=user_priors.get("Sigma"))
        pm.Deterministic("Sigma", chol @ chol.T)

        # --- B: (K, m) contemporaneous exogenous coefficients ----------------
        # Only added when exog is present. The prior is Normal(0, 1.0) —
        # same scale as alpha; exog columns are typically standardised before
        # fitting so a unit-scale prior is weakly informative.
        if exog is not None:
            n_exog = exog.shape[1]
            b_mat = _resolve_dist(
                name="B",
                user_spec=user_priors.get("B"),
                default_spec={"dist": "Normal", "mu": 0.0, "sigma": 1.0},
                shape=(n_vars, n_exog),
            )

        # --- Mean ------------------------------------------------------------
        # mu_t = alpha beta' y_{t-1} + Gamma Delta_x_t [+ B X_t], matrix form:
        #   (T_eff, K) = (T_eff, y_lag1_cols)(y_lag1_cols, r)(r, K)
        #              + (T_eff, delta_x_cols)(delta_x_cols, K)
        #              [+ (T_eff, m)(m, K)]
        ec_term = pm.math.dot(pm.math.dot(y_lag1, beta), alpha.T)
        mu = ec_term + pm.math.dot(delta_x, gamma.T) if delta_x_cols > 0 else ec_term
        if exog is not None:
            mu = mu + pm.math.dot(exog, b_mat.T)

        # --- Likelihood: row-wise MvNormal with shared Sigma ----------------
        # Using chol directly (rather than cov=Sigma) avoids redundant
        # factorisation inside MvNormal and is the recommended PyMC idiom.
        pm.MvNormal("delta_y_obs", mu=mu, chol=chol, observed=delta_y)

    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_priors_dict(priors: dict[str, Any] | None) -> dict[str, Any]:
    """Return a defensive copy of ``priors``, validating its keys.

    The public ``BayesianVECM`` constructor already checks that ``priors`` is
    a dict or ``None``; this pass adds the v0-specific check on which keys
    are recognised so a typo like ``"sigma"`` (lowercase) fails loudly
    instead of silently being ignored.
    """
    if priors is None:
        return {}
    if not isinstance(priors, dict):
        # Defensive — BayesianVECM.__init__ already enforces this, but
        # build_pymc_model is a public function in its own right.
        raise TypeError(
            f"priors must be a dict or None; got priors of type {type(priors).__name__}"
        )
    unknown = set(priors) - _VALID_PRIOR_KEYS
    if unknown:
        valid = sorted(_VALID_PRIOR_KEYS)
        raise ValueError(
            f"unknown prior key(s) {sorted(unknown)}; valid keys are {valid}. "
            "Keys are case-sensitive."
        )
    return dict(priors)


def _resolve_dist(
    *,
    name: str,
    user_spec: dict[str, Any] | None,
    default_spec: dict[str, Any],
    shape: tuple[int, ...],
) -> Any:
    """Create a PyMC random variable from a ``{"dist": "<Name>", ...}`` spec.

    The user spec wins entirely if provided — defaults are not merged in
    field-by-field, because doing so would make "I want a Laplace prior with
    just my own scale" require the user to know the default ``mu``. Each
    spec is a complete distribution description.
    """
    spec = user_spec if user_spec is not None else default_spec
    if not isinstance(spec, dict):
        raise TypeError(
            f"prior spec for {name!r} must be a dict of the form "
            f"{{'dist': '<DistName>', **kwargs}}; got {type(spec).__name__}"
        )
    spec = dict(spec)  # defensive copy — we mutate via pop()

    dist_name = spec.pop("dist", None)
    if dist_name is None:
        raise ValueError(
            f"prior spec for {name!r} must include a 'dist' key naming a PyMC distribution"
        )

    dist_cls = getattr(pm, dist_name, None)
    if dist_cls is None:
        raise ValueError(
            f"unknown PyMC distribution {dist_name!r} in prior spec for {name!r}; "
            f"check the name against the pymc API."
        )

    return dist_cls(name, shape=shape, **spec)


def _build_sigma(*, n_vars: int, user_spec: dict[str, Any] | None) -> Any:
    """Construct the residual-covariance Cholesky factor.

    Uses :class:`pm.LKJCholeskyCov` regardless of user spec; only ``eta`` and
    ``sd_sigma`` are configurable in v0. The packed-name ``"Sigma_chol"`` is
    introduced so the public-facing ``"Sigma"`` name can be reserved for the
    full covariance matrix (added as a :func:`pm.Deterministic` by the
    caller).
    """
    if user_spec is None:
        eta = 2.0
        sd_sigma = 1.0
    else:
        if not isinstance(user_spec, dict):
            raise TypeError(
                "prior spec for 'Sigma' must be a dict with optional 'eta' and "
                f"'sd_sigma' keys; got {type(user_spec).__name__}"
            )
        unknown = set(user_spec) - _VALID_SIGMA_KEYS
        if unknown:
            valid = sorted(_VALID_SIGMA_KEYS)
            raise ValueError(
                f"unknown Sigma-prior key(s) {sorted(unknown)}; "
                f"v0 accepts only {valid}. A richer covariance-prior API is a follow-up."
            )
        eta = float(user_spec.get("eta", 2.0))
        sd_sigma = float(user_spec.get("sd_sigma", 1.0))

    sd_dist = pm.HalfNormal.dist(sigma=sd_sigma, shape=n_vars)
    chol, _corr, _stds = pm.LKJCholeskyCov(
        "Sigma_chol",
        n=n_vars,
        eta=eta,
        sd_dist=sd_dist,
        compute_corr=True,
    )
    return chol
