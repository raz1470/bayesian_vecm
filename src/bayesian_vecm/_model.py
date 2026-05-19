"""Public ``BayesianVECM`` class — skeleton only.

This module defines the public-facing class that the rest of the package will
hang off. **No estimation is implemented yet**: every method that depends on a
fitted PyMC graph raises :class:`NotImplementedError`. The point of this slice
is to lock in the API shape before any PyMC code lands.

Design decisions captured here
------------------------------
* **Configuration in ``__init__``, data in ``fit``.** ``k_ar_diff``,
  ``coint_rank``, ``deterministic`` and ``priors`` all change the *shape* of
  the eventual PyMC graph, so they belong to the model object's identity.
  ``coint_rank`` in particular lives here rather than on ``fit`` because
  :math:`\\alpha` is :math:`K \\times r` and :math:`\\beta` is :math:`K \\times r` —
  changing ``r`` is a full graph rebuild. A rank-selection workflow is
  naturally expressed as a loop over fresh ``BayesianVECM(coint_rank=r)``
  instances, which keeps each fitted ``idata`` available for later
  Bayesian-model-averaging.
* **Dict-based priors.** ``priors`` is a plain ``dict`` mapping parameter
  names (``"alpha"``, ``"beta"``, ``"Gamma"``, ``"Sigma"``) to distribution
  specs of the form ``{"dist": "Normal", "mu": 0.0, "sigma": 1.0}``. This is
  inspired by ``pymc_marketing.MMM``'s pattern but kept one step simpler:
  no dedicated ``Prior`` class in v0. The dict is JSON-serialisable,
  trivial to document, and forward-compatible — we can later accept a
  ``Prior``-like object in addition without breaking anyone.
* **Store ``endog`` after fit.** ``sample_posterior_predictive`` is a
  forecast and needs the last :math:`k\\_ar\\_diff + 1` rows of the original
  series to seed the recursion. Requiring callers to re-pass the data at
  predict time is friction and a footgun (a different series silently gives
  nonsense forecasts). The class stores the input as ``self.endog_`` —
  sklearn-style trailing-underscore convention meaning "set during fit". The
  same array is also stashed inside ``self.idata_.constant_data`` so a
  serialised ``idata`` is self-contained for reproducibility independent of
  the live object.

Attributes set during ``fit`` (none defined yet — the methods are stubs)
-----------------------------------------------------------------------
The trailing-underscore convention distinguishes init-time config from
fit-time state, mirroring scikit-learn:

* ``endog_`` : ``ndarray`` of shape ``(T, K)`` — the input passed to ``fit``.
* ``idata_`` : ``arviz.InferenceData`` — the full posterior record.
* ``variable_names_`` : ``list[str] | None`` — column labels, if the input
  exposed them via a DataFrame-like ``.columns`` attribute.
"""

from __future__ import annotations

from typing import Any

# Valid deterministic-term codes for v0. Mirrors the set accepted by
# ``cointegration_design`` once the deterministic-terms work merges to main.
# Compound Johansen codes (cases 4 and 5) are deferred to a follow-up.
_VALID_DETERMINISTIC = frozenset({"n", "co", "ci", "lo", "li"})


class BayesianVECM:
    """Bayesian Vector Error Correction Model.

    A fixed-rank cointegrated VAR estimated by Bayesian inference. The model
    being targeted is

    .. math::

        \\Delta y_t = \\alpha \\beta' y_{t-1}
                     + \\sum_{i=1}^{k} \\Gamma_i \\, \\Delta y_{t-i}
                     + D_t \\Phi
                     + \\varepsilon_t,
                     \\quad \\varepsilon_t \\sim \\mathcal{N}(0, \\Sigma),

    where :math:`\\alpha, \\beta` are :math:`K \\times r`, the :math:`\\Gamma_i`
    are :math:`K \\times K` short-run dynamics, and :math:`D_t` collects any
    requested deterministic terms (constants, trends, inside or outside the
    cointegration relation).

    .. warning::

        This is a v0 skeleton. ``fit``, ``idata``, ``summary``, and
        ``sample_posterior_predictive`` all raise :class:`NotImplementedError`.
        The class exists to lock in the public API; the PyMC graph and
        sampling code arrive in the next slice.

    Parameters
    ----------
    k_ar_diff
        Number of lagged-difference blocks :math:`\\Delta y_{t-1}, \\dots,
        \\Delta y_{t-k}` to include. ``0`` means no short-run dynamics — a
        pure error-correction equation. Defaults to ``1``.
    coint_rank
        Cointegration rank :math:`r`. Sets the shared inner dimension of
        :math:`\\alpha` and :math:`\\beta`. Must be at least ``1``; cannot
        exceed :math:`K` (enforced at ``fit`` time when :math:`K` is known).
        Defaults to ``1``.
    deterministic
        Deterministic-term code, following the convention of
        :mod:`statsmodels.tsa.vector_ar.vecm`. Single codes in v0:

        * ``"n"`` — no deterministic terms (default).
        * ``"co"`` — constant outside the cointegration relation.
        * ``"ci"`` — constant inside the cointegration relation.
        * ``"lo"`` — linear trend outside the cointegration relation.
        * ``"li"`` — linear trend inside the cointegration relation.

        Compound codes (Johansen cases 4 and 5) are deferred to a follow-up.
    priors
        Optional mapping from parameter name to distribution spec. Keys
        recognised in v0: ``"alpha"``, ``"beta"``, ``"Gamma"``, ``"Sigma"``.
        Each value is a dict of the form ``{"dist": "<DistName>", **kwargs}``
        — for example ``{"dist": "Normal", "mu": 0.0, "sigma": 1.0}``. Any
        parameter omitted from the dict falls back to a weakly-informative
        default chosen at ``fit`` time. ``None`` (the default) means "use
        defaults for everything".

    Raises
    ------
    ValueError
        If ``k_ar_diff`` is negative, ``coint_rank`` is less than ``1``, or
        ``deterministic`` is not one of the recognised v0 codes.
    TypeError
        If ``priors`` is not ``None`` and not a ``dict``.

    Examples
    --------
    Construct a model and inspect its configuration::

        >>> from bayesian_vecm import BayesianVECM
        >>> model = BayesianVECM(k_ar_diff=2, coint_rank=1, deterministic="ci")
        >>> model.k_ar_diff
        2
        >>> model.deterministic
        'ci'

    Calling ``fit`` (or any other estimation method) raises until the next
    slice lands::

        >>> model.fit(...)                       # doctest: +SKIP
        Traceback (most recent call last):
            ...
        NotImplementedError: BayesianVECM.fit is not implemented yet ...
    """

    def __init__(
        self,
        k_ar_diff: int = 1,
        coint_rank: int = 1,
        deterministic: str = "n",
        priors: dict[str, Any] | None = None,
    ) -> None:
        if k_ar_diff < 0:
            raise ValueError(f"k_ar_diff must be non-negative; got k_ar_diff={k_ar_diff}")

        if coint_rank < 1:
            raise ValueError(f"coint_rank must be at least 1; got coint_rank={coint_rank}")

        if deterministic not in _VALID_DETERMINISTIC:
            valid = sorted(_VALID_DETERMINISTIC)
            raise ValueError(
                f"deterministic must be one of {valid}; got deterministic={deterministic!r}. "
                "Compound Johansen codes (cases 4 and 5) are a v0.x follow-up."
            )

        if priors is not None and not isinstance(priors, dict):
            raise TypeError(
                f"priors must be a dict or None; got priors of type {type(priors).__name__}"
            )

        self.k_ar_diff = k_ar_diff
        self.coint_rank = coint_rank
        self.deterministic = deterministic
        self.priors = priors

    def fit(self, endog: Any) -> BayesianVECM:
        """Fit the model to ``endog`` by running PyMC sampling.

        Parameters
        ----------
        endog
            Endogenous time series of shape ``(T, K)``. Accepts anything
            :func:`bayesian_vecm._data.validate_endog` accepts — including
            objects with a ``.to_numpy()`` method, such as a
            ``pandas.DataFrame``.

        Returns
        -------
        BayesianVECM
            The fitted model (``self``), to support method chaining.

        Raises
        ------
        NotImplementedError
            Always — estimation is not implemented in this slice. The PyMC
            graph and sampler arrive in the next slice
            (``feat/first-pymc-model``).
        """
        raise NotImplementedError(
            "BayesianVECM.fit is not implemented yet — arriving in the feat/first-pymc-model slice."
        )

    @property
    def idata(self) -> Any:
        """Posterior record from the most recent ``fit`` as an ``arviz.InferenceData``.

        Raises
        ------
        NotImplementedError
            Always in v0 — accessor exists to lock the attribute name. After
            ``fit`` lands this becomes a thin getter for ``self.idata_``.
        """
        raise NotImplementedError(
            "BayesianVECM.idata is not implemented yet — arriving in the "
            "feat/first-pymc-model slice."
        )

    def summary(self) -> Any:
        """Return a tabular summary of the posterior.

        Wraps :func:`arviz.summary` for the fitted parameters and adds
        VECM-specific diagnostics (β identification, error-correction
        coefficients, residual variances).

        Raises
        ------
        NotImplementedError
            Always in v0 — see :meth:`fit`.
        """
        raise NotImplementedError(
            "BayesianVECM.summary is not implemented yet — arriving after "
            "the feat/first-pymc-model slice."
        )

    def sample_posterior_predictive(self, steps: int) -> Any:
        """Forecast ``steps`` periods ahead from the fitted posterior.

        Uses the last :math:`k\\_ar\\_diff + 1` rows of ``self.endog_`` to
        seed the recursion, propagating posterior uncertainty in
        :math:`\\alpha, \\beta, \\Gamma_i, \\Sigma` through to the forecast
        distribution.

        Parameters
        ----------
        steps
            Number of periods to forecast. Must be at least ``1``.

        Returns
        -------
        arviz.InferenceData
            A new ``InferenceData`` whose ``posterior_predictive`` group
            holds the forecast draws.

        Raises
        ------
        NotImplementedError
            Always in v0 — see :meth:`fit`.
        """
        raise NotImplementedError(
            "BayesianVECM.sample_posterior_predictive is not implemented yet — "
            "arriving after the feat/first-pymc-model slice."
        )
