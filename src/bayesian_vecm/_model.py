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

Attributes set during ``fit``
-----------------------------
The trailing-underscore convention distinguishes init-time config from
fit-time state, mirroring scikit-learn:

* ``endog_`` : ``ndarray`` of shape ``(T, K)`` — the input passed to ``fit``.
* ``idata_`` : ``arviz.InferenceData`` — the full posterior record, with
  the raw ``endog`` and design matrices stashed inside ``constant_data`` so
  a serialised idata can stand on its own.
* ``variable_names_`` : ``list[str] | None`` — column labels, if the input
  exposed them via a DataFrame-like ``.columns`` attribute.

v0 scope
--------
``fit`` only estimates the simplest VECM that actually samples: known
cointegration rank :math:`r = 1`, ``deterministic="n"``, weakly-informative
defaults for :math:`\\alpha, \\beta, \\Gamma, \\Sigma`. Other configurations
were accepted at construction time to lock in the public API, but raise
:class:`NotImplementedError` from inside the PyMC graph builder.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import pymc as pm
import xarray as xr

from bayesian_vecm._data import validate_endog
from bayesian_vecm._design import cointegration_design
from bayesian_vecm._pymc import build_pymc_model

# Valid deterministic-term codes for v0. Mirrors the set accepted by
# ``cointegration_design``. Compound Johansen codes (cases 4 and 5) are
# deferred to a follow-up.
_VALID_DETERMINISTIC = frozenset({"n", "co", "ci", "lo", "li"})

# Default headline parameters shown by ``summary()``. ``Gamma`` is added
# conditionally when ``k_ar_diff > 0``.
_SUMMARY_VAR_NAMES_BASE = ("alpha", "beta", "Sigma")

_NOT_FITTED_MSG = (
    "BayesianVECM has not been fitted yet; call .fit(endog) first."
)


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

        This is a v0 implementation. ``fit``, ``idata`` and ``summary`` are
        live, but only for ``coint_rank=1`` + ``deterministic="n"``. Other
        configurations are accepted at construction time (the public API is
        locked in) but raise :class:`NotImplementedError` from inside the
        PyMC graph builder. ``sample_posterior_predictive`` is deferred to
        its own follow-up slice — forecasting through the VAR recursion is
        meaningfully its own design problem.

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

    For a non-default configuration (``coint_rank=1, deterministic="n"``
    is the only one ``fit`` currently estimates), ``fit`` raises a
    :class:`NotImplementedError` from the PyMC graph builder::

        >>> model.fit(...)                       # doctest: +SKIP
        Traceback (most recent call last):
            ...
        NotImplementedError: deterministic='ci' is not yet supported ...
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

    def fit(
        self,
        endog: Any,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        progressbar: bool = True,
        **sample_kwargs: Any,
    ) -> BayesianVECM:
        """Fit the model to ``endog`` by running PyMC sampling.

        v0 scope: only ``coint_rank=1`` and ``deterministic="n"`` are
        actually estimated. Other configurations were accepted at
        construction time to lock in the public API, but trigger a
        :class:`NotImplementedError` here. The PyMC graph for the wider
        envelope arrives in a follow-up slice.

        Parameters
        ----------
        endog
            Endogenous time series of shape ``(T, K)``. Accepts anything
            :func:`bayesian_vecm._data.validate_endog` accepts — including
            objects with a ``.to_numpy()`` method, such as a
            ``pandas.DataFrame``. Column labels (if present) are captured
            into ``self.variable_names_``.
        draws, tune, chains, target_accept, random_seed, progressbar
            Forwarded to :func:`pm.sample`. The defaults aim at "reasonable
            for a small VECM": four chains of 1000 draws after 1000 tuning
            iterations, with a slightly cautious ``target_accept=0.9``.
        **sample_kwargs
            Additional keyword arguments forwarded to :func:`pm.sample` —
            for example ``cores``, ``init``, or ``nuts_sampler``.

        Returns
        -------
        BayesianVECM
            The fitted model (``self``), to support method chaining.

        Raises
        ------
        NotImplementedError
            If ``coint_rank != 1`` or ``deterministic != "n"`` — the PyMC
            graph for those configurations is not yet implemented.
        ValueError
            If ``endog`` fails validation, or if ``priors`` is malformed.
        """
        # Capture column labels before validate_endog drops the DataFrame wrapper.
        variable_names: list[str] | None = None
        if hasattr(endog, "columns"):
            variable_names = [str(c) for c in endog.columns]

        endog_arr = validate_endog(endog)

        design = cointegration_design(
            endog_arr,
            k_ar_diff=self.k_ar_diff,
            deterministic=self.deterministic,
        )

        model = build_pymc_model(
            design,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic,
            priors=self.priors,
        )

        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                **sample_kwargs,
            )

        # Stash the raw endog in constant_data so a serialised idata is a
        # self-contained record — forecasting only needs the last few rows,
        # but anyone reloading the file can reconstruct everything.
        endog_da_kwargs: dict[str, Any] = {"dims": ("time", "variable")}
        if variable_names is not None:
            endog_da_kwargs["coords"] = {"variable": variable_names}
        idata.constant_data["endog"] = xr.DataArray(endog_arr, **endog_da_kwargs)

        # Fit-time state — sklearn-style trailing-underscore convention.
        self.endog_ = endog_arr
        self.idata_ = idata
        self.variable_names_ = variable_names
        return self

    @property
    def idata(self) -> az.InferenceData:
        """Posterior record from the most recent ``fit``.

        Returns
        -------
        arviz.InferenceData
            The full posterior trace, including ``constant_data`` with the
            raw ``endog`` and the design matrices.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)
        return self.idata_

    def summary(self, **summary_kwargs: Any) -> Any:
        """Return a tabular summary of the headline posterior parameters.

        Thin wrapper around :func:`arviz.summary`. By default reports
        :math:`\\alpha`, :math:`\\beta`, :math:`\\Sigma`, and — when
        ``k_ar_diff > 0`` — :math:`\\Gamma`. Pass ``var_names=...`` to
        override; any other ``arviz.summary`` keyword (``hdi_prob``,
        ``round_to``, etc.) is forwarded through.

        Returns
        -------
        pandas.DataFrame
            One row per scalar parameter, columns ``mean``, ``sd``,
            ``hdi_3%``, ``hdi_97%``, ``ess_bulk``, ``ess_tail``, ``r_hat``
            (defaults from ArviZ).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        if "var_names" not in summary_kwargs:
            var_names = list(_SUMMARY_VAR_NAMES_BASE)
            if self.k_ar_diff > 0:
                # Insert before Sigma so the printout reads alpha, beta, Gamma, Sigma.
                var_names.insert(2, "Gamma")
            summary_kwargs["var_names"] = var_names

        return az.summary(self.idata_, **summary_kwargs)

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
            "deferred to its own follow-up slice. The first-PyMC-model slice "
            "shipped fit, idata and summary; forecasting is meaningfully its "
            "own design problem (drawing innovations from Σ, rolling the "
            "VAR recursion forward) and is the next thing on the roadmap."
        )
