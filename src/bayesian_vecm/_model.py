"""Public ``BayesianVECM`` class.

This module defines the model class that the rest of the package hangs off.
``fit`` builds and samples the PyMC graph. The output methods (forecasts,
impulse responses, FEVD, Granger causality and residual checks) read the
fitted posterior.

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

Scope
-----
``fit`` estimates any fixed rank :math:`1 \\le r < K`, any single
deterministic code (``"n"``, ``"co"``, ``"ci"``, ``"lo"``, ``"li"``) and
optional exogenous regressors. The rank is chosen beforehand, for example
with :func:`~bayesian_vecm.select_coint_rank`. Compound deterministic codes
are not supported.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from bayesian_vecm._causality import CausalityResult

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

from bayesian_vecm._constants import VALID_DETERMINISTIC
from bayesian_vecm._data import validate_endog
from bayesian_vecm._design import cointegration_design
from bayesian_vecm._pymc import build_pymc_model

# Default headline parameters shown by ``summary()``. ``Gamma`` is added
# conditionally when ``k_ar_diff > 0``.
_SUMMARY_VAR_NAMES_BASE = ("alpha", "beta", "Sigma")


def _driver_list(items: Any, label: str) -> list | None:
    """Normalise an ``adstock`` / ``saturation`` argument to a list or ``None``."""
    if items is None:
        return None
    if isinstance(items, (str, int)):
        items = [items]
    items = list(items)
    if not items:
        raise ValueError(f"{label} must name at least one driver, or be None")
    if len(set(items)) != len(items):
        raise ValueError(f"{label} has duplicate entries: {items}")
    return items


def _saturation_scale(exog: Any, idx: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """In-sample maximum and mean of each saturated driver's raw spend."""
    arr = np.asarray(exog.to_numpy() if hasattr(exog, "to_numpy") else exog, dtype=np.float64)
    cols = arr[:, list(idx)]
    if (cols < 0).any():
        raise ValueError(
            "saturated drivers must be raw, non-negative spend; "
            "found negative values (pass the raw series, not deviations from the average)"
        )
    scale = cols.max(axis=0)
    if (scale <= 0).any():
        raise ValueError("every saturated driver needs some positive spend")
    return scale, cols.mean(axis=0)


_NOT_FITTED_MSG = "BayesianVECM has not been fitted yet; call .fit(endog) first."


class BayesianVECM:
    """Bayesian Vector Error Correction Model.

    A fixed-rank cointegrated VAR estimated by Bayesian inference. The model
    being targeted is

    .. math::

        \\Delta y_t = \\alpha \\beta' y_{t-1}
                     + \\sum_{i=1}^{k} \\Gamma_i \\, \\Delta y_{t-i}
                     + \\sum_{l=0}^{L} B_l x_{t-l}
                     + D_t \\Phi
                     + \\varepsilon_t,
                     \\quad \\varepsilon_t \\sim \\mathcal{N}(0, \\Sigma),

    where :math:`\\alpha, \\beta` are :math:`K \\times r`, the :math:`\\Gamma_i`
    are :math:`K \\times K` short-run dynamics, and :math:`D_t` collects any
    requested deterministic terms (constants, trends, inside or outside the
    cointegration relation). :math:`x_t` holds the exogenous regressors
    passed as ``exog`` to :meth:`fit`, and :math:`L` is ``exog_lags``.

    Parameters
    ----------
    k_ar_diff
        Number of lagged-difference blocks :math:`\\Delta y_{t-1}, \\dots,
        \\Delta y_{t-k}` to include. ``0`` means no short-run dynamics — a
        pure error-correction equation. Defaults to ``1``.
    coint_rank
        Cointegration rank :math:`r`. Sets the shared inner dimension of
        :math:`\\alpha` and :math:`\\beta`. Must be at least ``1`` and below
        :math:`K`.
        Defaults to ``1``.
    deterministic
        Deterministic-term code, following the convention of
        :mod:`statsmodels.tsa.vector_ar.vecm`. Single codes only:

        * ``"n"`` — no deterministic terms (default).
        * ``"co"`` — constant outside the cointegration relation.
        * ``"ci"`` — constant inside the cointegration relation.
        * ``"lo"`` — linear trend outside the cointegration relation.
        * ``"li"`` — linear trend inside the cointegration relation.

        Compound codes (Johansen cases 4 and 5) are not supported.
    priors
        Optional mapping from parameter name to distribution spec. Keys
        recognised: ``"alpha"``, ``"beta"``, ``"Gamma"``, ``"Sigma"``, ``"B"``,
        ``"adstock_alpha"``, ``"saturation_lam"``. With ``adstock`` or
        ``saturation``, ``"B"`` sets the prior on the total effects
        ``B_total`` (and on the free lags of other drivers).
        Each value is a dict of the form ``{"dist": "<DistName>", **kwargs}``
        — for example ``{"dist": "Normal", "mu": 0.0, "sigma": 1.0}``. Any
        parameter omitted from the dict falls back to a weakly-informative
        default chosen at ``fit`` time. ``None`` (the default) means "use
        defaults for everything".
    exog_lags
        Number of past periods of each ``exog`` driver to include,
        :math:`L`. ``0`` (default) keeps the same period only. Use ``1`` or
        ``2`` when a driver acts with a delay, for example on monthly data
        where spend late in a month moves next month's average. Lags of a
        flighted driver are highly correlated, so read the effect summed
        over lags (:meth:`exog_irf`, :meth:`granger_causality`) rather than
        each lag on its own.
    adstock
        Drivers whose effect carries over with geometric decay (adstock),
        given as ``exog`` column names or positions. For each, the lag-:math:`l`
        coefficient is :math:`b \\, w^l / \\sum_{i=0}^{L_a} w^i`: ``B_total``
        (:math:`b`, prior ``Normal(0, 1)``) is the total effect on each
        series and ``adstock_alpha`` (:math:`w`, prior ``Beta(1, 1)``, flat) is one
        decay per driver, shared across series. ``B`` keeps its usual layout,
        so :meth:`exog_irf`, forecasts and :meth:`granger_causality` read it
        as before. Drivers not listed keep free coefficients on lags
        ``0..exog_lags``. ``None`` (default) means no adstock.
    adstock_max_lag
        Length of the adstock window, :math:`L_a`. Defaults to ``8``, which
        suits weekly data with decays up to about 0.7 (at 0.9, lag 8 still
        carries 43% of the same-week weight). The first
        ``max(exog_lags, adstock_max_lag) - k_ar_diff - 1`` rows are dropped
        so every lag exists. Ignored without ``adstock``.
    saturation
        Drivers whose effect flattens as spend grows (logistic saturation),
        given as ``exog`` column names or positions. Their ``exog`` columns
        must be **raw, non-negative spend**, not deviations from the
        average; other columns are unchanged. Each acts through
        :math:`b \\, [f(a_t / s) - \\bar f]`, with :math:`a_t` the spend after
        adstock (same-week spend if not adstocked), :math:`s` the in-sample
        maximum of spend, :math:`f(u) = \\tanh(\\lambda u / 2)` and
        :math:`\\bar f` its in-sample mean. ``saturation_lam``
        (:math:`\\lambda`, prior ``Gamma(2, 0.3)``) sets how fast it bends.
        ``B_total`` (prior ``Normal(0, 1)``) is the total effect of one more
        unit at mean spend, the quantity the data pin down best, and
        ``B_saturation`` (:math:`b`, the effect at full saturation) is
        derived from it. The return on one more unit depends on the spend
        level: ``B``, :meth:`exog_irf` and :meth:`granger_causality` give it
        at mean spend, ``exog_irf(at=...)`` at another level and
        :meth:`saturation_curve` across levels. Forecasts use the exact
        curve. ``None`` (default) means no saturation.
    contemporaneous
        Same-week links between the brand metrics to estimate, as
        ``{effect: [causes]}`` with ``endog`` column names, for example
        ``{"brand_consideration": ["brand_awareness"]}``. The drivers then
        act through the structural form :math:`A_0 \\Delta y_t = \\dots +
        B_s x_t` with :math:`A_0 = I - L` and :math:`L` holding the links
        (``A0_links``, prior ``Normal(0, 0.5)``). ``B``, ``B_total`` and
        ``B_saturation`` stay reduced form, :math:`A_0^{-1} B_s`, so every
        other method reads them as before. ``B_structural`` holds the
        direct effects. :math:`\\Sigma` stays a full covariance: the links
        are identified only by ``excluded``. Links must not form a cycle.
        Run :func:`~bayesian_vecm.granger_pretest` with the same arguments
        first. ``None`` (default) means no links.
    excluded
        Drivers with no direct same-week effect on a brand metric, as
        ``{effect: [drivers]}`` with ``endog`` and ``exog`` column names.
        Their entries of :math:`B_s` are zero at every lag. Each effect in
        ``contemporaneous`` needs at least as many excluded drivers as
        causes. Can be used without ``contemporaneous`` to drop effects.

    Raises
    ------
    ValueError
        If ``k_ar_diff`` or ``exog_lags`` is negative, ``coint_rank`` is less
        than ``1``, ``deterministic`` is not one of the recognised codes,
        ``adstock_max_lag`` is below ``1``, or ``adstock`` or ``saturation``
        is empty or has duplicates. At fit time, if ``contemporaneous`` or
        ``excluded`` names an unknown column, the links form a cycle or a
        link is not identified.
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

    Fit, then forecast with the drivers at chosen values::

        >>> model.fit(endog, exog=exog)                        # doctest: +SKIP
        >>> fc = model.sample_posterior_predictive(52, exog_future=x_future)  # doctest: +SKIP
    """

    def __init__(
        self,
        k_ar_diff: int = 1,
        coint_rank: int = 1,
        deterministic: str = "n",
        priors: dict[str, Any] | None = None,
        exog_lags: int = 0,
        adstock: Sequence[str | int] | None = None,
        adstock_max_lag: int = 8,
        saturation: Sequence[str | int] | None = None,
        contemporaneous: dict[str, list[str]] | None = None,
        excluded: dict[str, list[str]] | None = None,
    ) -> None:
        """Configure the model. See the class docstring for parameter details.

        No data is required here: data is passed to :meth:`fit`. Building a
        model instance does not run any PyMC code. The graph is built lazily
        on the first call to :meth:`fit`, once :math:`K` (the number of
        endogenous series) is known.
        """
        if k_ar_diff < 0:
            raise ValueError(f"k_ar_diff must be non-negative; got k_ar_diff={k_ar_diff}")

        if coint_rank < 1:
            raise ValueError(f"coint_rank must be at least 1; got coint_rank={coint_rank}")

        if exog_lags < 0:
            raise ValueError(f"exog_lags must be non-negative; got exog_lags={exog_lags}")

        if adstock_max_lag < 1:
            raise ValueError(
                f"adstock_max_lag must be at least 1; got adstock_max_lag={adstock_max_lag}"
            )

        adstock = _driver_list(adstock, "adstock")
        saturation = _driver_list(saturation, "saturation")

        if deterministic not in VALID_DETERMINISTIC:
            valid = sorted(VALID_DETERMINISTIC)
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
        self.exog_lags = exog_lags
        self.adstock = adstock
        self.adstock_max_lag = adstock_max_lag
        self.saturation = saturation
        self.contemporaneous = contemporaneous
        self.excluded = excluded

    def fit(
        self,
        endog: Any,
        *,
        exog: Any = None,
        exog_coint: Any = None,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        progressbar: bool = True,
        **sample_kwargs: Any,
    ) -> BayesianVECM:
        """Fit the model to ``endog`` by running PyMC sampling.

        Parameters
        ----------
        endog
            Endogenous time series of shape ``(T, K)``. Accepts anything
            :func:`bayesian_vecm._data.validate_endog` accepts — including
            objects with a ``.to_numpy()`` method, such as a
            ``pandas.DataFrame``. Column labels (if present) are captured
            into ``self.variable_names_``.
        exog
            Optional exogenous regressors, shape ``(T, m)``. These enter
            the short-run equation as :math:`\\sum_{l=0}^{L} B_l X_{t-l}`
            with :math:`L` = ``exog_lags``. The posterior ``B`` is
            ``(K, m * (L + 1))``, lag-major, with a
            ``Normal(0, 1 / sqrt(L + 1))`` prior on each entry. With
            ``exog_lags > k_ar_diff + 1`` the first
            ``exog_lags - k_ar_diff - 1`` rows are dropped. Pass ``None``
            (default) to omit.
        exog_coint
            Optional exogenous variables inside the cointegrating relation,
            shape ``(T, m_c)``. Their columns are appended to :math:`y_{t-1}`
            before building the cointegration term — the same mechanism used
            by ``deterministic="ci"`` / ``"li"``. Pass ``None`` (default) to
            omit.
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
        ValueError
            If ``endog`` fails validation, if ``exog`` / ``exog_coint`` have
            the wrong shape, or if ``priors`` is malformed.
        """
        # Capture column labels before validate_endog drops the DataFrame wrapper.
        variable_names: list[str] | None = None
        if hasattr(endog, "columns"):
            variable_names = [str(c) for c in endog.columns]

        endog_arr = validate_endog(endog)

        adstock_idx = self._resolve_drivers(getattr(self, "adstock", None), exog, "adstock")
        saturation_idx = self._resolve_drivers(
            getattr(self, "saturation", None), exog, "saturation"
        )
        window = self._window_for(adstock_idx)
        sat_scale = sat_level = None
        if saturation_idx:
            sat_scale, sat_level = _saturation_scale(exog, saturation_idx)

        links, b_mask = self._resolve_same_week(variable_names, endog_arr.shape[1], exog)

        design = cointegration_design(
            endog_arr,
            k_ar_diff=self.k_ar_diff,
            deterministic=self.deterministic,
            exog=exog,
            exog_coint=exog_coint,
            exog_lags=window,
        )

        model = build_pymc_model(
            design,
            k_ar_diff=self.k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=self.deterministic,
            priors=self.priors,
            exog_lags=self.exog_lags,
            adstock=adstock_idx,
            adstock_max_lag=self.adstock_max_lag,
            saturation=saturation_idx,
            saturation_scale=sat_scale,
            saturation_level=sat_level,
            links=links,
            b_mask=b_mask,
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

        # Stash exog arrays in constant_data for fittedvalues / resid reuse.
        if design.exog is not None:
            idata.constant_data["exog"] = xr.DataArray(
                design.exog, dims=("time_eff", "exog_variable")
            )
        if saturation_idx:
            idata.constant_data["saturation_idx"] = xr.DataArray(
                np.array(saturation_idx), dims=("saturation_variable",)
            )
            idata.constant_data["saturation_scale"] = xr.DataArray(
                sat_scale, dims=("saturation_variable",)
            )
        if exog_coint is not None:
            # Store the raw (T, m_c) array — aligned slice is embedded in y_lag1.
            exog_coint_arr = np.asarray(
                exog_coint.to_numpy() if hasattr(exog_coint, "to_numpy") else exog_coint,
                dtype=np.float64,
            )
            idata.constant_data["exog_coint"] = xr.DataArray(
                exog_coint_arr, dims=("time", "exog_coint_variable")
            )

        # Fit-time state — sklearn-style trailing-underscore convention.
        self.endog_ = endog_arr
        self.idata_ = idata
        self.variable_names_ = variable_names
        self.exog_names_ = [str(c) for c in exog.columns] if hasattr(exog, "columns") else None
        self.exog_ = design.exog  # (T_eff, m * (window + 1)) or None
        self.adstock_idx_ = adstock_idx
        self.saturation_idx_ = saturation_idx
        # Drivers with a B_total column, in its order.
        self.total_idx_ = adstock_idx + tuple(j for j in saturation_idx if j not in adstock_idx)
        self.saturation_scale_ = sat_scale
        self.saturation_level_ = sat_level
        self.exog_window_ = window
        self.links_ = links
        # The last `window` rows of the raw drivers seed the lags in a forecast.
        self.exog_tail_ = None
        if exog is not None:
            exog_raw = np.asarray(
                exog.to_numpy() if hasattr(exog, "to_numpy") else exog, dtype=np.float64
            )
            self.exog_tail_ = exog_raw[exog_raw.shape[0] - window :]
            self._exog_recent = exog_raw[-13:].mean(axis=0)
        self.exog_coint_ = (
            None
            if exog_coint is None
            else np.asarray(
                exog_coint.to_numpy() if hasattr(exog_coint, "to_numpy") else exog_coint,
                dtype=np.float64,
            )
        )
        return self

    def _resolve_same_week(
        self, variable_names: list[str] | None, n_vars: int, exog: Any
    ) -> tuple[tuple[tuple[int, int], ...], np.ndarray | None]:
        """Turn ``contemporaneous`` and ``excluded`` into links and a ``B`` mask."""
        links_arg = getattr(self, "contemporaneous", None)
        excl_arg = getattr(self, "excluded", None)
        if not links_arg and not excl_arg:
            return (), None
        if exog is None:
            raise ValueError(
                "contemporaneous and excluded need exog: the links are identified "
                "by drivers excluded from an equation"
            )
        from bayesian_vecm._same_week import resolve_same_week

        names = variable_names or [f"y{i}" for i in range(n_vars)]
        n_exog = np.shape(exog)[1]
        exog_names = (
            [str(c) for c in exog.columns]
            if hasattr(exog, "columns")
            else [f"x{j}" for j in range(n_exog)]
        )
        spec = resolve_same_week(links_arg, excl_arg, names, exog_names, require_dag=True)
        for effect, causes in spec.links.items():
            if not spec.identified(effect):
                n_excl = len(spec.excluded.get(effect, ()))
                raise ValueError(
                    f"same-week links into {names[effect]!r} are not identified: "
                    f"{len(causes)} cause(s) need at least as many drivers in "
                    f"excluded[{names[effect]!r}]; got {n_excl}"
                )
        mask = np.ones((n_vars, n_exog))
        for effect, drivers in spec.excluded.items():
            mask[effect, list(drivers)] = 0.0
        links = tuple(
            (effect, cause) for effect, causes in sorted(spec.links.items()) for cause in causes
        )
        return links, mask

    def same_week_links(self, *, threshold: float = 0.95, ci_prob: float = 0.89) -> pd.DataFrame:
        """Posterior of each same-week link in ``contemporaneous``.

        Parameters
        ----------
        threshold
            Evidence at or above this is reported as a link.
        ci_prob
            Probability mass of the equal-tailed interval.

        Returns
        -------
        pandas.DataFrame
            One row per link. Columns ``cause``, ``effect``, ``mean``,
            ``lower``, ``upper``, ``evidence`` and ``link``. ``evidence`` is
            as in :meth:`granger_causality`: for one coefficient,
            ``1 - 2 * min(P(> 0), P(< 0))``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If the model has no ``contemporaneous`` links.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)
        links = getattr(self, "links_", ())
        if not links:
            raise ValueError("the model has no same-week links; set contemporaneous=")
        import pandas as pd

        from bayesian_vecm._causality import evidence_against_zero

        draws = np.asarray(self.idata_.posterior["A0_links"].values, dtype=np.float64)
        draws = draws.reshape(-1, draws.shape[-1])
        names = self.variable_names_ or [f"y{i}" for i in range(self.endog_.shape[1])]
        tail = (1.0 - ci_prob) / 2.0
        rows = []
        for i, (effect, cause) in enumerate(links):
            d = draws[:, i]
            lower, upper = np.quantile(d, [tail, 1.0 - tail])
            evidence = evidence_against_zero(d)
            rows.append(
                {
                    "cause": names[cause],
                    "effect": names[effect],
                    "mean": float(d.mean()),
                    "lower": float(lower),
                    "upper": float(upper),
                    "evidence": evidence,
                    "link": evidence >= threshold,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _resolve_drivers(items: list | None, exog: Any, label: str) -> tuple[int, ...]:
        """Map driver names or positions to ``exog`` column indices."""
        if items is None:
            return ()
        if exog is None:
            raise ValueError(f"{label} needs exog: pass the drivers to fit(exog=...)")
        n_cols = np.shape(exog)[1] if np.ndim(exog) == 2 else None
        names = [str(c) for c in exog.columns] if hasattr(exog, "columns") else None
        idx = []
        for item in items:
            if isinstance(item, str):
                if names is None or item not in names:
                    raise ValueError(
                        f"{label} driver {item!r} is not an exog column; columns are {names}"
                    )
                idx.append(names.index(item))
            else:
                if n_cols is None or not 0 <= int(item) < n_cols:
                    raise ValueError(f"{label} position {item} is out of range for exog")
                idx.append(int(item))
        if len(set(idx)) != len(idx):
            raise ValueError(f"{label} names the same driver twice: {items}")
        return tuple(idx)

    def _window_for(self, adstock_idx: tuple[int, ...]) -> int:
        """Number of past periods of ``exog`` in the design."""
        if adstock_idx:
            return max(self.exog_lags, self.adstock_max_lag)
        return self.exog_lags

    def _exog_window(self) -> int:
        """Lag window of a fitted model (falls back for older pickles)."""
        return getattr(self, "exog_window_", self.__dict__.get("exog_lags", 0))

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
            if getattr(self, "exog_", None) is not None:
                # Append B after Gamma (or after beta if no Gamma) but before Sigma.
                var_names.insert(-1, "B")
            extra = []
            if getattr(self, "total_idx_", getattr(self, "adstock_idx_", ())):
                extra.append("B_total")
            if getattr(self, "adstock_idx_", ()):
                extra.append("adstock_alpha")
            if getattr(self, "saturation_idx_", ()):
                extra.append("saturation_lam")
            if getattr(self, "links_", ()):
                extra.append("A0_links")
            var_names[-1:-1] = extra
            summary_kwargs["var_names"] = var_names

        return az.summary(self.idata_, **summary_kwargs)

    @property
    def fittedvalues(self) -> xr.DataArray:
        """Posterior in-sample fitted values of :math:`\\Delta y`.

        Computes :math:`\\hat{\\Delta y}_t = \\alpha\\beta^\\top y_{t-1} +
        \\Gamma\\Delta x_t` for every posterior draw.

        Returns
        -------
        xarray.DataArray
            Shape ``(chain, draw, time, variable)``.  ``time`` is zero-indexed
            over the :math:`T_{\\text{eff}}` in-sample observations.  Use
            ``.mean(("chain", "draw"))`` for the posterior-mean fitted values.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        from bayesian_vecm._output import compute_fittedvalues

        return compute_fittedvalues(self.idata_, self.k_ar_diff, self.variable_names_)

    @property
    def resid(self) -> xr.DataArray:
        """Posterior in-sample residuals :math:`\\Delta y - \\hat{\\Delta y}`.

        Returns
        -------
        xarray.DataArray
            Shape ``(chain, draw, time, variable)``.  Use
            ``.mean(("chain", "draw"))`` for posterior-mean residuals.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        from bayesian_vecm._output import compute_resid

        return compute_resid(self.idata_, self.k_ar_diff, self.variable_names_)

    @property
    def var_rep(self) -> xr.DataArray:
        """Posterior levels VAR(:math:`p`) coefficient matrices.

        Reconstructs :math:`A_1, \\dots, A_p` from the VECM posterior using
        the standard VECM-to-VAR(p) conversion
        (:math:`p = k_{\\text{ar\\_diff}} + 1`).

        Returns
        -------
        xarray.DataArray
            Shape ``(chain, draw, lag, response_variable, shock_variable)``.
            ``lag`` runs from ``1`` to ``p``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        from bayesian_vecm._output import compute_var_rep

        return compute_var_rep(self.idata_, self.k_ar_diff, self.variable_names_)

    def test_normality(self) -> pd.DataFrame:
        """Jarque-Bera normality test on posterior-mean residuals.

        Tests each endogenous variable independently.  H0: residuals are
        normally distributed.

        Returns
        -------
        pandas.DataFrame
            One row per variable, columns ``jb_stat`` and ``p_value``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        from bayesian_vecm._diagnostics import normality_test

        return normality_test(self.idata_, self.k_ar_diff, self.variable_names_)

    def test_whiteness(self, *, lags: int = 10) -> pd.DataFrame:
        """Ljung-Box whiteness test on posterior-mean residuals.

        Tests each endogenous variable independently for autocorrelation up
        to ``lags`` periods.  H0: no autocorrelation.

        Parameters
        ----------
        lags
            Lag order for the portmanteau Q-statistic.  Defaults to ``10``.

        Returns
        -------
        pandas.DataFrame
            One row per variable, columns ``lb_stat``, ``p_value``, ``lags``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``lags < 1``.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        from bayesian_vecm._diagnostics import whiteness_test

        return whiteness_test(
            self.idata_, self.k_ar_diff, lags=lags, variable_names=self.variable_names_
        )

    def irf(
        self,
        steps: int,
        *,
        method: str = "girf",
        order: Sequence[str | int] | None = None,
    ) -> xr.DataArray:
        """Compute posterior Impulse Response Functions for *steps* horizons.

        Returns a posterior distribution over IRF paths — the response of
        each variable to a unit shock in each other variable, for every
        posterior draw.

        Two identification schemes are available:

        * ``"girf"`` (default) — Generalised IRFs (Pesaran & Shin 1998).
          Order-invariant. The response to a shock in variable j equals the
          Cholesky IRF with j ordered first, so it is structural only if j
          leads the other variables within the period.
        * ``"cholesky"`` — Orthogonalised IRFs (Sims 1980).  Requires a
          defensible recursive causal ordering among the variables: a shock
          to one variable moves the variables after it in the same period,
          and not the ones before it. Set the ordering with ``order``.

        Parameters
        ----------
        steps
            Number of periods ahead to compute IRFs.  Horizons
            :math:`h = 0, 1, \\dots, \\text{steps}` are returned, giving
            ``steps + 1`` entries along the ``horizon`` dimension.
        method
            Identification scheme — ``"girf"`` or ``"cholesky"``.
        order
            Cholesky ordering, listing every endogenous variable once by
            name or column index. ``None`` uses the column order of
            ``endog``. The output keeps the column order either way. Only
            valid with ``method="cholesky"``. Changing the column order of
            ``endog`` instead would also change the normalisation of
            :math:`\\beta`.

        Returns
        -------
        xarray.DataArray
            Shape ``(chain, draw, horizon, response_variable, shock_variable)``.
            ``horizon`` runs from ``0`` (impact) to ``steps``.
            Entry ``[..., h, i, j]`` is the response of variable :math:`i`
            to a unit shock in variable :math:`j` at horizon :math:`h`.
            ``response_variable`` and ``shock_variable`` coordinates are set
            when ``self.variable_names_`` is available.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``steps < 1``, ``method`` is unrecognised, or ``order`` is
            invalid.
        TypeError
            If ``order`` is a string or holds entries that are not names or
            integers.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)
        if steps < 1:
            raise ValueError(f"steps must be at least 1; got steps={steps}")

        from bayesian_vecm._irf import compute_irf

        return compute_irf(
            idata=self.idata_,
            k_ar_diff=self.k_ar_diff,
            steps=steps,
            method=method,
            variable_names=self.variable_names_,
            order=order,
        )

    def exog_irf(self, steps: int, *, at: Any = "mean") -> xr.DataArray:
        """Posterior response of the levels to a one-period rise in each exogenous driver.

        Each exogenous variable is raised by one unit for one period, with
        every other input at its baseline. The path is
        :math:`\\Phi_h B` for :math:`h = 0, \\dots, \\text{steps}`. No
        identification scheme is needed: the driver is observed, so its
        impact is a column of :math:`B`.

        Use the result to value a driver. With log-transformed ``endog``,
        entry ``[..., h, i, j]`` is the log change in variable :math:`i`
        at horizon :math:`h` per unit of driver :math:`j`. The cumulative
        sum over ``horizon`` is the response to a rise held for every
        period. In a cointegrated system the path settles at :math:`C B`,
        so a one-period rise can move the levels for good.

        With ``exog_lags = L`` the lag-:math:`l` block :math:`B_l` acts
        :math:`l` periods after the rise, so the path is
        :math:`\\sum_{l} \\Phi_{h-l} B_l` and settles at
        :math:`C \\sum_l B_l`. The result has one column per driver, with
        all lags included.

        Without ``saturation`` the model is linear in the drivers, so the
        result is the same at every spend level. A saturated driver's return
        on one more unit depends on how much is already spent: the result
        is its slope at the level set by ``at``, carried through the same
        lags. For a change larger than a few units, or to see the whole
        curve, use :meth:`saturation_curve` or a forecast.

        Only ``exog`` is covered. Variables passed as ``exog_coint`` enter
        the cointegrating relation and have no response here.

        Parameters
        ----------
        steps
            Number of periods ahead. ``steps + 1`` horizons are returned,
            from ``0`` (impact) to ``steps``.
        at
            Spend level at which saturated drivers are valued. ``"mean"``
            (default) is the in-sample mean, ``"recent"`` the mean of the
            last 13 periods, and a dict maps driver names (or positions) to
            levels in raw units, with the mean for any left out. Ignored
            without ``saturation``.

        Returns
        -------
        xarray.DataArray
            Shape ``(chain, draw, horizon, response_variable, exog_variable)``.
            Coordinates are set when variable names are available.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``steps < 1`` or the model was fitted without ``exog``.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        from bayesian_vecm._irf import compute_exog_irf

        idata = self.idata_
        if getattr(self, "saturation_idx_", ()) and not (isinstance(at, str) and at == "mean"):
            from types import SimpleNamespace

            from bayesian_vecm._saturation import linearised_b

            b = linearised_b(idata, self._saturation_levels(at))
            posterior = idata.posterior
            posterior = (
                posterior.to_dataset() if hasattr(posterior, "to_dataset") else posterior
            ).copy()
            posterior["B"] = (posterior["B"].dims, b)
            idata = SimpleNamespace(posterior=posterior)
        return compute_exog_irf(
            idata=idata,
            k_ar_diff=self.k_ar_diff,
            steps=steps,
            variable_names=self.variable_names_,
            exog_names=getattr(self, "exog_names_", None),
            exog_lags=self._exog_window(),
        )

    def saturation_curve(
        self,
        driver: str | int,
        levels: Any,
        *,
        marginal: bool = False,
    ) -> xr.DataArray:
        """Posterior response curve of a saturated driver.

        The total effect on each series' weekly growth of holding the driver
        at each spend level for good, against its in-sample average
        contribution. Adstock does not change a level held for ever. The
        effect then passes through the rest of the system as any other
        shift in growth would; use a forecast for its path in the levels.

        Parameters
        ----------
        driver
            Name or position of a saturated driver.
        levels
            Spend levels in raw units.
        marginal
            Return the slope per unit at each level (the return on one more
            unit) instead of the effect.

        Returns
        -------
        xarray.DataArray
            Shape ``(chain, draw, level, response_variable)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``driver`` is not saturated or a level is negative.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)
        from bayesian_vecm._saturation import response_curve

        j = self._saturated_position(driver)
        levels_arr = np.atleast_1d(np.asarray(levels, dtype=np.float64))
        if (levels_arr < 0).any():
            raise ValueError("spend levels must be non-negative")
        curve = response_curve(self.idata_, j, levels_arr, marginal=marginal)
        coords: dict[str, Any] = {
            "chain": self.idata_.posterior.coords["chain"].values,
            "draw": self.idata_.posterior.coords["draw"].values,
            "level": levels_arr,
        }
        if self.variable_names_ is not None:
            coords["response_variable"] = self.variable_names_
        return xr.DataArray(
            curve, dims=["chain", "draw", "level", "response_variable"], coords=coords
        )

    def _saturated_position(self, driver: str | int) -> int:
        """Position in ``exog`` of a saturated driver given by name or position."""
        sat = tuple(getattr(self, "saturation_idx_", ()))
        names = getattr(self, "exog_names_", None)
        if isinstance(driver, str):
            if names is None or driver not in names:
                raise ValueError(f"{driver!r} is not an exog column")
            j = names.index(driver)
        else:
            j = int(driver)
        if j not in sat:
            raise ValueError(f"driver {driver!r} is not saturated")
        return j

    def _saturation_levels(self, at: Any) -> np.ndarray:
        """Spend level per saturated driver for ``exog_irf(at=...)``."""
        sat = list(self.saturation_idx_)
        levels = np.array(self.saturation_level_, dtype=np.float64)
        if isinstance(at, str):
            if at == "mean":
                return levels
            if at == "recent":
                recent = self._exog_recent
                return np.array([recent[j] for j in sat])
            raise ValueError(f'at must be "mean", "recent" or a dict; got {at!r}')
        if not isinstance(at, dict):
            raise ValueError(f'at must be "mean", "recent" or a dict; got {at!r}')
        for key, value in at.items():
            j = self._saturated_position(key)
            if value < 0:
                raise ValueError("spend levels must be non-negative")
            levels[sat.index(j)] = float(value)
        return levels

    def fevd(
        self,
        steps: int,
        *,
        method: str = "girf",
        order: Sequence[str | int] | None = None,
    ) -> xr.DataArray:
        """Compute the posterior forecast error variance decomposition.

        For each variable and horizon, the share of its forecast error
        variance that is due to shocks in each variable.

        Parameters
        ----------
        steps
            Longest forecast horizon. Horizons ``1, ..., steps`` are returned.
        method
            ``"girf"`` (default) does not depend on the column order. Rows are
            rescaled to sum to 1. ``"cholesky"`` uses the column order, the
            same as :meth:`irf`.
        order
            Cholesky ordering, as in :meth:`irf`. Only valid with
            ``method="cholesky"``.

        Returns
        -------
        xarray.DataArray
            Shape ``(chain, draw, horizon, response_variable, shock_variable)``.
            Each row sums to 1 over ``shock_variable``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``steps < 1``, ``method`` is unrecognised, or ``order`` is
            invalid.
        TypeError
            If ``order`` is a string or holds entries that are not names or
            integers.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        from bayesian_vecm._fevd import compute_fevd

        return compute_fevd(
            idata=self.idata_,
            k_ar_diff=self.k_ar_diff,
            steps=steps,
            method=method,
            variable_names=self.variable_names_,
            order=order,
        )

    def granger_causality(
        self,
        *,
        threshold: float = 0.95,
        ci_prob: float = 0.89,
    ) -> CausalityResult:
        """Summarise the posterior evidence on what helps predict what.

        Returns four tables: short-run Granger causality from
        :math:`\\Gamma`, long-run links from :math:`\\Pi = \\alpha
        \\beta^\\top`, weak exogeneity from the rows of :math:`\\alpha`,
        and exogenous drivers from :math:`B`. Print the result for a
        plain-English summary.

        The evidence for each link is a number between 0 and 1: the highest
        credible level at which zero lies outside the posterior region. These
        are statements about prediction, not about cause.

        Parameters
        ----------
        threshold
            Evidence at or above this counts as a link.
        ci_prob
            Probability mass of the equal-tailed interval around each effect.

        Returns
        -------
        CausalityResult

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``threshold`` or ``ci_prob`` is not strictly between 0 and 1.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)

        from bayesian_vecm._causality import compute_causality

        gamma_spec = (self.priors or {}).get("Gamma") or {}
        return compute_causality(
            self.idata_,
            self.k_ar_diff,
            variable_names=self.variable_names_,
            exog_names=getattr(self, "exog_names_", None),
            exog_lags=self._exog_window(),
            adstock=tuple(getattr(self, "total_idx_", getattr(self, "adstock_idx_", ()))),
            threshold=threshold,
            ci_prob=ci_prob,
            gamma_prior=str(gamma_spec.get("dist", "Normal")),
        )

    def sample_posterior_predictive(
        self,
        steps: int,
        *,
        exog_future: Any = None,
        random_seed: int | None = None,
    ) -> xr.DataTree:
        """Forecast ``steps`` periods ahead from the fitted posterior.

        Rolls the VECM recursion forward in levels:

        .. math::

            \\Delta y_{T+h} = \\alpha \\beta' y_{T+h-1}
                             + \\sum_{i=1}^{k} \\Gamma_i \\, \\Delta y_{T+h-i}
                             + \\sum_{l=0}^{L} B_l X_{T+h-l}
                             + d_{T+h}
                             + \\varepsilon_{T+h},
                             \\quad \\varepsilon_{T+h} \\sim \\mathcal{N}(0, \\Sigma)

            y_{T+h} = y_{T+h-1} + \\Delta y_{T+h}

        for :math:`h = 1, \\dots, \\text{steps}`. :math:`d_{T+h}` is the
        constant or trend set by ``deterministic``, continued from the fit.

        Parameters
        ----------
        steps
            Number of periods to forecast. Must be at least ``1``.
        exog_future
            Future values of the exogenous regressors, shape ``(steps, m)``.
            Required when the model was fitted with ``exog``; ignored
            otherwise. Must have exactly ``steps`` rows and the same number
            of columns as the ``exog`` passed to :meth:`fit`. With
            ``exog_lags`` the lags of the first steps are taken from the
            last rows of the ``exog`` passed to :meth:`fit`.
        random_seed
            Seed for the NumPy random generator used to draw innovations.
            Pass an integer for reproducible forecasts; ``None`` (the
            default) gives a fresh unseeded generator.

        Returns
        -------
        xarray.DataTree
            A DataTree whose ``posterior_predictive`` child node
            contains:

            * ``y`` — forecast levels, shape ``(chain, draw, steps, K)``.
            * ``delta_y`` — forecast first differences, same shape.

            Both arrays carry a ``forecast_step`` coordinate running from
            ``1`` to ``steps``, and a ``variable`` coordinate if
            ``self.variable_names_`` is set.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If ``steps`` is less than ``1``, or if the model was fitted with
            ``exog`` but ``exog_future`` is not provided (or has the wrong
            shape).

        Warns
        -----
        UserWarning
            If the model was fitted with ``exog_coint``. The forecast drops
            the ``exog_coint`` rows of :math:`\\beta`, so the
            error-correction term leaves those variables out.
        """
        if not hasattr(self, "idata_"):
            raise RuntimeError(_NOT_FITTED_MSG)
        if steps < 1:
            raise ValueError(f"steps must be at least 1; got steps={steps}")
        if getattr(self, "exog_coint_", None) is not None:
            warnings.warn(
                "This model was fitted with exog_coint, but the forecast does not "
                "use it. The exog_coint rows of beta are dropped, so the "
                "error-correction term leaves those variables out and the "
                "forecast is biased whenever their contribution is not zero.",
                UserWarning,
                stacklevel=2,
            )

        # Validate exog_future when the model was fitted with exog.
        exog_future_arr = None
        if getattr(self, "exog_", None) is not None:
            if exog_future is None:
                raise ValueError(
                    "This model was fitted with exog; exog_future must be provided "
                    "for forecasting. Pass an array of shape (steps, m)."
                )
            exog_future_arr = np.asarray(
                exog_future.to_numpy() if hasattr(exog_future, "to_numpy") else exog_future,
                dtype=np.float64,
            )
            if exog_future_arr.ndim != 2:
                raise ValueError(
                    f"exog_future must be 2-D (steps, m); got shape {exog_future_arr.shape}"
                )
            if exog_future_arr.shape[0] != steps:
                raise ValueError(
                    f"exog_future must have {steps} rows (one per forecast step); "
                    f"got {exog_future_arr.shape[0]}"
                )
            n_lags = self._exog_window() + 1
            m_fit = self.exog_.shape[1] // n_lags  # type: ignore[union-attr]
            if exog_future_arr.shape[1] != m_fit:
                raise ValueError(
                    f"exog_future has {exog_future_arr.shape[1]} columns but the model "
                    f"was fitted with {m_fit} exog column(s)"
                )
            sat_cols = list(getattr(self, "saturation_idx_", ()))
            if sat_cols and (exog_future_arr[:, sat_cols] < 0).any():
                raise ValueError("exog_future has negative values in a saturated driver")
            if n_lags > 1:
                # Row h stacks [X_{T+h}, X_{T+h-1}, ..., X_{T+h-L}], lag-major
                # as in the fit. Early lags come from the end of the fit sample.
                history = np.vstack([self.exog_tail_, exog_future_arr])
                exog_future_arr = np.concatenate(
                    [history[n_lags - 1 - lag : n_lags - 1 - lag + steps] for lag in range(n_lags)],
                    axis=1,
                )

        from bayesian_vecm._forecast import forecast_posterior

        rng = np.random.default_rng(random_seed)
        return forecast_posterior(
            idata=self.idata_,
            endog=self.endog_,
            k_ar_diff=self.k_ar_diff,
            steps=steps,
            variable_names=self.variable_names_,
            exog_future=exog_future_arr,
            rng=rng,
            deterministic=self.deterministic,
        )
