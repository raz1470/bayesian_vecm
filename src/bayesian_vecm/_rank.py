"""Cointegration rank selection via the Johansen trace test.

This module wraps :func:`statsmodels.tsa.vector_ar.vecm.select_coint_rank`
to provide a clean, readable interface that fits the ``bayesian_vecm``
workflow.  Use it before constructing a :class:`~bayesian_vecm.BayesianVECM`
to decide the ``coint_rank`` argument:

.. code-block:: python

    from bayesian_vecm import select_coint_rank

    result = select_coint_rank(endog_df, k_ar_diff=2)
    print(result)          # summary table
    print(result.rank)     # recommended rank

Then pass ``result.rank`` (or your own judgement) as ``coint_rank`` to
``BayesianVECM``.

Notes
-----
Rank selection is a *frequentist* pre-processing step — we use the Johansen
trace test to settle the structural question of how many cointegrating
relations exist before handing off to the Bayesian sampler.  The horseshoe
prior on :math:`\\Gamma` makes this less critical for lag selection (set
``k_ar_diff`` generously and let the prior shrink), but the rank :math:`r`
is not a continuous parameter and must still be specified explicitly.
"""

from __future__ import annotations

import numpy as np

from bayesian_vecm._data import validate_endog


class CointRankResult:
    """Result of a Johansen trace-test rank-selection procedure.

    Attributes
    ----------
    rank : int
        Recommended cointegration rank — the smallest :math:`r` for which
        the trace statistic falls below the 5% critical value.  If the
        statistic exceeds the critical value at all :math:`r < K`, the
        procedure recommends the full rank ``K`` (all series stationary in
        levels — unlikely in practice for a cointegrated system).
    test_stats : np.ndarray, shape (K,)
        Trace test statistics for each null hypothesis
        :math:`H_0: r \\leq 0, 1, \\dots, K-1`.
    crit_vals : np.ndarray, shape (K,)
        Corresponding 5% critical values.
    variable_names : list[str] | None
        Column names from the input, if available.
    det_order : int
        Deterministic-term order passed to the Johansen test (``-1``, ``0``,
        or ``1``).
    k_ar_diff : int
        Number of lagged-difference blocks used in the test.
    """

    def __init__(
        self,
        rank: int,
        test_stats: np.ndarray,
        crit_vals: np.ndarray,
        variable_names: list[str] | None,
        det_order: int,
        k_ar_diff: int,
    ) -> None:
        self.rank = rank
        self.test_stats = test_stats
        self.crit_vals = crit_vals
        self.variable_names = variable_names
        self.det_order = det_order
        self.k_ar_diff = k_ar_diff

    def __repr__(self) -> str:
        return (
            f"CointRankResult(rank={self.rank}, "
            f"k_ar_diff={self.k_ar_diff}, "
            f"det_order={self.det_order})"
        )

    def __str__(self) -> str:
        """Tabular summary of the trace test results."""
        lines = [
            "Johansen cointegration rank test (trace statistic, 5% critical value)",
            f"  k_ar_diff = {self.k_ar_diff}   det_order = {self.det_order}",
            "",
            f"  {'H0: r <=':>9}  {'Trace stat':>12}  {'Crit val (5%)':>14}  {'Reject?':>7}",
            "  " + "-" * 48,
        ]
        for i, (stat, cv) in enumerate(zip(self.test_stats, self.crit_vals, strict=True)):
            reject = "yes" if stat > cv else "no"
            lines.append(f"  {'r <= ' + str(i):>9}  {stat:>12.3f}  {cv:>14.3f}  {reject:>7}")
        lines += [
            "  " + "-" * 48,
            f"  Recommended rank: {self.rank}",
        ]
        return "\n".join(lines)


def select_coint_rank(
    endog,
    *,
    k_ar_diff: int = 1,
    det_order: int = 0,
) -> CointRankResult:
    """Select the cointegration rank via the Johansen trace test.

    Wraps :func:`statsmodels.tsa.vector_ar.vecm.select_coint_rank` and
    returns a :class:`CointRankResult` with a readable summary table and
    the recommended rank.

    Parameters
    ----------
    endog
        Endogenous time-series data.  Accepts the same inputs as
        :func:`~bayesian_vecm._data.validate_endog`: a 2-D ``numpy`` array
        or any array-like with a ``.to_numpy()`` method (e.g. a
        ``pandas.DataFrame``).  Shape ``(T, K)`` — rows are time periods,
        columns are variables.  At least two variables required.
    k_ar_diff
        Number of lagged-difference blocks in the VECM (same as the
        ``k_ar_diff`` you intend to pass to :class:`~bayesian_vecm.BayesianVECM`).
        Passed as the ``k_ar_diffs`` argument to statsmodels.
    det_order
        Deterministic-term order for the Johansen test:
        ``-1`` — no constant/trend,
        ``0``  — constant outside cointegrating relation (default),
        ``1``  — constant inside + trend outside.
        This controls the test's critical values; it does not need to match
        the ``deterministic`` code you use for fitting (though it should be
        in the same ballpark).

    Returns
    -------
    CointRankResult
        Contains the recommended rank, test statistics, critical values, and
        a printable summary table.

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.
    ValueError
        If ``endog`` fails validation or ``det_order`` is not in
        ``{-1, 0, 1}``.
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import select_coint_rank as _sm_select
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for select_coint_rank. "
            "Install it with: uv add --dev statsmodels"
        ) from exc

    if det_order not in (-1, 0, 1):
        raise ValueError(f"det_order must be -1, 0, or 1; got {det_order!r}.")

    data = validate_endog(endog)
    variable_names: list[str] | None = None
    if hasattr(endog, "columns"):
        variable_names = list(endog.columns)

    sm_result = _sm_select(data, det_order=det_order, k_ar_diff=k_ar_diff, signif=0.05)

    # statsmodels stores trace statistics and critical values on the result.
    # The recommended rank is sm_result.rank.
    test_stats = np.asarray(sm_result.test_stats)
    crit_vals = np.asarray(sm_result.crit_vals)

    return CointRankResult(
        rank=int(sm_result.rank),
        test_stats=test_stats,
        crit_vals=crit_vals,
        variable_names=variable_names,
        det_order=det_order,
        k_ar_diff=k_ar_diff,
    )
