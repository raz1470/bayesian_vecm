"""Posterior evidence on which variable helps predict which.

Four tables, all read from the fitted posterior:

``short_run``
    Granger causality. Does lagged growth in X help predict growth in Y? The
    quantity is the block of :math:`\\Gamma` that links X to Y, across all
    lags.
``long_run``
    Does the level of X pull on Y? The quantity is one entry of
    :math:`\\Pi = \\alpha \\beta^\\top`. Unlike :math:`\\alpha` alone,
    :math:`\\Pi` does not depend on how :math:`\\beta` is normalised.
``weak_exogeneity``
    Does the variable adjust to any long-run relation at all? The quantity is
    its whole row of :math:`\\alpha`. A variable that does not adjust drives
    the system without being pulled back by it.
``exog``
    Does an exogenous variable help predict Y? The quantity is one entry of
    :math:`B`. Present only when the model was fitted with ``exog``.

The evidence measure
--------------------
Each row carries one number between 0 and 1: the highest credible level at
which zero lies outside the posterior region for that quantity. The region is
the ellipsoid given by the posterior mean and covariance, and the level is
the share of draws that sit closer to the mean than zero does.

For a single coefficient with a symmetric posterior this equals
``1 - 2 * min(P(> 0), P(< 0))``. A value near 1 means the posterior is far
from zero. A value near 0 means zero is close to the centre.

Two limits to keep in mind:

* These are statements about prediction, not about cause. "X helps predict
  Y" is as far as the short-run table goes.
* A shrinkage prior on :math:`\\Gamma` pulls coefficients towards zero, so the
  short-run evidence depends on that prior. The result records which prior
  was used. A horseshoe posterior is also less elliptical than a normal one,
  which makes the measure approximate there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr


@dataclass(frozen=True)
class CausalityResult:
    """Tables of posterior evidence on what helps predict what.

    Printing the result gives a plain-English summary. Each attribute is a
    ``pandas.DataFrame`` for further work.

    Attributes
    ----------
    short_run : pandas.DataFrame
        One row per ordered pair of endogenous variables. Columns ``cause``,
        ``effect``, ``evidence``, ``mean``, ``lower``, ``upper``. The effect
        size is the sum of the :math:`\\Gamma` coefficients over lags. Empty
        when ``k_ar_diff = 0``.
    long_run : pandas.DataFrame
        Same columns. The effect size is the entry of :math:`\\Pi`.
    weak_exogeneity : pandas.DataFrame
        One row per endogenous variable. Columns ``variable``, ``evidence``
        and ``weakly_exogenous``. ``evidence`` is the evidence that the
        variable adjusts.
    exog : pandas.DataFrame or None
        Same columns as ``short_run``, with exogenous variables as causes.
        ``None`` when the model was fitted without ``exog``.
    threshold : float
        Evidence at or above this is reported as a link.
    ci_prob : float
        Probability mass of the equal-tailed interval in ``lower``/``upper``.
    gamma_prior : str
        Prior family used for :math:`\\Gamma`.
    """

    short_run: pd.DataFrame
    long_run: pd.DataFrame
    weak_exogeneity: pd.DataFrame
    exog: pd.DataFrame | None
    threshold: float
    ci_prob: float
    gamma_prior: str

    def __str__(self) -> str:
        lines = [
            "What helps predict what",
            f"  evidence threshold = {self.threshold:g}   Gamma prior = {self.gamma_prior}",
        ]
        lines += self._pair_table(
            "Short run: lagged growth in the cause (Gamma)", self.short_run, "helps predict"
        )
        lines += self._pair_table(
            "Long run: level of the cause (Pi = alpha beta')", self.long_run, "pulls on"
        )
        lines += ["", "Weak exogeneity: does the variable adjust? (alpha row)"]
        lines.append(f"  {'variable':<26}{'evidence':>9}  verdict")
        lines.append("  " + "-" * 62)
        for row in self.weak_exogeneity.itertuples(index=False):
            verdict = "weakly exogenous" if row.weakly_exogenous else "adjusts"
            lines.append(f"  {row.variable:<26}{row.evidence:>9.2f}  {verdict}")
        if self.exog is not None:
            lines += self._pair_table("Exogenous drivers (B)", self.exog, "helps predict")
        return "\n".join(lines)

    def _pair_table(self, title: str, table: pd.DataFrame, verb: str) -> list[str]:
        lines = ["", title]
        if len(table) == 0:
            return [*lines, "  (none: the model has no lagged differences)"]
        lines.append(f"  {'cause -> effect':<46}{'evidence':>9}{'mean':>9}  verdict")
        lines.append("  " + "-" * 82)
        for row in table.itertuples(index=False):
            pair = f"{row.cause} -> {row.effect}"
            verdict = verb if row.evidence >= self.threshold else "no clear evidence"
            lines.append(f"  {pair:<46}{row.evidence:>9.2f}{row.mean:>9.3f}  {verdict}")
        return lines


def evidence_against_zero(draws: NDArray[np.floating]) -> float:
    """Highest credible level at which zero is outside the posterior region.

    Parameters
    ----------
    draws
        Posterior draws, shape ``(n_samples,)`` for one coefficient or
        ``(n_samples, p)`` for a block of ``p`` coefficients.

    Returns
    -------
    float
        Share of draws closer to the posterior mean than zero is, measured in
        Mahalanobis distance. Between 0 and 1.
    """
    block = np.asarray(draws, dtype=np.float64)
    if block.ndim == 1:
        block = block[:, np.newaxis]
    mean = block.mean(axis=0)
    centred = block - mean
    cov = np.atleast_2d(np.cov(centred, rowvar=False))
    if not np.any(cov):
        # No posterior spread at all: the block is a constant.
        return float(np.any(mean != 0.0))
    # pinv copes with a partly degenerate block, such as one fixed coefficient.
    precision = np.linalg.pinv(cov)
    distance = np.einsum("ni,ij,nj->n", centred, precision, centred)
    distance_of_zero = float(mean @ precision @ mean)
    return float(np.mean(distance < distance_of_zero))


def compute_causality(
    idata: xr.DataTree,
    k_ar_diff: int,
    *,
    variable_names: list[str] | None = None,
    exog_names: list[str] | None = None,
    threshold: float = 0.95,
    ci_prob: float = 0.89,
    gamma_prior: str = "Normal",
) -> CausalityResult:
    """Build the four evidence tables from a fitted posterior.

    Parameters
    ----------
    idata
        Fitted posterior from ``BayesianVECM.fit``.
    k_ar_diff
        Number of lagged-difference blocks in the model.
    variable_names
        Labels for the endogenous variables. Defaults to ``y0, y1, ...``.
    exog_names
        Labels for the exogenous variables. Defaults to ``x0, x1, ...``.
    threshold
        Evidence at or above this counts as a link. Sets
        ``weakly_exogenous`` and the wording of the printed summary.
    ci_prob
        Probability mass of the equal-tailed interval around each effect.
    gamma_prior
        Name of the prior family on :math:`\\Gamma`, recorded on the result.

    Returns
    -------
    CausalityResult

    Raises
    ------
    ValueError
        If ``threshold`` or ``ci_prob`` is not strictly between 0 and 1.
    """
    import pandas as pd

    for name, value in (("threshold", threshold), ("ci_prob", ci_prob)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1; got {name}={value}")

    posterior = idata.posterior
    alpha = _stack(posterior["alpha"].values)  # (N, K, r)
    n_vars = alpha.shape[1]
    # beta carries an extra row when a deterministic term sits inside the
    # cointegrating relation. Only the first K rows belong to the variables.
    beta = _stack(posterior["beta"].values)[:, :n_vars, :]
    names = variable_names or [f"y{i}" for i in range(n_vars)]
    tail = (1.0 - ci_prob) / 2.0

    def effect_row(cause: str, effect: str, block: NDArray[np.float64]) -> dict[str, Any]:
        size = block.sum(axis=1)
        lower, upper = np.quantile(size, [tail, 1.0 - tail])
        return {
            "cause": cause,
            "effect": effect,
            "evidence": evidence_against_zero(block),
            "mean": float(size.mean()),
            "lower": float(lower),
            "upper": float(upper),
        }

    columns = ["cause", "effect", "evidence", "mean", "lower", "upper"]
    pairs = [(i, j) for j in range(n_vars) for i in range(n_vars) if i != j]

    short_rows = []
    if k_ar_diff > 0:
        # Outside deterministic terms add a trailing column to Gamma. Keep
        # only the K * k_ar_diff columns that multiply lagged differences.
        gamma = _stack(posterior["Gamma"].values)[:, :, : n_vars * k_ar_diff]
        for effect, cause in pairs:
            block = gamma[:, effect, cause::n_vars]  # one column per lag
            short_rows.append(effect_row(names[cause], names[effect], block))

    pi = np.einsum("nir,njr->nij", alpha, beta)  # alpha @ beta.T per draw
    long_rows = [
        effect_row(names[cause], names[effect], pi[:, effect, cause, np.newaxis])
        for effect, cause in pairs
    ]

    weak_rows = []
    for i in range(n_vars):
        evidence = evidence_against_zero(alpha[:, i, :])
        weak_rows.append(
            {"variable": names[i], "evidence": evidence, "weakly_exogenous": evidence < threshold}
        )

    exog_table = None
    if "B" in posterior:
        b = _stack(posterior["B"].values)  # (N, K, m)
        x_names = exog_names or [f"x{i}" for i in range(b.shape[2])]
        exog_rows = [
            effect_row(x_names[x], names[effect], b[:, effect, x, np.newaxis])
            for x in range(b.shape[2])
            for effect in range(n_vars)
        ]
        exog_table = pd.DataFrame(exog_rows, columns=columns)

    return CausalityResult(
        short_run=pd.DataFrame(short_rows, columns=columns),
        long_run=pd.DataFrame(long_rows, columns=columns),
        weak_exogeneity=pd.DataFrame(
            weak_rows, columns=["variable", "evidence", "weakly_exogenous"]
        ),
        exog=exog_table,
        threshold=threshold,
        ci_prob=ci_prob,
        gamma_prior=gamma_prior,
    )


def _stack(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Merge the chain and draw axes: ``(C, D, ...)`` to ``(C * D, ...)``."""
    arr = np.asarray(values, dtype=np.float64)
    return arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
