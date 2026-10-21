"""Same-week links between brand metrics: shared checks and the IV test.

A same-week link says growth in one brand metric moves another in the same
period. Regression cannot find it: the metrics share shocks, so least
squares of one on the other picks up the shared shock as well as any link.

The test uses instrumental variables. An instrument for a link from
``cause`` to ``effect`` is a driver that

* moves the cause (relevance, checked by the first-stage F),
* is not set in response to the same week's shocks (independence), and
* reaches the effect only through the cause (exclusion).

Only relevance can be checked from the data. Independence and exclusion come
from knowledge of the business. With more instruments than causes the
Sargan test checks that the instruments agree with each other.

On monthly data a same-month link also holds lagged effects that land inside
the month. A weekly lagged link shows up as a same-month link.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

#: First-stage F below this is reported as weak instruments (Staiger and Stock).
WEAK_F = 10.0


@dataclass(frozen=True)
class SameWeekSpec:
    """Validated same-week links and exclusions, as column indices.

    Attributes
    ----------
    links : dict[int, tuple[int, ...]]
        Effect index -> cause indices, both into the brand metrics.
    excluded : dict[int, tuple[int, ...]]
        Effect index -> driver indices with no direct effect on it.
    """

    links: dict[int, tuple[int, ...]]
    excluded: dict[int, tuple[int, ...]]

    def identified(self, effect: int) -> bool:
        """Order condition: at least as many excluded drivers as causes."""
        return len(self.excluded.get(effect, ())) >= len(self.links.get(effect, ()))


def resolve_same_week(
    contemporaneous: Mapping[str, Sequence[str]] | None,
    excluded: Mapping[str, Sequence[str]] | None,
    endog_names: Sequence[str],
    exog_names: Sequence[str],
    *,
    require_dag: bool,
) -> SameWeekSpec:
    """Check ``contemporaneous`` and ``excluded`` against the column names.

    Raises
    ------
    ValueError
        If a name is unknown, a metric is linked to itself, a cause or an
        excluded driver is listed twice, or (with ``require_dag``) the links
        form a cycle.
    """
    endog_names = list(endog_names)
    exog_names = list(exog_names)
    links: dict[int, tuple[int, ...]] = {}
    for effect, causes in (contemporaneous or {}).items():
        if isinstance(causes, str):
            causes = [causes]
        e = _index(effect, endog_names, "contemporaneous", "brand metric")
        idx = tuple(_index(c, endog_names, "contemporaneous", "brand metric") for c in causes)
        if e in idx:
            raise ValueError(f"a brand metric cannot have a same-week link to itself: {effect!r}")
        if len(set(idx)) != len(idx):
            raise ValueError(f"contemporaneous lists a cause twice for {effect!r}")
        if idx:
            links[e] = idx
    excl: dict[int, tuple[int, ...]] = {}
    for effect, drivers in (excluded or {}).items():
        if isinstance(drivers, str):
            drivers = [drivers]
        e = _index(effect, endog_names, "excluded", "brand metric")
        idx = tuple(_index(d, exog_names, "excluded", "driver") for d in drivers)
        if len(set(idx)) != len(idx):
            raise ValueError(f"excluded lists a driver twice for {effect!r}")
        if idx:
            excl[e] = idx
    if require_dag:
        k = len(endog_names)
        pattern = np.zeros((k, k))
        for e, causes in links.items():
            pattern[e, list(causes)] = 1.0
        # The links are acyclic if and only if their pattern is nilpotent.
        if np.linalg.matrix_power(pattern, k).any():
            raise ValueError("contemporaneous links must not form a cycle")
    return SameWeekSpec(links=links, excluded=excl)


def _index(name: str, names: list[str], arg: str, kind: str) -> int:
    if name not in names:
        raise ValueError(f"{arg} names an unknown {kind} {name!r}; choose from {names}")
    return names.index(name)


@dataclass(frozen=True)
class IVResult:
    """Two-stage least squares for one equation."""

    ols: NDArray[np.float64]
    estimate: NDArray[np.float64]
    std_err: NDArray[np.float64]
    p_value: NDArray[np.float64]
    first_stage_f: NDArray[np.float64]
    sargan: float | None
    sargan_df: int
    sargan_p: float | None


def two_stage_least_squares(
    y: NDArray[np.float64],
    endog: NDArray[np.float64],
    included: NDArray[np.float64],
    instruments: NDArray[np.float64],
) -> IVResult:
    """2SLS of ``y`` on ``endog`` and ``included``, instrumenting ``endog``.

    Parameters
    ----------
    y
        Target, shape ``(n,)``.
    endog
        Endogenous regressors, shape ``(n, k)``.
    included
        Exogenous regressors in the equation, shape ``(n, p)``.
    instruments
        Excluded instruments, shape ``(n, q)`` with ``q >= k``.

    The first-stage F is the Sanderson-Windmeijer conditional F, the usual
    partial F when ``k = 1``. Sargan is ``n R^2`` of the 2SLS residuals on
    all exogenous variables, ``None`` when exactly identified.
    """
    n, k = endog.shape
    q = instruments.shape[1]
    w = np.column_stack([included, instruments])
    x = np.column_stack([endog, included])
    ols = np.linalg.lstsq(x, y, rcond=None)[0][:k]
    x_hat = w @ np.linalg.lstsq(w, x, rcond=None)[0]
    coef = np.linalg.lstsq(x_hat, y, rcond=None)[0]
    resid = y - x @ coef
    df_resid = n - x.shape[1]
    s2 = resid @ resid / df_resid
    cov = s2 * np.linalg.pinv(x_hat.T @ x_hat)
    se = np.sqrt(np.diag(cov)[:k])
    t_stat = coef[:k] / se
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), df_resid)

    f_first = np.array([_first_stage_f(endog, j, included, instruments) for j in range(k)])

    sargan = sargan_p = None
    if q > k:
        fitted = w @ np.linalg.lstsq(w, resid, rcond=None)[0]
        centred = resid - resid.mean()
        sargan = float(n * (fitted @ fitted) / (centred @ centred))
        sargan_p = float(stats.chi2.sf(sargan, q - k))
    return IVResult(
        ols=ols,
        estimate=coef[:k],
        std_err=se,
        p_value=p_value,
        first_stage_f=f_first,
        sargan=sargan,
        sargan_df=q - k,
        sargan_p=sargan_p,
    )


def _first_stage_f(
    endog: NDArray[np.float64],
    j: int,
    included: NDArray[np.float64],
    instruments: NDArray[np.float64],
) -> float:
    """Sanderson-Windmeijer F for endogenous regressor ``j``."""
    n, k = endog.shape
    q = instruments.shape[1]
    w = np.column_stack([included, instruments])
    target = endog[:, j]
    if k > 1:
        # Partial out the other endogenous regressors by 2SLS.
        others = np.delete(endog, j, axis=1)
        x = np.column_stack([others, included])
        x_hat = w @ np.linalg.lstsq(w, x, rcond=None)[0]
        coef = np.linalg.lstsq(x_hat, target, rcond=None)[0]
        target = target - x @ coef
    restricted = target - included @ np.linalg.lstsq(included, target, rcond=None)[0]
    full = target - w @ np.linalg.lstsq(w, target, rcond=None)[0]
    df_num = q - k + 1
    df_den = n - w.shape[1]
    rss_full = full @ full
    return float(((restricted @ restricted - rss_full) / df_num) / (rss_full / df_den))


def verdict(
    identified: bool, f_stat: float, sargan_p: float | None, p_value: float, alpha: float
) -> str:
    """Plain-language reading of one link."""
    if not identified:
        return "not identified"
    if f_stat < WEAK_F:
        return "weak instruments"
    if sargan_p is not None and sargan_p < alpha:
        return "instruments disagree"
    return "same-week link" if p_value < alpha else "no clear evidence"
