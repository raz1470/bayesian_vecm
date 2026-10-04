"""Stationarity check: which series trend and which do not.

A VECM needs to know the order of integration of each series before it is
fitted:

* An **I(1)** series has no fixed level. It becomes stationary after one
  difference. These go in ``endog``, where they can share long-run relations.
* An **I(0)** series is stationary: it returns to a fixed level. These go in
  ``exog``, where they act on short-run growth.
* An **I(2)** series is still not stationary after one difference. Difference
  it once before using it.

:func:`check_stationarity` runs the Augmented Dickey-Fuller (ADF) test on each
column twice, once on the first difference and once on the levels. The null
hypothesis of the ADF test is a unit root, so a small p-value means
stationary. The difference is read first: if it still has a unit root the
series is I(2), whatever the levels test says.

.. code-block:: python

    from bayesian_vecm import check_stationarity

    result = check_stationarity(df)
    print(result)          # one line per column with a verdict
    result.table           # the same as a DataFrame

Notes
-----
The ADF test has low power against series that revert slowly. A stationary
series with strong persistence can be reported as I(1), most often in short
samples. Treat the verdict as a guide and overrule it when you know the
series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bayesian_vecm._data import validate_endog

if TYPE_CHECKING:
    import pandas as pd

_VALID_REGRESSION: frozenset[str] = frozenset({"c", "ct", "n"})

_VERDICTS: dict[str, str] = {
    "I(0)": "stationary: use as exog",
    "I(1)": "trending: use as endog",
    "I(2)": "still trending after one difference: difference it first",
}


@dataclass(frozen=True)
class StationarityResult:
    """Result of :func:`check_stationarity`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per column of the input, with columns:

        * ``variable``: column name.
        * ``p_level``: ADF p-value for the levels.
        * ``p_diff``: ADF p-value for the first difference.
        * ``order``: ``"I(0)"``, ``"I(1)"`` or ``"I(2)"``. ``"I(2)"`` means
          two or more.
        * ``role``: ``"exog"``, ``"endog"`` or ``"difference first"``.
    signif : float
        Significance level used for both tests.
    regression : str
        Deterministic terms in the levels test.
    """

    table: pd.DataFrame
    signif: float
    regression: str

    @property
    def endog(self) -> list[str]:
        """Names of the I(1) columns."""
        return self._names("I(1)")

    @property
    def exog(self) -> list[str]:
        """Names of the I(0) columns."""
        return self._names("I(0)")

    def _names(self, order: str) -> list[str]:
        return list(self.table.loc[self.table["order"] == order, "variable"])

    def __str__(self) -> str:
        width = max(len("variable"), *(len(str(v)) for v in self.table["variable"]))
        lines = [
            f"Stationarity check (ADF test, {self.signif:.0%} level)",
            f"  regression = {self.regression!r}",
            "",
            f"  {'variable':<{width}}  {'p (levels)':>10}  {'p (diff)':>9}  {'order':>5}  verdict",
            "  " + "-" * (width + 76),
        ]
        for row in self.table.itertuples(index=False):
            lines.append(
                f"  {row.variable:<{width}}  {row.p_level:>10.3f}  {row.p_diff:>9.3f}"
                f"  {row.order:>5}  {_VERDICTS[row.order]}"
            )
        return "\n".join(lines)


def check_stationarity(
    data: Any,
    *,
    signif: float = 0.05,
    regression: str = "c",
) -> StationarityResult:
    """Classify each column as I(0), I(1) or I(2) with the ADF test.

    Parameters
    ----------
    data
        Time series, shape ``(T, n)``. A ``pandas.DataFrame`` or a 2-D array.
        Column names are used when available.
    signif
        Significance level. A p-value below this rejects the unit root.
    regression
        Deterministic terms in the test on the levels:

        * ``"c"``: constant (default).
        * ``"ct"``: constant and linear trend. Use this when a series could
          be stationary around a straight line.
        * ``"n"``: none.

        The test on the first difference drops the trend, so ``"ct"`` becomes
        ``"c"`` there.

    Returns
    -------
    StationarityResult

    Raises
    ------
    ImportError
        If ``statsmodels`` is not installed.
    ValueError
        If ``data`` fails validation, a column is constant, ``signif`` is not
        strictly between 0 and 1, or ``regression`` is not recognised.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for check_stationarity. "
            "Install it with: uv add --dev statsmodels"
        ) from exc
    import numpy as np
    import pandas as pd

    if not 0.0 < signif < 1.0:
        raise ValueError(f"signif must be strictly between 0 and 1; got signif={signif}")
    if regression not in _VALID_REGRESSION:
        valid = sorted(_VALID_REGRESSION)
        raise ValueError(f"regression must be one of {valid}; got regression={regression!r}")

    values = validate_endog(data, min_obs=20)
    if hasattr(data, "columns"):
        names = [str(c) for c in data.columns]
    else:
        names = [f"y{i}" for i in range(values.shape[1])]

    diff_regression = "c" if regression == "ct" else regression
    rows = []
    for name, column in zip(names, values.T, strict=True):
        if np.ptp(column) == 0.0:
            raise ValueError(f"column {name!r} is constant; the ADF test is undefined")
        p_level = float(adfuller(column, regression=regression, autolag="AIC")[1])
        p_diff = float(adfuller(np.diff(column), regression=diff_regression, autolag="AIC")[1])
        # Test the difference first (Dickey & Pantula 1987). A series with two
        # unit roots can reject on the levels by chance, so the levels test is
        # only read once the difference is known to be stationary.
        if p_diff >= signif:
            order, role = "I(2)", "difference first"
        elif p_level >= signif:
            order, role = "I(1)", "endog"
        else:
            order, role = "I(0)", "exog"
        rows.append(
            {"variable": name, "p_level": p_level, "p_diff": p_diff, "order": order, "role": role}
        )

    return StationarityResult(table=pd.DataFrame(rows), signif=signif, regression=regression)
