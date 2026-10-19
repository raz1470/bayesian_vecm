"""Media transforms shared by the simulator, the model and the forecasts.

Logistic saturation, as in pymc-marketing:

.. math::

    f(u; \\lambda) = \\frac{1 - e^{-\\lambda u}}{1 + e^{-\\lambda u}}
                  = \\tanh(\\lambda u / 2),

for scaled spend :math:`u = x / \\max(x) \\ge 0`. It is 0 at no spend and
tends to 1 as spend grows. :math:`\\lambda` sets how fast it bends. The slope
at zero is :math:`\\lambda / 2`.

The NumPy versions live here. The model builds the PyTensor graph with the
same formula in ``_pymc.py``, and the tests check the two agree.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def logistic_saturation(u: ArrayLike, lam: ArrayLike) -> NDArray[np.float64]:
    """Logistic saturation :math:`\\tanh(\\lambda u / 2)` of scaled spend ``u``."""
    return np.tanh(np.asarray(lam, dtype=np.float64) * np.asarray(u, dtype=np.float64) / 2.0)


def logistic_saturation_slope(u: ArrayLike, lam: ArrayLike) -> NDArray[np.float64]:
    """Derivative of :func:`logistic_saturation` with respect to ``u``."""
    lam = np.asarray(lam, dtype=np.float64)
    f = logistic_saturation(u, lam)
    return lam / 2.0 * (1.0 - f**2)


def geometric_adstock_weights(w: ArrayLike, max_lag: int) -> NDArray[np.float64]:
    """Normalised geometric weights :math:`w^l / \\sum_i w^i` for ``l = 0..max_lag``.

    ``w`` may be an array; the lag axis is last.
    """
    w = np.asarray(w, dtype=np.float64)[..., None]
    powers = w ** np.arange(max_lag + 1, dtype=np.float64)
    return powers / powers.sum(axis=-1, keepdims=True)
