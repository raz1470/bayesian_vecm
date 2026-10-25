"""Synthetic brand-marketing data with known VECM parameters.

One data-generating process (DGP), shared by the practitioner notebook, the
tests and any demo: three cointegrated brand series driven by five stationary
exogenous variables.

.. code-block:: python

    from bayesian_vecm import simulate_brand_data

    data = simulate_brand_data()
    data.endog   # log levels: organic_sales, brand_awareness, brand_consideration
    data.exog    # brand_spend, pr_reach, sponsorship_reach, interest_rate, ...
    data.truth   # the parameters that generated them

The process
-----------
.. math::

    \\Delta y_t = c + \\alpha \\beta' y_{t-1} + \\Gamma \\Delta y_{t-1}
                 + B x_t + \\varepsilon_t, \\qquad
    \\varepsilon_t \\sim N(0, \\Sigma).

The causal story is a chain, and it is the ground truth for any "what drives
what" diagnostic run on this data:

    marketing channels -> awareness -> consideration -> sales

* **One common trend** (``K = 3``, cointegration rank 2). The series load on
  it with ``loading = [1.0, 0.9, 0.8]`` and ``beta`` is built so that
  ``beta' @ loading = 0`` exactly.
* **Awareness is weakly exogenous**: its row of ``alpha`` is zero, so it is the
  only series whose shocks last for ever. Consideration adjusts towards
  awareness, and sales adjust towards consideration.
* **Short-run dynamics** (``gamma``) follow the same chain: lagged awareness
  growth helps predict consideration, lagged consideration growth helps
  predict sales.
* **Exogenous effects** (``b``): the three marketing channels act on awareness
  only (``brand_spend`` > ``pr_reach`` > ``sponsorship_reach``); the two macro
  variables act on consideration only.

**Trending drivers.** ``exog_unit_root=("interest_rate",)`` makes the named
drivers random walks (I(1)). Their effect then runs through the change,
``b`` times :math:`\\Delta x_t`, while ``exog`` holds the level. Fit them as
``exog.diff()``: a trending level cannot explain stationary growth, so a
levels fit gives a meaningless coefficient. The default keeps all five
drivers stationary.

**Saturating channels.** ``exog_saturation={"brand_spend": 5.0}`` returns
the named channels as raw, non-negative spend (mean 1, log-normal) and makes
their effect bend: the driver acts through
:math:`b_s \\, [f(a_t / s) - \\bar f]`, where :math:`a_t` is spend after any
adstock, :math:`s` is the in-sample maximum of raw spend and :math:`f` is
logistic saturation with the given :math:`\\lambda`. :math:`b_s` is set so
the marginal effect of one unit at mean spend equals the linear ``b``, which
keeps a linear fit's local target unchanged.

The constant ``c`` gives the system an upward drift (sales double over 260
weeks on average) and fixes the level of the two cointegrating relations. It
is an unrestricted constant, so ``deterministic="co"`` is the matching
specification when fitting.

**Same-week links.** ``contemporaneous={"brand_consideration":
{"brand_awareness": 0.2}}`` adds a same-week effect of awareness growth on
consideration growth. The drivers act through the structural form
:math:`A_0 \\Delta y_t = \\dots + B_s x_t`, so the reduced form has
:math:`B = A_0^{-1} B_s`: the channels now reach consideration in the same
week through awareness. :math:`\\alpha`, :math:`\\Gamma`, :math:`\\Sigma` and
the constant are reduced-form and unchanged. The default has no links.

**Total sales.** ``total_sales=True`` adds the series an analyst actually
sees: weekly sales in £ that add up, week by week, as

.. math::

    S_t = \\underbrace{e^{y_{1t}}}_{\\text{base}}
          + \\text{brand}_t + \\text{search}_t + \\text{social}_t
          + \\text{promo}_t + \\text{season}_t + u_t .

The base is ``exp(organic_sales)``, the brand system above. On top of it sit
short-term effects that the brand system never sees: a same-week hit from
brand spend (centred on its average), two performance channels
``paid_search`` and ``paid_social`` (raw spend with mean 1, no effect on the
brand metrics), a promotion flag on about 8% of weeks, annual seasonality,
and weekly noise :math:`u_t` with standard deviation ``sales_noise`` times
the reference level. By default every short-term effect is linear and in
the same week, which a UCM with regressors can fit exactly.
``sales_adstock`` and ``sales_saturation`` add geometric carryover and
logistic saturation to them. ``paid_search_follows`` makes paid search
spend rise and fall with demand: its log spend gets a term in the standardised
four-week growth of consideration, as when search volume (and so spend) goes
up with interest in the brand. Sizes are set
against a reference level of ``1.44 * start_levels[0]``, the average of a
level that doubles. The extra series use their own random numbers, so
``endog``, ``exog`` and ``truth`` are unchanged by the option.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from bayesian_vecm._transform import logistic_saturation, logistic_saturation_slope

if TYPE_CHECKING:
    import pandas as pd

ENDOG_NAMES: tuple[str, ...] = ("organic_sales", "brand_awareness", "brand_consideration")
EXOG_NAMES: tuple[str, ...] = (
    "brand_spend",
    "pr_reach",
    "sponsorship_reach",
    "interest_rate",
    "consumer_confidence",
)

# How each series loads on the single common trend.
_LOADING = np.array([1.00, 0.90, 0.80])

# Cointegrating vectors, normalised so the top r x r block is the identity
# (the package's convention). Columns: sales - 1.25 * consideration and
# awareness - 1.125 * consideration. Both are orthogonal to _LOADING.
_BETA = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [-_LOADING[0] / _LOADING[2], -_LOADING[1] / _LOADING[2]],
    ]
)

# Adjustment speeds. Sales close 24% of their gap to consideration each week;
# consideration closes 18% of its gap to awareness. Awareness does not adjust.
_ALPHA = np.array(
    [
        [-0.24, 0.00],
        [0.00, 0.00],
        [0.00, 0.18],
    ]
)

# Short-run dynamics, gamma[i, j] = effect of lagged growth in j on growth in i.
_GAMMA = np.array(
    [
        [0.10, 0.00, 0.40],  # sales <- consideration
        [0.00, 0.10, 0.00],  # awareness: own lag only
        [0.00, 0.25, 0.10],  # consideration <- awareness
    ]
)

# Exogenous effects, rows = endogenous, columns = EXOG_NAMES.
_B = np.array(
    [
        [0.000, 0.000, 0.000, 0.000, 0.000],
        [0.018, 0.012, 0.006, 0.000, 0.000],
        [0.000, 0.000, 0.000, -0.020, 0.015],
    ]
)

# Innovation covariance = idiosyncratic part + a shock to the common trend.
_SIGMA_TREND = 0.006
_CHOL = np.array(
    [
        [0.0120, 0.0000, 0.0000],
        [0.0018, 0.0060, 0.0000],
        [0.0024, 0.0018, 0.0072],
    ]
)
_SIGMA = _CHOL @ _CHOL.T + _SIGMA_TREND**2 * np.outer(_LOADING, _LOADING)

# Expected weekly log growth of sales: doubling over 260 weeks.
_SALES_GROWTH = np.log(2.0) / 260.0

# Exogenous processes: AR(1), mean zero. The three channels share correlated
# innovations; the two macro variables are independent of everything else.
_EXOG_RHO = np.array([0.50, 0.50, 0.50, 0.70, 0.65])
_CHANNEL_SD = 0.30  # stationary standard deviation of each channel
_CHANNEL_CORR = 0.50  # pairwise correlation between channels
_MACRO_INNOV_SD = np.array([0.22, 0.30])
_EXOG_BURN_IN = 100

_START_LEVELS = (1000.0, 20.0, 10.0)  # sales, awareness %, consideration %

# Total sales: short-term parts in £, sized against a reference level of
# 1.44 * starting sales (the average of a level that doubles).
SALES_DRIVER_NAMES: tuple[str, ...] = ("paid_search", "paid_social", "promo")
SALES_PART_NAMES: tuple[str, ...] = (
    "base",
    "brand_spend",
    "paid_search",
    "paid_social",
    "promo",
    "seasonality",
    "noise",
)
_SALES_REF_MULT = 1.44
# Share of the reference level each part adds at mean spend (paid channels),
# per unit of brand spend above its average, or in a promo week.
_SALES_SHARE = {"brand_spend": 0.06, "paid_search": 0.08, "paid_social": 0.05, "promo": 0.10}
_SALES_ADSTOCK_NAMES = ("brand_spend", "paid_search", "paid_social")
_SALES_SATURATION_NAMES = ("paid_search", "paid_social")
_PAID_RHO = 0.5
_PAID_LOG_SD = 0.40
_PROMO_RATE = 0.08
# Annual seasonality: two harmonics, peak-to-trough about 16% of the reference.
_SEASON_AMPLITUDE = (0.07, 0.025)
_WEEKS_PER_YEAR = 365.25 / 7.0


@dataclass(frozen=True)
class BrandTruth:
    """The parameters that generated a :class:`BrandData` set.

    Shapes use ``K = 3`` endogenous variables, ``r = 2`` cointegrating
    relations and ``m = 5`` exogenous variables. Row and column order follows
    :attr:`BrandData.endog` and :attr:`BrandData.exog`.

    Attributes
    ----------
    alpha : np.ndarray, shape (K, r)
        Adjustment speeds. The awareness row is zero (weakly exogenous).
    beta : np.ndarray, shape (K, r)
        Cointegrating vectors, top ``r x r`` block the identity.
    gamma : np.ndarray, shape (K, K)
        Coefficients on :math:`\\Delta y_{t-1}`.
    b : np.ndarray, shape (K, m)
        Coefficients on the same-period exogenous variables. With adstock
        this is the same-week share ``(1 - w) * b_total``.
    sigma : np.ndarray, shape (K, K)
        Innovation covariance.
    const : np.ndarray, shape (K,)
        Unrestricted constant, what ``deterministic="co"`` estimates. Depends
        on the starting levels.
    loading : np.ndarray, shape (K,)
        How each series loads on the common trend. ``beta.T @ loading`` is zero.
    growth : np.ndarray, shape (K,)
        Expected log growth per period of each series.
    b_lag : np.ndarray, shape (K, m)
        Coefficients on the exogenous variables one period back. Zero unless
        ``exog_lag_share`` is set. The long-run effect of a driver depends on
        ``b + b_lag`` only.
    differenced : tuple of str
        Drivers simulated as random walks (``exog_unit_root``). For these,
        ``b`` and ``b_lag`` act on the change :math:`\\Delta x_t`, not the
        level. Empty by default.
    adstock : np.ndarray, shape (m,)
        Geometric adstock decay :math:`w` of each driver (``exog_adstock``).
        The effect at lag :math:`l` is ``b_total * (1 - w) * w**l``. Zero
        (no carryover) by default.
    saturation : np.ndarray, shape (m,)
        Logistic saturation :math:`\\lambda` of each driver
        (``exog_saturation``). Zero (linear) by default. For a saturated
        driver ``b`` and ``b_total`` are the marginal effects of one unit at
        mean spend (1.0).
    spend_scale : np.ndarray, shape (m,)
        In-sample maximum of raw spend for saturated drivers, the scale
        :math:`s` in :math:`f(a_t / s)`. One for the others.
    b_saturation : np.ndarray, shape (K, m)
        Coefficient on the saturated, centred regressor
        :math:`f(a_t / s) - \\bar f`: the effect at full saturation. Zero for
        linear drivers.
    saturation_mean : np.ndarray, shape (m,)
        In-sample mean :math:`\\bar f` of the saturated regressor. Zero for
        linear drivers.
    a0 : np.ndarray, shape (K, K)
        Same-week structure :math:`A_0`: ones on the diagonal and minus each
        same-week link off it, so ``a0[i, j] = -0.2`` means growth in ``j``
        raises growth in ``i`` by 0.2 in the same week. The identity unless
        ``contemporaneous`` is set.
    b_structural : np.ndarray, shape (K, m)
        Direct same-week effects :math:`B_s`, before the links pass them on.
        ``b`` is ``inv(a0) @ b_structural``. Equal to ``b`` without links.
    sales_parts : pandas.DataFrame or None
        With ``total_sales=True``: each part of total sales in £ per week,
        columns :data:`SALES_PART_NAMES`. They sum to
        ``BrandData.sales["total_sales"]``. ``None`` otherwise.
    sales_short_term : dict or None
        With ``total_sales=True``: the short-term settings, keys
        ``"reference"`` (£), ``"adstock"``, ``"saturation"``,
        ``"spend_scale"``, ``"coef"`` (£ per unit of the transformed
        driver) and ``"paid_search_follows"``. ``None`` otherwise.
    """

    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    gamma: NDArray[np.float64]
    b: NDArray[np.float64]
    sigma: NDArray[np.float64]
    const: NDArray[np.float64]
    loading: NDArray[np.float64]
    growth: NDArray[np.float64]
    b_lag: NDArray[np.float64]
    differenced: tuple[str, ...] = ()
    adstock: NDArray[np.float64] = field(default_factory=lambda: np.zeros(len(EXOG_NAMES)))
    saturation: NDArray[np.float64] = field(default_factory=lambda: np.zeros(len(EXOG_NAMES)))
    spend_scale: NDArray[np.float64] = field(default_factory=lambda: np.ones(len(EXOG_NAMES)))
    b_saturation: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((len(ENDOG_NAMES), len(EXOG_NAMES)))
    )
    saturation_mean: NDArray[np.float64] = field(default_factory=lambda: np.zeros(len(EXOG_NAMES)))
    a0: NDArray[np.float64] = field(default_factory=lambda: np.eye(len(ENDOG_NAMES)))
    b_structural: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((len(ENDOG_NAMES), len(EXOG_NAMES)))
    )
    sales_parts: pd.DataFrame | None = None
    sales_short_term: dict[str, Any] | None = None

    @property
    def links(self) -> NDArray[np.float64]:
        """Same-week links, ``links[i, j]`` = effect of growth in ``j`` on ``i``.

        ``I - a0``. Zero without ``contemporaneous``.
        """
        return np.eye(self.a0.shape[0]) - self.a0

    @property
    def b_total(self) -> NDArray[np.float64]:
        """Total (long-run summed) effect of each driver, shape ``(K, m)``.

        For a saturated driver this is the marginal effect of one unit at
        mean spend.
        """
        return (self.b + self.b_lag) / (1.0 - self.adstock)

    def response(self, driver: str, level: ArrayLike) -> NDArray[np.float64]:
        """Total effect on each series' growth of holding ``driver`` at ``level``.

        Relative to the driver's in-sample average contribution, so a linear
        driver held at 0 (its average) gives zero. Adstock does not change a
        level held for ever.

        Returns shape ``level.shape + (K,)``.
        """
        j = EXOG_NAMES.index(driver)
        level = np.asarray(level, dtype=np.float64)[..., None]
        lam = self.saturation[j]
        if lam == 0.0:
            return level * self.b_total[:, j]
        f = logistic_saturation(level / self.spend_scale[j], lam)
        return (f - self.saturation_mean[j]) * self.b_saturation[:, j]

    def marginal_effect(self, driver: str, level: ArrayLike) -> NDArray[np.float64]:
        """Total effect of one more unit of ``driver`` held at ``level``.

        The slope of :meth:`response`. Constant for a linear driver.
        Returns shape ``level.shape + (K,)``.
        """
        j = EXOG_NAMES.index(driver)
        level = np.asarray(level, dtype=np.float64)
        lam = self.saturation[j]
        if lam == 0.0:
            return np.broadcast_to(self.b_total[:, j], (*level.shape, self.b.shape[0])).copy()
        scale = self.spend_scale[j]
        slope = logistic_saturation_slope(level / scale, lam) / scale
        return slope[..., None] * self.b_saturation[:, j]


@dataclass(frozen=True)
class BrandData:
    """Simulated brand data and the parameters behind it.

    Attributes
    ----------
    endog : pandas.DataFrame, shape (n_obs, 3)
        **Log levels** of ``organic_sales``, ``brand_awareness`` and
        ``brand_consideration``, weekly. Use ``np.exp`` for original units.
    exog : pandas.DataFrame, shape (n_obs, 5)
        ``brand_spend``, ``pr_reach``, ``sponsorship_reach``, ``interest_rate``
        and ``consumer_confidence``, as stationary deviations from their
        long-run averages. A driver named in ``exog_unit_root`` is a random
        walk instead, starting near zero. A channel named in
        ``exog_saturation`` is raw, non-negative spend with mean 1.
    truth : BrandTruth
        The generating parameters.
    sales : pandas.DataFrame or None
        With ``total_sales=True``: one column, ``total_sales``, weekly sales
        in £ (not logs). ``None`` otherwise.
    sales_drivers : pandas.DataFrame or None
        With ``total_sales=True``: ``paid_search`` and ``paid_social`` (raw
        spend, mean 1) and ``promo`` (0 or 1). These drive total sales only.
        ``None`` otherwise.
    """

    endog: pd.DataFrame
    exog: pd.DataFrame
    truth: BrandTruth
    sales: pd.DataFrame | None = None
    sales_drivers: pd.DataFrame | None = None


def simulate_brand_data(
    n_obs: int = 260,
    *,
    seed: int = 14,
    start_levels: tuple[float, float, float] = _START_LEVELS,
    exog_lag_share: float = 0.0,
    exog_unit_root: tuple[str, ...] = (),
    exog_adstock: dict[str, float] | None = None,
    exog_saturation: dict[str, float] | None = None,
    spend_log_sd: float = 0.30,
    contemporaneous: dict[str, dict[str, float]] | None = None,
    total_sales: bool = False,
    sales_noise: float = 0.0,
    sales_adstock: dict[str, float] | None = None,
    sales_saturation: dict[str, float] | None = None,
    paid_search_follows: float = 0.0,
) -> BrandData:
    """Simulate weekly brand data from a VECM with known parameters.

    Parameters
    ----------
    n_obs
        Number of weekly observations. Defaults to 260 (five years).
    seed
        Seed for the random number generator. The same seed always gives the
        same data.
    start_levels
        Starting values of sales, awareness and consideration in original
        units (not logs). Awareness and consideration are percentages; start
        them low enough to leave room to grow.
    exog_lag_share
        Share of each exogenous effect that arrives one week late, between
        0 and 1. ``0`` (default) puts the whole effect in the same week. The
        same-week coefficients become ``(1 - share) * b`` and the lagged ones
        ``share * b``, so the long-run effect of each driver is unchanged.
    exog_unit_root
        Names of drivers to simulate as random walks (I(1)), for example
        ``("interest_rate",)``. Their weekly changes have the same
        innovations as the stationary version, and their effect acts
        through the change, so ``exog[name].diff()`` is the right regressor.
        Empty (default) keeps every driver stationary.
    exog_adstock
        Geometric adstock decay per driver, for example
        ``{"brand_spend": 0.5}``. The driver acts through
        :math:`a_t = (1 - w) x_t + w a_{t-1}`, so its effect at lag
        :math:`l` is :math:`b (1 - w) w^l` and the total effect is unchanged.
        Decays must be in ``[0, 1)``. Cannot be combined with
        ``exog_lag_share``. ``None`` (default) means no carryover.
    exog_saturation
        Logistic saturation :math:`\\lambda` per marketing channel, for
        example ``{"brand_spend": 5.0}``. The channel is returned as raw
        spend, ``exp(d - spend_log_sd**2 / 2)`` for the stationary deviation
        ``d``, so its mean is 1. It acts through
        :math:`b_s [f(a_t / s) - \\bar f]` after any adstock, with :math:`s`
        the in-sample maximum of spend. :math:`b_s` is set so one more unit
        at mean spend has the linear effect. Only the three channels can
        saturate. ``None`` (default) keeps every driver linear.
    spend_log_sd
        Standard deviation of log spend for saturated channels. Defaults to
        ``0.30`` (spend mostly between half and twice its mean). Larger
        values spread spend over more of the curve. Ignored without
        ``exog_saturation``.
    contemporaneous
        Same-week links between the brand metrics, as
        ``{effect: {cause: coefficient}}``. For example
        ``{"brand_consideration": {"brand_awareness": 0.2}}`` makes 1% more
        awareness growth add 0.2% to consideration growth in the same week.
        The links must not form a cycle. Every driver effect at every lag
        passes through them: the reduced-form ``b`` is
        ``inv(a0) @ b_structural``. ``None`` (default) means no links.
    total_sales
        Also return total sales in £ (``sales``), the drivers that act on it
        only (``sales_drivers``) and its parts (``truth.sales_parts``). The
        base is ``exp(organic_sales)``. ``False`` (default) leaves them
        ``None``.
    sales_noise
        Standard deviation of the weekly noise in total sales that is not
        part of the brand system, as a share of the reference level (for
        example ``0.02`` for 2%). Defaults to ``0.0``. Needs
        ``total_sales=True``.
    sales_adstock
        Geometric adstock decay of the short-term effects on total sales,
        for example ``{"paid_search": 0.3}``. Keys from ``brand_spend``,
        ``paid_search`` and ``paid_social``; decays in ``[0, 1)``. Separate
        from ``exog_adstock``, which acts on the brand metrics. ``None``
        (default) keeps every short-term effect in the same week. Needs
        ``total_sales=True``.
    sales_saturation
        Logistic saturation :math:`\\lambda` of the performance channels'
        effect on total sales, for example ``{"paid_search": 3.0}``. The
        coefficient is set so the effect at mean spend is unchanged.
        ``None`` (default) keeps them linear. Needs ``total_sales=True``.
    paid_search_follows
        How strongly paid search spend follows demand. Log spend gets
        ``paid_search_follows * z_t``, where :math:`z_t` is the four-week
        growth in log consideration, standardised. Spend is then rescaled to
        mean 1. The other paid channel is unchanged. ``0.0`` (default) keeps
        search spend independent of the brand. Must not be negative. Needs
        ``total_sales=True``.

    Returns
    -------
    BrandData
        ``endog`` (log levels), ``exog`` and ``truth``.

    Raises
    ------
    ValueError
        If ``n_obs`` is less than 3, any starting level is not positive,
        ``exog_lag_share`` is outside ``[0, 1]``, ``exog_unit_root`` or
        ``exog_adstock`` names an unknown driver, an adstock decay is outside
        ``[0, 1)``, ``exog_adstock`` or ``exog_saturation`` is combined with
        ``exog_lag_share``, ``exog_saturation`` names anything but a channel
        or a channel in ``exog_unit_root``, a saturation value is not
        positive, ``spend_log_sd`` is not positive, or ``contemporaneous``
        names an unknown series, links a series to itself or has a cycle,
        ``sales_noise`` is negative, ``sales_adstock`` or ``sales_saturation``
        names an unknown driver or has an invalid value,
        ``paid_search_follows`` is negative, or any of these is set without
        ``total_sales``.
    """
    import pandas as pd

    sales_w, sales_lam = _sales_options(total_sales, sales_noise, sales_adstock, sales_saturation)
    if paid_search_follows < 0.0:
        raise ValueError(f"paid_search_follows must not be negative; got {paid_search_follows}")
    if paid_search_follows and not total_sales:
        raise ValueError("paid_search_follows needs total_sales=True")

    if n_obs < 3:
        raise ValueError(f"n_obs must be at least 3; got n_obs={n_obs}")
    start = np.asarray(start_levels, dtype=np.float64)
    if start.shape != (3,) or not (start > 0).all():
        raise ValueError("start_levels must be three positive numbers")
    if not 0.0 <= exog_lag_share <= 1.0:
        raise ValueError(f"exog_lag_share must be between 0 and 1; got {exog_lag_share}")
    unit_root = tuple(exog_unit_root)
    unknown = sorted(set(unit_root) - set(EXOG_NAMES))
    if unknown:
        raise ValueError(
            f"unknown exog_unit_root name(s) {unknown}; choose from {list(EXOG_NAMES)}"
        )
    ur_mask = np.array([name in unit_root for name in EXOG_NAMES])
    decay = np.zeros(len(EXOG_NAMES))
    if exog_adstock:
        unknown = sorted(set(exog_adstock) - set(EXOG_NAMES))
        if unknown:
            raise ValueError(
                f"unknown exog_adstock name(s) {unknown}; choose from {list(EXOG_NAMES)}"
            )
        if exog_lag_share:
            raise ValueError("exog_adstock cannot be combined with exog_lag_share")
        for name, w in exog_adstock.items():
            if not 0.0 <= w < 1.0:
                raise ValueError(f"adstock decay for {name!r} must be in [0, 1); got {w}")
            decay[EXOG_NAMES.index(name)] = w
    lam = _saturation_values(exog_saturation, exog_lag_share, unit_root, spend_log_sd)
    sat = lam > 0.0
    a0 = _a0_matrix(contemporaneous)
    a0_inv = np.linalg.inv(a0)
    b_now = (1.0 - exog_lag_share) * _B
    b_lag = exog_lag_share * _B

    rng = np.random.default_rng(seed)
    x, x_reg = _simulate_exog(n_obs, rng, ur_mask)
    if sat.any():
        # Raw spend: log-normal with mean 1, from the same draws.
        log_dev = x[:, sat] * (spend_log_sd / _CHANNEL_SD)
        x[:, sat] = np.exp(log_dev - spend_log_sd**2 / 2.0)
        x_reg[:, sat] = x[:, sat]
    if decay.any():
        # Adstock state a_t = (1 - w) x_t + w a_{t-1}, started at x_0.
        stock = x_reg.copy()
        for t in range(1, n_obs):
            stock[t] = (1.0 - decay) * x_reg[t] + decay * stock[t - 1]
        x_reg = stock
    scale = np.ones(len(EXOG_NAMES))
    b_sat = np.zeros_like(_B)
    f_mean = np.zeros(len(EXOG_NAMES))
    b_eff = b_now.copy()
    if sat.any():
        # f(a / s) - mean, with b_s set so the slope per unit at spend 1 is b.
        scale[sat] = x[:, sat].max(axis=0)
        f = logistic_saturation(x_reg[:, sat] / scale[sat], lam[sat])
        f_mean[sat] = f.mean(axis=0)
        x_reg = x_reg.copy()
        x_reg[:, sat] = f - f_mean[sat]
        slope = logistic_saturation_slope(1.0 / scale[sat], lam[sat]) / scale[sat]
        b_sat[:, sat] = _B[:, sat] / slope
        b_eff[:, sat] = b_sat[:, sat]
    # Reduced form: the same-week links pass every driver effect on.
    b_eff = a0_inv @ b_eff
    b_lag = a0_inv @ b_lag
    b_sat = a0_inv @ b_sat
    eps = rng.multivariate_normal(np.zeros(3), _SIGMA, size=n_obs)

    y = np.zeros((n_obs, 3))
    y[0] = np.log(start)
    const = _constant(y[0])
    pi = _ALPHA @ _BETA.T
    dy_prev = np.zeros(3)
    for t in range(1, n_obs):
        exog_term = b_eff @ x_reg[t] + b_lag @ x_reg[t - 1]
        dy = const + pi @ y[t - 1] + _GAMMA @ dy_prev + exog_term + eps[t]
        y[t] = y[t - 1] + dy
        dy_prev = dy

    dates = pd.date_range("2021-01-04", periods=n_obs, freq="W-MON")
    truth = BrandTruth(
        alpha=_ALPHA.copy(),
        beta=_BETA.copy(),
        gamma=_GAMMA.copy(),
        b=a0_inv @ (b_now * (1.0 - decay)),
        sigma=_SIGMA.copy(),
        const=const,
        loading=_LOADING.copy(),
        growth=_SALES_GROWTH * _LOADING,
        b_lag=b_lag,
        differenced=tuple(n for n in EXOG_NAMES if n in unit_root),
        adstock=decay,
        saturation=lam,
        spend_scale=scale,
        b_saturation=b_sat,
        saturation_mean=f_mean,
        a0=a0,
        b_structural=b_now * (1.0 - decay),
    )
    endog = pd.DataFrame(y, index=dates, columns=list(ENDOG_NAMES))
    exog = pd.DataFrame(x, index=dates, columns=list(EXOG_NAMES))
    if not total_sales:
        return BrandData(endog=endog, exog=exog, truth=truth)
    parts, drivers, settings = _simulate_total_sales(
        np.exp(y[:, 0]),
        x[:, 0],
        dates,
        seed,
        float(start[0]),
        sales_noise,
        sales_w,
        sales_lam,
        _demand_signal(y[:, 2]) * paid_search_follows,
    )
    settings["paid_search_follows"] = float(paid_search_follows)
    truth = replace(truth, sales_parts=parts, sales_short_term=settings)
    sales = parts.sum(axis=1).to_frame("total_sales")
    return BrandData(endog=endog, exog=exog, truth=truth, sales=sales, sales_drivers=drivers)


def _sales_options(
    total_sales: bool,
    noise: float,
    adstock: dict[str, float] | None,
    saturation: dict[str, float] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Validate the total sales options; return decay and lambda per driver."""
    if noise < 0.0:
        raise ValueError(f"sales_noise must not be negative; got {noise}")
    for name, value in (("sales_noise", noise), ("sales_adstock", adstock)):
        if value and not total_sales:
            raise ValueError(f"{name} needs total_sales=True")
    if saturation and not total_sales:
        raise ValueError("sales_saturation needs total_sales=True")
    w = dict.fromkeys(_SALES_ADSTOCK_NAMES, 0.0)
    for name, value in (adstock or {}).items():
        if name not in _SALES_ADSTOCK_NAMES:
            raise ValueError(
                f"unknown sales_adstock name {name!r}; choose from {list(_SALES_ADSTOCK_NAMES)}"
            )
        if not 0.0 <= value < 1.0:
            raise ValueError(f"sales_adstock decay for {name!r} must be in [0, 1); got {value}")
        w[name] = float(value)
    lam = dict.fromkeys(_SALES_SATURATION_NAMES, 0.0)
    for name, value in (saturation or {}).items():
        if name not in _SALES_SATURATION_NAMES:
            raise ValueError(
                f"unknown sales_saturation name {name!r}; "
                f"choose from {list(_SALES_SATURATION_NAMES)}"
            )
        if not value > 0.0:
            raise ValueError(f"sales_saturation for {name!r} must be positive; got {value}")
        lam[name] = float(value)
    return w, lam


def _demand_signal(log_consideration: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standardised four-week growth in log consideration (0 in the first weeks)."""
    growth = np.zeros_like(log_consideration)
    growth[4:] = log_consideration[4:] - log_consideration[:-4]
    growth[:4] = growth[4:].mean()
    return (growth - growth.mean()) / growth.std()


def _adstock(x: NDArray[np.float64], w: float) -> NDArray[np.float64]:
    """Geometric adstock :math:`a_t = (1 - w) x_t + w a_{t-1}`, started at ``x_0``."""
    a = np.array(x, dtype=np.float64)
    for t in range(1, len(a)):
        a[t] = (1.0 - w) * x[t] + w * a[t - 1]
    return a


def _simulate_total_sales(
    base: NDArray[np.float64],
    brand_spend: NDArray[np.float64],
    dates: pd.DatetimeIndex,
    seed: int,
    start_sales: float,
    noise_sd: float,
    decay: dict[str, float],
    lam: dict[str, float],
    search_demand: NDArray[np.float64],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Total sales parts in £, the sales-only drivers and the settings.

    Uses its own generator, seeded from ``seed``, so the brand system's
    draws are untouched.
    """
    import pandas as pd

    n_obs = len(base)
    rng = np.random.default_rng([seed, 1])
    ref = _SALES_REF_MULT * start_sales

    # Paid channels: log-normal AR(1) spend with mean 1.
    n_total = n_obs + _EXOG_BURN_IN
    innov_sd = _PAID_LOG_SD * np.sqrt(1.0 - _PAID_RHO**2)
    d = np.zeros((n_total, 2))
    e = rng.normal(size=(n_total, 2)) * innov_sd
    # search_demand is zero unless search follows demand; it starts after burn-in.
    push = np.zeros(n_total)
    push[_EXOG_BURN_IN:] = search_demand
    for t in range(1, n_total):
        d[t] = _PAID_RHO * d[t - 1] + e[t]
        d[t, 0] += push[t]
    paid = np.exp(d[_EXOG_BURN_IN:] - _PAID_LOG_SD**2 / 2.0)
    if search_demand.any():
        paid[:, 0] /= paid[:, 0].mean()
    promo = (rng.random(n_obs) < _PROMO_RATE).astype(np.float64)
    noise = rng.normal(size=n_obs) * noise_sd * ref

    coef: dict[str, float] = {}
    scale: dict[str, float] = {}
    parts = {"base": base}
    # Brand spend's short-term hit, centred so it adds nothing on average.
    stock = _adstock(brand_spend, decay["brand_spend"])
    coef["brand_spend"] = _SALES_SHARE["brand_spend"] * ref
    parts["brand_spend"] = coef["brand_spend"] * (stock - stock.mean())
    for k, name in enumerate(("paid_search", "paid_social")):
        stock = _adstock(paid[:, k], decay[name])
        scale[name] = float(paid[:, k].max())
        if lam[name] == 0.0:
            # Linear: the contribution at mean spend (1.0) is the target share.
            coef[name] = _SALES_SHARE[name] * ref
            parts[name] = coef[name] * stock
            continue
        f = logistic_saturation(stock / scale[name], lam[name])
        at_mean = logistic_saturation(1.0 / scale[name], lam[name])
        coef[name] = float(_SALES_SHARE[name] * ref / at_mean)
        parts[name] = coef[name] * f
    coef["promo"] = _SALES_SHARE["promo"] * ref
    parts["promo"] = coef["promo"] * promo
    angle = 2.0 * np.pi * np.arange(n_obs) / _WEEKS_PER_YEAR
    parts["seasonality"] = ref * sum(
        a * np.sin((h + 1) * angle) for h, a in enumerate(_SEASON_AMPLITUDE)
    )
    parts["noise"] = noise

    settings = {
        "reference": ref,
        "adstock": decay,
        "saturation": lam,
        "spend_scale": scale,
        "coef": coef,
    }
    drivers = pd.DataFrame(
        {"paid_search": paid[:, 0], "paid_social": paid[:, 1], "promo": promo}, index=dates
    )
    return pd.DataFrame(parts, index=dates)[list(SALES_PART_NAMES)], drivers, settings


def _a0_matrix(contemporaneous: dict[str, dict[str, float]] | None) -> NDArray[np.float64]:
    """Validate ``contemporaneous`` and return :math:`A_0 = I - L`."""
    k = len(ENDOG_NAMES)
    links = np.zeros((k, k))
    for effect, causes in (contemporaneous or {}).items():
        for name in (effect, *causes):
            if name not in ENDOG_NAMES:
                raise ValueError(
                    f"unknown contemporaneous series {name!r}; choose from {list(ENDOG_NAMES)}"
                )
        for cause, value in causes.items():
            if cause == effect:
                raise ValueError(f"a series cannot have a same-week link to itself: {effect!r}")
            links[ENDOG_NAMES.index(effect), ENDOG_NAMES.index(cause)] = value
    # The links are acyclic if and only if their pattern is nilpotent.
    pattern = (links != 0).astype(np.float64)
    if np.linalg.matrix_power(pattern, k).any():
        raise ValueError("contemporaneous links must not form a cycle")
    return np.eye(k) - links


def _saturation_values(
    exog_saturation: dict[str, float] | None,
    exog_lag_share: float,
    unit_root: tuple[str, ...],
    spend_log_sd: float,
) -> NDArray[np.float64]:
    """Validate ``exog_saturation`` and return lambda per driver (0 = linear)."""
    lam = np.zeros(len(EXOG_NAMES))
    if not exog_saturation:
        return lam
    channels = EXOG_NAMES[:3]
    bad = sorted(set(exog_saturation) - set(channels))
    if bad:
        raise ValueError(f"exog_saturation can only name channels {list(channels)}; got {bad}")
    if exog_lag_share:
        raise ValueError("exog_saturation cannot be combined with exog_lag_share")
    both = sorted(set(exog_saturation) & set(unit_root))
    if both:
        raise ValueError(f"saturated channels cannot be in exog_unit_root: {both}")
    if not spend_log_sd > 0.0:
        raise ValueError(f"spend_log_sd must be positive; got {spend_log_sd}")
    for name, value in exog_saturation.items():
        if not value > 0.0:
            raise ValueError(f"saturation for {name!r} must be positive; got {value}")
        lam[EXOG_NAMES.index(name)] = value
    return lam


def _constant(y0: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unrestricted constant: drift along the trend plus the equilibrium level.

    Only awareness carries permanent shocks, so long-run growth is
    ``loading * drift[awareness] / ((I - gamma)[awareness] @ loading)``. The
    drift is scaled so sales grow at ``_SALES_GROWTH``. The second term pins
    the cointegrating relations at their starting values, so the simulation
    starts in equilibrium.
    """
    pass_through = _LOADING[1] / ((np.eye(3) - _GAMMA)[1] @ _LOADING)
    drift = (_SALES_GROWTH / pass_through) * _LOADING
    return drift - _ALPHA @ (_BETA.T @ y0)


def _simulate_exog(
    n_obs: int, rng: np.random.Generator, unit_root: NDArray[np.bool_] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Five mean-zero AR(1) series; the first three have correlated innovations.

    Columns flagged in ``unit_root`` are random walks built from the same
    innovations, starting at their first innovation. Returns the series and
    the regressors the DGP uses: the series itself, or its change for the
    random walks. The random draws do not depend on ``unit_root``.
    """
    n_total = n_obs + _EXOG_BURN_IN
    corr = np.full((3, 3), _CHANNEL_CORR)
    np.fill_diagonal(corr, 1.0)
    innov_var = _CHANNEL_SD**2 * (1.0 - _EXOG_RHO[0] ** 2)
    channel_innov = rng.multivariate_normal(np.zeros(3), corr * innov_var, size=n_total)
    macro_innov = rng.normal(size=(n_total, 2)) * _MACRO_INNOV_SD
    innov = np.column_stack([channel_innov, macro_innov])

    x = np.zeros((n_total, 5))
    for t in range(1, n_total):
        x[t] = _EXOG_RHO * x[t - 1] + innov[t]
    x = x[_EXOG_BURN_IN:]
    if unit_root is None or not unit_root.any():
        return x, x
    step = innov[_EXOG_BURN_IN:, unit_root]
    x = x.copy()
    x[:, unit_root] = np.cumsum(step, axis=0)
    x_reg = x.copy()
    x_reg[:, unit_root] = step
    return x, x_reg
