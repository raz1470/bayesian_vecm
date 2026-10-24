"""Tests for simulate_brand_data and the truth it exposes."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_vecm import BrandData, BrandTruth, select_coint_rank, simulate_brand_data
from bayesian_vecm._transform import logistic_saturation

ENDOG = ["organic_sales", "brand_awareness", "brand_consideration"]
EXOG = ["brand_spend", "pr_reach", "sponsorship_reach", "interest_rate", "consumer_confidence"]


@pytest.fixture(scope="module")
def data() -> BrandData:
    return simulate_brand_data()


# ---------------------------------------------------------------------------
# Shape, names, reproducibility
# ---------------------------------------------------------------------------


def test_shapes_and_column_names(data: BrandData) -> None:
    assert data.endog.shape == (260, 3)
    assert data.exog.shape == (260, 5)
    assert list(data.endog.columns) == ENDOG
    assert list(data.exog.columns) == EXOG
    assert data.endog.index.equals(data.exog.index)


def test_n_obs_is_respected() -> None:
    out = simulate_brand_data(120)
    assert len(out.endog) == 120
    assert len(out.exog) == 120


def test_no_missing_values(data: BrandData) -> None:
    assert not data.endog.isna().any().any()
    assert not data.exog.isna().any().any()


def test_same_seed_gives_same_data() -> None:
    a, b = simulate_brand_data(seed=3), simulate_brand_data(seed=3)
    np.testing.assert_array_equal(a.endog.to_numpy(), b.endog.to_numpy())
    np.testing.assert_array_equal(a.exog.to_numpy(), b.exog.to_numpy())


def test_different_seeds_give_different_data() -> None:
    a, b = simulate_brand_data(seed=3), simulate_brand_data(seed=4)
    assert not np.allclose(a.endog.to_numpy(), b.endog.to_numpy())


def test_starts_at_the_requested_levels() -> None:
    out = simulate_brand_data(start_levels=(500.0, 25.0, 12.0))
    np.testing.assert_allclose(np.exp(out.endog.iloc[0].to_numpy()), [500.0, 25.0, 12.0])


@pytest.mark.parametrize("n_obs", [0, 2])
def test_too_few_observations_raises(n_obs: int) -> None:
    with pytest.raises(ValueError, match="n_obs"):
        simulate_brand_data(n_obs)


def test_non_positive_start_level_raises() -> None:
    with pytest.raises(ValueError, match="start_levels"):
        simulate_brand_data(start_levels=(1000.0, 0.0, 10.0))


# ---------------------------------------------------------------------------
# The truth is internally consistent
# ---------------------------------------------------------------------------


def test_truth_shapes(data: BrandData) -> None:
    truth = data.truth
    assert isinstance(truth, BrandTruth)
    assert truth.alpha.shape == (3, 2)
    assert truth.beta.shape == (3, 2)
    assert truth.gamma.shape == (3, 3)
    assert truth.b.shape == (3, 5)
    assert truth.sigma.shape == (3, 3)
    assert truth.const.shape == (3,)


def test_beta_is_normalised_and_orthogonal_to_the_loading(data: BrandData) -> None:
    truth = data.truth
    np.testing.assert_array_equal(truth.beta[:2], np.eye(2))
    np.testing.assert_allclose(truth.beta.T @ truth.loading, 0.0, atol=1e-12)


def test_exactly_one_unit_root(data: BrandData) -> None:
    """K - r = 1 common trend: one companion eigenvalue at 1, the rest inside."""
    truth = data.truth
    a1 = np.eye(3) + truth.alpha @ truth.beta.T + truth.gamma
    companion = np.block([[a1, -truth.gamma], [np.eye(3), np.zeros((3, 3))]])
    moduli = np.sort(np.abs(np.linalg.eigvals(companion)))[::-1]
    assert moduli[0] == pytest.approx(1.0, abs=1e-10)
    assert (moduli[1:] < 0.9).all()


def test_sigma_is_a_valid_covariance(data: BrandData) -> None:
    sigma = data.truth.sigma
    np.testing.assert_allclose(sigma, sigma.T)
    assert (np.linalg.eigvalsh(sigma) > 0).all()


def test_causal_chain_is_encoded_in_the_truth(data: BrandData) -> None:
    """channels -> awareness -> consideration -> sales, and nothing else."""
    truth = data.truth
    sales, awareness, consideration = 0, 1, 2

    np.testing.assert_array_equal(truth.alpha[awareness], 0.0)  # weakly exogenous
    assert truth.alpha[sales, 0] < 0
    assert truth.alpha[consideration, 1] > 0

    off_diagonal = truth.gamma - np.diag(np.diag(truth.gamma))
    assert off_diagonal[consideration, awareness] > 0
    assert off_diagonal[sales, consideration] > 0
    assert np.count_nonzero(off_diagonal) == 2

    channels = truth.b[awareness, :3]
    assert channels[0] > channels[1] > channels[2] > 0
    assert np.count_nonzero(truth.b[awareness, 3:]) == 0
    assert np.count_nonzero(truth.b[sales]) == 0
    assert np.count_nonzero(truth.b[consideration, :3]) == 0
    assert truth.b[consideration, 3] < 0 < truth.b[consideration, 4]


def test_truth_arrays_are_copies() -> None:
    """Mutating a returned truth must not leak into the next simulation."""
    first = simulate_brand_data()
    first.truth.alpha[:] = 99.0
    assert simulate_brand_data().truth.alpha.max() < 1.0


def test_data_follow_the_stated_equation(data: BrandData) -> None:
    """Residuals from the true parameters have covariance close to sigma."""
    truth = data.truth
    y, x = data.endog.to_numpy(), data.exog.to_numpy()
    dy = np.diff(y, axis=0)
    resid = (
        dy[1:]
        - truth.const
        - y[1:-1] @ (truth.alpha @ truth.beta.T).T
        - dy[:-1] @ truth.gamma.T
        - x[2:] @ truth.b.T
    )
    assert np.abs(resid.mean(axis=0)).max() < 0.002
    np.testing.assert_allclose(np.cov(resid.T), truth.sigma, atol=4e-5)


# ---------------------------------------------------------------------------
# The data look the way the notebook needs them to
# ---------------------------------------------------------------------------


def test_default_data_trend_upward(data: BrandData) -> None:
    growth = np.exp(data.endog.iloc[-1] - data.endog.iloc[0])
    assert (growth > 1.3).all()
    assert growth["organic_sales"] < 3.0


def test_percentages_stay_below_100(data: BrandData) -> None:
    levels = np.exp(data.endog[["brand_awareness", "brand_consideration"]])
    assert levels.max().max() < 100.0


def test_most_seeds_trend_upward() -> None:
    ups = [
        (out.endog.iloc[-1] > out.endog.iloc[0]).all()
        for out in (simulate_brand_data(seed=s) for s in range(40))
    ]
    assert np.mean(ups) >= 0.9


def test_mean_growth_matches_the_truth() -> None:
    """Average log growth across seeds should match truth.growth * (n_obs - 1)."""
    total = np.array(
        [
            (out.endog.iloc[-1] - out.endog.iloc[0]).to_numpy()
            for out in (simulate_brand_data(seed=s) for s in range(200))
        ]
    )
    expected = simulate_brand_data().truth.growth * 259
    np.testing.assert_allclose(total.mean(axis=0), expected, atol=0.08)


def test_channels_are_moderately_correlated_and_macro_is_not() -> None:
    corr = simulate_brand_data(5000, seed=0).exog.corr().to_numpy()
    channel_pairs = corr[:3, :3][np.triu_indices(3, k=1)]
    assert ((channel_pairs > 0.4) & (channel_pairs < 0.6)).all()
    assert np.abs(corr[:3, 3:]).max() < 0.1
    assert abs(corr[3, 4]) < 0.1


def test_exog_are_centred() -> None:
    exog = simulate_brand_data(5000, seed=0).exog
    assert exog.mean().abs().max() < 0.05


# ---------------------------------------------------------------------------
# Rank recovery (needs statsmodels)
# ---------------------------------------------------------------------------


def test_default_data_recover_rank_two(data: BrandData) -> None:
    pytest.importorskip("statsmodels", reason="statsmodels not installed")
    assert select_coint_rank(data.endog, k_ar_diff=1).rank == 2
    assert select_coint_rank(data.endog, k_ar_diff=2).rank == 2


def test_most_seeds_recover_rank_two() -> None:
    pytest.importorskip("statsmodels", reason="statsmodels not installed")
    ranks = [select_coint_rank(simulate_brand_data(seed=s).endog).rank for s in range(40)]
    assert np.mean(np.array(ranks) == 2) >= 0.65


# ---------------------------------------------------------------------------
# Total sales
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sales_data() -> BrandData:
    return simulate_brand_data(total_sales=True, sales_noise=0.02)


def test_total_sales_off_by_default(data: BrandData) -> None:
    assert data.sales is None
    assert data.sales_drivers is None
    assert data.truth.sales_parts is None
    assert data.truth.sales_short_term is None


def test_total_sales_leaves_the_brand_system_unchanged(
    data: BrandData, sales_data: BrandData
) -> None:
    np.testing.assert_array_equal(data.endog.to_numpy(), sales_data.endog.to_numpy())
    np.testing.assert_array_equal(data.exog.to_numpy(), sales_data.exog.to_numpy())


def test_total_sales_shapes_and_names(sales_data: BrandData) -> None:
    assert list(sales_data.sales.columns) == ["total_sales"]
    assert list(sales_data.sales_drivers.columns) == ["paid_search", "paid_social", "promo"]
    assert list(sales_data.truth.sales_parts.columns) == [
        "base",
        "brand_spend",
        "paid_search",
        "paid_social",
        "promo",
        "seasonality",
        "noise",
    ]
    for frame in (sales_data.sales, sales_data.sales_drivers, sales_data.truth.sales_parts):
        assert frame.index.equals(sales_data.endog.index)
        assert not frame.isna().any().any()


def test_parts_add_up_to_total_sales(sales_data: BrandData) -> None:
    parts = sales_data.truth.sales_parts
    np.testing.assert_allclose(parts.sum(axis=1), sales_data.sales["total_sales"])


def test_base_is_organic_sales_in_pounds(sales_data: BrandData) -> None:
    np.testing.assert_allclose(
        sales_data.truth.sales_parts["base"], np.exp(sales_data.endog["organic_sales"])
    )


def test_total_sales_is_positive_and_mostly_base(sales_data: BrandData) -> None:
    total = sales_data.sales["total_sales"]
    assert (total > 0).all()
    share = (sales_data.truth.sales_parts["base"] / total).mean()
    assert 0.8 < share < 0.95


def test_sales_drivers_have_the_stated_form(sales_data: BrandData) -> None:
    drivers = sales_data.sales_drivers
    assert (drivers[["paid_search", "paid_social"]] > 0).all().all()
    np.testing.assert_allclose(drivers[["paid_search", "paid_social"]].mean(), 1.0, atol=0.15)
    assert set(np.unique(drivers["promo"])) <= {0.0, 1.0}
    assert 0.02 < drivers["promo"].mean() < 0.15


def test_brand_short_term_hit_is_centred(sales_data: BrandData) -> None:
    assert abs(sales_data.truth.sales_parts["brand_spend"].mean()) < 1e-8


def test_promo_part_is_the_flag_times_its_coefficient(sales_data: BrandData) -> None:
    coef = sales_data.truth.sales_short_term["coef"]["promo"]
    np.testing.assert_allclose(
        sales_data.truth.sales_parts["promo"], coef * sales_data.sales_drivers["promo"]
    )


def test_short_term_effects_are_linear_and_same_week_by_default(sales_data: BrandData) -> None:
    st = sales_data.truth.sales_short_term
    parts, drivers = sales_data.truth.sales_parts, sales_data.sales_drivers
    assert st["reference"] == pytest.approx(1.44 * 1000.0)
    assert set(st["adstock"].values()) == {0.0}
    assert set(st["saturation"].values()) == {0.0}
    for name, share in (("paid_search", 0.08), ("paid_social", 0.05)):
        assert st["coef"][name] == pytest.approx(share * st["reference"])
        np.testing.assert_allclose(parts[name], st["coef"][name] * drivers[name])
    spend = sales_data.exog["brand_spend"]
    np.testing.assert_allclose(
        parts["brand_spend"], st["coef"]["brand_spend"] * (spend - spend.mean())
    )


def test_sales_adstock_and_saturation() -> None:
    out = simulate_brand_data(
        total_sales=True,
        sales_adstock={"paid_search": 0.4, "brand_spend": 0.3},
        sales_saturation={"paid_search": 3.0},
    )
    st, parts, drivers = out.truth.sales_short_term, out.truth.sales_parts, out.sales_drivers
    assert st["adstock"]["paid_search"] == 0.4
    # Carryover: this week's search contribution depends on last week's spend.
    stock = drivers["paid_search"].to_numpy().copy()
    for t in range(1, len(stock)):
        stock[t] = 0.6 * drivers["paid_search"].iloc[t] + 0.4 * stock[t - 1]
    scale = st["spend_scale"]["paid_search"]
    np.testing.assert_allclose(
        parts["paid_search"], st["coef"]["paid_search"] * logistic_saturation(stock / scale, 3.0)
    )
    at_mean = st["coef"]["paid_search"] * logistic_saturation(1.0 / scale, 3.0)
    assert at_mean == pytest.approx(0.08 * st["reference"])
    np.testing.assert_allclose(
        parts["paid_social"], st["coef"]["paid_social"] * drivers["paid_social"]
    )
    assert abs(parts["brand_spend"].mean()) < 1e-8


def test_sales_noise_sets_the_noise_part() -> None:
    quiet = simulate_brand_data(total_sales=True)
    assert (quiet.truth.sales_parts["noise"] == 0).all()
    loud = simulate_brand_data(n_obs=2000, total_sales=True, sales_noise=0.05)
    sd = loud.truth.sales_parts["noise"].std() / loud.truth.sales_short_term["reference"]
    assert sd == pytest.approx(0.05, rel=0.1)


def test_total_sales_is_reproducible() -> None:
    a = simulate_brand_data(seed=5, total_sales=True, sales_noise=0.01)
    b = simulate_brand_data(seed=5, total_sales=True, sales_noise=0.01)
    np.testing.assert_array_equal(a.sales.to_numpy(), b.sales.to_numpy())
    c = simulate_brand_data(seed=6, total_sales=True, sales_noise=0.01)
    assert not np.allclose(a.sales_drivers.to_numpy(), c.sales_drivers.to_numpy())


def test_total_sales_works_with_saturated_brand_spend() -> None:
    out = simulate_brand_data(total_sales=True, exog_saturation={"brand_spend": 5.0})
    assert abs(out.truth.sales_parts["brand_spend"].mean()) < 1e-8
    assert (out.sales["total_sales"] > 0).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"total_sales": True, "sales_noise": -0.01}, "must not be negative"),
        ({"sales_noise": 0.01}, "sales_noise needs total_sales"),
        ({"sales_adstock": {"paid_search": 0.3}}, "sales_adstock needs total_sales"),
        ({"sales_saturation": {"paid_search": 3.0}}, "sales_saturation needs total_sales"),
        ({"total_sales": True, "sales_adstock": {"pr_reach": 0.3}}, "unknown sales_adstock"),
        ({"total_sales": True, "sales_adstock": {"paid_search": 1.0}}, "must be in"),
        ({"total_sales": True, "sales_saturation": {"brand_spend": 3.0}}, "unknown sales_sat"),
        ({"total_sales": True, "sales_saturation": {"paid_social": 0.0}}, "must be positive"),
    ],
)
def test_total_sales_option_errors(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        simulate_brand_data(**kwargs)
