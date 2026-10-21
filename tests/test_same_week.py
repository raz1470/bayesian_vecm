"""Same-week links between brand metrics."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_vecm import simulate_brand_data
from bayesian_vecm._simulate import ENDOG_NAMES, EXOG_NAMES

SALES, AWARE, CONS = (ENDOG_NAMES.index(n) for n in ENDOG_NAMES)
LINK = {"brand_consideration": {"brand_awareness": 0.2}}

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_no_links_by_default() -> None:
    truth = simulate_brand_data().truth
    np.testing.assert_array_equal(truth.a0, np.eye(3))
    np.testing.assert_array_equal(truth.links, np.zeros((3, 3)))
    np.testing.assert_array_equal(truth.b_structural, truth.b)


def test_empty_links_change_nothing() -> None:
    plain = simulate_brand_data(seed=3)
    empty = simulate_brand_data(seed=3, contemporaneous={})
    np.testing.assert_array_equal(plain.endog.to_numpy(), empty.endog.to_numpy())
    np.testing.assert_array_equal(plain.exog.to_numpy(), empty.exog.to_numpy())


def test_link_passes_driver_effects_on() -> None:
    plain = simulate_brand_data().truth
    truth = simulate_brand_data(contemporaneous=LINK).truth
    assert truth.links[CONS, AWARE] == 0.2
    np.testing.assert_array_equal(truth.b_structural, plain.b)
    np.testing.assert_allclose(truth.b[CONS], plain.b[CONS] + 0.2 * plain.b[AWARE])
    np.testing.assert_array_equal(truth.b[[SALES, AWARE]], plain.b[[SALES, AWARE]])
    np.testing.assert_allclose(truth.b, np.linalg.inv(truth.a0) @ truth.b_structural)


def test_link_leaves_the_other_parameters_alone() -> None:
    plain = simulate_brand_data().truth
    truth = simulate_brand_data(contemporaneous=LINK).truth
    for name in ("alpha", "beta", "gamma", "sigma", "const"):
        np.testing.assert_array_equal(getattr(truth, name), getattr(plain, name))


def test_links_chain() -> None:
    links = {
        "brand_consideration": {"brand_awareness": 0.2},
        "organic_sales": {"brand_consideration": 0.5},
    }
    truth = simulate_brand_data(contemporaneous=links).truth
    brand = EXOG_NAMES.index("brand_spend")
    rate = EXOG_NAMES.index("interest_rate")
    b_aware = truth.b_structural[AWARE, brand]
    assert truth.b[SALES, brand] == pytest.approx(0.5 * 0.2 * b_aware)
    assert truth.b[SALES, rate] == pytest.approx(0.5 * truth.b_structural[CONS, rate])


def test_data_with_a_link_follow_the_reduced_form() -> None:
    data = simulate_brand_data(2000, seed=1, contemporaneous=LINK)
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
    np.testing.assert_allclose(np.cov(resid.T), truth.sigma, atol=4e-5)


def test_link_reaches_lags_adstock_and_saturation() -> None:
    plain = simulate_brand_data(exog_lag_share=0.4).truth
    lagged = simulate_brand_data(exog_lag_share=0.4, contemporaneous=LINK).truth
    for name in ("b_lag", "b_total"):
        np.testing.assert_allclose(
            getattr(lagged, name)[CONS],
            getattr(plain, name)[CONS] + 0.2 * getattr(plain, name)[AWARE],
        )

    sat = simulate_brand_data(
        exog_adstock={"brand_spend": 0.5},
        exog_saturation={"brand_spend": 5.0},
        contemporaneous=LINK,
    ).truth
    brand = EXOG_NAMES.index("brand_spend")
    assert sat.b_saturation[CONS, brand] == pytest.approx(0.2 * sat.b_saturation[AWARE, brand])
    assert sat.b_total[CONS, brand] == pytest.approx(0.2 * sat.b_total[AWARE, brand])
    slope = sat.marginal_effect("brand_spend", 1.0)
    assert slope[CONS] == pytest.approx(0.2 * slope[AWARE])


def test_link_shows_up_in_the_data() -> None:
    """Brand spend moves consideration in the same week only with the link."""
    brand = EXOG_NAMES.index("brand_spend")

    def same_week_effect(links: dict | None) -> float:
        data = simulate_brand_data(3000, seed=2, contemporaneous=links)
        truth = data.truth
        y, x = data.endog.to_numpy(), data.exog.to_numpy()
        dy = np.diff(y, axis=0)
        target = (
            dy[1:, CONS]
            - truth.const[CONS]
            - y[1:-1] @ (truth.alpha @ truth.beta.T)[CONS]
            - dy[:-1] @ truth.gamma[CONS]
        )
        coef = np.linalg.lstsq(x[2:], target, rcond=None)[0]
        return coef[brand]

    assert abs(same_week_effect(None)) < 0.002
    assert same_week_effect(LINK) == pytest.approx(0.2 * 0.018, abs=0.002)


@pytest.mark.parametrize(
    ("links", "match"),
    [
        ({"brand_trust": {"brand_awareness": 0.2}}, "unknown contemporaneous series"),
        ({"brand_consideration": {"awareness": 0.2}}, "unknown contemporaneous series"),
        ({"brand_awareness": {"brand_awareness": 0.2}}, "to itself"),
        (
            {
                "brand_consideration": {"brand_awareness": 0.2},
                "brand_awareness": {"brand_consideration": 0.1},
            },
            "cycle",
        ),
        (
            {
                "brand_consideration": {"brand_awareness": 0.2},
                "organic_sales": {"brand_consideration": 0.1},
                "brand_awareness": {"organic_sales": 0.1},
            },
            "cycle",
        ),
    ],
)
def test_bad_links_raise(links: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        simulate_brand_data(contemporaneous=links)
