"""Same-week links between brand metrics."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_vecm import granger_pretest, simulate_brand_data
from bayesian_vecm._same_week import resolve_same_week, two_stage_least_squares, verdict
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


# ---------------------------------------------------------------------------
# Two-stage least squares
# ---------------------------------------------------------------------------

CHANNELS = ["brand_spend", "pr_reach", "sponsorship_reach"]
MACRO = ["interest_rate", "consumer_confidence"]


def _iv_data(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, ...]:
    """y = 0.5 x + u with x correlated with u, instrumented by z1, z2."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    w = np.column_stack([np.ones(n), rng.normal(size=n)])
    u = rng.normal(size=n)
    x = z @ [0.6, 0.4] + 0.8 * u + rng.normal(size=n) + 0.3 * w[:, 1]
    y = 0.5 * x + 0.2 * w[:, 1] + u
    return y, x[:, None], w, z


def test_2sls_recovers_the_coefficient_and_ols_does_not() -> None:
    y, x, w, z = _iv_data()
    res = two_stage_least_squares(y, x, w, z)
    assert res.estimate[0] == pytest.approx(0.5, abs=0.05)
    assert res.ols[0] > 0.7
    assert res.first_stage_f[0] > 100
    assert res.sargan_df == 1
    assert res.sargan_p > 0.01


def test_2sls_matches_the_textbook_formula() -> None:
    y, x, w, z = _iv_data(n=300, seed=1)
    res = two_stage_least_squares(y, x, w, z[:, :1])
    # Exactly identified: beta = (Z'X)^-1 Z'y with Z = [z, w], X = [x, w].
    big_z = np.column_stack([z[:, :1], w])
    big_x = np.column_stack([x, w])
    beta = np.linalg.solve(big_z.T @ big_x, big_z.T @ y)
    assert res.estimate[0] == pytest.approx(beta[0])
    assert res.sargan is None
    assert res.sargan_p is None


def test_first_stage_f_is_the_partial_f_for_one_regressor() -> None:
    y, x, w, z = _iv_data(n=300, seed=2)
    res = two_stage_least_squares(y, x, w, z)
    r_res = x[:, 0] - w @ np.linalg.lstsq(w, x[:, 0], rcond=None)[0]
    full = np.column_stack([w, z])
    u_res = x[:, 0] - full @ np.linalg.lstsq(full, x[:, 0], rcond=None)[0]
    f = ((r_res @ r_res - u_res @ u_res) / 2) / (u_res @ u_res / (300 - 4))
    assert res.first_stage_f[0] == pytest.approx(f)


def test_sargan_rejects_an_invalid_instrument() -> None:
    y, x, w, z = _iv_data()
    y = y + 0.5 * z[:, 1]  # z2 now acts on y directly
    assert two_stage_least_squares(y, x, w, z).sargan_p < 0.001


def test_verdicts() -> None:
    assert verdict(False, 50.0, None, 0.0, 0.05) == "not identified"
    assert verdict(True, 5.0, None, 0.0, 0.05) == "weak instruments"
    assert verdict(True, 50.0, 0.01, 0.0, 0.05) == "instruments disagree"
    assert verdict(True, 50.0, 0.5, 0.01, 0.05) == "same-week link"
    assert verdict(True, 50.0, None, 0.5, 0.05) == "no clear evidence"


# ---------------------------------------------------------------------------
# granger_pretest same_week table
# ---------------------------------------------------------------------------


def _pretest(value: float | None, n_obs: int = 1500, seed: int = 4, **kwargs):
    links = {"brand_consideration": {"brand_awareness": value}} if value else None
    data = simulate_brand_data(n_obs, seed=seed, contemporaneous=links)
    return granger_pretest(
        data.endog,
        data.exog,
        contemporaneous=kwargs.pop("contemporaneous", {"brand_consideration": ["brand_awareness"]}),
        excluded=kwargs.pop("excluded", {"brand_consideration": CHANNELS}),
        **kwargs,
    )


def test_no_same_week_table_by_default() -> None:
    data = simulate_brand_data()
    assert granger_pretest(data.endog, data.exog).same_week is None


def test_pretest_finds_a_link() -> None:
    row = _pretest(0.2).same_week.iloc[0]
    assert (row.cause, row.effect) == ("brand_awareness", "brand_consideration")
    assert row.estimate == pytest.approx(0.2, abs=0.07)
    assert row.ols > row.estimate + 0.1
    assert row.n_instruments == 3
    assert row.first_stage_f > 10
    assert row.verdict == "same-week link"


def test_pretest_finds_no_link_where_there_is_none() -> None:
    row = _pretest(None).same_week.iloc[0]
    assert abs(row.estimate) < 0.08
    assert row.ols > 0.2
    assert row.verdict == "no clear evidence"


def test_pretest_handles_two_causes() -> None:
    table = _pretest(
        0.2,
        contemporaneous={
            "brand_consideration": "brand_awareness",
            "organic_sales": ["brand_consideration", "brand_awareness"],
        },
        excluded={"brand_consideration": CHANNELS, "organic_sales": CHANNELS + MACRO},
    ).same_week
    assert len(table) == 3
    sales = table[table["effect"] == "organic_sales"]
    assert (sales["n_instruments"] == 5).all()
    assert sales["estimate"].abs().max() < 0.15
    assert sales["sargan_p"].notna().all()


def test_pretest_reports_unidentified_links() -> None:
    table = _pretest(
        None,
        contemporaneous={
            "brand_awareness": ["organic_sales"],
            "brand_consideration": ["brand_awareness"],
        },
        excluded={"brand_consideration": CHANNELS},
    ).same_week
    row = table.set_index("effect").loc["brand_awareness"]
    assert row.verdict == "not identified"
    assert np.isnan(row.estimate)
    assert row.n_instruments == 0


def test_pretest_without_exog_cannot_identify() -> None:
    data = simulate_brand_data()
    table = granger_pretest(
        data.endog, contemporaneous={"brand_consideration": ["brand_awareness"]}
    ).same_week
    assert table["verdict"].tolist() == ["not identified"]


def test_pretest_flags_weak_instruments_on_a_short_sample() -> None:
    table = _pretest(
        0.2, n_obs=60, excluded={"brand_consideration": ["sponsorship_reach"]}
    ).same_week
    assert table["first_stage_f"].iloc[0] < 10
    assert table["verdict"].iloc[0] == "weak instruments"


def test_pretest_prints_the_same_week_table() -> None:
    text = str(_pretest(0.2))
    assert "Same week" in text
    assert "brand_awareness -> brand_consideration" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"contemporaneous": {"trust": ["brand_awareness"]}}, "unknown brand metric 'trust'"),
        ({"contemporaneous": {"brand_consideration": ["brand_spend"]}}, "unknown brand metric"),
        ({"contemporaneous": {"brand_awareness": ["brand_awareness"]}}, "to itself"),
        (
            {"contemporaneous": {"brand_consideration": ["brand_awareness", "brand_awareness"]}},
            "cause twice",
        ),
        ({"excluded": {"brand_consideration": ["tv"]}}, "unknown driver 'tv'"),
        ({"excluded": {"brand_consideration": ["brand_spend", "brand_spend"]}}, "driver twice"),
    ],
)
def test_pretest_bad_arguments_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _pretest(None, n_obs=100, **kwargs)


def test_cycles_allowed_in_the_test_but_not_the_model() -> None:
    names = list(ENDOG_NAMES)
    cyc = {"brand_consideration": ["brand_awareness"], "brand_awareness": ["brand_consideration"]}
    resolve_same_week(cyc, None, names, CHANNELS, require_dag=False)
    with pytest.raises(ValueError, match="cycle"):
        resolve_same_week(cyc, None, names, CHANNELS, require_dag=True)
