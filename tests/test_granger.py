"""Tests for granger_pretest and GrangerPretestResult."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_vecm import GrangerPretestResult, granger_pretest, simulate_brand_data
from bayesian_vecm._granger import _deterministic, _largest_lag, select_lag_order

DRIVERS = ["brand_spend", "interest_rate", "consumer_confidence"]


@pytest.fixture(scope="module")
def brand():
    return simulate_brand_data()


@pytest.fixture(scope="module")
def result(brand):
    return granger_pretest(brand.endog, brand.exog[DRIVERS])


def _p(result, cause, effect):
    row = result.tests.query("cause == @cause and effect == @effect")
    return float(row["p_value"].iloc[0])


# ---------------------------------------------------------------------------
# Shape of the result
# ---------------------------------------------------------------------------


def test_returns_result(result):
    assert isinstance(result, GrangerPretestResult)
    assert "GrangerPretestResult(lags=" in repr(result)


def test_one_row_per_ordered_pair(result):
    n = 6
    assert len(result.tests) == n * (n - 1)
    assert len(result.same_period) == n * (n - 1) // 2
    assert set(result.tests.columns) == {
        "cause",
        "effect",
        "group",
        "f_stat",
        "df_num",
        "df_den",
        "p_value",
    }


def test_groups(result):
    counts = result.tests["group"].value_counts().to_dict()
    assert counts == {
        "driver -> metric": 9,
        "metric -> driver": 9,
        "metric -> metric": 6,
        "driver -> driver": 6,
    }


def test_n_obs_drops_tested_and_extra_lags(brand):
    res = granger_pretest(brand.endog, brand.exog[DRIVERS], lags=2, extra_lags=1)
    assert res.n_obs == len(brand.endog) - 3
    assert (res.tests["df_num"] == 2).all()


def test_endog_only(brand):
    res = granger_pretest(brand.endog)
    assert set(res.tests["group"]) == {"metric -> metric"}
    text = str(res)
    assert "Links:" in text
    assert "Drivers:" not in text
    assert "Feedback:" not in text


def test_numpy_inputs_get_default_names(brand):
    res = granger_pretest(brand.endog.to_numpy(), brand.exog[DRIVERS].to_numpy())
    assert set(res.tests["cause"]) == {"y0", "y1", "y2", "x0", "x1", "x2"}


def test_str_lists_every_section(result):
    text = str(result)
    for heading in ("Drivers:", "Links:", "Feedback:", "Same period:"):
        assert heading in text
    assert "brand_awareness -> brand_consideration" in text
    assert "brand_spend -> brand_awareness" in text


# ---------------------------------------------------------------------------
# Recovers the simulated chain: spend -> awareness -> consideration -> sales
# ---------------------------------------------------------------------------


def test_finds_the_funnel(result):
    assert _p(result, "brand_awareness", "brand_consideration") < 0.01
    assert _p(result, "brand_consideration", "organic_sales") < 0.01
    assert _p(result, "brand_spend", "brand_awareness") < 0.01


def test_finds_the_macro_drivers(result):
    assert _p(result, "interest_rate", "brand_consideration") < 0.01
    assert _p(result, "consumer_confidence", "brand_consideration") < 0.01


def test_no_feedback_to_spend(result):
    for metric in ("organic_sales", "brand_awareness", "brand_consideration"):
        assert _p(result, metric, "brand_spend") > 0.05


def test_same_period_spend_and_awareness(result):
    row = result.same_period.query(
        "variable_1 == 'brand_awareness' and variable_2 == 'brand_spend'"
    )
    assert float(row["corr"].iloc[0]) > 0.3
    assert float(row["p_value"].iloc[0]) < 0.01


def test_detects_feedback_when_present():
    """Spend that follows last period's sales is flagged as feedback."""
    rng = np.random.default_rng(0)
    n = 300
    sales = np.cumsum(rng.normal(size=n))
    spend = np.zeros(n)
    for t in range(1, n):
        spend[t] = 0.5 * spend[t - 1] + 0.8 * (sales[t - 1] - sales[t - 2] if t > 1 else 0)
        spend[t] += rng.normal(scale=0.3)
    res = granger_pretest(pd.DataFrame({"sales": sales}), pd.DataFrame({"spend": spend}), lags=1)
    assert _p(res, "sales", "spend") < 0.01


# ---------------------------------------------------------------------------
# Agreement with statsmodels
# ---------------------------------------------------------------------------


def test_f_stat_matches_statsmodels(brand):
    sm = pytest.importorskip("statsmodels.tsa.api")
    data = pd.concat([brand.endog, brand.exog[["brand_spend"]]], axis=1)
    ours = granger_pretest(brand.endog, brand.exog[["brand_spend"]], lags=2, extra_lags=0)
    fitted = sm.VAR(data).fit(2, trend="c")
    for cause, effect in (
        ("brand_spend", "brand_awareness"),
        ("brand_awareness", "brand_consideration"),
        ("organic_sales", "brand_spend"),
    ):
        expected = fitted.test_causality(effect, [cause], kind="f").test_statistic
        row = ours.tests.query("cause == @cause and effect == @effect")
        assert float(row["f_stat"].iloc[0]) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lag_order_matches_statsmodels(seed):
    sm = pytest.importorskip("statsmodels.tsa.api")
    data = simulate_brand_data(seed=seed).endog.to_numpy()
    ours = select_lag_order(data, _deterministic(len(data), None), max_lags=6)
    theirs = sm.VAR(data).select_order(maxlags=6, trend="c").selected_orders["hqic"]
    assert ours == max(theirs, 1)


# ---------------------------------------------------------------------------
# Lag cap and seasonal dummies
# ---------------------------------------------------------------------------


def test_lag_cap_keeps_half_the_rows():
    # 60 rows, 6 series, constant, one extra lag: 4 lags in the test
    # regression leaves 56 - 25 = 31 >= 28; 5 lags leaves 55 - 31 = 24 < 27.5.
    assert _largest_lag(60, 6, 1, 1) == 3


def test_short_sample_picks_short_lags():
    """60 rows of 6 series: the cap stops HQ running to the longest lag."""
    rng = np.random.default_rng(1)
    data = np.cumsum(rng.normal(size=(60, 6)), axis=0)
    res = granger_pretest(data[:, :3], data[:, 3:])
    assert res.lags <= 3


def test_seasonal_dummies_are_centred():
    det = _deterministic(24, 12)
    assert det.shape == (24, 12)
    np.testing.assert_allclose(det[:, 1:].sum(axis=0), 0.0, atol=1e-12)


def test_seasonal_keeps_n_obs(brand):
    plain = granger_pretest(brand.endog, lags=1)
    seasonal = granger_pretest(brand.endog, lags=1, seasonal=4)
    assert seasonal.n_obs == plain.n_obs
    assert seasonal.seasonal == 4
    assert "seasonal = 4" in str(seasonal)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_mismatched_rows(brand):
    with pytest.raises(ValueError, match="same number of rows"):
        granger_pretest(brand.endog, brand.exog.iloc[:-1])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"lags": 0}, "lags must be"),
        ({"extra_lags": -1}, "extra_lags must be"),
        ({"alpha": 1.5}, "alpha must be"),
        ({"seasonal": 1}, "seasonal must be"),
    ],
)
def test_rejects_bad_arguments(brand, kwargs, match):
    with pytest.raises(ValueError, match=match):
        granger_pretest(brand.endog, **kwargs)


def test_rejects_duplicate_names(brand):
    exog = brand.exog[["brand_spend"]].rename(columns={"brand_spend": "organic_sales"})
    with pytest.raises(ValueError, match="unique"):
        granger_pretest(brand.endog, exog)


def test_rejects_too_many_lags(brand):
    with pytest.raises(ValueError, match="Too few observations"):
        granger_pretest(brand.endog.iloc[:20], brand.exog.iloc[:20], lags=3)


# ---------------------------------------------------------------------------
# Delay: past spend once current spend is known
# ---------------------------------------------------------------------------


def test_delay_table_shape(result):
    assert len(result.delay) == 9
    assert set(result.delay["group"]) == {"delay"}
    assert "Delay:" in str(result)


def test_delay_absent_without_exog(brand):
    assert granger_pretest(brand.endog).delay is None


def test_no_delay_when_effect_is_same_period(result):
    """The simulated spend effect is same-week only."""
    row = result.delay.query("cause == 'brand_spend' and effect == 'brand_awareness'")
    assert float(row["p_value"].iloc[0]) > 0.05


def _delayed_brand(delay: bool, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = 260
    spend = np.zeros(n)
    for t in range(1, n):
        spend[t] = 0.5 * spend[t - 1] + rng.normal(scale=0.26)
    aware = np.zeros(n)
    for t in range(1, n):
        push = spend[t - 1] if delay else spend[t]
        aware[t] = aware[t - 1] + 0.05 * push + rng.normal(scale=0.01)
    return pd.DataFrame({"aware": aware}), pd.DataFrame({"spend": spend})


def test_delay_found_when_effect_is_lagged():
    res = granger_pretest(*_delayed_brand(delay=True), lags=1)
    assert float(res.delay["p_value"].iloc[0]) < 0.01


def test_persistent_spend_passes_driver_test_but_not_delay():
    """Same-period effect plus persistent spend: driver test fires, delay does not."""
    res = granger_pretest(*_delayed_brand(delay=False), lags=1)
    assert _p(res, "spend", "aware") < 0.05
    assert float(res.delay["p_value"].iloc[0]) > 0.05
