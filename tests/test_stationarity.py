"""Tests for check_stationarity and StationarityResult."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bayesian_vecm import StationarityResult, check_stationarity, simulate_brand_data

pytest.importorskip("statsmodels", reason="statsmodels not installed")


def _series(n_obs: int = 400, seed: int = 0) -> pd.DataFrame:
    """One clearly I(0), one I(1) and one I(2) series."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n_obs, 3))
    stationary = np.zeros(n_obs)
    for t in range(1, n_obs):
        stationary[t] = 0.5 * stationary[t - 1] + noise[t, 0]
    walk = np.cumsum(noise[:, 1])
    return pd.DataFrame(
        {
            "stationary": stationary,
            "walk": walk,
            "double_walk": np.cumsum(np.cumsum(noise[:, 2])),
        }
    )


@pytest.fixture(scope="module")
def result() -> StationarityResult:
    return check_stationarity(_series())


def test_orders_and_roles(result: StationarityResult) -> None:
    table = result.table.set_index("variable")
    assert table["order"].to_dict() == {
        "stationary": "I(0)",
        "walk": "I(1)",
        "double_walk": "I(2)",
    }
    assert table["role"].to_dict() == {
        "stationary": "exog",
        "walk": "endog",
        "double_walk": "difference first",
    }


def test_table_columns_and_p_values(result: StationarityResult) -> None:
    assert list(result.table.columns) == ["variable", "p_level", "p_diff", "order", "role"]
    assert result.table["p_level"].between(0.0, 1.0).all()
    assert result.table["p_diff"].between(0.0, 1.0).all()


def test_endog_and_exog_lists(result: StationarityResult) -> None:
    assert result.endog == ["walk"]
    assert result.exog == ["stationary"]


def test_printed_summary(result: StationarityResult) -> None:
    text = str(result)
    assert "ADF test, 5% level" in text
    assert "trending: use as endog" in text
    assert "stationary: use as exog" in text
    assert "difference it first" in text
    assert "double_walk" in text


def test_array_input_gets_default_names() -> None:
    out = check_stationarity(_series().to_numpy())
    assert list(out.table["variable"]) == ["y0", "y1", "y2"]


def test_signif_changes_the_verdict() -> None:
    """A borderline p-value flips with the significance level."""
    data = _series()[["walk"]]
    p_level = check_stationarity(data).table["p_level"].iloc[0]
    above = min(p_level + 0.01, 0.999)
    assert check_stationarity(data, signif=above).table["order"].iloc[0] == "I(0)"


def test_trend_stationary_series_needs_the_trend_option() -> None:
    rng = np.random.default_rng(1)
    data = pd.DataFrame({"trend": 0.05 * np.arange(400) + rng.standard_normal(400)})
    assert check_stationarity(data, regression="ct").table["order"].iloc[0] == "I(0)"
    assert check_stationarity(data, regression="ct").regression == "ct"


def test_brand_data_are_classified_for_the_model() -> None:
    data = simulate_brand_data()
    out = check_stationarity(pd.concat([data.endog, data.exog], axis=1))
    assert out.endog == list(data.endog.columns)
    assert out.exog == list(data.exog.columns)


@pytest.mark.parametrize("signif", [0.0, 1.0, -0.1, 2.0])
def test_bad_signif_raises(signif: float) -> None:
    with pytest.raises(ValueError, match="signif"):
        check_stationarity(_series(), signif=signif)


def test_bad_regression_raises() -> None:
    with pytest.raises(ValueError, match="regression"):
        check_stationarity(_series(), regression="ctt")


def test_constant_column_raises() -> None:
    data = _series()
    data["flat"] = 3.0
    with pytest.raises(ValueError, match="flat"):
        check_stationarity(data)


def test_too_few_observations_raises() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        check_stationarity(_series(10))


def test_one_dimensional_input_raises() -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        check_stationarity(np.arange(50.0))
