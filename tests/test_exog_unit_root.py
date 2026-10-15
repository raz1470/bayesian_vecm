"""Tests for ``simulate_brand_data(exog_unit_root=...)``: trending drivers."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_vecm import check_stationarity, simulate_brand_data

UR = ("interest_rate",)


def test_default_is_unchanged() -> None:
    base = simulate_brand_data(seed=3)
    empty = simulate_brand_data(seed=3, exog_unit_root=())
    np.testing.assert_array_equal(base.endog.to_numpy(), empty.endog.to_numpy())
    np.testing.assert_array_equal(base.exog.to_numpy(), empty.exog.to_numpy())
    assert base.truth.differenced == ()


def test_other_drivers_are_untouched() -> None:
    base = simulate_brand_data(seed=3)
    ur = simulate_brand_data(seed=3, exog_unit_root=UR)
    others = [c for c in base.exog.columns if c != "interest_rate"]
    np.testing.assert_array_equal(base.exog[others].to_numpy(), ur.exog[others].to_numpy())
    assert ur.truth.differenced == UR


def test_named_driver_is_a_random_walk() -> None:
    ur = simulate_brand_data(seed=3, exog_unit_root=UR).exog["interest_rate"]
    base = simulate_brand_data(seed=3).exog["interest_rate"]
    # Its changes are the stationary version's innovations: x_t - rho x_{t-1}.
    innov = (base - 0.70 * base.shift()).iloc[1:]
    np.testing.assert_allclose(ur.diff().iloc[1:], innov, atol=1e-12)
    assert ur.iloc[0] == pytest.approx(ur.diff().iloc[1:].std(), abs=1.0)


def test_check_stationarity_flags_the_level_and_passes_the_change() -> None:
    exog = simulate_brand_data(seed=3, exog_unit_root=UR).exog
    level = check_stationarity(exog[["interest_rate"]])
    change = check_stationarity(exog[["interest_rate"]].diff().dropna())
    assert level.endog == ["interest_rate"]
    assert change.exog == ["interest_rate"]


def test_data_follow_the_equation_in_changes() -> None:
    data = simulate_brand_data(n_obs=2000, seed=5, exog_unit_root=UR)
    truth = data.truth
    y = data.endog.to_numpy()
    x = data.exog.copy()
    x["interest_rate"] = x["interest_rate"].diff()
    x = x.to_numpy()
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


def test_works_with_a_lag_share() -> None:
    data = simulate_brand_data(seed=3, exog_unit_root=UR, exog_lag_share=0.5)
    assert np.isfinite(data.endog.to_numpy()).all()


def test_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown exog_unit_root"):
        simulate_brand_data(exog_unit_root=("bitcoin",))
