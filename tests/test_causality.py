"""Tests for the causality tables, the evidence measure and the FEVD.

Most tests build a synthetic posterior around known parameters, so they need
no sampling. One small fitted model checks the ``BayesianVECM`` wiring.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from bayesian_vecm import BayesianVECM, CausalityResult, simulate_brand_data
from bayesian_vecm._causality import compute_causality, evidence_against_zero
from bayesian_vecm._fevd import compute_fevd

ENDOG = ["organic_sales", "brand_awareness", "brand_consideration"]
EXOG = ["brand_spend", "pr_reach", "sponsorship_reach", "interest_rate", "consumer_confidence"]
SALES, AWARENESS, CONSIDERATION = 0, 1, 2


def _posterior(
    *,
    noise: float = 0.01,
    n_chains: int = 2,
    n_draws: int = 400,
    gamma: np.ndarray | None = None,
    with_exog: bool = True,
    with_gamma: bool = True,
    seed: int = 0,
) -> xr.DataTree:
    """Posterior draws scattered around the brand DGP's true parameters."""
    truth = simulate_brand_data(10).truth
    rng = np.random.default_rng(seed)

    def scatter(value: np.ndarray, scale: float = noise) -> np.ndarray:
        return value + scale * rng.standard_normal((n_chains, n_draws, *value.shape))

    beta = scatter(truth.beta)
    beta[:, :, :2, :] = np.eye(2)  # normalised block is fixed, as in a real fit
    data = {
        "alpha": (("chain", "draw", "variable", "relation"), scatter(truth.alpha)),
        "beta": (("chain", "draw", "variable", "relation"), beta),
        "Sigma": (
            ("chain", "draw", "variable", "variable_bis"),
            np.broadcast_to(truth.sigma, (n_chains, n_draws, 3, 3)).copy(),
        ),
    }
    if with_gamma:
        gamma_true = truth.gamma if gamma is None else gamma
        data["Gamma"] = (("chain", "draw", "variable", "regressor"), scatter(gamma_true))
    if with_exog:
        data["B"] = (("chain", "draw", "variable", "exog_variable"), scatter(truth.b, noise / 10))
    dataset = xr.Dataset(data, coords={"chain": np.arange(n_chains), "draw": np.arange(n_draws)})
    return xr.DataTree.from_dict({"posterior": dataset})


def _links(table, threshold: float = 0.95) -> set[tuple[str, str]]:
    found = table[table["evidence"] >= threshold]
    return set(zip(found["cause"], found["effect"], strict=True))


# ---------------------------------------------------------------------------
# The evidence measure
# ---------------------------------------------------------------------------


def test_evidence_is_near_one_far_from_zero() -> None:
    draws = np.random.default_rng(0).normal(5.0, 1.0, size=4000)
    assert evidence_against_zero(draws) > 0.999


def test_evidence_is_near_zero_when_centred_on_zero() -> None:
    draws = np.random.default_rng(0).normal(0.0, 1.0, size=4000)
    assert evidence_against_zero(draws) < 0.05


def test_scalar_evidence_matches_the_two_sided_tail_formula() -> None:
    draws = np.random.default_rng(0).normal(1.0, 1.0, size=200_000)
    expected = 1.0 - 2.0 * min(np.mean(draws > 0), np.mean(draws < 0))
    assert evidence_against_zero(draws) == pytest.approx(expected, abs=0.005)
    assert evidence_against_zero(draws) == pytest.approx(0.6827, abs=0.005)


def test_one_and_two_dimensional_input_agree() -> None:
    draws = np.random.default_rng(0).normal(0.5, 1.0, size=1000)
    assert evidence_against_zero(draws) == evidence_against_zero(draws[:, np.newaxis])


def test_block_evidence_picks_up_one_strong_coefficient() -> None:
    rng = np.random.default_rng(0)
    block = np.column_stack([rng.normal(0, 1, 4000), rng.normal(6, 1, 4000)])
    assert evidence_against_zero(block) > 0.999
    assert evidence_against_zero(block[:, :1]) < 0.05


def test_block_evidence_uses_the_correlation() -> None:
    """Two coefficients that are each unclear can be jointly clear."""
    rng = np.random.default_rng(0)
    common = rng.normal(0, 1, 4000)
    block = np.column_stack([1.0 + common, -1.0 + common + rng.normal(0, 0.1, 4000)])
    assert evidence_against_zero(block[:, 0]) < 0.8
    assert evidence_against_zero(block) > 0.999


def test_constant_block() -> None:
    assert evidence_against_zero(np.zeros(50)) == 0.0
    assert evidence_against_zero(np.full(50, 0.3)) == 1.0


def test_partly_constant_block_does_not_fail() -> None:
    rng = np.random.default_rng(0)
    block = np.column_stack([np.zeros(2000), rng.normal(4, 1, 2000)])
    assert evidence_against_zero(block) > 0.99


# ---------------------------------------------------------------------------
# The tables, checked against the DGP's causal chain
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result() -> CausalityResult:
    return compute_causality(_posterior(), 1, variable_names=ENDOG, exog_names=EXOG)


def test_table_shapes_and_columns(result: CausalityResult) -> None:
    columns = ["cause", "effect", "evidence", "mean", "lower", "upper"]
    assert list(result.short_run.columns) == columns
    assert list(result.long_run.columns) == columns
    assert list(result.exog.columns) == columns
    assert len(result.short_run) == 6  # K * (K - 1) ordered pairs
    assert len(result.long_run) == 6
    assert len(result.exog) == 15  # m * K
    assert list(result.weak_exogeneity["variable"]) == ENDOG


def test_no_variable_is_paired_with_itself(result: CausalityResult) -> None:
    assert (result.short_run["cause"] != result.short_run["effect"]).all()
    assert (result.long_run["cause"] != result.long_run["effect"]).all()


def test_evidence_is_a_probability(result: CausalityResult) -> None:
    for table in (result.short_run, result.long_run, result.exog, result.weak_exogeneity):
        assert table["evidence"].between(0.0, 1.0).all()


def test_interval_brackets_the_mean(result: CausalityResult) -> None:
    for table in (result.short_run, result.long_run, result.exog):
        assert (table["lower"] <= table["mean"]).all()
        assert (table["mean"] <= table["upper"]).all()


def test_short_run_recovers_the_chain(result: CausalityResult) -> None:
    assert _links(result.short_run) == {
        ("brand_awareness", "brand_consideration"),
        ("brand_consideration", "organic_sales"),
    }


def test_long_run_recovers_the_chain(result: CausalityResult) -> None:
    assert _links(result.long_run) == {
        ("brand_awareness", "brand_consideration"),
        ("brand_consideration", "organic_sales"),
    }


def test_only_awareness_is_weakly_exogenous(result: CausalityResult) -> None:
    flags = dict(
        zip(
            result.weak_exogeneity["variable"],
            result.weak_exogeneity["weakly_exogenous"],
            strict=True,
        )
    )
    assert flags == {
        "organic_sales": False,
        "brand_awareness": True,
        "brand_consideration": False,
    }


def test_exog_table_recovers_the_true_drivers(result: CausalityResult) -> None:
    assert _links(result.exog) == {
        ("brand_spend", "brand_awareness"),
        ("pr_reach", "brand_awareness"),
        ("sponsorship_reach", "brand_awareness"),
        ("interest_rate", "brand_consideration"),
        ("consumer_confidence", "brand_consideration"),
    }


def test_effect_sizes_are_close_to_the_truth(result: CausalityResult) -> None:
    truth = simulate_brand_data(10).truth
    short = result.short_run.set_index(["cause", "effect"])["mean"]
    assert short["brand_awareness", "brand_consideration"] == pytest.approx(0.25, abs=0.005)
    assert short["brand_consideration", "organic_sales"] == pytest.approx(0.40, abs=0.005)
    pi = truth.alpha @ truth.beta.T
    long = result.long_run.set_index(["cause", "effect"])["mean"]
    assert long["brand_consideration", "organic_sales"] == pytest.approx(
        pi[SALES, CONSIDERATION], abs=0.01
    )
    assert long["brand_awareness", "brand_consideration"] == pytest.approx(
        pi[CONSIDERATION, AWARENESS], abs=0.01
    )


def test_default_names() -> None:
    out = compute_causality(_posterior(), 1)
    assert list(out.weak_exogeneity["variable"]) == ["y0", "y1", "y2"]
    assert set(out.exog["cause"]) == {"x0", "x1", "x2", "x3", "x4"}


def test_no_exog_gives_none() -> None:
    assert compute_causality(_posterior(with_exog=False), 1).exog is None


def test_no_lagged_differences_gives_an_empty_short_run_table() -> None:
    out = compute_causality(_posterior(with_gamma=False), 0, variable_names=ENDOG)
    assert len(out.short_run) == 0
    assert len(out.long_run) == 6
    assert "no lagged differences" in str(out)


def test_deterministic_column_in_gamma_is_ignored() -> None:
    """An outside constant adds a trailing Gamma column. It is not a lag."""
    idata = _posterior()
    base = compute_causality(idata, 1, variable_names=ENDOG)

    posterior = idata.posterior.to_dataset()
    gamma = posterior["Gamma"].values
    constant = np.full((*gamma.shape[:3], 1), 5.0)
    posterior = posterior.drop_vars("Gamma")
    posterior["Gamma"] = (
        ("chain", "draw", "variable", "regressor_bis"),
        np.concatenate([gamma, constant], axis=-1),
    )
    out = compute_causality(
        xr.DataTree.from_dict({"posterior": posterior}), 1, variable_names=ENDOG
    )
    np.testing.assert_array_equal(out.short_run["mean"], base.short_run["mean"])
    np.testing.assert_array_equal(out.short_run["evidence"], base.short_run["evidence"])


def test_a_link_at_the_second_lag_only_is_found() -> None:
    gamma = np.zeros((3, 6))
    gamma[SALES, 3 + AWARENESS] = 0.3  # lag 2: awareness -> sales
    out = compute_causality(_posterior(gamma=gamma), 2, variable_names=ENDOG)
    assert _links(out.short_run) == {("brand_awareness", "organic_sales")}


def test_long_run_table_does_not_depend_on_the_beta_normalisation() -> None:
    """Pi = alpha beta' is unchanged when beta is rotated and alpha compensates."""
    idata = _posterior()
    base = compute_causality(idata, 1, variable_names=ENDOG)

    rotation = np.array([[2.0, 0.5], [-1.0, 1.5]])
    posterior = idata.posterior.to_dataset()
    posterior["beta"] = (posterior["beta"].dims, posterior["beta"].values @ rotation)
    posterior["alpha"] = (
        posterior["alpha"].dims,
        posterior["alpha"].values @ np.linalg.inv(rotation).T,
    )
    rotated = compute_causality(
        xr.DataTree.from_dict({"posterior": posterior}), 1, variable_names=ENDOG
    )
    np.testing.assert_allclose(rotated.long_run["mean"], base.long_run["mean"], atol=1e-10)
    np.testing.assert_allclose(rotated.long_run["evidence"], base.long_run["evidence"])
    np.testing.assert_allclose(
        rotated.weak_exogeneity["evidence"], base.weak_exogeneity["evidence"], atol=0.01
    )


def test_threshold_changes_the_verdict_not_the_evidence() -> None:
    idata = _posterior(noise=0.1)
    strict = compute_causality(idata, 1, variable_names=ENDOG, threshold=0.999)
    loose = compute_causality(idata, 1, variable_names=ENDOG, threshold=0.5)
    np.testing.assert_array_equal(strict.short_run["evidence"], loose.short_run["evidence"])
    assert strict.weak_exogeneity["weakly_exogenous"].sum() >= (
        loose.weak_exogeneity["weakly_exogenous"].sum()
    )


def test_wider_interval_for_higher_ci_prob() -> None:
    idata = _posterior()
    narrow = compute_causality(idata, 1, ci_prob=0.5).short_run
    wide = compute_causality(idata, 1, ci_prob=0.95).short_run
    assert ((wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])).all()


@pytest.mark.parametrize("name", ["threshold", "ci_prob"])
@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.5])
def test_probabilities_outside_the_unit_interval_raise(name: str, value: float) -> None:
    with pytest.raises(ValueError, match=name):
        compute_causality(_posterior(), 1, **{name: value})


def test_printed_summary(result: CausalityResult) -> None:
    text = str(result)
    assert "Gamma prior = Normal" in text
    assert "evidence threshold = 0.95" in text
    assert "brand_awareness -> brand_consideration" in text
    assert "helps predict" in text
    assert "no clear evidence" in text
    assert "weakly exogenous" in text
    assert "causes" not in text


def test_gamma_prior_is_recorded() -> None:
    out = compute_causality(_posterior(), 1, gamma_prior="Horseshoe")
    assert out.gamma_prior == "Horseshoe"
    assert "Gamma prior = Horseshoe" in str(out)


# ---------------------------------------------------------------------------
# FEVD
# ---------------------------------------------------------------------------


def _exact_posterior() -> xr.DataTree:
    """Every draw equals the truth, so the FEVD has a closed form."""
    return _posterior(noise=0.0, n_chains=1, n_draws=2, with_exog=False)


def _phi(steps: int) -> list[np.ndarray]:
    """MA matrices of the true process by direct VAR(2) recursion."""
    truth = simulate_brand_data(10).truth
    a1 = np.eye(3) + truth.alpha @ truth.beta.T + truth.gamma
    a2 = -truth.gamma
    phi = [np.eye(3), a1]
    for _ in range(2, steps):
        phi.append(a1 @ phi[-1] + a2 @ phi[-2])
    return phi[:steps]


@pytest.mark.parametrize("method", ["girf", "cholesky"])
def test_fevd_shape_and_coords(method: str) -> None:
    fevd = compute_fevd(_posterior(), 1, 12, method=method, variable_names=ENDOG)
    assert fevd.dims == ("chain", "draw", "horizon", "response_variable", "shock_variable")
    assert fevd.shape == (2, 400, 12, 3, 3)
    assert list(fevd.coords["horizon"].values) == list(range(1, 13))
    assert list(fevd.coords["shock_variable"].values) == ENDOG


@pytest.mark.parametrize("method", ["girf", "cholesky"])
def test_fevd_rows_sum_to_one_and_shares_are_non_negative(method: str) -> None:
    fevd = compute_fevd(_posterior(noise=0.05), 1, 24, method=method)
    np.testing.assert_allclose(fevd.sum("shock_variable").values, 1.0, atol=1e-12)
    assert (fevd.values >= 0).all()


def test_cholesky_fevd_matches_the_closed_form() -> None:
    steps = 20
    sigma = simulate_brand_data(10).truth.sigma
    chol = np.linalg.cholesky(sigma)
    num, mse, expected = np.zeros((3, 3)), np.zeros(3), []
    for phi in _phi(steps):
        num += (phi @ chol) ** 2
        mse += np.diag(phi @ sigma @ phi.T)
        expected.append(num / mse[:, np.newaxis])
    fevd = compute_fevd(_exact_posterior(), 1, steps, method="cholesky")
    np.testing.assert_allclose(fevd.values[0, 0], np.array(expected), atol=1e-12)


def test_generalised_fevd_matches_the_closed_form() -> None:
    steps = 20
    sigma = simulate_brand_data(10).truth.sigma
    num, mse, expected = np.zeros((3, 3)), np.zeros(3), []
    for phi in _phi(steps):
        num += (phi @ sigma) ** 2 / np.diag(sigma)
        mse += np.diag(phi @ sigma @ phi.T)
        raw = num / mse[:, np.newaxis]
        expected.append(raw / raw.sum(axis=1, keepdims=True))
    fevd = compute_fevd(_exact_posterior(), 1, steps, method="girf")
    np.testing.assert_allclose(fevd.values[0, 0], np.array(expected), atol=1e-12)


def test_one_step_cholesky_fevd_is_the_impact_decomposition() -> None:
    sigma = simulate_brand_data(10).truth.sigma
    chol = np.linalg.cholesky(sigma)
    fevd = compute_fevd(_exact_posterior(), 1, 1, method="cholesky")
    np.testing.assert_allclose(
        fevd.values[0, 0, 0], chol**2 / np.diag(sigma)[:, np.newaxis], atol=1e-12
    )
    assert fevd.values[0, 0, 0, SALES, SALES] == pytest.approx(1.0)  # first in the ordering


def test_methods_agree_when_shocks_are_uncorrelated() -> None:
    idata = _posterior(noise=0.02)
    posterior = idata.posterior.to_dataset()
    diagonal = np.broadcast_to(np.diag([1.0, 2.0, 0.5]), posterior["Sigma"].shape).copy()
    posterior["Sigma"] = (posterior["Sigma"].dims, diagonal)
    idata = xr.DataTree.from_dict({"posterior": posterior})
    np.testing.assert_allclose(
        compute_fevd(idata, 1, 15, method="girf").values,
        compute_fevd(idata, 1, 15, method="cholesky").values,
        atol=1e-12,
    )


def test_long_horizon_fevd_is_driven_by_the_permanent_shock() -> None:
    """Only the awareness innovation is permanent, so every row converges to
    the split of that innovation across the orthogonal shocks."""
    sigma = simulate_brand_data(10).truth.sigma
    chol = np.linalg.cholesky(sigma)
    limit = chol[AWARENESS] ** 2 / sigma[AWARENESS, AWARENESS]
    fevd = compute_fevd(_exact_posterior(), 1, 4000, method="cholesky").values[0, 0, -1]
    for row in fevd:
        np.testing.assert_allclose(row, limit, atol=0.01)


def test_fevd_without_lagged_differences() -> None:
    fevd = compute_fevd(_posterior(with_gamma=False), 0, 5)
    assert fevd.shape == (2, 400, 5, 3, 3)


def test_fevd_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="method"):
        compute_fevd(_posterior(), 1, 5, method="sign")
    with pytest.raises(ValueError, match="steps"):
        compute_fevd(_posterior(), 1, 0)


# ---------------------------------------------------------------------------
# BayesianVECM wiring
# ---------------------------------------------------------------------------


def test_methods_require_a_fitted_model() -> None:
    model = BayesianVECM(k_ar_diff=1, coint_rank=2)
    with pytest.raises(RuntimeError):
        model.fevd(steps=5)
    with pytest.raises(RuntimeError):
        model.granger_causality()


@pytest.fixture(scope="module")
def fitted_model() -> BayesianVECM:
    data = simulate_brand_data(80)
    model = BayesianVECM(
        k_ar_diff=1,
        coint_rank=2,
        deterministic="co",
        priors={"Gamma": {"dist": "Horseshoe"}},
    )
    model.fit(
        data.endog,
        exog=data.exog,
        draws=20,
        tune=20,
        chains=1,
        cores=1,
        progressbar=False,
        random_seed=0,
        compute_convergence_checks=False,
    )
    return model


def test_model_granger_causality(fitted_model: BayesianVECM) -> None:
    out = fitted_model.granger_causality(threshold=0.9)
    assert isinstance(out, CausalityResult)
    assert out.gamma_prior == "Horseshoe"
    assert out.threshold == 0.9
    assert list(out.weak_exogeneity["variable"]) == ENDOG
    assert set(out.short_run["cause"]) == set(ENDOG)
    assert set(out.exog["cause"]) == set(EXOG)
    assert len(out.short_run) == 6


def test_model_fevd(fitted_model: BayesianVECM) -> None:
    fevd = fitted_model.fevd(steps=8)
    assert fevd.shape == (1, 20, 8, 3, 3)
    assert list(fevd.coords["response_variable"].values) == ENDOG
    np.testing.assert_allclose(fevd.sum("shock_variable").values, 1.0, atol=1e-10)
    assert not np.allclose(fevd.values, fitted_model.fevd(steps=8, method="cholesky").values)
