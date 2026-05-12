# Project notes & handoff

Working notes on direction, decisions, and where to pick up.
Read this first if you (or a fresh Claude session) are coming back after a break.

## Goal

A Python package for **Bayesian Vector Error Correction Models (VECM)**, built on PyMC.
Headed for PyPI eventually. Author: Ryan O'Sullivan.

## API we're aiming for

Combines two reference patterns:

- **`statsmodels.tsa.vector_ar.vecm.VECM`** — for the public interface, parameter names, and econometric methodology (alpha, beta, Gamma, Sigma; `k_ar_diff`, `coint_rank`, `deterministic`).
- **`pymc_marketing`** model classes (e.g. `MMM`) — for the Bayesian patterns: a class that builds a PyMC model under the hood, exposes `fit()` / `idata` / `sample_posterior_predictive()`, accepts user-supplied priors with sensible defaults.

Target shape:

```python
from bayesian_vecm import BayesianVECM

model = BayesianVECM(
    k_ar_diff=1,
    coint_rank=1,
    deterministic="ci",
    priors={...},  # optional
)
model.fit(endog_df)             # runs PyMC sampling
model.idata                     # arviz.InferenceData
model.summary()
model.sample_posterior_predictive(steps=12)
```

## Decisions locked in

| Area | Choice | Why |
| --- | --- | --- |
| Package layout | `src/bayesian_vecm/` | Modern best practice; forces testing against installed package. |
| Build backend | `hatchling` | Lightweight, modern, well-supported. |
| Env / deps | `uv` + `uv.lock` | Fast, reproducible, current best practice. |
| Lint + format | `ruff` | Replaces flake8 / isort / black. Configured in `pyproject.toml`. |
| Tests | `pytest` + `pytest-cov` | Standard. Configured in `pyproject.toml`. |
| Min Python | 3.11 | Matches modern scientific Python. |
| License | MIT | Permissive, standard for OSS Python. |
| CI | GitHub Actions, matrix on Py 3.11 + 3.12 | `.github/workflows/ci.yml` runs ruff + pytest on push to main and PRs. |

## Status as of last session (2026-05-12)

**Done:**

- Full scaffold: `pyproject.toml`, `src/bayesian_vecm/__init__.py`, `tests/test_package.py`, README, LICENSE, `.gitignore`.
- Local env via `uv sync --all-extras`. Sanity tests pass (`uv run pytest`).
- Repo on GitHub at <https://github.com/raz1470/bayesian_vecm>.
- CI green on `main`.
- **Branch protection on `main` is live** — direct pushes are rejected; changes must go through a PR with the 2 required status checks (ruff + pytest) green before merge.
- **Data utilities slice shipped** (PR `feat/data-utilities`):
  - `numpy>=1.26` added as first runtime dep.
  - `src/bayesian_vecm/_data.py` with `validate_endog`, `difference`, `lag_matrix` (lag-major ordering, statsmodels-compatible).
  - `tests/test_data.py` with 22 unit tests, all passing.

**Not yet done:**

- Any actual VECM model code (just preprocessing primitives so far).
- Pandas integration tests (we duck-type via `.to_numpy()`; haven't tested against a real pandas DataFrame).

## Workflow reminder

Pushes to `main` are rejected. Always work on a feature branch:

```bash
git switch -c feat/<slice-name>
# ...commits...
git push -u origin feat/<slice-name>
# Open PR on GitHub, wait for CI, merge, then locally:
git switch main && git pull && git branch -d feat/<slice-name>
```

## Picking up next session — recommended candidates

Pick one (in roughly this order):

1. **`cointegration_design` helper.** Combines `difference` + `lag_matrix` into the regression triple a VECM actually needs: `(ΔY, ΔX, Y_{-1})` where `ΔY` is the LHS, `ΔX` stacks lagged differences, and `Y_{-1}` is the level-lag for the error-correction term. Still a pure function; another easy TDD slice.
2. **`BayesianVECM` class skeleton.** Mirror the `pymc_marketing.MMM` shape: `__init__` accepts `k_ar_diff`, `coint_rank`, `deterministic`, `priors`; `fit()` / `idata` / `summary()` / `sample_posterior_predictive()` as stubs that `raise NotImplementedError`. This forces the API decisions before any PyMC code, and `raise NotImplementedError` is already excluded from coverage.
3. **First PyMC model.** Start with the simplest case: known cointegration rank `r=1`, no deterministic terms, weakly-informative priors on `α`, `β`, `Γ`, and `Σ`. This is where the real econometrics begins (identification of `β` is the tricky part — beta is only identified up to a rotation without a normalisation).

Why option 1 next: it's the natural follow-on to `_data.py`, keeps the testable-pure-functions streak going, and the result is the exact input the eventual PyMC model needs.

## Useful commands

```bash
# Activate the venv (or use `uv run <cmd>` to skip activation)
source .venv/bin/activate

# Run tests
uv run pytest

# Lint + format check
uv run ruff check .
uv run ruff format --check .

# Auto-format
uv run ruff format .

# Add a runtime dep
uv add numpy

# Add a dev-only dep
uv add --dev mypy

# Build distributions
uv build
```

## Domain-learning track

Ryan is **learning VECMs as we build**, so explanations of the econometrics (cointegration, error-correction term, identification, lag selection, etc.) should accompany the code as it's written.
