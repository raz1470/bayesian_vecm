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

**Update 2026-05-15:**

- **Docs/learning track kicked off.** Added `notebooks/01_data_utilities_walkthrough.ipynb` — a beginner-friendly walkthrough of `validate_endog`, `difference`, and `lag_matrix` with synthetic-data demos and a primer on where each helper fits into the VECM equation. Convention: one numbered notebook per public-API slice.

**Not yet done:**

- Any actual VECM model code (just preprocessing primitives so far).
- Pandas integration tests (we duck-type via `.to_numpy()`; haven't tested against a real pandas DataFrame).
- **CI execution of notebooks.** Notebooks aren't run in CI yet, so the docs/learning track is at risk of silent drift. Add a job that runs `jupyter nbconvert --to notebook --execute notebooks/*.ipynb` and fails on errors. Also decide on outputs strategy — leaning toward `nbstripout` as a git filter so committed `.ipynb` files have outputs cleared and diffs stay small, but worth revisiting once we have 2–3 notebooks.

## Workflow reminder

Pushes to `main` are rejected. Always work on a feature branch:

```bash
git switch -c feat/<slice-name>
# ...commits...
git push -u origin feat/<slice-name>
# Open PR on GitHub, wait for CI, merge, then locally:
git switch main && git pull && git branch -d feat/<slice-name>
```

## Next slice — `cointegration_design` (decisions locked in)

**Branch:** `feat/cointegration-design`

**Goal.** A single helper that turns raw `(T, K)` data into the three regression matrices the VECM equation actually consumes. The equation:

```
Δy_t = α β' y_{t-1} + Γ_1 Δy_{t-1} + … + Γ_k Δy_{t-k} + ε_t
```

so we need, for each usable time `t`:

- `delta_y`  : Δy_t                                  — shape `(T_eff, K)`     (LHS)
- `delta_x`  : [Δy_{t-1}, …, Δy_{t-k}] stacked       — shape `(T_eff, K*k)`   (Γ blocks)
- `y_lag1`   : y_{t-1}                               — shape `(T_eff, K)`     (β feeds on this; this is what makes it a VECM vs a VAR-in-differences)

Effective sample size: `T_eff = T − k − 1`. Earliest usable `t` is `k + 2`.

**API decisions (simplicity-first):**

- Return type: a `typing.NamedTuple` called `CointegrationDesign(delta_y, delta_x, y_lag1)`. Self-documenting, unpackable, no methods.
- Deterministic terms (`"n" / "co" / "ci" / "lo" / "li"`) **deferred** to a follow-up slice. This PR ships only the "no constants, no trends" case.
- `validate_endog` is called *inside* `cointegration_design` — so callers can pass raw DataFrame/ndarray without ceremony.
- New module: `src/bayesian_vecm/_design.py`. Keeps `_data.py` as pure preprocessing; design matrices are a separate concept.

**Signature sketch:**

```python
from typing import NamedTuple
from numpy.typing import NDArray
import numpy as np

class CointegrationDesign(NamedTuple):
    delta_y: NDArray[np.float64]  # (T_eff, K)
    delta_x: NDArray[np.float64]  # (T_eff, K * k_ar_diff)
    y_lag1: NDArray[np.float64]   # (T_eff, K)

def cointegration_design(data, k_ar_diff: int) -> CointegrationDesign: ...
```

**Tests to write first (TDD):**

- Hand-built tiny example: `T=5, K=2, k_ar_diff=1`, verify all three matrices row-by-row.
- Shape: `delta_y.shape == (T - k - 1, K)`, `delta_x.shape == (T - k - 1, K*k)`, `y_lag1.shape == (T - k - 1, K)`.
- `k_ar_diff = 0` → `delta_x.shape[1] == 0`, but `delta_y` and `y_lag1` still work.
- Rejects `k_ar_diff < 0` and `k_ar_diff >= T - 1`.
- Accepts DataFrame-like input (uses the `_FakeFrame` trick from `test_data.py`).
- Alignment check: row `i` of all three matrices corresponds to the same time index in the original data.

## After cointegration_design — candidates

1. **Deterministic-terms follow-up.** Add `deterministic` parameter to `cointegration_design` covering at least `"n"` (current default), `"co"` (constant in Γ block) and `"ci"` (constant in cointegration relation).
2. **`BayesianVECM` class skeleton.** Mirror the `pymc_marketing.MMM` shape: `__init__` accepts `k_ar_diff`, `coint_rank`, `deterministic`, `priors`; `fit()` / `idata` / `summary()` / `sample_posterior_predictive()` as stubs that `raise NotImplementedError`. Forces API decisions before any PyMC code.
3. **First PyMC model.** Simplest case: known cointegration rank `r=1`, no deterministic terms, weakly-informative priors on `α`, `β`, `Γ`, `Σ`. Real econometrics starts here — identification of `β` is the tricky part (β is only identified up to a rotation without a normalisation, e.g. `β[:r, :] = I_r`).

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

**Convention.** One numbered Jupyter notebook per public-API slice, living in `notebooks/`:

- Filename pattern: `NN_<topic>.ipynb` (e.g. `01_data_utilities_walkthrough.ipynb`).
- Each notebook explains *what* each helper does, *why* a VECM needs it, and demos it on small synthetic data — written for a reader meeting VECMs for the first time.
- Trigger for a new notebook: "did this slice ship something a learner needs to understand?" Internal refactors don't need one.
- Notebooks are runnable docs *and* lightweight integration tests — when CI execution lands (see TODO in the status section), a broken explanation becomes a failing build.
- Once the catalogue grows, consider graduating to a docs site (Sphinx + nbsphinx, or MkDocs + mkdocs-jupyter). Defer until the `BayesianVECM` skeleton is in.
