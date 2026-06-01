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
| Min Python | 3.12 | Required by arviz 1.x; bumped 2026-05-20. |
| License | MIT | Permissive, standard for OSS Python. |
| CI | GitHub Actions, matrix on Py 3.11 + 3.12 | `.github/workflows/ci.yml` runs ruff + pytest on push to main and PRs. |

## Status as of last session (2026-05-30)

**Update 2026-05-30 — `feat/wider-graph` complete (merged to `main`):**

- **Full v0 envelope now live.** The PyMC graph supports any `coint_rank >= 1` and all five deterministic codes (`"n"`, `"co"`, `"ci"`, `"lo"`, `"li"`).
- **Implementation insight:** the graph reads shapes directly off the design matrices rather than branching on `deterministic`. `cointegration_design` already appended the right columns, so:
  - Inside terms (`"ci"`, `"li"`): `y_lag1` gains a column → β widens from `(K, r)` to `(K+1, r)`, extra row is free.
  - Outside terms (`"co"`, `"lo"`): `delta_x` gains a column → Γ widens from `(K, Kk)` to `(K, Kk+1)`.
  - Γ condition changed from `k_ar_diff > 0` to `delta_x_cols > 0` (handles `k=0` + outside term).
- **`test_pymc.py`:** removed `TestScopeGuards`; added `TestDeterministicTerms` (18 tests across all 5 codes × r=1,2).
- **Notebook 06** shipped: trivariate `r=2` example (β-pin is `I_2`, free row recovery) + `"ci"` and `"co"` demos showing β and Γ shape changes.
- **nbstripout pre-commit hook migration** done (separate chore PR): moved from git clean filter (parallel, crashes with 5 notebooks) to pre-commit hook (sequential). Workflow: stage notebook → pre-commit strips outputs → re-stage stripped file → commit.
- **Suite: 173 passed.**

## Status as of last session (2026-05-28)

**Update 2026-05-28 — housekeeping sprint (all merged to `main`):**

- **`_constants.py` introduced** (PR `chore/consolidate-deterministic-codes`): `VALID_DETERMINISTIC` moved from `_design.py` and `_model.py` into a new `src/bayesian_vecm/_constants.py`. Both modules now import from there. `tests/test_model.py` updated to import from `_constants` too.
- **Pre-commit hook** (PR `chore/pre-commit-hook`): `.pre-commit-config.yaml` added with `ruff-format` and `ruff` hooks from `astral-sh/ruff-pre-commit`. `pre-commit` added as a dev dep. Run `pre-commit install` once after cloning to activate.
- **Notebook CI + nbstripout** (PR `chore/notebook-ci`):
  - New `notebooks` job in `.github/workflows/ci.yml` runs `jupyter nbconvert --execute --inplace` on all notebooks with a 5-minute timeout. This is the CI guard that would have caught the 2026-05-20 dep-drift incident immediately.
  - `nbstripout` added as a dev dep and wired via `.gitattributes` — committed notebooks always have outputs stripped, keeping diffs small.
  - `notebooks/04_first_pymc_model_walkthrough.ipynb` gained a `FAST_SAMPLING` config cell (matching the pattern from notebook 05): `FAST_SAMPLING=True` → 200 draws / 200 tune / 2 chains; `False` → 1000/1000/4.
  - `notebooks/03_bayesian_vecm_skeleton_walkthrough.ipynb` §6 updated: "honest stubs" section replaced with a "pre-fit guard" demo (all four estimation methods are now live; the section now shows that calling `idata`/`summary`/`sample_posterior_predictive` before `fit` raises `RuntimeError`, which is still the correct contract).
  - CI now has 3 required status checks: `Lint & test`, `Execute notebooks`, and the pre-commit hook.

**Not yet done:**

- **Higher cointegration rank (`r > 1`) and deterministic terms in the PyMC graph** — see Option 1 below. This is the next slice.
- Pandas integration tests.
- `_pymc.py` coverage is 63% — the uncovered lines are the `NotImplementedError` scope guards that `feat/wider-graph` will delete, plus some prior-override branches.

## Status as of earlier sessions

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

**Update 2026-05-15 (earlier):**

- **Docs/learning track kicked off.** Added `notebooks/01_data_utilities_walkthrough.ipynb` — a beginner-friendly walkthrough of `validate_endog`, `difference`, and `lag_matrix` with synthetic-data demos and a primer on where each helper fits into the VECM equation. Convention: one numbered notebook per public-API slice. Shipped via PR `feat/notebook-data-utilities-walkthrough`.

**Update 2026-05-15 (later):**

- **Cointegration design slice shipped** (PR `feat/cointegration-design`):
  - `src/bayesian_vecm/_design.py` with `CointegrationDesign` NamedTuple and `cointegration_design(data, k_ar_diff)` function. Calls `validate_endog` internally, then `difference` once, then slices and `lag_matrix` to produce three matrices aligned to `T_eff = T - k_ar_diff - 1` rows.
  - `tests/test_design.py` with 21 unit tests covering the hand-built row-by-row spec, shape contract, `k_ar_diff = 0` edge case, validation, DataFrame-like input, alignment across all three outputs, and lag-major column ordering.
  - Kept private (no `__init__.py` re-export), mirroring `_data.py`.
- **Walkthrough notebook 02** added: `notebooks/02_cointegration_design_walkthrough.ipynb`. Frames the alignment problem, derives `T_eff = T - k - 1`, demos the hand-built tiny example, the `k = 0` and `k = 2` cases, and an end-to-end synthetic cointegrated example. First mention of the β-identification problem in the docs — flagged as the natural place to expand once model code lands.

**Update 2026-05-18 (in-flight, not yet on `main`):**

- **Deterministic-terms follow-up** lives on `origin/docs/future-directions`, awaiting a PR + merge. `cointegration_design` gains a `deterministic: str = "n"` argument with the v0 codes `{"n", "co", "ci", "lo", "li"}`; compound Johansen codes (cases 4 and 5) explicitly rejected with a v0.x follow-up message. `tests/test_design.py` grew from 21 to 47 tests. Notebook 02 gained a `§6 Deterministic terms` section. Same branch also captured the future-directions parking lot. Folded into one branch to avoid two adjacent PRs.

**Update 2026-05-19:**

- **`BayesianVECM` class skeleton shipped** (this branch: `feat/bayesian-vecm-skeleton`):
  - `src/bayesian_vecm/_model.py` with the public `BayesianVECM` class. All estimation methods (`fit`, `idata`, `summary`, `sample_posterior_predictive`) raise `NotImplementedError` per design — the skeleton's job is to lock the API shape ahead of the PyMC work, not to estimate anything.
  - Re-exported from the package root: `from bayesian_vecm import BayesianVECM` now works (matches the target API at the top of this file).
  - `tests/test_model.py` with 34 unit tests covering default + custom construction, every supported deterministic code, eager validation of bad `k_ar_diff` / `coint_rank` / `deterministic` / `priors`, and the four `NotImplementedError`-raising methods. Full test suite at 77 (was 43).
- **Three API decisions locked in this slice:**
  - **Priors are a plain `dict[str, dict]`**, inspired by `pymc_marketing.MMM`'s pattern but without adopting their `Prior` class. Keys are parameter names (`"alpha"`, `"beta"`, `"Gamma"`, `"Sigma"`); values are `{"dist": "<Name>", **kwargs}` distribution specs. JSON-serialisable, easy to document, forward-compatible if we later want a richer `Prior` class. `priors=None` and `priors={}` are both legal — both mean "use defaults", chosen at `fit` time.
  - **`coint_rank` lives in `__init__`**, not `fit()`. Changing `r` is a full PyMC-graph rebuild (α and β are both `K×r`), so "re-fit with a different rank" was never cheap. A rank-selection loop `for r in [...]: BayesianVECM(coint_rank=r).fit(data)` is barely longer and keeps each fitted `idata` available for the eventual rank-uncertainty / model-averaging work.
  - **`endog` will be stored on the fitted object as `self.endog_`** (sklearn-style trailing-underscore convention for "set during fit") *and* inside `idata.constant_data`. Forecasting needs the last `k_ar_diff + 1` rows to seed the recursion — making callers re-pass them is friction and a footgun. Two storage locations serve different needs: live access vs. self-contained serialised record.
- **Cross-branch caveat — merge ordering.** The skeleton validates `deterministic` against `{"n", "co", "ci", "lo", "li"}`. On `main` today `cointegration_design` doesn't accept a `deterministic` argument at all — that support is on `origin/docs/future-directions` waiting for its PR. Functionally the two are independent right now (because `fit` raises `NotImplementedError`, the design helper isn't actually called), but for the cleanest history: **merge `docs/future-directions` first, then rebase this branch onto the new `main` before opening its PR.** *(Resolved — both PRs are on `main` as of 2026-05-19 later.)*

**Update 2026-05-19 (later) — first PyMC model shipped** (branch `feat/first-pymc-model`):

- **`BayesianVECM.fit` is live** for the v0 envelope (`coint_rank=1` + `deterministic="n"`). Runs validate → design → build → sample → store. Sampler defaults: 4 chains × 1000 draws after 1000 tune, `target_accept=0.9`. Outside the v0 envelope the call raises `NotImplementedError` from inside the PyMC graph builder — the public API has accepted the wider configuration since the skeleton, but the graph isn't there yet.
- **β-identification problem solved** the Johansen way: pin `β[:r, :] = I_r` inside the graph. For `r = 1` the first entry of β is a `pt.eye(1)` block stacked on top of a `(K - 1, r)` free RV; the fixed entry isn't a random variable at all, so it can't drift. The full `(K, r)` β matrix is exposed as a `pm.Deterministic` so downstream consumers don't have to remember the normalisation. Closes the loop on the identification narrative kicked off in notebook 02 §5 and notebook 03 §7.
- **`model.idata` (property)** and **`model.summary()`** are live — thin wrappers around `idata_` and `arviz.summary`. Both raise `RuntimeError("BayesianVECM has not been fitted yet")` if called before `fit`. Fit-time state lives on `self` as `endog_`, `idata_`, `variable_names_` (sklearn convention); `endog` is also stashed inside `idata.constant_data` so a serialised file is self-contained.
- **New private module `src/bayesian_vecm/_pymc.py`** owns the graph: `build_pymc_model(design, *, k_ar_diff, coint_rank, deterministic, priors)`. Keeps `_model.py` thin — `fit` is just orchestration. Default priors: `α ~ Normal(0, 1.0)`, `β_free ~ Normal(0, 5.0)`, `Γ ~ Normal(0, 0.5)`, `Σ ~ LKJCholeskyCov(η=2, sd_dist=HalfNormal(1.0))`. User `priors` dict overrides any of the four; Σ has its own narrower override surface (`eta`, `sd_sigma`) because LKJCholeskyCov doesn't fit the `{"dist": ..., **kwargs}` pattern used by the others.
- **Tests:** new `tests/test_pymc.py` with ~22 unit tests (graph construction, scope guards, prior plumbing, β-pin sanity). `tests/test_model.py` grew with end-to-end integration tests on the synthetic cointegrated series — these actually run `pm.sample` (tiny `draws=20, chains=1` to keep wall time manageable). A module-scoped `fitted_model` fixture pays the PyTensor compile cost once and shares the result across the integration tests.
- **Notebook 04** shipped: `notebooks/04_first_pymc_model_walkthrough.ipynb`. Bivariate cointegrated DGP with `β = (1, -0.5)`, `α = (-0.4, 0.2)`. Demonstrates the identification pin (`β[0, 0] == 1.0` bit-exact in every draw), parameter recovery via `model.summary()`, and the "Γ posterior is just noise" property since the DGP has no short-run dynamics.

**Update 2026-05-18:**

- **Future directions parking lot added** (this branch): captures `bvhar` as a reference, the Medium-article brand-marketing use case, and a sequenced list of modelling extensions (sparse priors → stochastic volatility → uncertain cointegration rank). Non-binding planning section — see "Future directions (parking lot)" below.
- **Deterministic-terms follow-up shipped** (this branch; originally planned as `feat/cointegration-design-deterministic`, folded into `docs/future-directions` to avoid a second PR for adjacent work):
  - `cointegration_design` now accepts `deterministic: str = "n"`. Single codes in v0: `"n"`, `"co"`, `"ci"`, `"lo"`, `"li"`. Compound codes (Johansen cases 4 and 5) are rejected with a clear v0.x-follow-up message.
  - Outside terms (`"co"`, `"lo"`) append a column to `delta_x`; inside terms (`"ci"`, `"li"`) append to `y_lag1`. Trend columns are 1-indexed (`[1, 2, …, T_eff]`).
  - `tests/test_design.py` grew from 21 to 47 tests (26 new, parametrised across codes and lag counts). 100% coverage held on `_design.py`.
  - Notebook 02 gained `§6 Deterministic terms` — inside-vs-outside explained economically, demo cell showing each code's effect on the design, and a quick example of the compound-code rejection message. `§6` "What this unlocks" renumbered to `§7`.

**Update 2026-05-20 — dep drift fix, folded into the same branch:**

A `uv sync` between sessions crossed two major version boundaries: PyMC 5 → 6 (removes `pm.ConstantData`) and ArviZ 0.23 → 1.1 (DataTree rewrite — `idata.groups` is now a property, not a method). Notebook 04 broke on both; tests had stayed green because they use `chains=1` and never call `idata.groups()`.

- `_pymc.py`: three `pm.ConstantData(...)` calls → `pm.Data(...)`. Same non-mutable semantics; design matrices still land in `idata.constant_data`.
- `notebooks/04_first_pymc_model_walkthrough.ipynb`: `model.idata_.groups()` → `model.idata_.groups`.
- `pyproject.toml`: floors bumped to the now-known-good versions — `pymc>=6.0`, `arviz>=1.1`. No upper caps (standard practice for libraries heading to PyPI).
- **Min Python bumped 3.11 → 3.12.** ArviZ 1.x requires `>=3.12`; reflected in `requires-python`, classifiers, and the CI matrix (now `["3.12"]` only).

**Update 2026-05-26 — `sample_posterior_predictive` shipped** (branch `feat/posterior-predictive`):

- **`BayesianVECM.sample_posterior_predictive(steps, *, random_seed)` is live.** Rolls the VAR recursion forward `steps` periods for every posterior draw simultaneously. New private module `src/bayesian_vecm/_forecast.py` owns the logic: `forecast_posterior(idata, endog, k_ar_diff, steps, variable_names, rng)`.
- **Implementation:** NumPy-only (no PyTensor). The (chain × draw) dimension is fully vectorised with einsum; only a single Python loop over forecast steps is needed. Innovations drawn from the posterior Cholesky of Σ, pre-computed once. Returns an `xr.DataTree` with a `posterior_predictive` child node holding `y` (levels) and `delta_y` (differences), both shape `(chain, draw, steps, K)`, with `forecast_step` coord (1…steps) and `variable` coord if column names were provided at fit time.
- **ArviZ 1.x note:** `az.InferenceData(group=ds)` constructor kwargs are gone — `az.InferenceData` is now just an alias for `xr.DataTree`. Use `xr.DataTree.from_dict({"group_name": ds})` to construct a new InferenceData-equivalent.
- **Tests:** 9 new tests in `test_model.py` covering pre-fit RuntimeError, bad-steps ValueError, return type, shape, finite values, `forecast_step` coord, seed reproducibility, and variable names in coords. Total test count: 150 (was 141).
- **Notebook 05** added: `notebooks/05_posterior_predictive_walkthrough.ipynb`. Same DGP as nb04, T=100 split 80/20, fan chart with 80%/94% HDI bands and held-out actuals overlaid, band-width table showing uncertainty compounding, coverage check, and error-correction term plot. Includes a `FAST_SAMPLING` flag (draws=200/tune=200/chains=2 when True) so `nbconvert --execute` finishes in under 2 minutes.
- **`matplotlib` added as a dev dependency** (`uv add --dev matplotlib`) — required by notebook 05; not a runtime dep of the package itself.

**Not yet done:**

- **Higher cointegration rank (`r > 1`) and deterministic terms in the PyMC graph** — the next slice (`feat/wider-graph`). See Option 1 below.
- Pandas integration tests (we duck-type via `.to_numpy()`; the new integration tests use a `_FakeDF` test double rather than pandas itself).

## Workflow reminder

Pushes to `main` are rejected. Always work on a feature branch:

```bash
git switch -c feat/<slice-name>
# ...commits...
git push -u origin feat/<slice-name>
# Open PR on GitHub, wait for CI, merge, then locally:
git switch main && git pull && git branch -d feat/<slice-name>
```

**Cowork / Claude session rule:** The very first thing to do after reading this file is to confirm which branch is active (`cat .git/HEAD`) and create a feature branch if on `main` — before touching any source files. Do not write code directly on `main`; the branch guard will reject the push and the working-tree changes will be stranded on the wrong branch.

## Next slice

**Impulse Response Functions (IRFs)**

**Branch:** `feat/irf`

**Goal.** Add `BayesianVECM.irf(steps)` — a posterior distribution over IRF paths. This is one of the primary outputs practitioners use a VECM for: understanding how a shock to one variable propagates through the system over time.

**Why now.** IRFs are a core output method (like `sample_posterior_predictive`), not a modelling extension. Users will want this as soon as the model works. It also makes a natural notebook 07.

**Substantive pieces:**

- **`src/bayesian_vecm/_irf.py`** — `compute_irf(idata, k_ar_diff, steps)` function. Vectorised over posterior draws (chain × draw). The shock is a Cholesky-orthogonalised unit shock — shocks are uncorrelated, and the Cholesky of Σ is already in `idata`. Returns an array of shape `(chain, draw, steps, K, K)` — response of variable `i` to shock `j` at horizon `h`.
- **`BayesianVECM.irf(steps)` method** — thin wrapper, returns an `xr.DataArray` with `horizon`, `response_variable`, and `shock_variable` coordinates.
- **Tests** — shape `(chain, draw, steps, K, K)`, `h=0` response is the identity for orthogonalised shocks, long-run response reflects the cointegrating restriction (EC mechanism pulls the system back), pre-fit `RuntimeError`.
- **Notebook 07** — IRF plots with posterior HDI bands. Same bivariate DGP as notebooks 04–05; show how a shock to `y0` propagates to both variables, and how the error-correction term damps the response back to the cointegrating relation.

**Implementation notes:**

- The VAR companion form is useful for computing IRFs efficiently: stack the VECM into a levels VAR, then iterate the companion matrix. For each posterior draw: reconstruct the companion matrix from `(alpha, beta, Gamma, Sigma)`, then multiply out `h` steps.
- Orthogonalised IRFs require a Cholesky ordering assumption — document this clearly. The default ordering is the variable order in `endog`; a future extension could expose a `cholesky_order` argument.
- Cumulative IRFs (summed over horizons) are also useful for I(1) systems — worth adding as an option.

## Exogenous regressors (`exog`) support

**Motivation.** In brand marketing applications some effects are contemporaneous — brand spend at time $t$ affects the outcome at the same time $t$, not just through lagged dynamics. The standard VECM only has lagged regressors on the right-hand side, so these effects get absorbed into the residuals and are invisible to the model.

**The extension.** Add an optional `exog` matrix of contemporaneous variables $X_t$ (shape `(T, m)`) to the short-run equation:

$$
\Delta y_t = \alpha\beta^\top y_{t-1} + \Gamma\,\Delta x_t + \mathbf{B}\,X_t + \varepsilon_t
$$

$\mathbf{B}$ is $(K, m)$ — one column per exogenous variable, one row per endogenous variable.

**API design.** Match statsmodels — pass `exog` at `fit` time (not `__init__`), since it's data not a model structural choice. Also expose `exog_coint` for the case where the exogenous variable belongs *inside* the cointegrating relation (e.g. a slowly-moving brand equity index that determines the long-run equilibrium):

```python
model.fit(endog, exog=brand_spend_df)           # contemporaneous effect in short-run eq
model.fit(endog, exog_coint=brand_equity_df)    # effect inside cointegrating relation
```

**Implementation.** `cointegration_design` gains optional `exog` and `exog_coint` arguments; validated and aligned to `T_eff` rows. The graph adds a `B` RV (shape `(K, m)`, prior `Normal(0, 1.0)`) and appends `B @ X_t.T` to the mean. `exog_coint` appends columns to `y_lag1` (same mechanism as inside deterministic terms). `sample_posterior_predictive` and `irf` both need updating to accept future `exog` paths.

**Sequencing.** After IRF and output methods — the baseline needs to be solid before adding exogenous regressors, and IRF with exog requires passing future exog paths to the forecast recursion.

## Post-IRF output methods (statsmodels parity)

Small slice after `feat/irf` merges. These are all post-fit properties/methods that statsmodels exposes and that are low-effort additions:

- **`fittedvalues` / `resid`** — in-sample fitted values and residuals. Properties that compute `alpha @ beta.T @ y_lag1 + Gamma @ delta_x` over posterior draws (or just the posterior mean for a point summary). Useful for diagnostic plots.
- **`var_rep`** — levels VAR(p) coefficient matrices reconstructed from VECM parameters. Will be computed internally for IRF anyway; worth exposing so users can inspect the companion form directly.
- **Diagnostic tests** — `test_normality` (Jarque-Bera on residuals) and `test_whiteness` (Portmanteau autocorrelation test). Classical post-estimation checks; Bayesian posterior predictive checks are richer but these are familiar to practitioners.
- **Granger causality** — skip. In a Bayesian model you just inspect the posterior of the relevant Γ coefficients; a separate test method adds little.

## Future directions (parking lot)

Forward-looking items raised during planning on 2026-05-18. Not committed to and not on the critical path — captured here so they don't get lost. Tackle step by step, after the baseline estimation slice (option 2 in the next-slice list — first PyMC model) lands.

### Known issues

- **macOS + Jupyter `pm.sample` parallel-mode `EOFError`.** First multi-chain fit from a Jupyter cell on macOS died with a bare `EOFError` from `ProcessAdapter.recv_draw` — worker process died during `"spawn"` startup, parent saw closed pipe with no traceback. `cores=1` works fine. Possible causes: BLAS/OpenMP fork-safety, PyTensor compile in worker, Jupyter `__main__` weirdness. Worth investigating because users will hit this; workaround for now is `cores=1`. Could be as simple as setting `mp_ctx="forkserver"` as a default on macOS inside `BayesianVECM.fit`.

### References to mine later

- **`bvhar`** — Python package for Bayesian VAR / VHAR with shrinkage priors. Doesn't do VECM/cointegration, but a useful reference for Bayesian time-series patterns in PyMC-adjacent territory: prior specification, hyperparameter handling, posterior summaries, and what a "good" Bayesian time-series API looks like in 2026.
- **VECM in brand marketing** — Ryan's Medium article: <https://medium.com/@raz1470/capturing-the-long-term-causal-effect-of-brand-marketing-bc577621a627>. The motivating use case for the whole package: brand investment has long-term effects that plain regression / MMM smears over short windows; VECM captures the cointegrating relationship between brand spend and the outcome variable. Worth linking from the README once the package is usable.

### Brand marketing applied notebook

After `exog` support ships, add a notebook that tells the full applied story — separate from the methodology walkthroughs, aimed at a practitioner who already knows marketing but is new to VECMs.

**Planned as notebook 08** (after notebook 07 — IRFs). Outline:

- **The problem.** Brand marketing has long-term effects that short-window regression and standard MMM miss. Show a synthetic DGP where brand spend and revenue are cointegrated — brand spend drifts up over years, revenue follows. A plain OLS or VAR-in-differences on this data gives wrong elasticities.
- **Why VECM.** The cointegrating relation *is* the long-run brand equity equation. The error-correction term tells you how fast the system corrects when brand spend and revenue fall out of equilibrium — i.e. the speed at which brand investment translates to revenue.
- **Contemporaneous effects via `exog`.** Some brand effects are immediate (a TV burst drives same-week sales). Show how `exog` captures this on top of the long-run cointegrating relationship.
- **IRF as the key output.** The IRF tells the story practitioners need: "if we increase brand spend by 1 unit today, what happens to revenue over the next 52 weeks?" With posterior HDI bands. Compare to what a naive regression would say.
- **Link to the Medium article** — frame this as the Bayesian version of the methodology described there.

This notebook is the "why does this package exist" moment — the one to share when pitching the package to practitioners.

### Modelling extensions

In rough order of when to attempt them, once the baseline estimator lands.

- **Sparse priors (horseshoe).** With `K` variables and `k` lags, the `Γ` block alone has `K²·k` parameters; `α` and `β` scale with `K` and `r`. Most entries are likely near zero in practice. A horseshoe prior (Carvalho, Polson & Scott 2010) or regularised horseshoe (Piironen & Vehtari 2017) on the `Γ` matrices — and possibly on `α` — would shrink the irrelevant ones toward zero while keeping real signals. More adaptive than the classical Minnesota prior, and doesn't require hand-tuning a shrinkage hyperparameter. **When:** after the fixed-rank constant-`Σ` model samples cleanly — otherwise you can't tell whether sampling pathologies come from the prior or the parameterisation.
- **Stochastic volatility.** Replace constant `Σ` with time-varying covariance. Standard recipes: Cogley-Sargent / Primiceri (2005) Cholesky-SV, factor SV, or univariate SV on each residual. Largely orthogonal to `α` / `β` / `Γ` estimation — can be layered on as an additional block. **When:** after horseshoe. Becomes important if this is ever pointed at finance data, where heteroskedasticity is the rule.
- **Uncertain cointegration rank.** Current plan fixes `r` at the class level. Inferring `r` jointly is meaningfully harder. Two viable routes: (i) fit at each plausible `r` and Bayesian-model-average via marginal likelihoods; (ii) put a shrinkage prior on the singular values of `αβ′` so `r` emerges from the posterior — see Strachan & Inder (2004), Villani (2005, 2006). **When:** last. Research-grade; defer until everything else is solid so there's a known-good fixed-`r` estimator to validate against.

### Sequencing thought

Full v0 envelope (`feat/wider-graph`) first — `r > 1` and all deterministic codes. Then layer extensions: horseshoe → stochastic volatility → rank uncertainty. Each extension should ship behind a flag or as an optional argument rather than replacing the baseline, so the baseline stays available as both a teaching example and a sampling-diagnostic reference.

## Session learnings (2026-05-29)

- **`uv run pytest` hangs on macOS in this environment.** Use `.venv/bin/pytest` directly. `uv run` appears to deadlock during pytest startup — possibly a conflict between uv's process management and PyTensor/PyMC imports. Workaround: activate the venv (`source .venv/bin/activate`) and call `pytest` directly, or use `.venv/bin/pytest` without activating.
- **`git add` doesn't persist across shell sessions.** When working with Cowork, each bash call is independent — staging done in one call is lost before the next. Always do `git add ... && git commit ...` in a single command.
- **nbstripout git clean filter crashes when multiple notebooks are staged simultaneously.** `rfc3987_syntax 1.1.0` rebuilds Lark parsers from scratch on every Python process startup. With 5 notebooks modified, git spawns 5 filter processes in parallel and SIGINT is sent mid-init. Workaround: temporarily swap the filter to `cat` for the commit (`git config --local filter.nbstripout.clean "cat"`, commit, restore). Long-term fix: move nbstripout to a pre-commit hook (sequential) rather than a git clean filter (parallel).
- **`rm -rf .venv && uv sync --all-extras` fixes a broken venv** but requires re-running `uv run python -m ipykernel install --sys-prefix` afterwards to restore the Jupyter kernelspec.

## Session learnings (2026-05-28)

Lessons from the housekeeping sprint:

- **`nbstripout --install` uses `.git/info/attributes` by default (local only).** To share the filter with other contributors and CI, run `nbstripout --install --attributes .gitattributes` instead — this writes a `.gitattributes` file that can be committed to the repo.
- **`nbconvert --inplace` doesn't prevent the glob from picking up `.nbconvert.ipynb` sidecar files** left over from a previous failed run. Clean up `notebooks/*.nbconvert*.ipynb` before re-running if a previous attempt errored partway through.
- **New code cells added by script must include `"execution_count": null`** — nbconvert validates against the nbformat schema and will reject a code cell that's missing this field, even if `outputs` is an empty list.
- **Notebook CI catches API drift that unit tests miss.** Notebook 03's "honest stubs" section was calling `m.fit(endog=None)` and expecting `NotImplementedError` — but `fit` now validates input first, so it raised `ValueError` instead. Tests were green; notebook CI would have caught this on the first run. Update notebook sections that document "not yet implemented" behaviour whenever the implementation catches up.

## Session learnings (2026-05-15)

Lessons worth not re-learning:

- **Pre-push checklist.** Both ruff failures in this session would have been caught by `uv run ruff format . && uv run ruff check .` before push. Adding this to a pre-commit hook is now an action item. Until then, run it manually before every `git push`.
- **ruff format checks notebooks too.** Aligned-dict whitespace, `t ** 2` vs `t**2`, blank lines between class methods — all the rules ruff applies to `.py` files apply inside `.ipynb` cells too.
- **`uv add --dev ipykernel` is not enough to make notebooks runnable via `uv run jupyter nbconvert`.** Jupyter looks up the `python3` kernelspec in user/system dirs and falls back to whatever non-venv Python is registered globally. After `uv add --dev ipykernel jupyter`, run `uv run python -m ipykernel install --sys-prefix` once to write the kernelspec into `.venv/share/jupyter/kernels/python3/`. Then `uv run jupyter` picks it up automatically.
- **Naming conventions.** Ruff (with `select = [..., "N", ...]`) enforces lowercase function and variable names. `T`, `K`, `T_eff` in tests will fail N806/N802 — use `n_obs`, `n_vars`, `n_eff` even though the maths uses uppercase. Module docstrings can still say `(T, K)` since that's prose.
- **Greek letters in docstrings.** Literal `α`, `β`, `ε` trigger RUF002 (ambiguous with Latin lookalikes). Either use LaTeX commands (`\\alpha`, `\\beta`) inside `:math:` directives — which is what `_data.py` does — or avoid them in inline prose. `Δ` (uppercase delta) is fine; it has no Latin lookalike.

## Session learnings (2026-05-19)

Lessons from the skeleton-shipping session:

- **Run git operations from the local terminal, not from inside Cowork.** The Cowork shell sandbox can read, write, and *rename* files in the worktree (including `.git/`), but it cannot **unlink** them — even ones it just created. That breaks every destructive git operation: `git switch` (can't replace worktree files), `git branch -d` (can't remove the ref file), `git restore .`, lock cleanup. File reads/writes/edits via Cowork are fine for *code* changes; for branch management, commits, and any `rm`-flavoured cleanup, do it from `~/Documents/repos/claude/bayesian_vecm` in a normal terminal. Workaround if you ever get a stale `.git/index.lock` you can't delete: `mv .git/index.lock .git/index.lock.OLD` works where `rm` doesn't, and gets git unblocked.
- **macOS zsh doesn't treat `#` as a comment in interactive mode** unless you've opted in. If you paste a block that mixes commands and `# comments`, an apostrophe later in a comment (e.g. "they're") opens a string that never closes, dropping you into `quote>`. Either strip comments from pasted blocks, or add `setopt interactivecomments` to `~/.zshrc` once. Ctrl+C escapes the `quote>` prompt; no harm done if nothing has run yet.
- **iCloud Drive silently corrupts the venv.** The repo currently lives at `~/Documents/repos/claude/bayesian_vecm`, and "Documents in iCloud" is enabled, so iCloud sync touches `.venv/`. Symptom: tests fail to collect with `ModuleNotFoundError: No module named 'bayesian_vecm'` even though `uv pip list` shows the package as installed. Diagnosis: iCloud was duplicating files into `site-packages/` with " 2", " 3", " 4" name suffixes whenever it detected a sync conflict, and the editable-install `.pth` file (`_editable_impl_bayesian_vecm.pth`, which should point at `src/`) was getting clobbered — missing trailing newline, multiple conflicting copies. Quick mitigation applied this session: `xattr -w com.apple.fileprovider.ignore#P 1 ~/Documents/repos/claude/bayesian_vecm/.venv` to stop iCloud touching the venv (undocumented but effective). **Real fix:** move the repo out of `~/Documents/` entirely — e.g. `~/code/bayesian_vecm` or `~/Developer/bayesian_vecm`. Until that happens, *any* time tests start failing with import errors and `uv pip list` says the package is installed, suspect iCloud first: `rm -rf .venv && uv sync --all-extras` is the recovery command.

## Session learnings (2026-05-20)

Lessons from the dep-drift firefight:

- **Open lower bounds + `uv sync` = silent major-version drift.** `pymc>=5.28.5` and `arviz>=0.23.4` happily resolved to PyMC 6 and ArviZ 1.x once those landed on PyPI. The lockfile recorded the change but no human review caught it. Pin floors at the known-good *current* version after every dep work session — the floor is documentation of "I tested against this", not just a minimum. No upper caps on a library going to PyPI (causes downstream resolution headaches); instead lean on notebook-CI to catch the next major bump fast.
- **Tests-green isn't notebooks-green.** The integration tests in `test_model.py` use `chains=1` (single-process) and don't probe `idata.groups()`, so they sailed past both today's bugs. Argues for executing notebooks in CI sooner rather than later — see the "Not yet done" item, which just earned a sharp justification.
- **macOS + Jupyter + `pm.sample` parallel mode** can die with a bare `EOFError` from the multiprocessing pipe — a worker dies during `"spawn"` startup and the parent just sees a closed pipe with no traceback. Diagnostic: re-run with `cores=1`. If that succeeds, the model is fine; if it fails, you get the real error. Didn't root-cause today (the synthetic-data fit takes 4 seconds with `cores=1`, so it's not pressing) — see new parking-lot item under "Future directions".

## Session learnings (2026-05-26)

Lessons from the `sample_posterior_predictive` + notebook 05 session:

- **ArviZ 1.x dropped the `InferenceData(**group_kwargs)` constructor.** `az.InferenceData` is now a deprecated alias for `xr.DataTree`. Constructing a new InferenceData-equivalent with a named group requires `xr.DataTree.from_dict({"group_name": ds})` — the old `az.InferenceData(posterior_predictive=ds)` pattern raises `TypeError: DataTree.__init__() got an unexpected keyword argument`.
- **`replace_all=True` on a quoted string also replaces string literals.** When using a bulk find-and-replace to remove quotes from type annotations (UP037), double-check that the target string doesn't also appear as a value in `hasattr(obj, "ClassName")` or `"ClassName" in __all__` — those need to stay quoted. The safe approach is to fix UP037 violations one at a time or run `uv run ruff check --fix .` to let ruff do it.
- **`matplotlib` is not a transitive dependency of PyMC/ArviZ in the venv.** Even though PyMC's full install pulls it in on many systems, `uv sync` only installs what's explicitly declared. Any notebook that uses `matplotlib` needs `uv add --dev matplotlib` — otherwise nbconvert fails immediately with `ModuleNotFoundError`.
- **`nbconvert --execute` kernel startup takes 30–60 seconds.** The asyncio selector is waiting for the kernel process to finish importing PyMC/ArviZ before it responds. Do not Ctrl+C during this phase — it looks stuck but isn't. If execution genuinely hangs beyond ~2 minutes on a tiny notebook, check that the kernelspec is registered: `uv run python -m ipykernel install --sys-prefix`.
- **Add a `FAST_SAMPLING` flag to every notebook that calls `pm.sample`.** Default to `True` (small draws/tune) so nbconvert and CI finish in a reasonable time; set to `False` for publication-quality runs. Document the flag at the top of the sampling config cell.

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

**Pre-push checklist** — run this before every `git push` until a pre-commit hook lands. Catches both CI failure modes we hit on 2026-05-15:

```bash
uv run ruff format .          # auto-fix formatting (including notebooks)
uv run ruff check .           # lint
uv run pytest                 # tests
# optional: execute notebooks if you've edited them
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## Domain-learning track

Ryan is **learning VECMs as we build**, so explanations of the econometrics (cointegration, error-correction term, identification, lag selection, etc.) should accompany the code as it's written.

**Convention.** One numbered Jupyter notebook per public-API slice, living in `notebooks/`:

- Filename pattern: `NN_<topic>.ipynb` (e.g. `01_data_utilities_walkthrough.ipynb`).
- Each notebook explains *what* each helper does, *why* a VECM needs it, and demos it on small synthetic data — written for a reader meeting VECMs for the first time.
- Trigger for a new notebook: "did this slice ship something a learner needs to understand?" Internal refactors don't need one.
- Notebooks are runnable docs *and* lightweight integration tests — when CI execution lands (see TODO in the status section), a broken explanation becomes a failing build.
- Once the catalogue grows, consider graduating to a docs site (Sphinx + nbsphinx, or MkDocs + mkdocs-jupyter). Defer until the `BayesianVECM` skeleton is in.
