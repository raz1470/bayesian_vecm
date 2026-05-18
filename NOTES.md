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

**Update 2026-05-15 (earlier):**

- **Docs/learning track kicked off.** Added `notebooks/01_data_utilities_walkthrough.ipynb` — a beginner-friendly walkthrough of `validate_endog`, `difference`, and `lag_matrix` with synthetic-data demos and a primer on where each helper fits into the VECM equation. Convention: one numbered notebook per public-API slice. Shipped via PR `feat/notebook-data-utilities-walkthrough`.

**Update 2026-05-15 (later):**

- **Cointegration design slice shipped** (PR `feat/cointegration-design`):
  - `src/bayesian_vecm/_design.py` with `CointegrationDesign` NamedTuple and `cointegration_design(data, k_ar_diff)` function. Calls `validate_endog` internally, then `difference` once, then slices and `lag_matrix` to produce three matrices aligned to `T_eff = T - k_ar_diff - 1` rows.
  - `tests/test_design.py` with 21 unit tests covering the hand-built row-by-row spec, shape contract, `k_ar_diff = 0` edge case, validation, DataFrame-like input, alignment across all three outputs, and lag-major column ordering.
  - Kept private (no `__init__.py` re-export), mirroring `_data.py`.
- **Walkthrough notebook 02** added: `notebooks/02_cointegration_design_walkthrough.ipynb`. Frames the alignment problem, derives `T_eff = T - k - 1`, demos the hand-built tiny example, the `k = 0` and `k = 2` cases, and an end-to-end synthetic cointegrated example. First mention of the β-identification problem in the docs — flagged as the natural place to expand once model code lands.

**Not yet done:**

- Any actual VECM model code (preprocessing + design matrices only; no estimation yet).
- Pandas integration tests (we duck-type via `.to_numpy()`; haven't tested against a real pandas DataFrame).
- **CI execution of notebooks.** Notebooks aren't run in CI yet, so the docs/learning track is at risk of silent drift. Add a job that runs `uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb` and fails on errors. Also decide on outputs strategy — leaning toward `nbstripout` as a git filter so committed `.ipynb` files have outputs cleared and diffs stay small, but worth revisiting once we have 3+ notebooks.
- **Pre-commit hook.** A local hook running `ruff format` and `ruff check` would have caught two CI bounces this session. Smaller than the notebook-CI work; could land first.

## Workflow reminder

Pushes to `main` are rejected. Always work on a feature branch:

```bash
git switch -c feat/<slice-name>
# ...commits...
git push -u origin feat/<slice-name>
# Open PR on GitHub, wait for CI, merge, then locally:
git switch main && git pull && git branch -d feat/<slice-name>
```

## Next slice — pick from these

Three real candidates, in roughly increasing order of scope and risk. **Recommended starting point: option 1** — it cleanly extends the helper we just shipped, the API surface is already partly designed (we know what `deterministic="n"` means because it's the implicit current behaviour), and it knocks off a TODO before we start touching PyMC.

### Option 1 (recommended) — Deterministic-terms follow-up

**Branch:** `feat/cointegration-design-deterministic`

**Goal.** Add a `deterministic` parameter to `cointegration_design` so it can produce design matrices for the full statsmodels family of cases:

| Code | Meaning | Where the term sits |
| --- | --- | --- |
| `"n"` | No deterministic terms (current behaviour) | — |
| `"co"` | Constant only, outside the cointegration relation | Added as a column to `delta_x` |
| `"ci"` | Constant restricted to the cointegration relation | Added as a column to `y_lag1` |
| `"lo"` | Linear trend, outside the cointegration relation | Added as a column to `delta_x` |
| `"li"` | Linear trend restricted to the cointegration relation | Added as a column to `y_lag1` |

Statsmodels supports combinations (e.g. `"colo"`) — decide whether to support those or keep it to single-character codes for v0.

**Signature change:**

```python
def cointegration_design(
    data,
    k_ar_diff: int,
    deterministic: str = "n",
) -> CointegrationDesign: ...
```

**Tests to write first (TDD):**

- Each code produces the right additional column(s) and they end up in the right matrix.
- Shapes update: `delta_x.shape[1]` and `y_lag1.shape[1]` reflect the added columns.
- Unknown code (e.g. `"xyz"`) raises a clear `ValueError`.
- `deterministic="n"` (default) still produces the exact output of the current implementation. Use the existing hand-built example here to make sure nothing regressed.

**Domain learning to capture in a notebook:** *what* each deterministic case means economically. "Constant in cointegration relation" vs "constant outside" sounds like jargon until you realise it's the difference between "the equilibrium has a non-zero level" and "each variable has a drift on top of the equilibrium". Worth a short notebook 03 — possibly as a section appended to notebook 02 rather than a fresh notebook, depending on how much there is to say.

### Option 2 — `BayesianVECM` class skeleton

**Branch:** `feat/bayesian-vecm-skeleton`

**Goal.** Lay out the public-facing class with **no estimation yet**. All methods raise `NotImplementedError` (which is already excluded from coverage in `pyproject.toml`). The point is to force the API decisions before any PyMC code.

```python
class BayesianVECM:
    def __init__(
        self,
        k_ar_diff: int = 1,
        coint_rank: int = 1,
        deterministic: str = "n",
        priors: dict | None = None,
    ) -> None: ...

    def fit(self, endog) -> "BayesianVECM": ...      # raise NotImplementedError
    @property
    def idata(self): ...                              # raise NotImplementedError
    def summary(self): ...                            # raise NotImplementedError
    def sample_posterior_predictive(self, steps: int): ...   # raise NotImplementedError
```

**Open decisions to make in this slice:**

- How are priors specified? `pymc_marketing.MMM` uses a `dict` of distribution names plus parameters; copying that pattern is the path of least resistance.
- Does `coint_rank` go in `__init__` or in `fit()`? Statsmodels passes it to the class; pymc_marketing-style would put it in `fit()` so the same class can be re-fit with different ranks. Worth picking deliberately.
- Does the class store the input `endog` after `fit()`, or only `idata`?

This slice depends on option 1 only if you want `deterministic` in the constructor signature to be meaningful — and even then, the skeleton can accept the parameter and just pass it through to a not-yet-existent design step.

### Option 3 — First PyMC model

**Branch:** `feat/first-pymc-model`

**Goal.** Simplest possible VECM: known cointegration rank `r = 1`, `deterministic = "n"`, weakly-informative priors on `α`, `β`, `Γ_1, ..., Γ_k`, and `Σ`. Targets a small synthetic cointegrated dataset (the one in notebook 02 is a good starting point).

**This is where the real econometrics begins.** The piece to think hardest about is the **identification of β**: $\alpha \beta'$ is invariant under $(\alpha, \beta) \to (\alpha R^{-1}, \beta R^{\top})$ for any invertible $R$, so without a normalisation (the standard choice is `β[:r, :] = I_r`) the posterior over `β` alone is non-identified and sampling will struggle. Worth reading the relevant section of *Johansen 1995* before writing any PyMC code.

Should NOT be tackled until option 2 is in — the class skeleton needs to exist before there's anywhere to wire the PyMC graph into.

## Future directions (parking lot)

Forward-looking items raised during planning on 2026-05-18. Not committed to and not on the critical path — captured here so they don't get lost. Tackle step by step, after the baseline estimation slice (option 3) lands.

### References to mine later

- **`bvhar`** — Python package for Bayesian VAR / VHAR with shrinkage priors. Doesn't do VECM/cointegration, but a useful reference for Bayesian time-series patterns in PyMC-adjacent territory: prior specification, hyperparameter handling, posterior summaries, and what a "good" Bayesian time-series API looks like in 2026.
- **VECM in brand marketing** — Ryan's Medium article: <https://medium.com/@raz1470/capturing-the-long-term-causal-effect-of-brand-marketing-bc577621a627>. The motivating use case for the whole package: brand investment has long-term effects that plain regression / MMM smears over short windows; VECM captures the cointegrating relationship between brand spend and the outcome variable. Worth linking from the README once the package is usable, and worth distilling into an "applied example" notebook later — separate from the methodology walkthroughs in `notebooks/`.

### Modelling extensions

In rough order of when to attempt them, once the baseline estimator lands.

- **Sparse priors (horseshoe).** With `K` variables and `k` lags, the `Γ` block alone has `K²·k` parameters; `α` and `β` scale with `K` and `r`. Most entries are likely near zero in practice. A horseshoe prior (Carvalho, Polson & Scott 2010) or regularised horseshoe (Piironen & Vehtari 2017) on the `Γ` matrices — and possibly on `α` — would shrink the irrelevant ones toward zero while keeping real signals. More adaptive than the classical Minnesota prior, and doesn't require hand-tuning a shrinkage hyperparameter. **When:** after the fixed-rank constant-`Σ` model samples cleanly — otherwise you can't tell whether sampling pathologies come from the prior or the parameterisation.
- **Stochastic volatility.** Replace constant `Σ` with time-varying covariance. Standard recipes: Cogley-Sargent / Primiceri (2005) Cholesky-SV, factor SV, or univariate SV on each residual. Largely orthogonal to `α` / `β` / `Γ` estimation — can be layered on as an additional block. **When:** after horseshoe. Becomes important if this is ever pointed at finance data, where heteroskedasticity is the rule.
- **Uncertain cointegration rank.** Current plan fixes `r` at the class level. Inferring `r` jointly is meaningfully harder. Two viable routes: (i) fit at each plausible `r` and Bayesian-model-average via marginal likelihoods; (ii) put a shrinkage prior on the singular values of `αβ′` so `r` emerges from the posterior — see Strachan & Inder (2004), Villani (2005, 2006). **When:** last. Research-grade; defer until everything else is solid so there's a known-good fixed-`r` estimator to validate against.

### Sequencing thought

Fixed-`r`, constant-`Σ`, weakly-informative-prior VECM first (option 3 in the next-slice list). Then layer extensions: horseshoe → stochastic volatility → rank uncertainty. Each extension should ship behind a flag or as an optional argument rather than replacing the baseline, so the baseline stays available as both a teaching example and a sampling-diagnostic reference.

## Session learnings (2026-05-15)

Lessons worth not re-learning:

- **Pre-push checklist.** Both ruff failures in this session would have been caught by `uv run ruff format . && uv run ruff check .` before push. Adding this to a pre-commit hook is now an action item. Until then, run it manually before every `git push`.
- **ruff format checks notebooks too.** Aligned-dict whitespace, `t ** 2` vs `t**2`, blank lines between class methods — all the rules ruff applies to `.py` files apply inside `.ipynb` cells too.
- **`uv add --dev ipykernel` is not enough to make notebooks runnable via `uv run jupyter nbconvert`.** Jupyter looks up the `python3` kernelspec in user/system dirs and falls back to whatever non-venv Python is registered globally. After `uv add --dev ipykernel jupyter`, run `uv run python -m ipykernel install --sys-prefix` once to write the kernelspec into `.venv/share/jupyter/kernels/python3/`. Then `uv run jupyter` picks it up automatically.
- **Naming conventions.** Ruff (with `select = [..., "N", ...]`) enforces lowercase function and variable names. `T`, `K`, `T_eff` in tests will fail N806/N802 — use `n_obs`, `n_vars`, `n_eff` even though the maths uses uppercase. Module docstrings can still say `(T, K)` since that's prose.
- **Greek letters in docstrings.** Literal `α`, `β`, `ε` trigger RUF002 (ambiguous with Latin lookalikes). Either use LaTeX commands (`\\alpha`, `\\beta`) inside `:math:` directives — which is what `_data.py` does — or avoid them in inline prose. `Δ` (uppercase delta) is fine; it has no Latin lookalike.

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
