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
- **Cross-branch caveat — merge ordering.** The skeleton validates `deterministic` against `{"n", "co", "ci", "lo", "li"}`. On `main` today `cointegration_design` doesn't accept a `deterministic` argument at all — that support is on `origin/docs/future-directions` waiting for its PR. Functionally the two are independent right now (because `fit` raises `NotImplementedError`, the design helper isn't actually called), but for the cleanest history: **merge `docs/future-directions` first, then rebase this branch onto the new `main` before opening its PR.**

**Not yet done:**

- **Actual VECM estimation.** The class skeleton is in but every estimation method (`fit`, `idata`, `summary`, `sample_posterior_predictive`) raises `NotImplementedError`. The PyMC graph is the next slice — see "Next slice" below.
- **`_VALID_DETERMINISTIC` will be duplicated** in `_model.py` (now) and `_design.py` (once `docs/future-directions` merges). Consolidate into a shared private constant — likely a new `src/bayesian_vecm/_constants.py` that both import from. Small cleanup, not blocking.
- Pandas integration tests (we duck-type via `.to_numpy()`; haven't tested against a real pandas DataFrame).
- **CI execution of notebooks.** Notebooks aren't run in CI yet, so the docs/learning track is at risk of silent drift. Add a job that runs `uv run jupyter nbconvert --to notebook --execute notebooks/*.ipynb` and fails on errors. Also decide on outputs strategy — leaning toward `nbstripout` as a git filter so committed `.ipynb` files have outputs cleared and diffs stay small, but worth revisiting once we have 3+ notebooks.
- **Pre-commit hook.** A local hook running `ruff format` and `ruff check` would have caught two CI bounces in the 2026-05-15 session. Smaller than the notebook-CI work; could land first.

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

The two earlier candidates have both shipped: deterministic-terms is on `origin/docs/future-directions` awaiting its PR, and the `BayesianVECM` skeleton landed on this branch. **Recommended starting point: option 1** — the skeleton just locked the API, so this is where the real econometrics finally begins. Option 2 is a small cleanup tax that should land alongside or shortly after.

### Option 1 (recommended) — First PyMC model

**Branch:** `feat/first-pymc-model`

**Goal.** Simplest possible VECM that actually samples: known cointegration rank `r = 1`, `deterministic = "n"`, weakly-informative priors on `α`, `β`, `Γ_1, ..., Γ_k`, and `Σ`. Targets a small synthetic cointegrated dataset (the one in notebook 02 is a good starting point).

**Where it plugs in.** `BayesianVECM.fit(endog)` should:

1. Call `validate_endog(endog)` (already exists in `_data.py`).
2. Call `cointegration_design(endog, k_ar_diff=self.k_ar_diff, deterministic=self.deterministic)` to get the three aligned matrices.
3. Build a PyMC model that ties those matrices to `α`, `β`, `Γ_i`, `Σ`.
4. Run `pm.sample(...)` and store the result as `self.idata_`.
5. Store the raw `endog` as `self.endog_` and inside `idata_.constant_data`.

**The piece to think hardest about is the identification of β.** $\alpha \beta'$ is invariant under $(\alpha, \beta) \to (\alpha R^{-1}, \beta R^{\top})$ for any invertible $R$, so without a normalisation (the standard choice is `β[:r, :] = I_r`) the posterior over `β` alone is non-identified and sampling will be a horror show. Worth reading the relevant section of *Johansen 1995* before writing PyMC code; notebook 02 §5 has a starter explanation.

**Dependencies to add when this slice starts:** `pymc`, `arviz`. Both go in the main runtime deps in `pyproject.toml` (`uv add pymc arviz`). PyMC pulls in pytensor + numpyro/jax transitively; expect the lockfile to grow.

**Walkthrough notebook 03:** the first PyMC model is a perfect candidate for a notebook — fit on the notebook-02 synthetic data, plot the posteriors of `α` and `β`, eyeball whether they recover the true cointegrating vector. Frame the β-identification choice as a teaching moment.

### Option 2 — Consolidate `_VALID_DETERMINISTIC`

**Branch:** `chore/consolidate-deterministic-codes`

**Goal.** Remove the duplication that opens up once `docs/future-directions` and the skeleton are both on `main`. Introduce a private `src/bayesian_vecm/_constants.py` (or similar) holding `VALID_DETERMINISTIC = frozenset({"n", "co", "ci", "lo", "li"})`. Both `_design.py` and `_model.py` import from it.

Trivial in scope; the real value is doing it *before* either set drifts.

**Order of operations.** This slice can only land after both prerequisites are on `main`: the `feat/bayesian-vecm-skeleton` PR and the `docs/future-directions` PR. Both can go in either order; this cleanup chases them.

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
