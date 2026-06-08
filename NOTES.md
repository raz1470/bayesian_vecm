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

## Practitioner guide notebook plan

**File:** `notebooks/guides/01_brand_vecm_walkthrough.ipynb`
**Audience:** Marketing analysts / brand managers with no VECM background.
**Purpose:** End-to-end walkthrough from raw data to business outputs. One swap cell at the top for real data; everything else flows from it.

### Reference paper
Cain (2022) "Modelling short and long-term marketing effects in the consumer purchase journey", IJRM 39, 96–116.
Key alignment: endogenous system = base sales + awareness + consideration + ops metric; paid media as exogenous; Johansen cointegration; GIRF for IRFs.

### Variable structure

**Endogenous (potentially cointegrated — all log-transformed):**
- `organic_sales` — MMM baseline output (spend-decomposed), or total organic revenue
- `brand_awareness` — unaided brand awareness survey score
- `brand_consideration` — brand consideration survey score
- `csat_score` — average CSAT score (ops metric; included if I(1), else moves to exog)

**Exogenous:**
- `brand_spend` — total brand investment (TV, digital brand, OOH etc.)
- `interest_rate` — Bank of England base rate or equivalent
- `consumer_confidence` — GfK or equivalent consumer confidence index

**Model config:**
```python
BayesianVECM(
    k_ar_diff=2,                              # generous; horseshoe shrinks spurious lags
    coint_rank=...,                           # from select_coint_rank
    deterministic="ci",                       # trend inside cointegration space
    priors={"Gamma": {"dist": "Horseshoe"}},  # shrink over-specified lags
)
model.fit(endog, exog=exog, cores=1)
model.irf(steps=52, method="girf")           # GIRF — order-invariant, standard in literature
```

**Why GIRF not Cholesky:** The awareness → consideration → csat → organic_sales causal chain is theoretically sound, but these variables have feedback loops (CSAT drives word-of-mouth which drives awareness). GIRF handles this without imposing a strict ordering. Cholesky is noted in §10 as an alternative for users who want to impose the recursive structure.

**Why log-transform:** Gives elasticity interpretation on parameters and natural diminishing returns on the brand response curve. All endogenous and continuous exogenous variables should be logged before fitting.

**Why `deterministic="ci"`:** Brand and ops metrics typically have long-run trends (Monzo growing, CSAT improving). Restricting the trend to the cointegration space is theoretically correct and matches Cain's treatment.

### Notebook sections

**§1 — The problem** (~3 cells)
- Plain-English motivation: why does brand investment have effects that last years, and why does standard MMM miss them?
- One synthetic chart: flat MMM response vs persistent VECM response to a brand spend shock
- No code, just markdown + one illustrative plot

**§2 — Your data** (~2 cells)
- THE SWAP CELL: load a DataFrame with `date` index and the 7 columns above
- Synthetic Monzo-style DGP provided as the default so notebook runs out of the box
- DGP: common I(1) trends for organic_sales, awareness, consideration, csat; brand_spend as exog driver; interest_rate and consumer_confidence as macro exog
- Plot all variables

**§3 — Do your variables have long-run trends?** (~4 cells)
- Plain-English framing: "Is this variable on a persistent upward/downward journey, or does it bounce around a stable level?"
- Intuition: a variable with a unit root has no 'home' it returns to; a stationary variable always mean-reverts
- ADF test for each variable with plain-English verdict: "organic_sales: persistent trend ✓" / "interest_rate: mean-reverting — will treat as exogenous"
- Decision rule table: I(1) variables → endogenous VECM; I(0) variables → exog
- Note on CSAT: if it tests as I(0), move to exog and re-run

**§4 — How many long-run relationships?** (~3 cells)
- Plain-English framing: "How many equilibrium forces are holding your system together?"
- Intuition: one cointegrating relationship means there's one long-run equilibrium (e.g. sales and brand awareness move together in the long run). Two means two separate equilibria.
- `select_coint_rank` output with annotation of each row
- Decision: use recommended rank, or override with domain knowledge

**§5 — Fit the model** (~3 cells)
- `BayesianVECM` construction with horseshoe prior
- `model.fit()` — show sampling progress
- `model.summary(var_names=["alpha", "beta"])` with plain-English annotation of each parameter

**§6 — What are α and β telling you?** (~4 cells)
- β (cointegrating vector): "This is the long-run equilibrium equation for your brand system. β[0]=1 means organic sales is the normalising variable; the other β entries tell you the long-run relationship between sales and each brand metric."
- α (error-correction loadings): "This is how fast each variable returns to equilibrium after a shock. A large negative α for organic_sales means sales adjusts quickly when the system goes out of balance — good for a healthy brand."
- Plain-English table: one row per variable, α value, and what it means
- Annotated plot of the cointegrating relation over time ("the gap that the error-correction closes")

**§7 — The key output: IRF** (~4 cells)
- Plain-English framing: "If brand spend increases by 10% today, what happens to organic sales over the next 52 weeks?"
- GIRF fan chart for brand_spend shock → organic_sales response with 80%/95% HDI
- Comparison chart: VECM long-run tail vs what a standard MMM would say (flat after ~4 weeks)
- Note: GIRF is order-invariant — no assumption about causal chain required

**§8 — Valuing ops: the CSAT story** (~3 cells)
- Plain-English framing: "If we improve CSAT by 1 point, what is the long-run impact on organic sales?"
- GIRF fan chart for csat_score shock → organic_sales response
- Revenue translation: multiply HDI bands by average revenue per customer
- Business takeaway: "ops improvements have a measurable long-run brand equity effect"

**§9 — Counterfactual ROI** (~4 cells)
- Plain-English framing: "What is the difference in organic sales between a high brand spend scenario and a low brand spend scenario?"
- Two spend paths: base (current) and uplift (+20% brand spend)
- `sample_posterior_predictive` for both; difference with HDI
- Bayesian ROI: (revenue uplift HDI) / (incremental spend) → "Our best estimate is £X return per £1 spent, with 80% credible interval £Y–£Z"

**§9a — Brand response curve** (~3 cells)
- Plain-English framing: "How does long-run organic sales respond as we increase brand spend from £0 to £10m?"
- Loop `sample_posterior_predictive` across 15 spend levels (£0–£10m annualised)
- Plot: X = annual brand spend, Y = cumulative 52-week organic sales uplift vs £0 baseline
- Posterior median + 80% credible band — shows where the curve flattens and where uncertainty widens
- Business takeaway: marginal return and optimal spend region with honest uncertainty

**§10 — Practical tips** (~1 cell, pure markdown)
- When to use horseshoe (set k_ar_diff=2 or 3 on real data; let horseshoe shrink)
- What convergence warnings mean (divergences → increase target_accept; rhat > 1.01 → more draws)
- How to think about coint_rank (domain knowledge can override the test; r=1 is the most common in marketing)
- CSAT as I(0): if ADF says stationary, move to exog — still useful, just a short-run driver
- Cholesky alternative: if you are confident in awareness → consideration → csat → organic_sales ordering, use `method="cholesky"` with variables in that order in the endog DataFrame
- Log-transformation reminder: always log-transform before fitting; back-transform for business outputs

### DGP design for synthetic data

K=4 endogenous, 3 exog, n_obs=300 weekly observations (~6 years).

Common stochastic trends: 2 (so coint_rank=2):
- Trend 1: brand equity trend (organic_sales and awareness share it)
- Trend 2: ops/product trend (consideration and csat share it, driven by product quality improving over time)

Exog effects:
- brand_spend → +awareness (B[awareness, brand_spend] > 0), +consideration (smaller)
- interest_rate → -consideration (higher rates reduce financial product consideration)
- consumer_confidence → +awareness, +consideration

Log-transform all endogenous variables before fitting.

### Key decisions locked in for this notebook

| Decision | Choice | Reason |
|---|---|---|
| IRF method | GIRF | Order-invariant; handles feedback loops; standard in literature |
| Deterministic | `"ci"` | Trends restricted to cointegration space; correct for trending brand data |
| Prior on Γ | Horseshoe | Real marketing data has unknown lag structure; set k generously |
| CSAT treatment | Endogenous if I(1), exog if I(0) | Test first; show both paths |
| Log transform | Yes, before fitting | Elasticity interpretation; natural diminishing returns on response curve |
| Variable names | Real marketing names throughout | Audience is practitioners, not econometricians |

### Real Monzo data — variables and sources

**Data availability note:** Ryan has 5 years of monthly brand and consideration data (60 observations). Monthly frequency, so use `k_ar_diff=1` or `k_ar_diff=2` — no need for higher lags on monthly data. 60 obs is tight but workable with horseshoe shrinkage; Bayesian approach handles this far better than OLS CVAR would.

#### Free macro data — easy to download as CSV

| Variable | Source | URL | Notes |
|---|---|---|---|
| Bank of England base rate | Bank of England | bankofengland.co.uk/boeapps/database | Search "Bank Rate". Monthly, goes back decades. |
| GfK Consumer Confidence | ONS | ons.gov.uk | Search "consumer confidence". Published monthly by GfK for ONS. |
| CPI / inflation | ONS | ons.gov.uk | Cost-of-living pressure affects propensity to switch to fee-free accounts. Alternative/complement to consumer confidence. |
| Unemployment rate | ONS | ons.gov.uk | Affects Monzo's core demographic (younger workers) disproportionately. |

All of the above are also on FRED (fred.stlouisfed.org) which has a clean API for automated pulls.

#### Marketing / brand data

| Variable | Source | Notes |
|---|---|---|
| Brand spend | Internal (Monzo) | Total monthly brand investment — TV, digital brand, OOH, etc. Primary exog driver. |
| Google Trends — "Monzo" | trends.google.com | Free, monthly. Organic brand search interest. Proxy for earned awareness. Could be exog or endogenous (test stationarity). |
| Press / media mentions | Meltwater / Brandwatch | Volume of Monzo press coverage monthly. Cain's equivalent of positive social mentions. If trending → endogenous candidate. |
| Feature launch dummies | Internal (Monzo) | Binary variables for major product launches: Monzo Plus, Monzo Premium, pots, salary sorter, etc. Equivalent to Cain's PR event dummies. Ryan knows these dates. |

#### Ops / product data

| Variable | Source | Notes |
|---|---|---|
| CSAT score | Internal (Monzo) | Average monthly CSAT. Test for stationarity: if I(1)/trending → endogenous; if I(0)/mean-reverting → exog. |
| App store rating | data.ai or scraped | Average monthly rating on iOS/Android. Monzo's equivalent of Cain's "product ratings". Strong theoretical link to CSAT and consideration. |
| NPS score | Internal (Monzo) | If available monthly. Very similar role to CSAT — use whichever is more complete. |

#### What Cain used (for reference)

Long-term CVAR endogenous: base sales, unaided brand awareness, brand consideration, positive social mentions (treated as exogenous for stability).

Long-term CVAR exogenous (stationary, entered short-run equations only): PR events around new product launches, retailer circulars, product ratings, total offline media experience, in-store special display experience.

#### Recommended starting point for first real-data run

Keep it simple — add variables iteratively once the core cointegrating structure is confirmed.

**Phase 1 (minimal viable model):**
- Endogenous: `organic_sales`, `brand_awareness`, `brand_consideration`, `csat_score`
- Exog: `brand_spend`, `interest_rate`, `consumer_confidence`
- Run `select_coint_rank`, check stationarity of CSAT, fit with `k_ar_diff=1`

**Phase 2 (enrich if Phase 1 converges cleanly):**
- Add Google Trends or press mentions as exog (or endogenous if trending)
- Add feature launch dummies as stationary exog
- Try `k_ar_diff=2` with horseshoe

**Phase 3 (refinement):**
- Add CPI or unemployment if consumer confidence alone is insufficient
- Test app store rating as alternative/supplement to CSAT

### Folder structure change (same session or next)

Move existing notebooks 01–10 to `notebooks/build/`.
Create `notebooks/guides/` for practitioner-facing content.
Update `.github/workflows/ci.yml` notebook execution path accordingly.

## Status as of last session (2026-06-05, chore/practitioner-guide-plan)

**Update 2026-06-05 — pre-PyPI housekeeping + practitioner guide notebook. PR open on `chore/practitioner-guide-plan`.**

- **Version bumped 0.0.1 → 0.1.0** in `pyproject.toml` and `src/bayesian_vecm/__init__.py`. `uv build` confirmed clean wheel + sdist.
- **README rewritten.** Covers: what the package does, `select_coint_rank` workflow, `BayesianVECM` quick start, horseshoe priors, variable naming / log-transform guide, development setup. First mention of the brand marketing motivation.
- **Notebooks reorganised.** `notebooks/01–10_*.ipynb` moved to `notebooks/build/`. New `notebooks/guides/` folder created for practitioner-facing content. CI updated to execute both folders.
- **`notebooks/guides/01_brand_vecm_walkthrough.ipynb` built.** End-to-end practitioner guide (§1–§10 + §9a): synthetic Monzo-style DGP, ADF tests, rank selection, horseshoe fit, α/β interpretation, GIRF fan charts, CSAT valuation, counterfactual ROI, brand response curve. Written for marketing analysts with no VECM background.
- **Two package bugs found and fixed during notebook testing:**
  - `_irf.py`: `beta_draws` has shape `(C, D, K+1, r)` when `deterministic="ci"/"li"` (extra trend row); code was trying to reshape as `(n_total, K, r)` → `ValueError`. Fix: slice `beta_draws[:, :, :n_vars, :]` before reshape. Regression test added to `tests/test_irf.py`.
  - `_forecast.py`: same bug in `forecast_posterior`. Fix: same slice. Regression test still to add.

**Key decisions / learnings this session:**

- **Use the common-trend DGP for practitioner notebooks, not a pure VECM simulation.** A VECM simulation (ΔVECM + EC correction) is I(1) with a unit root. The random walk component accumulates stddev ≈ σ × √T. With σ=0.08 and T=260, that's ±1.3 log-units of noise, easily overwhelming a small drift. The series can trend strongly downward despite a positive drift parameter. Common-trend DGP (`y = base + L @ trend + small_noise`) guarantees upward trends because the trend dominates by design. Rank=2 is also guaranteed regardless of seed. Use this pattern for any notebook where visual upward trends are needed.
- **Common-trend DGP with EC correction on top.** The hybrid approach (`dy_t = L @ d_trend[t] + alpha @ (beta.T @ y[t-1]) + gamma @ dy_lag1 + B @ exog + noise`) gives the best of both worlds: guaranteed upward trends from the common-trend innovations, plus identifiable α/β/Γ from the EC correction dynamics. Loading matrix L defines the cointegrating structure; `beta_true` entries should be consistent with `L[i,0]/L[j,0]` ratios so the EC terms stay near zero as the series grow.
- **DGP initial values must satisfy the cointegrating relations.** `y[0]` must satisfy `beta_true.T @ y[0] ≈ 0` to avoid a large initial EC shock. With the common-trend DGP, set `y[0]` from the loading ratios: if `beta = [1, -1.11, 0, 0]` then `organic_sales[0] = awareness[0] × 1.11`.
- **Exog variables must be centred (mean≈0) for the B matrix to work correctly.** The B coefficients are designed for deviations from mean. Adding a non-zero mean to exog (e.g. consumer_confidence = −10 + AR1) creates a large systematic B × mean contribution per step that swamps the trend. Keep all exog AR(1) centred at zero; express counterfactual uplifts as additive log-unit increases above the mean rather than multiplicative %.
- **Non-diagonal sigma_chol improves multi-hop IRF stories.** With diagonal innovations, h=0 GIRF for cross-variable shocks is exactly 0. Off-diagonal entries give contemporaneous covariance (e.g. csat ↔ consideration), making the CSAT → organic_sales IRF chain cleaner and more stable with fewer draws.
- **FAST_SAMPLING (200 draws / 300 tune / 2 chains) is inadequate for K=4 + horseshoe.** 57–74 divergences, r-hat > 1.01, near-zero α posteriors. Always use `FAST_SAMPLING = False` (1000/1000/4) when validating notebook outputs. FAST_SAMPLING is for CI only.
- **Two package bugs found and fixed during notebook testing:**
  - `_irf.py` + `_forecast.py`: `beta_draws` has shape `(C, D, K+1, r)` when `deterministic="ci"/"li"` adds a trend row inside the cointegrating space. Both `compute_irf` and `forecast_posterior` were reshaping it as `(n_total, K, r)` → `ValueError`. Fix: slice `beta_draws[:, :, :n_vars, :]` before reshape. Regression test added to `test_irf.py`; still to add for `_forecast.py`.
- **Module reload required after source edits in Jupyter.** Patching `_irf.py` or `_forecast.py` requires a full kernel restart. Re-running cells reuses the cached module.
- **MMM comparison chart: anchor to week-4 cumulative response.** Using the h=0 GIRF as the initial MMM impulse fails when cross-variable h=0 GIRF is near zero. Anchoring both curves to the same week-4 cumulative response makes the shape comparison robust.
- **Notebook folder structure:** `notebooks/build/` = implementation walkthroughs (01–10), `notebooks/guides/` = practitioner-facing notebooks. CI executes both.

## Next slice

**Finish and merge `chore/practitioner-guide-plan`**

- **NEXT STEP: full sample run.** Set `FAST_SAMPLING = False` in `notebooks/guides/01_brand_vecm_walkthrough.ipynb` and run end-to-end. Verify: upward-trending series, rank=2 detected, positive GIRF, positive counterfactual ROI, sensible α/β tables, no divergences.
- Add regression test for `_forecast.py` beta-slice fix (same pattern as `test_irf.py` test added this session)
- Commit final notebook outputs (strip before commit via nbstripout), push, open PR, wait for CI green, merge

**Then: real-data validation on Monzo data**

Branch: `feat/monzo-validation` (create from `main` after practitioner guide merges)

See "Next slice" section below for full spec.

## Status as of last session (2026-06-04, feat/horseshoe)

**Update 2026-06-04 — `feat/horseshoe` complete, PR open.**

- **Regularised horseshoe prior on Γ shipped.** Opt-in via `priors={"Gamma": {"dist": "Horseshoe"}}`. Optional kwargs: `tau_scale` (default 1.0), `slab_scale` (default 2.0), `slab_df` (default 4.0). Auxiliary RVs added to graph: `Gamma_tau`, `Gamma_lambda`, `Gamma_c2`. Default Normal(0, 0.5) prior unchanged — horseshoe is purely opt-in.
- **`select_coint_rank` shipped.** Wraps statsmodels Johansen trace test. Returns `CointRankResult` with `.rank`, `.test_stats`, `.crit_vals`, and a printable summary table. Exported from `bayesian_vecm.__init__`. Recommended workflow: `select_coint_rank` → set `coint_rank` → fit with generous `k_ar_diff` + horseshoe.
- **Notebook 10** (`10_horseshoe_prior_walkthrough.ipynb`): common-trend DGP (guaranteed rank=1), fit at k=3 (over-specified), side-by-side Normal vs horseshoe comparison — bar chart, density plots, τ posterior, SD ratio table.
- **Test count: 313 passed, 2 skipped** (macOS SIGINT expected skips). Coverage 96%.

**Key decisions / learnings this session:**

- **statsmodels `select_coint_rank` API:** positional arg order is `(endog, det_order, k_ar_diff)`, not keyword `k_ar_diffs`. Result attributes are `.test_stats` and `.crit_vals` (not `.trace_stat`).
- **EC-based DGPs fail Johansen rank detection.** Strong EC loadings (α = −0.4, 0.2) make series appear stationary (rank = K = 2 for K=2). Use common-trend DGP (`y = a * trend + noise`) for rank-detection tests and notebooks — guarantees rank = K − n_trends.
- **Horseshoe divergences with FAST_SAMPLING.** 57 divergences with 200 tune steps / 2 chains is expected — the funnel geometry needs more tuning. With full sampling (FAST_SAMPLING=False) divergences should be near zero. Documented in notebook note cell.
- **Pre-commit ruff E402.** `pytest.importorskip(...)` placed before a `from bayesian_vecm import ...` triggers E402 in the pre-commit ruff version. Fix: move `importorskip` call after all imports (safe since `bayesian_vecm` doesn't import statsmodels at module level).
- **Notebook JSON editing from Cowork.** Notebooks must be edited via Python `json` manipulation (not the `Edit` tool, which rejects `.ipynb` files). Use `ruff format` + `ruff check --fix` on notebooks before committing — ruff catches F401 (unused imports) and F541 (f-strings without placeholders) inside notebook cells.

## Next slice

**Real-data validation on Monzo data**

Branch: `feat/monzo-validation` (create from `main` after `feat/horseshoe` merges)

**Goal.** Fit the package on real Monzo brand/revenue data. This is the first real-world test — everything so far has been synthetic DGPs. Key questions:
- Does `select_coint_rank` give a sensible answer on real data?
- Does the horseshoe converge cleanly with a generous `k_ar_diff`?
- Do the GIRF fan charts tell a coherent brand-spend story?
- Are there any practical API gaps that only appear with real data?

**After validation:**
- Pre-PyPI housekeeping: README update (horseshoe + rank selection not yet mentioned), version bump to 0.1.0, confirm `uv build` produces a clean wheel.
- Medium article: package on PyPI + Monzo validation story + GIRF output = the hook.
- Docs site (MkDocs + mkdocs-jupyter): still deferred until notebook catalogue is stable post-validation.

## Status as of last session (2026-06-03, feat/horseshoe planning)

**Update 2026-06-03 — `feat/exog` merged (pending CI green); `feat/horseshoe` is next.**

- `feat/exog` PR open. CI had two minor ruff fixes after push (unused variable, Unicode minus sign); both patched and pushed. Waiting for CI green before merging.
- Notebook 09 (`09_exog_brand_marketing.ipynb`) is the applied brand marketing story — synthetic Monzo-style DGP, parameter recovery, GIRF, counterfactual ROI forecast diff. Plots verified manually.
- **Next slice decided: horseshoe priors on Γ (`feat/horseshoe`).** See "Next slice" section below.

**Key decisions made this session:**

- **Skip `select_lag_order` as a first-class feature.** Using a frequentist VAR to pick k before handing off to a Bayesian VECM is a philosophical inconsistency. More importantly, horseshoe makes it unnecessary — set k generously and let the prior shrink irrelevant lags toward zero.
- **Horseshoe before real-data testing.** Ryan will test the package on real Monzo data. Over-specified k is the most likely practical problem; horseshoe fixes it more gracefully than hard lag selection.
- **`select_coint_rank` still needed.** Horseshoe doesn't help with r — that's a structural graph decision, not a continuous parameter. A thin wrapper around `statsmodels.tsa.vector_ar.vecm.select_coint_rank` (Johansen trace test) should be added alongside or just after horseshoe. Workflow: run rank selection → set `coint_rank` → fit with horseshoe at generous k.
- **Docs site deferred.** MkDocs + mkdocs-jupyter is the right choice when the time comes; defer until after horseshoe and real-data validation.

## Broader vision: Bayesian marketing science stack

Discussed 2026-06-03. The VECM package is the first piece of a coherent stack of packages, each solving a real gap in the marketing science toolkit. Together they form a consulting proposition — Bayesian marketing science end to end — not just a GitHub portfolio.

### The four packages

**1. `bayesian_vecm` (this package)**
Captures long-run brand effects that standard MMMs miss. The VECM models the cointegrating relationship between brand spend and revenue; the EC mechanism quantifies the long-run tail. Differentiator: feeds directly into the MMM as a baseline input.

**2. Bayesian MMM**
Crowded space (Robyn, PyMC-Marketing) so needs a clear differentiator. The angle is tight integration with `bayesian_vecm` — an MMM that uses the VECM EC term as a revenue baseline, capturing both short-run channel attribution and long-run brand effects in one model. Nobody else is doing this combination.

**3. Bayesian budget optimiser**
Under-served gap. Most practitioners optimise on point estimates (scipy.optimize, Excel solver), ignoring posterior uncertainty. A Bayesian optimiser propagates uncertainty through to the allocation recommendation — "here's the optimal budget and here's the credible interval on that recommendation." Natural downstream consumer of the MMM posteriors.

**4. Collinearity diagnostic + budget adjuster (most original idea)**
The problem: real marketing data has highly correlated channels (TV and digital move together because budgets move together). MMM can't distinguish individual channel effects — the elasticities are wrong. The solution has three parts:
- **DGP simulation:** generate synthetic correlated spend data, fit an MMM, show the elasticities are wrong. Proves the problem rather than asserting it.
- **Budget perturbation recommendation:** design future spend plans that break the collinearity — deliberate de-correlation via channel mix variation. This is the "experiment design" step.
- **Sampling weights:** upweight recent de-correlated observations in the MMM so the model benefits quickly without waiting years for the new data to dominate the history.

The result: MMM practitioners can get materially better elasticity estimates within one or two budget cycles rather than waiting for natural variation to accumulate.

**Positioning note:** this is in the same spirit as the Vaver & Koehler geo-experiment literature from Google (using geographic variation to identify causal effects) but applied to channel mix rather than geography, and with a Bayesian twist. Worth knowing that literature when positioning the package — you're solving the same identification problem with a different instrument.

### The narrative arc

All four packages address the same root cause: marketing data is generated by business decisions rather than experiments, so causal identification is hard. The stack solves this end to end:
- VECM: recovers long-run effects the MMM baseline misses
- MMM: short-run channel attribution, integrated with VECM baseline
- Collinearity adjuster: fixes the identification problem in the MMM inputs
- Budget optimiser: makes spend recommendations with honest uncertainty

### Go-to-market thoughts

- Ship `bayesian_vecm` to PyPI first; validate on Monzo data; write the Medium article
- Second Medium article: "how I used AI as a co-pilot to build a non-trivial econometrics package" — the process story, not just the technical one
- The collinearity/budget adjuster idea is strong enough to be an academic paper, not just a package. Could drive significant LinkedIn traction in the MMM/causal inference community
- Ryan has an existing MMM/causal inference following on LinkedIn — the packages feed that audience directly
- Consulting proposition: "Bayesian marketing science end to end" — sells a methodology, not a bag of tools

## Next slice

**Horseshoe priors on Γ — `feat/horseshoe`**

**Branch:** create with `git switch -c feat/horseshoe` from `main` after `feat/exog` merges.

**Why.** With real marketing data the right lag order k is unknown. Setting k too high gives an over-parameterised Γ block with K²·k entries, most near zero. A horseshoe prior (Carvalho, Polson & Scott 2010) or regularised horseshoe (Piironen & Vehtari 2017) shrinks irrelevant Γ entries toward zero while keeping genuine short-run dynamics. This is more principled than hard lag selection via information criteria — the model expresses uncertainty about which lags matter rather than making a binary include/exclude decision.

**Scope:**
- Horseshoe applies to `Γ` (short-run lag coefficients). Possibly also `α` (EC loadings), but start with Γ only.
- Default prior stays `Normal(0, 0.5)` — horseshoe is opt-in via `priors={"Gamma": {"dist": "Horseshoe"}}` or a dedicated flag. Keeps the existing API and tests intact.
- PyMC has `pm.HalfStudentT` and `pm.HalfNormal` for the local/global scale hierarchy. The regularised horseshoe (RHS) uses a Student-T slab rather than a pure Cauchy — better sampling behaviour.
- Add a notebook (10) demonstrating horseshoe vs. Normal prior on a DGP where k is deliberately over-specified — show that horseshoe recovers the true sparse Γ while Normal smears mass over the irrelevant lags.

**Implementation sketch:**
```
Γ_ij ~ Normal(0, τ · λ_ij)   # local-global scale mixture
λ_ij ~ HalfCauchy(1)          # local scales (per-entry shrinkage)
τ    ~ HalfCauchy(scale)       # global scale (overall sparsity)
```
For the regularised horseshoe replace HalfCauchy local scales with a Student-T slab capped by a slab width `c²`.

**Also in this slice:** `select_coint_rank` — thin wrapper around statsmodels Johansen trace test. Returns a readable table (test statistic, 5% critical value, recommended r). Always runs before fitting on real data.

## Status as of last session (2026-06-02, feat/exog)

**Update 2026-06-02 — `feat/exog` in progress (not yet merged to `main`):**

- **Exogenous regressor support shipped.** Five files touched: `_design.py`, `_pymc.py`, `_model.py`, `_forecast.py`, `_output.py`.
- **`cointegration_design`** gains `exog` and `exog_coint` args. `exog` is aligned to `T_eff` rows and stored in `CointegrationDesign.exog`. `exog_coint` is appended to `y_lag1` (same mechanism as inside deterministic terms).
- **PyMC graph:** `B ~ Normal(0, 1)` RV of shape `(K, m)` added when `design.exog` is not None. `exog @ B.T` added to the mean. `pm.Data("exog", ...)` stored so `_output.py` can read it back.
- **`fit(endog, *, exog=None, exog_coint=None, ...)`** — aligned exog stashed in `idata.constant_data["exog"]` with dim `"time_eff"` (not `"time"` — avoids clash with the full-length `endog` array). `self.exog_` set for downstream use.
- **`sample_posterior_predictive(steps, *, exog_future=None, ...)`** — raises `ValueError` if model was fitted with exog but `exog_future` not provided.
- **`_forecast.py`:** `Gamma` now sliced to `K * k_ar_diff` dynamic columns before the forecast loop (fixes a pre-existing outside-deterministic bug). `B @ X_{T+h}` added at each step.
- **`_output.py`:** `compute_fittedvalues` includes `exog @ B.T` when both `"exog"` and `"B"` are available.
- **`tests/test_exog.py`:** 34 tests — 14 passed locally (design + graph); sampling tests deferred to CI (macOS SIGINT issue).
- **Notebook 09** written: brand marketing applied story. DGP: organic sales (MMM baseline) and brand awareness are cointegrated; brand spend drives awareness only (`B[0,0]=0`, `B[1,0]=0.15`). Covers OLS vs VECM, parameter recovery, GIRF, counterfactual forecast diff. **Not yet executed** — run cell by cell in VS Code to verify, then execute via nbconvert before opening PR.
- **Branch status:** `feat/exog` pushed. Open PR once CI is green and notebook executes.

**Key design decisions locked in this slice:**
- `exog` stored with dim `"time_eff"` in `constant_data` to avoid clash with `endog`'s `"time"` dim (different lengths: T vs T_eff).
- `B` is a separate RV from `Gamma` — cleaner semantics, avoids the outside-deterministic column-stripping complication.
- `exog_coint` absorbed into `y_lag1` at design time — same mechanism as `"ci"`/`"li"` deterministic terms.
- `exog_future` is required when the model was fitted with `exog` — silent zeros would give misleading forecasts.

**Brand marketing framing (important for notebook 09 and future applied work):**
- Organic sales is the **MMM baseline output** — spend-decomposed. Brand spend has **no direct contemporaneous effect** on organic sales (`B[0,:] = 0`).
- Brand spend drives **awareness** contemporaneously (`B[1,0]`). Awareness is cointegrated with organic sales. The EC mechanism propagates the awareness gain into organic sales over time — the long-run tail that standard MMMs miss.
- The counterfactual forecast diff (uplift minus baseline) gives a Bayesian ROI estimate including that long-run tail.

## Status as of last session (2026-06-01)

**Update 2026-06-01 — `feat/irf` in progress (not yet merged to `main`):**

- **IRF support shipped.** `src/bayesian_vecm/_irf.py` with `compute_irf(idata, k_ar_diff, steps, method, variable_names)`.
- **Implementation:** VAR companion form — VECM is converted to levels VAR(p) with `p = k_ar_diff + 1`. Companion matrix `F` is iterated to produce `Phi_h = top-left K×K block of F^h` for `h = 0..steps`. Fully vectorised over (chain × draw); single Python loop over horizons.
- **Two identification schemes:**
  - `"girf"` (default) — Generalised IRFs (Pesaran & Shin 1998). `GIRF_h = Phi_h @ Sigma @ diag(Sigma)^{-1/2}`. Order-invariant; the right choice for systems with contemporaneous feedback (awareness ↔ consideration ↔ organic sales).
  - `"cholesky"` — Orthogonalised IRFs (Sims 1980). `OIR_h = Phi_h @ P`. Available for recursive systems but not the default.
- **Key design decision:** outside deterministic terms (`"co"`, `"lo"`) append an extra column to `Gamma` in the fitted posterior. That column is a constant/trend coefficient, not a VAR lag, and must not enter the companion matrix. `compute_irf` always slices `Gamma` to `K * k_ar_diff` columns before building `F`.
- **`BayesianVECM.irf(steps, method="girf")`** — thin wrapper with pre-fit guard and steps validation. Returns `xr.DataArray (chain, draw, horizon, response_variable, shock_variable)`. Horizon coord runs 0..steps inclusive.
- **`tests/test_irf.py`:** 19 tests — shape, dims, coords, finite values, h=0 mathematical identities for both methods, Cholesky upper-triangle zeros, own-shock positivity, determinism, long-run I(1) non-decay. All passed.
- **Notebook 07** shipped: bivariate awareness/sales DGP, GIRF fan charts with 80%/95% HDI bands, I(1) non-decay table, GIRF vs Cholesky comparison under two orderings.
- **Ruff N806 lesson:** uppercase variable names (`Kp`, `F`, `A1`, `G_prev`, `P`) fail the N806 rule in function scope — rename to lowercase (`kp`, `companion`, `a1`, `g_prev`, `p_chol`). Pre-commit hook caught these before push.
- **Branch status:** `feat/irf` is ready to merge. Run `.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/07_irf_walkthrough.ipynb` before committing the notebook, then open a PR.
- **Suite: 192 passed** (173 + 19 new IRF tests).

**Motivation for GIRF as default (important for the brand marketing use case):**

The target system is:
- **Endogenous (cointegrated):** organic sales, brand awareness, consideration.
- **Exogenous (future slice):** brand spend, interest rates / demand driver.

Within the endogenous system the causal flow has feedback loops: awareness → consideration → sales, but also sales → awareness (word-of-mouth from buyers) and consideration → awareness (word-of-mouth from considerers). Any Cholesky ordering imposes a zero contemporaneous restriction that is wrong. GIRF is order-invariant and handles this correctly.

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

**Exogenous regressors (`exog`) — contemporaneous brand spend effects**

**Branch:** `feat/exog` (not yet created)

**Why now.** The brand marketing use case is the primary motivation for this package. The endogenous system (organic sales, brand awareness, consideration) is now fully estimable and has IRF output. The missing piece is brand spend entering as a contemporaneous exogenous driver. Once `exog` lands, the key marketing deliverable becomes a **counterfactual forecast diff** — run `sample_posterior_predictive` with two spend paths and take the difference — which is the Bayesian version of the methodology in Ryan's Medium article.

**Goal.** See the "Exogenous regressors (`exog`) support" section below for the full spec. Short version:
- `model.fit(endog, exog=brand_spend_df)` — contemporaneous effect in the short-run equation.
- `model.fit(endog, exog_coint=brand_equity_df)` — effect inside the cointegrating relation.
- `sample_posterior_predictive` and `irf` both need updating to accept future `exog` paths.

**Sequencing note.** The post-IRF output methods (fittedvalues, resid, var_rep, diagnostic tests) are small and can slot in before or after `exog` — they don't block anything.

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

## Session learnings (2026-06-02, feat/exog)

- **`idata.constant_data` dim name clash.** `endog` is stored with dim `"time"` (T rows). Storing the aligned `exog` (T_eff rows) with the same dim name raises `AlignmentError: conflicting dimension sizes`. Fix: use `"time_eff"` for any T_eff-length arrays added to `constant_data` after sampling.
- **`_forecast.py` had a latent outside-deterministic bug.** When `deterministic="co"/"lo"`, `Gamma` in the posterior has `K*k + 1` columns (extra outside-term column). The forecast loop was using the full width. Fixed in this slice: slice `gamma[:, :, :n_vars * k_ar_diff]` before the loop — same fix already in `_irf.py`.
- **Module-scoped fixtures with two `pm.sample` calls hit the double-sample SIGINT.** `fitted_model_with_exog` and `fitted_model_no_exog` both call `pm.sample`. Running them in the same pytest session triggers SIGINT. No local fix; rely on CI for the sampling tests.
- **Brand marketing DGP framing.** Organic sales is the MMM baseline (spend-decomposed), so brand spend must have zero direct effect on it (`B[0,:] = 0`). Spend drives awareness only; awareness is cointegrated with organic sales. The EC mechanism carries the awareness gain into organic sales over time — this is the long-run tail the MMM misses.

## Session learnings (2026-06-02, output-methods notebook)

- **`az.style.use("arviz-darkgrid")` is gone in arviz 1.x.** Available styles: `arviz-cetrino`, `arviz-tenui`, `arviz-tumma`, `arviz-variat`, `arviz-vibrant`. Simplest fix: remove the call (notebooks 04–07 don't use one).
- **`az.hdi` API changed in arviz 1.x.** Old: `az.hdi(da.stack(sample=("chain", "draw")), hdi_prob=0.80)`. New: `az.hdi(da, prob=0.80, dim=["chain", "draw"])` returning a DataArray with `ci_bound` coord. Access bounds via `.sel(ci_bound="lower")` / `.sel(ci_bound="upper")` (note: `"higher"` → `"upper"`).
- **`statsmodels` is not a transitive dep.** Any notebook using `plot_acf` needs `uv add --dev statsmodels`.
- **Local nbconvert still broken by macOS SIGINT.** CI (Linux) is unaffected — treat CI as the execution gate for notebooks.

## Session learnings (2026-06-01)

- **Ruff N806 — uppercase variable names in function scope.** `Kp`, `F`, `A1`, `G_prev`, `P` all fail N806 even when they're standard mathematical notation. Rename to `kp`, `companion`, `a1`, `g_prev`, `p_chol`. This catches you on commit via the pre-commit hook; easier to rename upfront than fix after staging.
- **IRF companion matrix: outside deterministic columns must be stripped.** When `deterministic="co"` or `"lo"`, `Gamma` in the fitted posterior has `K*k + 1` columns — the last column is the constant/trend, not a VAR lag. Passing the full matrix to the companion construction gives a wrong `A_1` and wrong IRFs. Always slice `gamma[:, :, :n_vars * k_ar_diff]` before building `F`.
- **GIRF is the right default for feedback systems.** The brand marketing system (awareness ↔ consideration ↔ organic sales) has contemporaneous feedback loops that violate the Cholesky recursive ordering assumption. The two Cholesky orderings give materially different long-run responses; GIRF is order-invariant. Document this clearly for users — the choice of `method` matters a lot in practice.
- **IRFs are fully deterministic** — no random innovations, unlike `sample_posterior_predictive`. The same idata gives the same IRF every time. No `random_seed` argument needed.
- **VECM-to-VAR conversion formula (for reference):** `A_1 = I + alpha @ beta.T + Gamma_1`, `A_j = Gamma_j - Gamma_{j-1}` for `j = 2..k`, `A_{k+1} = -Gamma_k`. For `k=0`: `A_1 = I + alpha @ beta.T`, `p=1`.

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

## Docs site (future)

Notebooks committed with outputs stripped (nbstripout) keep diffs clean but means GitHub can't render them. Options when the package is closer to PyPI:

- **nbviewer** — quick workaround now: paste any raw GitHub notebook URL into [nbviewer.org](https://nbviewer.org) to render it.
- **Docs site** — Sphinx + nbsphinx or MkDocs + mkdocs-jupyter. Would execute notebooks and publish rendered HTML. Worth doing once the notebook catalogue is stable (post-`exog`).

## Domain-learning track

Ryan is **learning VECMs as we build**, so explanations of the econometrics (cointegration, error-correction term, identification, lag selection, etc.) should accompany the code as it's written.

**Convention.** One numbered Jupyter notebook per public-API slice, living in `notebooks/`:

- Filename pattern: `NN_<topic>.ipynb` (e.g. `01_data_utilities_walkthrough.ipynb`).
- Each notebook explains *what* each helper does, *why* a VECM needs it, and demos it on small synthetic data — written for a reader meeting VECMs for the first time.
- Trigger for a new notebook: "did this slice ship something a learner needs to understand?" Internal refactors don't need one.
- Notebooks are runnable docs *and* lightweight integration tests — when CI execution lands (see TODO in the status section), a broken explanation becomes a failing build.
- Once the catalogue grows, consider graduating to a docs site (Sphinx + nbsphinx, or MkDocs + mkdocs-jupyter). Defer until the `BayesianVECM` skeleton is in.
