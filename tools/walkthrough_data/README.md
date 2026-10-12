# Data behind `docs/walkthrough.html`

`export.py` computes every number and chart series in the walkthrough and
writes them to `results/walkthrough.json`. `page/build.py` writes that JSON
into the page with the CSS and chart script, so the published file is
self-contained. Nothing here runs in CI.

Run from the repo root:

```
uv run python tools/walkthrough_data/export.py   # refit, about four minutes
python tools/walkthrough_data/page/build.py      # rebuild docs/walkthrough.html
```

The prose lives in `page/body.html`. Numbers quoted in the prose are typed
in, so check them against the JSON after a refit.

## Page sources

| File | What it holds |
|---|---|
| `page/body.html` | Header, nav, every section, FAQ and footer |
| `page/charts.js` | The SVG chart functions, one per figure |
| `page/style.css` | Layout and components, following `how_wrong_is_your_mmm/docs/overview.html` |
| `page/scrollspy.js` | Keeps the sticky nav in step with the section in view |
| `page/build.py` | Assembles `docs/walkthrough.html` |

`export.py` fits the model with the same settings as the fast run of
`notebooks/guides/01_brand_vecm_walkthrough.ipynb`: 2 chains, 1000 tune,
1000 draws, seed 42. About two minutes on two cores. Intervals are 89%
equal-tailed.

| Walkthrough section | JSON keys | Notebook section |
|---|---|---|
| 1 · The problem | `illustration` | §1 |
| 2 · What a VECM is | `spurious` | none |
| 3 · The data | `meta`, `series` | §2 |
| 4 · Tests before fitting | `stationarity`, `rank` | §3, §4 |
| 5 · The fit | `sampler`, `beta`, `alpha`, `relations`, `fit`, `holdout` | §5, §6 |
| 6 · The awareness effect | `irf_awareness`, `irf_cumulative`, `irf_week52_sales` | §7 |
| 7 · Which links exist | `causality` | §8 |
| 8 · A campaign | `campaign`, `response_curve` | §9, §9a |
| 9 · Return per channel | `return_per_channel` | §9b |

`fevd` is still exported but no longer drawn. `holdout` refits on the first
208 weeks, so the export takes about four minutes.

## What has been checked

Every headline number matches the notebook's committed outputs:

- Sales at week 52 after an awareness shock: 1.08% [0.94%, 1.24%].
- Cumulative response against matched adstock at week 52: 9.7x.
- Brand spend one standard deviation up for 26 weeks: 20.8% [14.8%, 27.0%] at week 52.
- Brand spend per £1: £2.00 [£1.41, £2.58] in year 1, £11.34 [£8.27, £14.48] over five years.
- Sampler: 0 divergences, max r-hat 1.00, min bulk ESS 850.
- Hold-out 2025: sales error 2.3% against 3.7% for a random walk with drift.

The charts were checked with `how_wrong_is_your_mmm/tools/check_charts.js` at
desktop and phone widths: every chart draws, no text under 9px, no clipping
and no overlapping labels.
