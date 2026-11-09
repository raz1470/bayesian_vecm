# The walkthrough article

Source for `docs/_extra/walkthrough.html`, published at
[raz1470.github.io/bayesian_vecm/walkthrough.html](https://raz1470.github.io/bayesian_vecm/walkthrough.html).
Every number comes from simulated data with a known answer.

| File | What it does |
|---|---|
| `article.md` | The text, one `## N. Title` heading per section, charts placed with `[[chart ...]]` |
| `build.py` | Assembles the page from the text, `style.css`, `charts.js` and the chart data, then bakes the drawn charts into the HTML |
| `style.css` | Page styles |
| `charts.js` | Draws each chart as inline SVG from its data, and redraws it at the reader's width |
| `figures/fig*.py` | One script per chart, written as JSON to `data/` |
| `data/*.json` | Chart data: title, legend, panels and series |

Rebuild from the repo root:

```bash
for f in tools/article/figures/fig*.py; do uv run python "$f"; done
uv run --with playwright python tools/article/build.py
```

Baking needs a Chromium. Run `uv run --with playwright python -m playwright install chromium` once, or set `CHROME` to a browser binary. Pass `--no-bake` to skip baking.

The chart scripts that fit models (`fig08_response_curves.py`, `fig08a_lasting_curve.py`) use fixed seeds and take a few minutes each.
