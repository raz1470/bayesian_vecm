# The walkthrough article

Source for `docs/_extra/walkthrough.html`, published at
[raz1470.github.io/bayesian_vecm/walkthrough.html](https://raz1470.github.io/bayesian_vecm/walkthrough.html).
Every number comes from simulated data with a known answer.

| File | What it does |
|---|---|
| `article.md` | The text, one `## §N` heading per section, figures placed with `[[fig ...]]` |
| `build.py` | Assembles the page from the text, `style.css` and the figures |
| `style.css` | Page styles |
| `figures/fig*.py` | One script per chart, written to `docs/_extra/walkthrough/` |
| `figures/render_html.py` | Renders the diagrams and tables in `figures/html/` to PNG with Playwright |

Rebuild from the repo root:

```bash
for f in tools/article/figures/fig*.py; do uv run python "$f"; done
uv run python tools/article/figures/render_html.py
uv run python tools/article/build.py
```

The chart scripts that fit models (`fig07_shocks.py`, `fig08_response_curves.py`, `fig08a_lasting_curve.py`) use fixed seeds and take a few minutes each.
