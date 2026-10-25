# The walkthrough article

Source for `docs/_extra/walkthrough.html`, published at
[raz1470.github.io/bayesian_vecm/walkthrough.html](https://raz1470.github.io/bayesian_vecm/walkthrough.html).
Every number comes from simulated data with a known answer.

| File | What it does |
|---|---|
| `article.md` | The text, one `## §N` heading per section |
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

`figures/fig12_mmm_handover.py` fits a PyMC-Marketing MMM. It needs
`pymc-marketing==1.3.0` in a separate environment, so skip it in the loop above
unless that is installed. The other chart scripts refit the package's models
with fixed seeds. They also read the validation results in `tools/*_results.csv`.
