"""Build docs/walkthrough.html from the page sources and results/walkthrough.json.

Run from the repo root::

    python tools/walkthrough_data/page/build.py

The page is self-contained: the CSS, the chart script and the numbers it
draws are all written inline.
"""

import json
from pathlib import Path

here = Path(__file__).parent
root = here.parents[2]
data = json.loads((here.parent / "results" / "walkthrough.json").read_text())
keep = [
    "meta",
    "series",
    "illustration",
    "relations",
    "alpha",
    "irf_awareness",
    "irf_cumulative",
    "causality",
    "spurious",
    "holdout",
    "campaign",
    "response_curve",
    "return_per_channel",
]
data = {k: data[k] for k in keep}
css = (here / "style.css").read_text()
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>How Long Does Brand Last? by Ryan O'Sullivan</title>
<style>{css}</style>
</head>
{(here / "body.html").read_text()}
<script>
// Numbers from tools/walkthrough_data/results/walkthrough.json (export.py).
const DATA = {json.dumps(data, ensure_ascii=False, separators=(",", ":"))};

{(here / "charts.js").read_text()}

{(here / "scrollspy.js").read_text()}
</script>
</body>
</html>
"""
out = root / "docs" / "walkthrough.html"
out.write_text(html)
print(f"Wrote {out} ({len(html):,} bytes)")
