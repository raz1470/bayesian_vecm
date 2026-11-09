"""Build docs/_extra/walkthrough.html from article.md, style.css and the figures.

Run the scripts in figures/ first: they write each chart's data to tools/article/data/.
The page draws the charts as inline SVG with charts.js, then this script bakes the drawn
SVG into the HTML so readers without JavaScript still see them. Baking needs Playwright;
set CHROME to a browser binary if Playwright's own Chromium isn't installed. Pass
--no-bake to skip it.

article.md uses a small subset of Markdown:

* ``# Title`` and an italic line under it for the subtitle.
* ``## N. Title`` starts a section; ``### Heading`` a sub-heading.
* ``PARTS`` below places a part banner before the section that opens each part.
* Paragraphs, ``- `` and ``1. `` lists, ``| a | b |`` tables, ``**bold**``, ``*italic*``,
  ``[text](url)`` links and ``<sub>`` / ``<sup>``.
* A block that starts with ``<`` is passed through as HTML.
* ``[[chart NAME | ALT | CAPTION]]`` places the chart in ``data/NAME.json``.
* ``<!-- ... -->`` comments are dropped.
"""

import html
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "docs" / "_extra"
source = (HERE / "article.md").read_text()
source = re.sub(r"<!--.*?-->", "", source, flags=re.S)

NAV = {
    "1": "Introduction",
    "2": "Adstock",
    "3": "The brand",
    "4": "Trends",
    "5": "Cointegration",
    "6": "The model",
    "7": "Response curves",
    "8": "Fading lifts",
    "9": "Estimating it",
    "10": "Bringing it together",
}

# Section that opens each part -> (part label, part title)
PARTS = {
    "2": ("Part 1", "The VECM"),
    "8": ("Part 2", "Letting lifts fade"),
    "9": ("Part 3", "Can we estimate it?"),
}


def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    for tag in ("sub", "sup", "b"):
        t = t.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    t = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+(?:\([^)\s]*\))?[^)\s]*)\)",
        lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">{m.group(1)}</a>',
        t,
    )
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    return t


CHARTS: dict[str, dict] = {}


def chart(spec: str) -> str:
    name, alt, cap = (p.strip() for p in spec.split("|", 2))
    path = HERE / "data" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing chart data {path.name}: run tools/article/figures")
    CHARTS[name] = json.loads(path.read_text())
    return (
        f'<figure class="fig chart-card" role="group" aria-label="{html.escape(alt)}">'
        f'<div class="fig-h">{html.escape(CHARTS[name]["title"])}</div>'
        f'<div class="panels" id="{name}"></div><div class="tip" aria-hidden="true"></div>'
        f'<figcaption class="fig-cap">{inline(cap)}</figcaption></figure>'
    )


def table(block: str) -> str:
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in block.splitlines()
        if not re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())
    ]
    head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
    labels = [html.escape(re.sub(r"[*,].*", "", c).strip(), quote=True) for c in rows[0]]
    body = "".join(
        "<tr>"
        + "".join(f'<td data-label="{h}">{inline(c)}</td>' for h, c in zip(labels, r, strict=True))
        + "</tr>"
        for r in rows[1:]
    )
    cls = "tbl stack" if len(rows[0]) > 2 else "tbl"
    table_html = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return f'<div class="{cls}">{table_html}</div>'


def block_html(b: str, in_refs: bool) -> str:
    if b.startswith("[[chart ") and b.endswith("]]"):
        return chart(b[8:-2])
    if b.startswith("<"):
        return b
    if b.startswith("|"):
        return table(b)
    if b.startswith("### "):
        title = b[4:].strip()
        anchor = ' id="references"' if title == "References" else ""
        return f"<h3{anchor}>{inline(title)}</h3>"
    if b.startswith("- "):
        items = [inline(x[2:]) for x in b.split("\n") if x.startswith("- ")]
        cls = "refs" if in_refs else "probs"
        return f'<ul class="{cls}">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
    if re.match(r"1\. ", b):
        items = [inline(re.sub(r"^\d+\. ", "", x)) for x in b.split("\n") if re.match(r"\d+\. ", x)]
        return '<ol class="aims">' + "".join(f"<li>{x}</li>" for x in items) + "</ol>"
    return f"<p>{inline(' '.join(b.split()))}</p>"


head, *rest = re.split(r"^## (\d+)\. (.+?)\s*$", source, flags=re.M)
title = re.search(r"^# (.+)$", head, flags=re.M).group(1)
subtitle = re.search(r"^\*(.+)\*\s*$", head, flags=re.M).group(1)

body = []
for i in range(0, len(rest), 3):
    num, name, text = rest[i], rest[i + 1], rest[i + 2]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out, in_refs = [], False
    for b in blocks:
        in_refs = in_refs or b.startswith("### References")
        out.append(block_html(b, in_refs))
    if num in PARTS:
        label, ptitle = PARTS[num]
        body.append(
            f'<div class="part"><p class="part-n">{label}</p>'
            f'<p class="part-t">{html.escape(ptitle)}</p></div>'
        )
    body.append(
        f'<section id="s{num}"><p class="s-label">{num} · {html.escape(NAV[num])}</p>'
        f"<h2>{inline(name)}</h2>" + "\n".join(out) + "</section>"
    )

nav = "".join(f'<a href="#s{n}">{n} · {html.escape(t)}</a>' for n, t in NAV.items())
css = (HERE / "style.css").read_text()
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} by Ryan O'Sullivan</title>
<style>{css}</style>
</head>
<body>
<header class="hdr"><div class="hdr-in">
<h1>{html.escape(title)}</h1>
<p class="tagline">{inline(subtitle)}</p>
<p class="byline">By Ryan O'Sullivan</p>
</div></header>
<nav class="snav" aria-label="Sections"><div class="snav-in">{nav}</div></nav>
<main>
{chr(10).join(body)}
</main>
<footer><p>Simulated data throughout. Built with
<a href="https://github.com/raz1470/bayesian_vecm" target="_blank" rel="noopener">bayesian_vecm</a>.
</p></footer>
<script type="application/json" id="DATA">{json.dumps(CHARTS, separators=(",", ":"))}</script>
<script>{(HERE / "charts.js").read_text()}</script>
</body>
</html>
"""


def bake(page: str) -> str:
    """Render the page in a headless browser and write each drawn chart into the HTML."""
    from playwright.sync_api import sync_playwright

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        path = Path(tmp) / "page.html"
        path.write_text(page)
        exe = os.environ.get("CHROME")
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        tab = browser.new_page(viewport={"width": 1000, "height": 900})
        errors: list[str] = []
        tab.on("pageerror", lambda e: errors.append(str(e)))
        tab.goto(path.as_uri())
        tab.wait_for_timeout(300)
        if errors:
            raise SystemExit(f"page errors: {errors}")
        drawn = tab.evaluate(
            "Object.fromEntries([...document.querySelectorAll('div.panels[id]')]"
            ".map(d => [d.id, d.innerHTML]))"
        )
        browser.close()
    for name, inner in drawn.items():
        page = page.replace(
            f'<div class="panels" id="{name}"></div>',
            f'<div class="panels" id="{name}">{inner}</div>',
        )
    return page


if "--no-bake" not in sys.argv:
    page = bake(page)
out = OUT / "walkthrough.html"
out.write_text(page)
print(f"Wrote {out.relative_to(REPO)}")
