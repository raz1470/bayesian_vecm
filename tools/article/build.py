"""Build docs/_extra/walkthrough.html from article.md, style.css and the figures.

Run the scripts in figures/ first: they write the PNGs to docs/_extra/walkthrough/.

article.md uses a small subset of Markdown:

* ``# Title`` and an italic line under it for the subtitle.
* ``## §N Title`` starts a section; ``### Heading`` a sub-heading.
* Paragraphs, ``- `` lists, ``| a | b |`` tables, ``**bold**``, ``*italic*``,
  ``[text](url)`` links and ``<sub>`` / ``<sup>``.
* A block that starts with ``<`` is passed through as HTML.
* ``[[fig FILE | ALT | CAPTION]]`` places a figure.
* ``<!-- ... -->`` comments are dropped.
"""

import html
import re
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
    "7": "Leaders",
    "8": "Response curves",
    "9": "What can go wrong",
    "10": "Open questions",
    "11": "Run it yourself",
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


def figure(spec: str) -> str:
    file, alt, cap = (p.strip() for p in spec.split("|", 2))
    if not (OUT / "walkthrough" / file).exists():
        raise FileNotFoundError(f"Missing figure {file}: run the scripts in tools/article/figures")
    return (
        f'<figure class="fig"><img src="walkthrough/{file}" alt="{html.escape(alt)}" '
        f'loading="lazy" decoding="async"><figcaption class="fig-cap">{inline(cap)}'
        "</figcaption></figure>"
    )


def table(block: str) -> str:
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in block.splitlines()
        if not re.fullmatch(r"\|?[\s:|-]+\|?", line.strip())
    ]
    head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
    return (
        f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


def block_html(b: str, in_refs: bool) -> str:
    if b.startswith("[[fig ") and b.endswith("]]"):
        return figure(b[6:-2])
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
    return f"<p>{inline(' '.join(b.split()))}</p>"


head, *rest = re.split(r"^## §(\d+) (.+?)\s*$", source, flags=re.M)
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
    body.append(
        f'<section id="s{num}"><p class="s-label">§{num} · {html.escape(NAV[num])}</p>'
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
</body>
</html>
"""
out = OUT / "walkthrough.html"
out.write_text(page)
print(f"Wrote {out.relative_to(REPO)}")
