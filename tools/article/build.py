"""Build docs/_extra/walkthrough.html from article.md, style.css and the figures.

Run the scripts in figures/ first: they write the PNGs to docs/_extra/walkthrough/.
"""

import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = REPO / "docs" / "_extra"
draft = (HERE / "article.md").read_text()

# Figures by section: (start of the paragraph the figure follows, file, alt text, caption)
FIGS = {
    "0": [
        (
            "Here is the shape",
            "fig00_response_curve.png",
            "Response curve of a 13-week brand campaign, revenue within one year and within five "
            "years",
            "Extra revenue from a 13-week brand campaign at each spend level, against no campaign. "
            "Simulated brand, full model, 89% intervals.",
        )
    ],
    "1": [
        (
            "So the long-term model",
            "fig01_adstock_illusion.png",
            "One burst of spend with a short adstock, a long adstock and a lasting brand-system "
            "response",
            "Illustration, not an estimate. Adstock spreads an effect over time and then lets it "
            "go. Brand building moves the level sales return to.",
        )
    ],
    "2": [
        (
            "Around the system sit the drivers",
            "fig02_system_diagram.png",
            "Diagram of the brand system: channels into awareness, macro drivers into "
            "consideration, then sales",
            "Drivers act from outside. The three brand series are tied together and trend.",
        ),
        (
            "To check every step",
            "fig02_brand_series.png",
            "Base sales, awareness and consideration over five years, indexed to 100",
            "The three brand series on the simulated brand, indexed to 100 in the first week.",
        ),
    ],
    "3": [
        (
            "This is not bad luck",
            "fig03_spurious.png",
            "Two unrelated random walks and the residuals of regressing one on the other",
            "Two random walks built independently. The regression residuals never settle back to "
            "zero.",
        ),
        (
            "So the first step",
            "fig03_stationarity_table.png",
            "Table of ADF p-values and verdicts for the eight series",
            "ADF test on each series in levels, 5% level.",
        ),
    ],
    "4": [
        (
            "In our brand one lead",
            "fig04_cointegration.png",
            "Base sales and 1.27 times consideration over five years, and the gap between them",
            "Left: both series as % from their average, in logs. Right: the gap stays within 8% of "
            "zero.",
        ),
        (
            "The number of leads",
            "fig04_rank_test.png",
            "Johansen trace statistics against their critical values",
            "Johansen trace test, one lag, constant outside the relationships.",
        ),
    ],
    "5": [
        (
            "[Equation figure]",
            "fig05_equation.png",
            "The VECM equation with each term labelled",
            None,
        ),
        (
            "[Three-equation figure]",
            "fig05_three_equations.png",
            "The VECM written out as one equation for each series",
            None,
        ),
        (
            "We fit the model",
            "fig05_terms_table.png",
            "Table of the five parts of the model and what each means for the brand",
            None,
        ),
    ],
    "6": [
        (
            "In our brand sales close",
            "fig06_gap_closing.png",
            "Share of each gap still open week by week, with half-lives",
            "Implied by the fitted adjustment speeds.",
        ),
        (
            "The fitted model is not perfect",
            "fig06_alpha_forest.png",
            "Adjustment speeds with 89% intervals against the true values",
            "Posterior mean with 89% interval. Black ticks mark the truth.",
        ),
        (
            "The scores recover the funnel",
            "fig06_evidence_table.png",
            "Evidence scores for the six links between brand metrics",
            None,
        ),
    ],
    "7": [
        (
            "In our brand sales are about 1% higher",
            "fig07_awareness_shock.png",
            "Responses of sales, awareness and consideration to a surprise rise in awareness",
            "Generalised impulse response to a one standard deviation awareness shock, 89% "
            "intervals.",
        ),
        (
            "So a GIRF is safe",
            "fig07_orderings.png",
            "The sales response under three orderings",
            None,
        ),
    ],
    "8": [
        (
            "In our brand one more £1",
            "fig08_return_brand.png",
            "Cumulative revenue per £1 of brand spend over five years with the truth",
            "One more £1 of brand spend for one week, valued at recent weekly sales. 89% interval.",
        ),
        (
            "The same calculation works",
            "fig08_return_channels.png",
            "Cumulative return for brand spend, PR and sponsorship with the truth",
            None,
        ),
    ],
    "9": [
        (
            "Both act on the drivers",
            "fig09_where_they_sit.png",
            "Diagram: raw spend, adstock, saturation, awareness, then the leads",
            None,
        ),
        (
            "The impact is large",
            "fig09_impact_table.png",
            "Table of errors with and without adstock and saturation",
            None,
        ),
    ],
    "10": [
        (
            "A regression of consideration growth",
            "fig10_ols_vs_iv.png",
            "Histograms of the regression and instrument estimates across 200 simulated brands",
            "200 simulated brands with no same-week link.",
        ),
        (
            "The fix, as in Cain",
            "fig10_exclusion.png",
            "Diagram of the exclusion: brand spend reaches consideration only through awareness",
            None,
        ),
    ],
    "11": [
        (
            "The second reason",
            "fig11_coverage.png",
            "Share of 90% intervals holding the truth, Bayesian against statsmodels",
            "30 simulated brands at each length. statsmodels intervals for the return per £ come "
            "from a residual bootstrap.",
        )
    ],
    "12": [
        (
            "The common mistake",
            "fig12_mmm_handover.png",
            "Total sales, the true base, total minus contributions and the MMM's trend",
            "PyMC-Marketing with a time-varying intercept on one simulated brand.",
        ),
        (
            "The short run did not",
            "fig12_base_sales.png",
            "Errors in the long-run return and adjustment speed for four ways of building base "
            "sales",
            "A handful of simulated brands with carryover and saturation in sales.",
        ),
    ],
    "13": [
        (
            "The package ships with a notebook",
            "fig13_check_table.png",
            "The table of checks the plug-in notebook prints",
            None,
        )
    ],
}
NAV = {
    "0": None,
    "1": "The problem",
    "2": "The system",
    "3": "Trends",
    "4": "Cointegration",
    "5": "The VECM",
    "6": "Leaders",
    "7": "Shocks",
    "8": "Spend to £",
    "9": "Carryover and saturation",
    "10": "Same-week links",
    "11": "Why Bayesian",
    "12": "Base sales",
    "13": "Your data",
    "14": "Bringing it together",
}


def inline(t):
    t = (
        html.escape(t, quote=False)
        .replace("&lt;sub&gt;", "<sub>")
        .replace("&lt;/sub&gt;", "</sub>")
    )
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)
    t = re.sub(
        r"(github\.com/raz1470/bayesian_vecm)",
        r'<a href="https://\1" target="_blank" rel="noopener">\1</a>',
        t,
    )
    t = re.sub(
        r"(raz1470\.github\.io/bayesian_vecm)",
        r'<a href="https://\1/" target="_blank" rel="noopener">\1</a>',
        t,
    )
    return t


def fig_html(file, alt, cap):
    if not (OUT / "walkthrough" / file).exists():
        raise FileNotFoundError(
            f"Missing figure {file}: run the scripts in tools/article/figures first"
        )
    c = f'<figcaption class="fig-cap">{inline(cap)}</figcaption>' if cap else ""
    return (
        f'<figure class="fig"><img src="walkthrough/{file}" alt="{html.escape(alt)}" '
        f'loading="lazy" decoding="async">{c}</figure>'
    )


# Split the draft into sections.
parts = re.split(r"^## §(\d+) (.+?)\s*$", draft, flags=re.M)
sections = {}
for i in range(1, len(parts) - 1, 3):
    sections[parts[i]] = (parts[i + 1], parts[i + 2])

body = []
for num in [str(n) for n in range(15)]:
    title, text = sections[num]
    text = text.split("\n## ")[0]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out, figs = [], list(FIGS.get(num, []))
    in_refs = False
    for b in blocks:
        if b.startswith("<!--"):
            continue
        if b.startswith("### References"):
            in_refs = True
            out.append('<h3 id="references">References</h3>')
            continue
        if b.startswith("[") and b.endswith("]"):
            html_b = ""
        elif b.startswith("- "):
            items = [inline(x[2:]) for x in b.split("\n") if x.startswith("- ")]
            cls = "refs" if in_refs else "probs"
            html_b = f'<ul class="{cls}">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
        else:
            html_b = f"<p>{inline(' '.join(b.split()))}</p>"
        if html_b:
            out.append(html_b)
        for f in list(figs):
            if b.startswith(f[0]):
                out.append(fig_html(f[1], f[2], f[3]))
                figs.remove(f)
    assert not figs, (num, figs)
    if num == "0":
        body.append('<section class="intro">' + "\n".join(out) + "</section>")
    else:
        body.append(
            f'<section id="s{num}"><p class="s-label">§{num} · {html.escape(NAV[num])}</p>'
            f"<h2>{inline(title)}</h2>" + "\n".join(out) + "</section>"
        )

nav = "".join(f'<a href="#s{n}">{n} · {html.escape(NAV[str(n)])}</a>' for n in range(1, 15))
css = (HERE / "style.css").read_text()
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>How Long Does Brand Last? by Ryan O'Sullivan</title>
<style>{css}</style>
</head>
<body>
<header class="hdr"><div class="hdr-in">
<h1>How Long Does Brand Last?</h1>
<p class="tagline">Measuring the long-term effect of brand marketing with a Bayesian VECM</p>
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
