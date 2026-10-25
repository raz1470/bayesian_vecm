"""Render the HTML-built figures (diagrams and tables) to PNG with Playwright."""

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)
# file stem -> (element to capture, viewport width)
FIGS = {
    "fig02_system_diagram": ("figure", 880),
    "fig03_stationarity_table": ("#f", 680),
    "fig05_equation": ("#f1", 880),
    "fig05_three_equations": ("#f3", 960),
    "fig05_terms_table": ("#f2", 820),
    "fig06_evidence_table": ("#f", 700),
    "fig09_where_they_sit": ("#f1", 960),
    "fig09_impact_table": ("#f2", 760),
    "fig13_check_table": ("#f", 960),
}

with sync_playwright() as p:
    browser = p.chromium.launch()
    for stem, (selector, width) in FIGS.items():
        page = browser.new_page(viewport={"width": width, "height": 900}, device_scale_factor=2)
        page.goto((HERE / "html" / f"{stem}.html").as_uri())
        page.locator(selector).screenshot(path=str(OUT / f"{stem}.png"))
        page.close()
    browser.close()
