"""Render the HTML-built figures (diagrams and tables) to PNG with Playwright."""

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "docs" / "_extra" / "walkthrough"
OUT.mkdir(parents=True, exist_ok=True)
# file stem -> (element to capture, viewport width)
FIGS = {
    "fig02_system_diagram": ("figure", 880),
}

with sync_playwright() as p:
    browser = p.chromium.launch()
    for stem, (selector, width) in FIGS.items():
        page = browser.new_page(viewport={"width": width, "height": 900}, device_scale_factor=2)
        page.goto((HERE / "html" / f"{stem}.html").as_uri())
        page.locator(selector).screenshot(path=str(OUT / f"{stem}.png"))
        page.close()
    browser.close()
