"""Regenerate the README screenshots in docs/screenshots/.

Start the app first, then run this in a second terminal:

    streamlit run app.py
    python scripts/capture_screenshots.py            # default: localhost:8501
    python scripts/capture_screenshots.py --url http://localhost:8504

Requires `pip install -r requirements-dev.txt`. Uses an installed Chrome or
Edge through Playwright, so no browser download is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def settle(page, extra_ms: int = 3500) -> None:
    """Wait until Streamlit has rendered and finished running the script."""
    page.wait_for_selector("h1", timeout=60_000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidgetRunningIcon\"]')",
        timeout=60_000,
    )
    # Plotly charts animate in after the script completes.
    page.wait_for_timeout(extra_ms)


def scroll_to_text(page, text: str) -> None:
    page.get_by_text(text, exact=False).first.scroll_into_view_if_needed()
    page.wait_for_timeout(2500)


def launch(p):
    for channel in ("chrome", "msedge"):
        try:
            return p.chromium.launch(channel=channel, headless=True)
        except PlaywrightError:
            continue
    # Falls back to Playwright's bundled Chromium (`playwright install chromium`).
    return p.chromium.launch(headless=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8501")
    base = ap.parse_args().url.rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = launch(p)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1.5,
            color_scheme="dark",
        )
        page = ctx.new_page()

        page.goto(f"{base}/?page=overview")
        settle(page)
        page.screenshot(path=OUT / "overview.png")

        page.goto(f"{base}/?page=data-explorer")
        settle(page)
        page.get_by_role("tab", name="Relationship to target").click()
        page.wait_for_timeout(4000)
        page.screenshot(path=OUT / "data-explorer.png")

        page.goto(f"{base}/?page=model-performance")
        settle(page)
        scroll_to_text(page, "Candidate leaderboard")
        page.screenshot(path=OUT / "model-performance.png")

        page.goto(f"{base}/?page=explainability")
        settle(page)
        page.screenshot(path=OUT / "explainability.png")

        page.goto(f"{base}/?page=predict")
        settle(page, 5000)
        scroll_to_text(page, "Percentile")
        page.screenshot(path=OUT / "predict.png")

        browser.close()

    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
