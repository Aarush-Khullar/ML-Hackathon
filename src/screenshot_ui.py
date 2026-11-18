"""Screenshot the running evaluator UI after predicting one trip (for the report).
Needs the API (python api/main.py) and the UI (python ui/serve_ui.py --no-browser) running locally.
Usage: python src/screenshot_ui.py [trip_id]  -> results/deployment/ui_screenshot.png"""
import os, sys
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trip = sys.argv[1] if len(sys.argv) > 1 else "00000094"
os.makedirs(os.path.join(ROOT, "results", "deployment"), exist_ok=True)
with sync_playwright() as p:
    try:
        b = p.chromium.launch(channel="msedge")
    except Exception:
        b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 950})
    pg.goto("http://localhost:8080/?api=http://localhost:8000")
    pg.wait_for_selector(f"#t{trip}", timeout=30000); pg.click(f"#t{trip}")
    pg.wait_for_function(f"document.querySelector('#status').textContent.includes('done')", timeout=60000)
    pg.wait_for_timeout(3000)
    out = os.path.join(ROOT, "results", "deployment", "ui_screenshot.png"); pg.screenshot(path=out); print(out)
    b.close()
