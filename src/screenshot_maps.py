"""Render the Folium HTML maps in results/maps/ to PNG screenshots for the report (uses installed Microsoft Edge, else: playwright install chromium).
Usage: python src/screenshot_maps.py"""
import glob, os
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
maps = sorted(glob.glob(os.path.join(ROOT, "results", "maps", "*.html")))
with sync_playwright() as p:
    try:
        b = p.chromium.launch(channel="msedge")          # use the Edge already installed on Windows
    except Exception:
        b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1400, "height": 900})
    for f in maps:
        pg.goto("file:///" + f.replace("\\", "/"))
        pg.wait_for_timeout(4000)                      # let OSM tiles load
        out = f[:-5] + ".png"; pg.screenshot(path=out); print(out)
    b.close()
