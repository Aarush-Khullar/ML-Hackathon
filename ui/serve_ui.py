"""Serve the PathSnap evaluator UI (LAPTOP 2).
Usage: python ui/serve_ui.py [--api http://<laptop1-ip>:8000] [--port 8080]
Opens http://localhost:8080/?api=<api> in the default browser. The page talks to the API directly from the browser,
so the API address must be reachable from this laptop (same Wi-Fi / hotspot)."""
import argparse, functools, http.server, os, socketserver, webbrowser
from urllib.parse import quote

ap = argparse.ArgumentParser()
ap.add_argument("--api", default="http://localhost:8000", help="PathSnap API base URL (laptop 1)")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--no-browser", action="store_true")
args = ap.parse_args()
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.dirname(os.path.abspath(__file__)))
url = f"http://localhost:{args.port}/?api={quote(args.api, safe=':/')}"
with socketserver.TCPServer(("", args.port), handler) as httpd:
    print(f"PathSnap UI on {url}  (API {args.api})  - Ctrl+C to stop")
    if not args.no_browser: webbrowser.open(url)
    httpd.serve_forever()
