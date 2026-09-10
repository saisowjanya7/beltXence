"""
SmartBelt — Phase 6: Web Dashboard Server

Serves the interactive industrial SCADA dashboard (index.html) over local HTTP.
Zero external dependencies required (uses standard library http.server).

Usage:
    python dashboard_web.py [--port 8050]

Then open in browser:
    http://localhost:8050
"""

import sys
import os
import argparse
import webbrowser
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

SCRIPT_DIR = Path(__file__).resolve().parent

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SCRIPT_DIR), **kwargs)

    def log_message(self, format, *args):
        # Clean server logging
        sys.stderr.write(f"[SCADA Server] {self.address_string()} - {args[0]}\n")

def run_server(port: int = 8050, open_browser: bool = False):
    server_address = ("", port)
    httpd = HTTPServer(server_address, CustomHandler)
    url = f"http://localhost:{port}/index.html"
    print("=" * 70)
    print("SmartBelt — Industrial Web SCADA Dashboard Server")
    print("=" * 70)
    print(f"  Server listening on: http://localhost:{port}")
    print(f"  Dashboard URL:       {url}")
    print("  Press Ctrl+C to halt the server.\n")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SCADA Server] Server stopped by operator.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartBelt Web SCADA Dashboard")
    parser.add_argument("--port", "-p", type=int, default=8050, help="HTTP port (default: 8050)")
    parser.add_argument("--open", "-o", action="store_true", help="Automatically open in default web browser")
    args = parser.parse_args()

    run_server(port=args.port, open_browser=args.open)
