"""Renders technical_report.html to PDF via Chrome DevTools Protocol directly
(Page.printToPDF with displayHeaderFooter=False), since the --print-to-pdf
CLI flag alone was still showing the file:// URL footer/header.
"""
import base64
import json
import subprocess
import time
from pathlib import Path

import requests
import websocket

ROOT = Path(__file__).resolve().parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9333
HTML_PATH = ROOT / "technical_report.html"
PDF_PATH = ROOT / "technical_report.pdf"

proc = subprocess.Popen([
    EDGE, "--headless=new", "--disable-gpu", "--no-sandbox",
    f"--remote-debugging-port={PORT}", "--remote-allow-origins=*",
    "--user-data-dir=" + str(ROOT / "_edge_profile"),
    f"file:///{HTML_PATH.as_posix()}",
])

try:
    for _ in range(30):
        time.sleep(0.5)
        try:
            tabs = requests.get(f"http://127.0.0.1:{PORT}/json").json()
            break
        except Exception:
            tabs = None
    if not tabs:
        raise SystemExit("could not connect to CDP endpoint")

    page = next(t for t in tabs if t.get("type") == "page")
    ws_url = page["webSocketDebuggerUrl"]
    ws = websocket.create_connection(ws_url)

    ws.send(json.dumps({
        "id": 1,
        "method": "Page.printToPDF",
        "params": {
            "displayHeaderFooter": False,
            "printBackground": True,
            "preferCSSPageSize": False,
        },
    }))
    result = json.loads(ws.recv())
    data = result["result"]["data"]
    PDF_PATH.write_bytes(base64.b64decode(data))
    print("wrote", PDF_PATH, PDF_PATH.stat().st_size, "bytes")
    ws.close()
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
