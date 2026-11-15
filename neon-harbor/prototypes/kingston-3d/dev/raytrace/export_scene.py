#!/usr/bin/env python3
"""The same export as export.js, driven from Python (for machines with Python Playwright but no Node).

    python3 dev/raytrace/export_scene.py <page.html> <out.json> [--dusk 0.45] [--at 1010,345] [--delivered 3] [--carry]

Set NH_CHROMIUM to a Chromium executable to use that browser instead of Playwright's own download.
"""
import json
import os
import sys

from playwright.sync_api import sync_playwright

a = sys.argv[1:]


def opt(name, default):
    return a[a.index(name) + 1] if name in a else default


if len(a) < 2:
    sys.exit("usage: export_scene.py <page.html> <out.json> [--dusk 0.45] [--at x,y] [--delivered n] [--carry]")
here = os.path.dirname(os.path.abspath(__file__))
setup = """(o) => { const g = window.game, sim = window.NH.sim; g.start(); const s = g.state,
  set = () => { s.t = sim.config.clockSeconds.value * o.dusk; s.player.x = o.at[0]; s.player.y = o.at[1]; s.carrying = o.carry; };
  for (let i = 0; i < o.delivered; i++) { s.carrying = true; s.player.x = sim.DROP.x; s.player.y = sim.DROP.y; g.step(1); }
  set(); g.step(40); set(); g.render(0.016); g.render(0.016); }"""
with sync_playwright() as pw:
    exe = os.environ.get("NH_CHROMIUM")
    browser = pw.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else pw.chromium.launch(args=["--no-sandbox"])
    page = browser.new_context(viewport={"width": 360, "height": 640}, device_scale_factor=1).new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("file://" + os.path.abspath(a[0]))
    page.evaluate(setup, {"dusk": float(opt("--dusk", "0.45")), "at": [float(v) for v in opt("--at", "1010,345").split(",")],
                          "delivered": int(opt("--delivered", "3")), "carry": "--carry" in a})
    page.add_script_tag(path=os.path.join(here, "inpage.js"))
    text = page.evaluate("window.__nhExportScene()")
    browser.close()
if errors:
    sys.exit("page errors: " + " | ".join(errors))
os.makedirs(os.path.dirname(os.path.abspath(a[1])), exist_ok=True)
open(a[1], "w").write(text)
d = json.loads(text)
print(json.dumps({"out": a[1], "megabytes": round(len(text) / 1e6, 1), "meshes": len(d["meshes"]),
                  "triangles": sum(m["triangles"] for m in d["meshes"]), "materials": len(d["materials"]),
                  "textures": len(d["textures"]), "dusk": d["dusk"]}))
