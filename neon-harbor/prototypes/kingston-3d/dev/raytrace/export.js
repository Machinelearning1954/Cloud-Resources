#!/usr/bin/env node
/* Export the prototype's 3D scene, exactly as the game builds it, to one JSON file that an
   offline ray tracer can load (see render.py). Needs Playwright.
     node dev/raytrace/export.js <page.html> <out.json> [--dusk 0.45] [--at 1010,345] [--delivered 3] [--carry] [--size 1080x1920]
   What is exported: every visible mesh (instances expanded, joined per material), materials
   (colour, roughness, metal, glow, textures drawn in code), lights, the game camera, the time of day.
   What is left out: game markers (vision cones, drop ring, arrow, see-through shapes), additive
   "pool of light" decals (a ray tracer makes real ones), sky domes, lines.
   Axes are converted from the game's (x east, y up, z south) to (x east, y north, z up). */
"use strict";
const fs = require("node:fs"), path = require("node:path");
const { chromium } = require("playwright");
const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i < 0 ? d : a[i + 1]; };

(async () => {
  if (a.length < 2) { console.error("usage: node dev/raytrace/export.js <page.html> <out.json> [--dusk 0.45] [--at x,y] [--delivered n] [--carry] [--size WxH]"); process.exit(2); }
  const [w, h] = opt("--size", "1080x1920").split("x").map(Number);
  const b = await chromium.launch(), cx = await b.newContext({ viewport: { width: Math.round(w / 3), height: Math.round(h / 3) }, deviceScaleFactor: 1 }), p = await cx.newPage(), errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  await p.goto("file://" + path.resolve(a[0]));
  await p.evaluate((o) => { const g = window.game, sim = window.NH.sim; g.start(); const s = g.state, set = () => { s.t = sim.config.clockSeconds.value * o.dusk; s.player.x = o.at[0]; s.player.y = o.at[1]; s.carrying = o.carry; };
    for (let i = 0; i < o.delivered; i++) { s.carrying = true; s.player.x = sim.DROP.x; s.player.y = sim.DROP.y; g.step(1); }
    set(); g.step(40); set(); g.render(0.016); g.render(0.016); },
    { dusk: +opt("--dusk", "0.45"), at: opt("--at", "1010,345").split(",").map(Number), delivered: +opt("--delivered", "3"), carry: a.includes("--carry") });
  await p.addScriptTag({ path: path.join(__dirname, "inpage.js") });
  const json = await p.evaluate(() => window.__nhExportScene());
  await b.close();
  if (errors.length) { console.error("page errors: " + errors.join(" | ")); process.exit(1); }
  fs.mkdirSync(path.dirname(path.resolve(a[1])), { recursive: true }); fs.writeFileSync(a[1], json);
  const d = JSON.parse(json);
  console.log(JSON.stringify({ out: a[1], megabytes: +(json.length / 1e6).toFixed(1), meshes: d.meshes.length, triangles: d.meshes.reduce((n, m) => n + m.triangles, 0), vertices: d.meshes.reduce((n, m) => n + m.vertices, 0),
    materials: d.materials.length, textures: d.textures.length, lights: d.lights.map((l) => l.type), dusk: d.dusk, skipped: d.skipped }));
})().catch((e) => { console.error(e); process.exit(2); });
