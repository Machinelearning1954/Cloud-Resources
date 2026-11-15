#!/usr/bin/env node
/* The standard set of screenshots, in one run.
     node dev/sheet.js <page.html> <out-dir> [only-these-names...]
   Writes <out-dir>/<name>.png for each view below and prints one JSON line per shot with the
   draw calls, triangles and any page errors. Software WebGL: correct, but its timings mean nothing.
   For a view that is not here, use dev/shot.js. */
"use strict";
const path = require("node:path"), fs = require("node:fs");
const { chromium } = require("playwright");
const PHONE = { width: 390, height: 844 };
const VIEWS = [ // name, what it is for, viewport, options
  ["day-street",   "4:18 PM at a junction, carrying, two delivered: the main gameplay view", PHONE, { dusk: 0.1, delivered: 2, at: [340, 430], carry: true }],
  ["day-midblock", "4:18 PM in the middle of an east-west street: is the street hidden by anything?", PHONE, { dusk: 0.1, delivered: 1, at: [600, 420], carry: true }],
  ["gold-street",  "5:30 PM golden hour on the southern cross street", PHONE, { dusk: 0.5, delivered: 3, at: [860, 940] }],
  ["night-yard",   "6:51 PM in the soundhall yard, four delivered: lamps, string lights, speakers", PHONE, { dusk: 0.95, delivered: 4, at: [1030, 300], carry: true }],
  ["night-street", "6:51 PM on a street with patrols about: can the player read it?", PHONE, { dusk: 0.95, delivered: 4, at: [340, 940] }],
  ["day-quay",     "4:18 PM on the quay: crate pile, boatworks, skiffs, palms, water", PHONE, { dusk: 0.1, delivered: 0, at: [300, 1410] }],
  ["cast",         "the Cast screen: every figure and prop side by side", PHONE, { scene: "cast" }],
  ["cast-close",   "the top of the Cast screen, close up, at 2x", PHONE, { scene: "cast", zoom: 0.5, pan: [0, 0.6, -4.2], dpr: 2 }],
  ["player-3x",    "the player and the people nearby exactly as a phone draws them (3x crop)", PHONE, { dusk: 0.1, delivered: 1, at: [1030, 300], carry: true, dpr: 3, clip: [75, 302, 240, 240] }],
  ["unfolded",     "an unfolded foldable, 690x829", { width: 690, height: 829 }, { dusk: 0.3, delivered: 2, at: [860, 420], carry: true }],
  ["landscape",    "a phone held sideways, 844x390", { width: 844, height: 390 }, { dusk: 0.6, delivered: 3, at: [600, 940] }],
];
(async () => {
  const [page, outDir, ...only] = process.argv.slice(2);
  if (!page || !outDir) { console.error("usage: node dev/sheet.js <page.html> <out-dir> [names...]\nviews: " + VIEWS.map((v) => v[0]).join(", ")); process.exit(2); }
  fs.mkdirSync(outDir, { recursive: true });
  const b = await chromium.launch(); let bad = 0;
  for (const [name, what, size, o] of VIEWS) {
    if (only.length && !only.includes(name)) continue;
    const cx = await b.newContext({ viewport: size, deviceScaleFactor: o.dpr || 1 }), p = await cx.newPage(), errors = [];
    p.on("pageerror", (e) => errors.push(e.message)); p.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
    await p.goto("file://" + path.resolve(page));
    const started = await p.evaluate(() => !!window.game && document.getElementById("fail").hidden);
    let stats = null;
    if (started) stats = await p.evaluate((o) => {
      const g = window.game, sim = NH.sim;
      if (o.scene === "cast") document.getElementById("cast").click();
      else {
        g.start(); const s = g.state, set = () => { s.t = sim.config.clockSeconds.value * o.dusk; s.player.x = o.at[0]; s.player.y = o.at[1]; s.carrying = !!o.carry; };
        for (let i = 0; i < o.delivered; i++) { s.carrying = true; s.player.x = sim.DROP.x; s.player.y = sim.DROP.y; g.step(1); }
        set(); g.step(40); set(); s.heat = Math.min(s.heat, 55);
      }
      g.view({ zoom: o.zoom, pan: o.pan }); g.render(0.016); g.render(0.016);
      const i = g.info(); return { calls: i.calls, triangles: i.triangles, textures: i.textures, geometries: i.geometries };
    }, o);
    const file = path.join(outDir, name + ".png");
    await p.waitForTimeout(120);
    await p.screenshot(o.clip ? { path: file, clip: { x: o.clip[0], y: o.clip[1], width: o.clip[2], height: o.clip[3] } } : { path: file });
    if (!started || errors.length) bad++;
    console.log(JSON.stringify({ name, what, file, started, errors, stats }));
    await cx.close();
  }
  await b.close(); process.exit(bad ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
