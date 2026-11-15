#!/usr/bin/env node
/* Screenshot the 3D page in a headless browser (software WebGL: slow but correct).
     node dev/shot.js <page.html> <out.png> [options]
   Options:
     --scene play|title|cast     default play
     --size 390x844              viewport in CSS pixels (keep it small: software rendering is slow)
     --dusk 0..1                 time of day: 0 = 4 PM, 0.5 = dusk, 0.95 = night
     --delivered n               crates already delivered (adds patrols and speakers)
     --at x,y                    player position in sim pixels (default 860,420)
     --carry                     player is carrying a crate
     --zoom k                    camera distance multiplier (0.3 = close up)
     --pitch deg                 camera pitch in degrees (default: the game's own)
     --pan x,y,z                 move the point the camera looks at, in metres (east, up, south)
     --dpr n                     device pixel ratio (default 1; 3 = what a phone really draws; makes a big image)
     --clip x,y,w,h              save only this rectangle of the screen (CSS pixels), e.g. a crop around the player
     --steps n                   run n rule steps before the shot (default 30)
   Prints one JSON line: errors, warnings, renderer stats and how long the frame took. */
"use strict";
const path = require("node:path");
const { chromium } = require("playwright");
const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i < 0 ? d : a[i + 1]; };
(async () => {
  const [w, h] = opt("--size", "390x844").split("x").map(Number), scene = opt("--scene", "play");
  const b = await chromium.launch(), cx = await b.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: +opt("--dpr", "1") }), p = await cx.newPage();
  const errors = [], warnings = [];
  p.on("pageerror", (e) => errors.push(e.message)); p.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); else if (m.type() === "warning" && !/deprecated with r150/.test(m.text())) warnings.push(m.text().slice(0, 200)); });
  await p.goto("file://" + path.resolve(a[0]));
  const failed = await p.evaluate(() => !document.getElementById("fail").hidden && document.getElementById("failmsg").textContent);
  if (!failed && !(await p.evaluate(() => !!window.game))) { console.log(JSON.stringify({ out: null, failed: "the page did not finish starting", errors })); await b.close(); process.exit(1); }
  let stats = null;
  if (!failed) stats = await p.evaluate(({ scene, dusk, delivered, at, carry, zoom, pitch, pan, steps }) => {
    const g = window.game, sim = NH.sim;
    if (scene === "cast") { document.getElementById("cast").click(); }
    else if (scene === "play") {
      g.start(); const s = g.state;
      for (let i = 0; i < delivered; i++) { s.carrying = true; s.player.x = sim.DROP.x; s.player.y = sim.DROP.y; g.step(1); }
      s.t = sim.config.clockSeconds.value * dusk; s.player.x = at[0]; s.player.y = at[1]; s.carrying = carry; g.step(steps);
      s.t = sim.config.clockSeconds.value * dusk; s.player.x = at[0]; s.player.y = at[1]; s.carrying = carry; s.heat = Math.min(s.heat, 60);
    }
    const v = {}; if (zoom) v.zoom = zoom; if (pitch) v.pitch = pitch * Math.PI / 180; if (pan) v.pan = pan; g.view(v);
    g.render(0.016); const t0 = performance.now(); g.render(0.016); const ms = performance.now() - t0;
    return Object.assign(g.info(), { frameMs: Math.round(ms), scene: g.scene });
  }, { scene, pan: opt("--pan", "") ? opt("--pan", "").split(",").map(Number) : null, dusk: +opt("--dusk", "0.1"), delivered: +opt("--delivered", "0"), at: opt("--at", "860,420").split(",").map(Number), carry: a.includes("--carry"),
       zoom: +opt("--zoom", "0"), pitch: +opt("--pitch", "0"), steps: +opt("--steps", "30") });
  const clip = opt("--clip", "") ? opt("--clip", "").split(",").map(Number) : null;
  await p.waitForTimeout(150); await p.screenshot(clip ? { path: a[1], clip: { x: clip[0], y: clip[1], width: clip[2], height: clip[3] } } : { path: a[1] });
  console.log(JSON.stringify({ out: a[1], failed: failed || null, errors, warnings, stats }));
  await b.close(); process.exit(errors.length || failed ? 1 : 0);
})();
