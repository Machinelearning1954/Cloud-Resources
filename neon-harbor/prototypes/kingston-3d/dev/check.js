#!/usr/bin/env node
/* Browser checks for the built page, run from file:// in a headless browser (software WebGL).
     node dev/check.js [page.html]        default index.html
   These need Playwright, so they are run by hand and by the build workflow, not in CI.
   CI runs test/sim.test.js (the rules) and build.js --check (the page matches its sources). */
"use strict";
const path = require("node:path");
const { chromium } = require("playwright");
const file = path.resolve(process.argv[2] || path.join(__dirname, "..", "index.html"));
const results = [];
const ok = (name, pass, detail) => { results.push([!!pass, name, detail]); };
(async () => {
  const b = await chromium.launch();
  const open = async (size, touch) => {
    const cx = await b.newContext({ viewport: size, deviceScaleFactor: 1, hasTouch: !!touch }), p = await cx.newPage(), errors = [];
    p.on("pageerror", (e) => errors.push(e.message)); p.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
    await p.goto("file://" + file); return { p, errors, cx };
  };
  const st = (p) => p.evaluate(() => { const s = window.game.state; return { scene: window.game.scene, x: s.player.x, y: s.player.y, carrying: s.carrying, delivered: s.delivered, heat: s.heat, patrols: s.patrols.length, result: s.result, t: s.t, cheated: s.cheated, stash: !!s.stash }; });

  // 1. loads cleanly and shows the title
  let { p, errors, cx } = await open({ width: 390, height: 844 });
  ok("page loads from file:// with no errors and no failure message", errors.length === 0 && await p.evaluate(() => document.getElementById("fail").hidden), errors.join(" | "));
  ok("title card is showing and HUD is hidden", await p.evaluate(() => window.game.scene === "title" && !document.getElementById("card").hidden && document.getElementById("hud").hidden));
  ok("WebGL is drawing with shadows on", await p.evaluate(() => { window.game.render(0); const i = window.game.info(); return i.shadowMap && i.calls > 5 && i.triangles > 500; }));
  await cx.close();

  // 2. real keyboard: Enter starts, arrow keys move, Space drops and picks up
  ({ p, errors, cx } = await open({ width: 390, height: 844 }));
  await p.keyboard.press("Enter"); let a = await st(p);
  ok("Enter starts the game", a.scene === "play");
  await p.keyboard.down("ArrowRight"); await p.waitForFunction((x0) => window.game.state.player.x > x0 + 25, a.x, { timeout: 15000 }).catch(() => {}); await p.keyboard.up("ArrowRight");
  let c = await st(p);
  ok("holding the right arrow walks right (real key events, real frames)", c.x > a.x + 25, `x ${a.x.toFixed(0)} -> ${c.x.toFixed(0)}`);
  await p.keyboard.down("ArrowRight"); await p.waitForFunction(() => window.game.state.carrying, null, { timeout: 15000 }).catch(() => {}); await p.keyboard.up("ArrowRight");
  ok("walking into the pile picks up a crate", (await st(p)).carrying);
  await p.keyboard.press(" "); c = await st(p); ok("Space puts the crate down", !c.carrying && c.stash);
  await p.keyboard.press(" "); c = await st(p); ok("Space picks it back up", c.carrying && !c.stash);
  await p.keyboard.press("p"); ok("P pauses", (await st(p)).scene === "pause");
  for (const k of ["ArrowUp", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "ArrowDown"]) await p.keyboard.press(k);
  c = await st(p); ok("a code typed on the pause screen works and marks the run", c.cheated && (await p.evaluate(() => document.getElementById("codemsg").textContent)).includes("Clean Slate"));
  await p.keyboard.press("p"); ok("P resumes", (await st(p)).scene === "play");
  ok("no errors during keyboard play", errors.length === 0, errors.join(" | ")); await cx.close();

  // 3. touch: drag on the street moves the player, the Drop button works
  ({ p, errors, cx } = await open({ width: 390, height: 844 }, true));
  await p.click("#go"); a = await st(p);
  await p.mouse.move(120, 600); await p.mouse.down(); await p.mouse.move(190, 600, { steps: 4 });
  const stickShown = await p.evaluate(() => !document.getElementById("stick").hidden);
  await p.waitForFunction(() => window.game.state.carrying, null, { timeout: 15000 }).catch(() => {}); await p.mouse.up();
  c = await st(p);
  ok("dragging on the street shows the stick and walks the player to the crates", stickShown && c.x > a.x + 20 && c.carrying, `x ${a.x.toFixed(0)} -> ${c.x.toFixed(0)}`);
  ok("the stick hides when the finger lifts", await p.evaluate(() => document.getElementById("stick").hidden));
  await p.click("#drop"); c = await st(p); ok("the Drop button puts the crate down", !c.carrying && c.stash);
  ok("no errors during touch play", errors.length === 0, errors.join(" | ")); await cx.close();

  // 4. scripted game through the same hooks the 2D prototype exposes
  ({ p, errors, cx } = await open({ width: 390, height: 844 }));
  const run = await p.evaluate(() => {
    const g = window.game, P = () => g.state.player, go = (k, until, max) => { g.hold(k, true); let n = 0; while (!until() && n++ < max) g.step(1); g.hold(k, false); };
    g.start();
    go("ArrowRight", () => P().x >= 340, 400); go("ArrowUp", () => P().y <= 420, 900); go("ArrowRight", () => P().x >= 860, 600);
    go("ArrowUp", () => P().y <= 290, 300); go("ArrowRight", () => g.state.delivered >= 1 || g.state.over, 300);
    g.render(0.016);
    return { delivered: g.state.delivered, patrols: g.state.patrols.length, t: g.state.t, crates: document.getElementById("crates").textContent };
  });
  ok("the scripted route delivers a crate in under 20 s and a second patrol arrives", run.delivered === 1 && run.patrols === 2 && run.t < 20 && run.crates === "1/5", JSON.stringify(run));
  const won = await p.evaluate(() => { const g = window.game, s = g.state, d = NH.sim.DROP; while (!s.over) { s.carrying = true; s.player.x = d.x; s.player.y = d.y; g.step(1); } g.render(0.016);
    return { scene: g.scene, result: s.result, card: document.getElementById("box").textContent.slice(0, 40) }; });
  ok("five deliveries win and the win card shows", won.scene === "won" && won.result === "won" && /Doors/.test(won.card), JSON.stringify(won));
  const lost = await p.evaluate(() => { const g = window.game; g.start(); const s = g.state; g.step(1); s.carrying = true; s.heat = 99.5; const q = s.patrols[0];
    for (let i = 0; i < 30 + Math.round(NH.sim.tune.graceSeconds * 60) && !s.over; i++) { s.player.x = q.x + Math.cos(q.ang) * 60; s.player.y = q.y + Math.sin(q.ang) * 60; g.step(1); } g.render(0.016);
    return { scene: g.scene, result: s.result }; });
  ok("being watched to 100 heat loses", lost.scene === "lost" && lost.result === "lost-heat", JSON.stringify(lost));
  const late = await p.evaluate(() => { const g = window.game; g.start(); g.state.t = NH.sim.config.clockSeconds.value - 0.02; g.step(4); return { scene: g.scene, result: g.state.result }; });
  ok("the clock running out loses", late.scene === "lost" && late.result === "lost-time", JSON.stringify(late));
  const q = await p.evaluate(() => { const g = window.game; g.start(); const q0 = g.quality; g.frameTime(40, 200); return [q0, g.quality]; });
  ok("slow frames lower the drawing resolution (never the frame rate target)", q[0] === 1 && q[1] < 1, q.join(" -> "));
  ok("no errors during the scripted game", errors.length === 0, errors.join(" | ")); await cx.close();

  // 5. other screen shapes: folded cover screen, unfolded inner screen, landscape
  for (const [name, size] of [["narrow cover screen 344x882", { width: 344, height: 882 }], ["unfolded 690x829", { width: 690, height: 829 }], ["landscape 844x390", { width: 844, height: 390 }]]) {
    ({ p, errors, cx } = await open(size));
    const r = await p.evaluate(() => { const g = window.game; g.start(); g.step(5); g.render(0.016); const c = document.getElementById("c"), d = document.getElementById("drop").getBoundingClientRect(), h = document.getElementById("hud").getBoundingClientRect();
      return { cw: c.clientWidth, ch: c.clientHeight, drop: d.right <= innerWidth + 1 && d.bottom <= innerHeight + 1 && d.width >= 44, hud: h.right <= innerWidth + 1 && h.left >= -1, scroll: document.documentElement.scrollWidth <= innerWidth + 1 }; });
    ok(`${name}: canvas fills the screen, HUD and Drop stay on it, no sideways scroll`, r.cw === size.width && r.ch === size.height && r.drop && r.hud && r.scroll && errors.length === 0, JSON.stringify(r) + errors.join(" | "));
    await cx.close();
  }
  await b.close();
  let failed = 0; for (const [pass, name, detail] of results) { console.log(`  ${pass ? "PASS" : "FAIL"}  ${name}${pass || !detail ? "" : " — " + detail}`); failed += !pass; }
  console.log(`\nBrowser checks: ${results.length - failed} passed, ${failed} failed (of ${results.length}).`); process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
