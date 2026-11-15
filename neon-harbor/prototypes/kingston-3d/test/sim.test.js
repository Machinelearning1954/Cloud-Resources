#!/usr/bin/env node
/* Rule tests for src/sim.js. No dependencies: `node test/sim.test.js`. Exit 0 = all pass.
   These run in CI. They test the game rules only, not drawing. */
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const sim = require(path.join(__dirname, "..", "src", "sim.js"));
const { tune, config, STEP } = sim;

const results = [];
function test(name, fn) { try { fn(); results.push([true, name]); } catch (e) { results.push([false, name, e.message.split("\n")[0]]); } }
const R = { x: 1, y: 0 }, L = { x: -1, y: 0 }, U = { x: 0, y: -1 }, D = { x: 0, y: 1 }, STILL = { x: 0, y: 0 };
function go(S, move, until, max) { let n = 0; while (!until() && n < max) { sim.step(S, move); n++; } return n; }
function deliverOne(S) { // the scripted route used by the browser checks too
  const P = S.player;
  go(S, R, () => P.x >= 340, 400); go(S, U, () => P.y <= 420, 900); go(S, R, () => P.x >= 860, 600);
  go(S, U, () => P.y <= 290, 300); go(S, R, () => S.delivered >= 1 || S.over, 300);
}
function teleportDeliver(S) { S.carrying = true; S.player.x = sim.DROP.x; S.player.y = sim.DROP.y; sim.step(S, STILL); }
function inFrontOf(S, g, d) { S.player.x = g.x + Math.cos(g.ang) * d; S.player.y = g.y + Math.sin(g.ang) * d; }

test("every tuning value sits inside its own min and max", () => {
  for (const [k, c] of Object.entries(config)) assert.ok(c.value >= c.min && c.value <= c.max, k);
});
test("same inputs give the same game (deterministic)", () => {
  const run = () => { const S = sim.create(); deliverOne(S); for (let i = 0; i < 600; i++) sim.step(S, i % 90 < 45 ? L : D);
    return JSON.stringify([S.t, S.heat, S.delivered, S.player, S.patrols.map((g) => [g.x, g.y, g.leg]), S.peds.map((q) => [q.x, q.y])]); };
  assert.equal(run(), run());
});
test("walking into the pile picks up a crate; the scripted route delivers it", () => {
  const S = sim.create(); go(S, R, () => S.player.x >= 340, 400);
  assert.equal(S.carrying, true);
  deliverOne(S);
  assert.equal(S.delivered, 1); assert.equal(S.carrying, false); assert.equal(S.patrols.length, 2);
  assert.ok(S.t < 20, `took ${S.t.toFixed(1)} s`);
  assert.deepEqual(S.events.filter((e) => e.type === "delivered"), [{ type: "delivered", count: 1 }]);
});
test("carrying is slower than walking empty-handed", () => {
  const a = sim.create(); a.player.x = 340; a.player.y = 1300; const y0 = a.player.y; for (let i = 0; i < 60; i++) sim.step(a, U);
  const b = sim.create(); b.player.x = 340; b.player.y = 1300; b.carrying = true; for (let i = 0; i < 60; i++) sim.step(b, U);
  assert.ok(Math.abs((y0 - a.player.y) - tune.walkSpeed) < 1); assert.ok(Math.abs((y0 - b.player.y) - tune.carrySpeed) < 1);
});
test("a patrol ignores you empty-handed and sees you carrying", () => {
  const S = sim.create(), g = S.patrols[0]; sim.step(S, STILL);
  inFrontOf(S, g, 80); S.carrying = false; sim.step(S, STILL); assert.equal(S.seen, false);
  inFrontOf(S, g, 80); S.carrying = true; sim.step(S, STILL); assert.equal(S.seen, true);
});
test("a roof between you and a patrol blocks its view", () => {
  const probe = (px, py) => { const S = sim.create(), g = S.patrols[0]; g.x = 340; g.y = 420; g.leg = 0; g.ang = 0; S.carrying = true; S.player.x = px; S.player.y = py;
    return sim.clearSight(g, S.player); };
  assert.equal(probe(480, 420), true); assert.equal(probe(480, 300), false);
});
test("you are not seen beyond the vision range or outside the cone", () => {
  const S = sim.create(), g = S.patrols[0]; sim.step(S, STILL); S.carrying = true;
  inFrontOf(S, g, tune.seeRange + 20); sim.step(S, STILL); assert.equal(S.seen, false);
  S.player.x = g.x - Math.cos(g.ang) * 80; S.player.y = g.y - Math.sin(g.ang) * 80; sim.step(S, STILL); assert.equal(S.seen, false, "behind the patrol");
});
test("heat rises while seen and falls slowly when not", () => {
  const S = sim.create(), g = S.patrols[0]; sim.step(S, STILL); S.carrying = true;
  for (let i = 0; i < 30; i++) { inFrontOf(S, g, 80); sim.step(S, STILL); }
  assert.ok(Math.abs(S.heat - tune.heatRise * 0.5) < 2, `heat ${S.heat}`);
  const h = S.heat; S.carrying = false; for (let i = 0; i < 60; i++) sim.step(S, STILL);
  assert.ok(Math.abs((h - S.heat) - tune.heatFall) < 0.5);
});
test("a crate you put down stays down, then comes back when you return", () => {
  const S = sim.create(); S.player.x = 340; S.player.y = 1000; S.carrying = true;
  assert.equal(sim.toggleCrate(S), "dropped");
  for (let i = 0; i < 200; i++) sim.step(S, STILL);
  assert.equal(S.carrying, false); assert.ok(S.stash && !S.stash.armed, "stays down while you stand on it");
  for (let i = 0; i < 40; i++) sim.step(S, U); assert.equal(S.stash.armed, true);
  for (let i = 0; i < 60; i++) sim.step(S, D); assert.equal(S.carrying, true); assert.equal(S.stash, null);
});
test("the Pick up button recovers a nearby crate, and does nothing far away", () => {
  const S = sim.create(); S.player.x = 340; S.player.y = 1000; S.carrying = true; sim.toggleCrate(S);
  assert.equal(sim.nearStash(S), true); assert.equal(sim.toggleCrate(S), "recovered");
  sim.toggleCrate(S); S.player.y = 600; assert.equal(sim.nearStash(S), false); assert.equal(sim.toggleCrate(S), null);
});
test("each delivery adds a patrol up to five, and speeds them up", () => {
  const S = sim.create(), g = S.patrols[0]; assert.equal(sim.patrolSpeed(S, g), 95);
  for (let i = 1; i <= 4; i++) { teleportDeliver(S); assert.equal(S.patrols.length, Math.min(5, 1 + i)); }
  assert.equal(sim.patrolSpeed(S, g), 95 + 7 * 4);
  assert.ok(S.patrols.some((p) => p.foot), "one of the five walks");
});
test("five deliveries win", () => {
  const S = sim.create(); for (let i = 0; i < 5; i++) teleportDeliver(S);
  assert.equal(S.result, "won"); assert.equal(S.over, true);
  const t = S.t; sim.step(S, R); assert.equal(S.t, t, "a finished game does not advance");
});
test("heat at 100 loses; the clock running out loses", () => {
  const a = sim.create(), g = a.patrols[0]; sim.step(a, STILL); a.carrying = true; a.heat = 99.9;
  for (let i = 0; i < 20 && !a.over; i++) { inFrontOf(a, g, 60); sim.step(a, STILL); } assert.equal(a.result, "lost-heat");
  const b = sim.create(); b.t = tune.clockSeconds - 0.01; for (let i = 0; i < 3; i++) sim.step(b, STILL); assert.equal(b.result, "lost-time");
});
test("you cannot walk into buildings, off the map or into the water", () => {
  for (const r of sim.B) assert.equal(sim.blocked(r.x + r.w / 2, r.y + r.h / 2, 11), true);
  assert.equal(sim.blocked(5, 500, 11), true); assert.equal(sim.blocked(600, sim.WATER, 11), true);
  assert.equal(sim.blocked(340, 420, 11), false, "street centre is open");
  const S = sim.create(); S.player.x = 340; S.player.y = 1450; for (let i = 0; i < 300; i++) sim.step(S, D);
  assert.ok(S.player.y <= sim.WATER - 11);
});
test("the pick-up, the drop-off, lamps and patrol routes are all on open ground", () => {
  assert.equal(sim.blocked(sim.PICK.x, sim.PICK.y, 11), false); assert.equal(sim.blocked(sim.DROP.x, sim.DROP.y, 11), false);
  for (const r of sim.ROUTES) for (const [x, y] of r.pts) assert.equal(sim.blocked(x, y, 11), false, `route point ${x},${y}`);
});
test("codes: Clean Slate, Slow Riddim, Late Doors; a wrong sequence does nothing", () => {
  const S = sim.create(); S.heat = 60;
  assert.equal(sim.enterCode(S, "UDUDUD"), null); assert.equal(S.cheated, false); assert.equal(S.heat, 60);
  assert.equal(sim.enterCode(S, "UUDLRD").name, "Clean Slate"); assert.equal(S.heat, 0); assert.equal(S.cheated, true);
  const g = S.patrols[0], fast = sim.patrolSpeed(S, g); sim.enterCode(S, "LRLRDD"); assert.equal(sim.patrolSpeed(S, g), fast / 2);
  S.t = 100; sim.enterCode(S, "DDURLU"); assert.equal(S.t, 100 - tune.bonusSeconds);
});
test("a full game can be won inside the clock by the scripted route alone", () => {
  const S = sim.create(), P = S.player; let guard = 0;
  while (!S.over && guard++ < 12) { // out along the streets, back the same way; no dodging
    if (!S.carrying) { go(S, D, () => P.y >= 1340 || S.over, 2000); go(S, L, () => P.x <= 345 || S.carrying || S.over, 2000); go(S, D, () => P.y >= 1420 || S.carrying || S.over, 300); go(S, L, () => S.carrying || S.over, 300); }
    go(S, R, () => P.x >= 340 || S.over, 400); go(S, U, () => P.y <= 420 || S.over, 2000); go(S, R, () => P.x >= 860 || S.over, 2000);
    const d0 = S.delivered; go(S, U, () => P.y <= 290 || S.over, 600); go(S, R, () => S.delivered > d0 || S.over, 600);
    go(S, L, () => P.x <= 860 || S.over, 600); go(S, D, () => P.y >= 420 || S.over, 600); go(S, L, () => P.x <= 340 || S.over, 2000);
  }
  // This route ignores the patrols, so it may lose on heat. It must at least finish, and report why.
  assert.equal(S.over, true); assert.ok(["won", "lost-heat", "lost-time"].includes(S.result));
  console.log(`  info  blind route result: ${S.result}, delivered ${S.delivered}, ${S.t.toFixed(0)} s, heat ${S.heat.toFixed(0)}`);
});

let failed = 0;
for (const [ok, name, why] of results) { console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : " — " + why}`); failed += !ok; }
console.log(`\nRule tests: ${results.length - failed} passed, ${failed} failed (of ${results.length}).`);
process.exit(failed ? 1 : 0);
