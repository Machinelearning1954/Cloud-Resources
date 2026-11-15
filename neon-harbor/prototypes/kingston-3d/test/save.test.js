#!/usr/bin/env node
/* Tests for src/save.js: how finished runs are counted. `node test/save.test.js`. Runs in CI. */
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const save = require(path.join(__dirname, "..", "src", "save.js"));
const lock = require(path.join(__dirname, "..", "src", "lock.js"));
const results = [], test = (name, fn) => results.push([name, fn]);
const fakeStorage = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => { m.set(k, String(v)); }, removeItem: (k) => { m.delete(k); } }; };
const tune = { clockSeconds: 240 };

test("an empty record reads as no runs", () => { assert.equal(save.line(save.fresh()), "No runs saved yet."); assert.equal(save.line(null), "No runs saved yet."); });
test("a clean win counts and sets the best", () => {
  const r = save.add(save.fresh(), { won: true, cheated: false, spare: 41.6, heat: 12.2 });
  assert.deepEqual(r, { v: 1, runs: 1, wins: 1, best: { spare: 42, heat: 12 } });
  assert.equal(save.line(r), "Best: doors open with 42 min to spare, heat 12. 1 win in 1 run.");
});
test("a loss counts as a run, not a win, and never changes the best", () => {
  let r = save.add(save.fresh(), { won: true, cheated: false, spare: 30, heat: 5 });
  r = save.add(r, { won: false, cheated: false, spare: 0, heat: 100 });
  assert.deepEqual(r, { v: 1, runs: 2, wins: 1, best: { spare: 30, heat: 5 } });
});
test("a win with codes never counts as a win or a best", () => {
  const r = save.add(save.fresh(), { won: true, cheated: true, spare: 170, heat: 0 });
  assert.deepEqual(r, { v: 1, runs: 1, wins: 0, best: null });
});
test("the best is the most time to spare; a tie goes to the lower heat", () => {
  let r = save.add(save.fresh(), { won: true, cheated: false, spare: 30, heat: 40 });
  r = save.add(r, { won: true, cheated: false, spare: 20, heat: 0 }); assert.deepEqual(r.best, { spare: 30, heat: 40 });
  r = save.add(r, { won: true, cheated: false, spare: 30, heat: 10 }); assert.deepEqual(r.best, { spare: 30, heat: 10 });
  r = save.add(r, { won: true, cheated: false, spare: 55, heat: 90 }); assert.deepEqual(r.best, { spare: 55, heat: 90 });
});
test("add never changes the record it was given, and repairs a damaged one", () => {
  const before = save.add(save.fresh(), { won: true, cheated: false, spare: 30, heat: 5 }), copy = JSON.stringify(before);
  save.add(before, { won: true, cheated: false, spare: 99, heat: 0 }); assert.equal(JSON.stringify(before), copy);
  for (const bad of [null, {}, { v: 2, runs: 3, wins: 1 }, { v: 1, runs: 1, wins: 5, best: null }, { v: 1, runs: -1, wins: 0 }, "x"])
    assert.deepEqual(save.add(bad, { won: false }), { v: 1, runs: 1, wins: 0, best: null });
});
test("a finished game is stored through the lock module (plain when the lock is off)", async () => {
  lock.use(fakeStorage()); await save.init(lock);
  await save.finish({ result: "won", cheated: false, t: 180, heat: 20 }, tune);
  assert.deepEqual(lock.read(), { v: 1, runs: 1, wins: 1, best: { spare: 45, heat: 20 } });
  await save.finish({ result: "lost-heat", cheated: false, t: 100, heat: 100 }, tune);
  assert.equal(lock.read().runs, 2); assert.equal(lock.read().wins, 1);
});
test("the panel escapes text and shows why the lock is unavailable outside a browser", async () => {
  lock.use(fakeStorage()); await save.init(lock);
  const html = save.panel(); assert.ok(html.includes("No runs saved yet.")); assert.ok(/Lock: /.test(html)); assert.ok(!html.includes("sv-lock"));
});
test("while the save is locked, finished games wait instead of overwriting it", async () => {
  const s = fakeStorage(); lock.use(s);
  s.setItem("nh.k3d.save", JSON.stringify({ mode: "sealed", cred: "AAAA", prfSalt: "AAAA", blob: await lock.core.seal(new Uint8Array(32).fill(3), save.fresh()) }));
  const sealed = s.getItem("nh.k3d.save"); await save.init(lock);
  await save.finish({ result: "won", cheated: false, t: 100, heat: 0 }, tune);
  assert.equal(s.getItem("nh.k3d.save"), sealed); assert.ok(save.panel().includes("1 new run will be added when you unlock"));
});

(async () => {
  let failed = 0;
  for (const [name, fn] of results) { try { await fn(); console.log(`  PASS  ${name}`); } catch (e) { failed++; console.log(`  FAIL  ${name} — ${String(e.message).split("\n")[0]}`); } }
  console.log(`\nSave tests: ${results.length - failed} passed, ${failed} failed (of ${results.length}).`);
  process.exit(failed ? 1 : 0);
})();
