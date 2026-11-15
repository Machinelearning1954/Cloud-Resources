#!/usr/bin/env node
/* Tests for the crypto and storage parts of src/lock.js. No dependencies: `node test/lock.test.js`.
   These run in CI. The passkey step itself needs a browser: see dev/lock-check.js. */
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const lock = require(path.join(__dirname, "..", "src", "lock.js"));
const { core } = lock;

const results = [];
const test = (name, fn) => results.push([name, fn]);
const secret = (fill) => new Uint8Array(32).fill(fill);
const fakeStorage = () => { const m = new Map(); return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => { m.set(k, String(v)); }, removeItem: (k) => { m.delete(k); }, _m: m }; };
const SAVE = { wins: 3, best: { spare: 42, heat: 7 }, note: "Zinc Roof Soundhall — ünïcode ok" };

test("a sealed save opens again with the same secret", async () => {
  const blob = await core.seal(secret(7), SAVE);
  assert.deepEqual(await core.unseal(secret(7), blob), SAVE);
});
test("the sealed form does not contain the save's text", async () => {
  const blob = await core.seal(secret(7), SAVE), raw = JSON.stringify(blob);
  assert.ok(!raw.includes("wins") && !raw.includes("Soundhall"));
  assert.ok(!Buffer.from(blob.ct, "base64").toString("latin1").includes("wins"));
});
test("a different secret cannot open it", async () => {
  const blob = await core.seal(secret(7), SAVE);
  await assert.rejects(core.unseal(secret(8), blob), /could not be opened/);
});
test("changing one byte of the stored data is detected", async () => {
  const blob = await core.seal(secret(7), SAVE);
  for (const field of ["ct", "iv", "salt"]) {
    const bytes = Buffer.from(blob[field], "base64"); bytes[0] ^= 1;
    await assert.rejects(core.unseal(secret(7), Object.assign({}, blob, { [field]: bytes.toString("base64") })), /could not be opened/, field);
  }
});
test("sealing the same save twice never repeats a nonce, a salt or the output", async () => {
  const seen = new Set();
  for (let i = 0; i < 200; i++) { const b = await core.seal(secret(7), SAVE); for (const f of ["iv", "salt", "ct"]) { assert.ok(!seen.has(f + b[f]), f); seen.add(f + b[f]); } }
});
test("the nonce is 12 bytes, the salt 16, and the key is 256-bit AES-GCM with a 16-byte tag", async () => {
  const b = await core.seal(secret(7), { a: 1 });
  assert.equal(Buffer.from(b.iv, "base64").length, 12); assert.equal(Buffer.from(b.salt, "base64").length, 16);
  assert.equal(Buffer.from(b.ct, "base64").length, JSON.stringify({ a: 1 }).length + 16);
});
test("an independent implementation opens it: HKDF-SHA-256 into AES-256-GCM, with the bound label", async () => {
  // Decrypt with Node's own crypto, not with lock.js. This fails if the key is not 256-bit,
  // if the derivation changes, or if the label that binds the data to this save format is dropped.
  const nodeCrypto = require("node:crypto"), LABEL = Buffer.from("nh-crate-run-3d save v1");
  const blob = await core.seal(secret(7), SAVE), ct = Buffer.from(blob.ct, "base64");
  const key = Buffer.from(nodeCrypto.hkdfSync("sha256", secret(7), Buffer.from(blob.salt, "base64"), LABEL, 32));
  const d = nodeCrypto.createDecipheriv("aes-256-gcm", key, Buffer.from(blob.iv, "base64"));
  d.setAAD(LABEL); d.setAuthTag(ct.subarray(ct.length - 16));
  const plain = Buffer.concat([d.update(ct.subarray(0, ct.length - 16)), d.final()]).toString("utf8");
  assert.deepEqual(JSON.parse(plain), SAVE);
});
test("a short or missing secret is refused, never padded", async () => {
  await assert.rejects(core.seal(new Uint8Array(16), SAVE), /no usable secret/);
  await assert.rejects(core.seal(null, SAVE), /no usable secret/);
});
test("a save from another version is refused with a clear message", async () => {
  const blob = await core.seal(secret(7), SAVE);
  await assert.rejects(core.unseal(secret(7), Object.assign({}, blob, { v: 2 })), /different version/);
});
test("plain mode: write, read, forget", async () => {
  lock.use(fakeStorage()); assert.equal(lock.status(), "none"); assert.equal(lock.read(), null);
  await lock.write(SAVE); assert.equal(lock.status(), "plain"); assert.deepEqual(lock.read(), SAVE);
  lock.forget(); assert.equal(lock.status(), "none"); assert.equal(lock.read(), null);
});
test("a locked save cannot be read or overwritten without opening it", async () => {
  const s = fakeStorage(); lock.use(s);
  s.setItem("nh.k3d.save", JSON.stringify({ mode: "sealed", cred: "AAAA", prfSalt: "AAAA", blob: await core.seal(secret(7), SAVE) }));
  const before = s.getItem("nh.k3d.save");
  assert.equal(lock.status(), "locked"); assert.equal(lock.read(), null);
  await assert.rejects(lock.write({ wins: 99 }), /locked/);
  assert.equal(s.getItem("nh.k3d.save"), before, "what is stored must not change");
  lock.forget(); assert.equal(lock.status(), "none");
});
test("broken or blocked storage never throws", async () => {
  lock.use({ getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }, removeItem() { throw new Error("blocked"); } });
  assert.equal(lock.status(), "none"); assert.equal(lock.read(), null); await lock.write(SAVE); lock.forget();
  lock.use({ getItem: () => "{not json", setItem() {}, removeItem() {} }); assert.equal(lock.status(), "none");
});
test("outside a browser the lock reports that it is not available, with a reason", async () => {
  const a = await lock.available(); assert.equal(a.ok, false); assert.ok(a.why.length > 10);
});

(async () => {
  let failed = 0;
  for (const [name, fn] of results) { try { await fn(); console.log(`  PASS  ${name}`); } catch (e) { failed++; console.log(`  FAIL  ${name} — ${String(e.message).split("\n")[0]}`); } }
  console.log(`\nLock tests: ${results.length - failed} passed, ${failed} failed (of ${results.length}).`);
  process.exit(failed ? 1 : 0);
})();
