#!/usr/bin/env node
/* Browser checks for the save lock (src/lock.js), using a SOFTWARE authenticator that the test
   browser provides. This proves the page's own logic: it is not a test on a real phone or a real
   fingerprint reader.   node dev/lock-check.js [page.html]     (default: the built index.html)
   Needs Playwright, so it is run by hand, not in CI (CI runs test/lock.test.js). */
"use strict";
const http = require("node:http"), fs = require("node:fs"), path = require("node:path");
const { chromium } = require("playwright");
const lockJs = fs.readFileSync(path.join(__dirname, "..", "src", "lock.js"), "utf8");
const PAGE = `<!doctype html><meta charset="utf-8"><title>lock check</title><script>${lockJs}</script>`;
const results = [], ok = (name, pass, detail) => results.push([!!pass, name, detail]);
const AUTH = { protocol: "ctap2", ctap2Version: "ctap2_1", transport: "internal", hasResidentKey: true, hasUserVerification: true, automaticPresenceSimulation: true };
(async () => {
  const gamePath = path.resolve(process.argv[2] || path.join(__dirname, "..", "index.html"));
  const server = http.createServer((req, res) => { res.writeHead(200, { "content-type": "text/html" }); res.end(req.url.startsWith("/game") ? "<!doctype html><meta charset=\"utf-8\">" + fs.readFileSync(gamePath, "utf8") : PAGE); });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const url = `http://localhost:${server.address().port}/`, b = await chromium.launch();
  const SAVE = { wins: 2, best: { spare: 31, heat: 12 } };
  const call = (p, fn, arg) => p.evaluate(async ({ fn, arg }) => { try { const f = new Function("lock", "arg", "return (async () => { " + fn + " })()"); return { value: await f(window.NH.lock, arg) }; } catch (e) { return { error: e.message }; } }, { fn, arg });

  // 1. a device with a screen lock and a passkey that can derive keys
  let cx = await b.newContext(), p = await cx.newPage(); await p.goto(url);
  let cdp = await cx.newCDPSession(p); await cdp.send("WebAuthn.enable");
  let { authenticatorId } = await cdp.send("WebAuthn.addVirtualAuthenticator", { options: Object.assign({ isUserVerified: true, hasPrf: true }, AUTH) });
  let r = await call(p, "return lock.available()"); ok("a device with a screen lock: the lock is offered", r.value && r.value.ok, JSON.stringify(r));
  r = await call(p, "await lock.enable(arg); return [lock.status(), lock.read(), localStorage.getItem('nh.k3d.save')]", SAVE);
  ok("turning the lock on seals the save", r.value && r.value[0] === "open" && JSON.stringify(r.value[1]) === JSON.stringify(SAVE), JSON.stringify(r).slice(0, 300));
  ok("what is stored on the device does not contain the save in the clear", r.value && !/wins|spare|heat/.test(r.value[2]) && /"mode":"sealed"/.test(r.value[2]));
  await p.reload();
  r = await call(p, "return [lock.status(), lock.read()]"); ok("after a reload the save is locked and unreadable", r.value && r.value[0] === "locked" && r.value[1] === null, JSON.stringify(r));
  r = await call(p, "try { await lock.write({ wins: 99 }); return 'wrote'; } catch (e) { return e.message; }"); ok("a locked save cannot be overwritten", /locked/.test(r.value), JSON.stringify(r));
  r = await call(p, "const d = await lock.open(); return [lock.status(), d]"); ok("the device check opens it again", r.value && r.value[0] === "open" && JSON.stringify(r.value[1]) === JSON.stringify(SAVE), JSON.stringify(r));
  r = await call(p, "await lock.write({ wins: 3 }); return localStorage.getItem('nh.k3d.save')"); ok("a new result is stored sealed too", r.value && /"mode":"sealed"/.test(r.value) && !/wins/.test(r.value));
  await p.reload(); r = await call(p, "return lock.open()"); ok("and comes back after another reload", r.value && r.value.wins === 3, JSON.stringify(r));

  // 2. lock off again, and forget
  r = await call(p, "await lock.disable(); return [lock.status(), lock.read()]"); ok("turning the lock off stores the save plain again", r.value && r.value[0] === "plain" && r.value[1].wins === 3, JSON.stringify(r));
  r = await call(p, "lock.forget(); return [lock.status(), localStorage.getItem('nh.k3d.save')]"); ok("forget deletes the save", r.value && r.value[0] === "none" && r.value[1] === null);

  // 3. the passkey is gone (deleted from the device): the sealed save cannot be opened, forget still works
  r = await call(p, "await lock.enable(arg); return lock.status()", SAVE); await p.reload();
  await cdp.send("WebAuthn.clearCredentials", { authenticatorId });
  const shut = await call(p, "try { await lock.open(); return 'opened'; } catch (e) { return e.message; }");
  ok("with the passkey deleted, the save stays shut", r.value === "open" && shut.value !== "opened" && (await call(p, "return lock.status()")).value === "locked", JSON.stringify([r, shut]));
  r = await call(p, "lock.forget(); return lock.status()"); ok("and can still be thrown away to start over", r.value === "none");

  // 3b. the device refuses (wrong finger, cancelled). Last on this authenticator: the test browser's software
  //     authenticator keeps refusing once it has refused, which a real device does not.
  r = await call(p, "await lock.enable(arg); return lock.status()", SAVE); await p.reload();
  await cdp.send("WebAuthn.setUserVerified", { authenticatorId, isUserVerified: false });
  const refused = await call(p, "try { await lock.open(); return 'opened'; } catch (e) { return e.message; }");
  ok("a failed device check leaves the save locked, with a plain message", r.value === "open" && /cancelled or refused/.test(refused.value) && (await call(p, "return [lock.status(), lock.read()]")).value.join() === "locked,", JSON.stringify([r, refused]));
  await cx.close();

  // 4. a device whose passkeys cannot derive keys: the lock stays off and nothing is half-written
  cx = await b.newContext(); p = await cx.newPage(); await p.goto(url); cdp = await cx.newCDPSession(p); await cdp.send("WebAuthn.enable");
  await cdp.send("WebAuthn.addVirtualAuthenticator", { options: Object.assign({ isUserVerified: true }, AUTH) });
  r = await call(p, "await lock.write(arg); try { await lock.enable(arg); return 'enabled'; } catch (e) { return [e.message, lock.status(), lock.read()]; }", SAVE);
  ok("a device without key derivation: the lock stays off and the plain save is untouched", Array.isArray(r.value) && /cannot derive/.test(r.value[0]) && r.value[1] === "plain" && r.value[2].wins === 2, JSON.stringify(r).slice(0, 300));
  await cx.close();

  // 5. no screen lock at all, and a page opened from a file
  cx = await b.newContext(); p = await cx.newPage(); await p.goto(url);
  r = await call(p, "return lock.available()"); ok("no screen lock on the device: not offered, with the reason", r.value && !r.value.ok && /screen lock/.test(r.value.why), JSON.stringify(r));
  const tmp = path.join(fs.mkdtempSync(path.join(require("node:os").tmpdir(), "nh-lock-")), "lock-file-check.html"); fs.writeFileSync(tmp, PAGE);
  await p.goto("file://" + tmp); r = await call(p, "return lock.available()"); ok("opened from a file: not offered, with the reason", r.value && !r.value.ok && /https address/.test(r.value.why), JSON.stringify(r));
  await cx.close();

  // 6. the whole thing through the game's own title card: win a run, lock it, reload, unlock
  cx = await b.newContext({ viewport: { width: 390, height: 844 } }); p = await cx.newPage(); const pageErrors = []; p.on("pageerror", (e) => pageErrors.push(e.message));
  await p.goto(url + "game"); cdp = await cx.newCDPSession(p); await cdp.send("WebAuthn.enable");
  await cdp.send("WebAuthn.addVirtualAuthenticator", { options: Object.assign({ isUserVerified: true, hasPrf: true }, AUTH) });
  await p.reload(); await p.waitForSelector("#savepanel");
  const panel = () => p.evaluate(() => document.getElementById("savepanel").textContent);
  ok("in the game: the title card says there are no saved runs yet", /No runs saved yet/.test(await panel()), await panel());
  await p.evaluate(() => { const g = window.game, d = NH.sim.DROP; g.start(); const s = g.state; while (!s.over) { s.carrying = true; s.player.x = d.x; s.player.y = d.y; g.step(1); } });
  await p.waitForFunction(() => /1 win in 1 run/.test((document.getElementById("savepanel") || {}).textContent || ""), null, { timeout: 5000 }).catch(() => {});
  ok("in the game: a clean win is recorded and shown on the win card", /Best: doors open with \d+ min to spare, heat \d+\. 1 win in 1 run\./.test(await panel()), await panel());
  await p.waitForSelector("#sv-lock", { timeout: 5000 }).catch(() => {}); await p.click("#sv-lock").catch(() => {});
  await p.waitForFunction(() => /Locked to this device/.test(document.getElementById("savepanel").textContent), null, { timeout: 8000 }).catch(() => {});
  ok("in the game: the Lock button seals the saved runs", /Locked to this device/.test(await panel()) && !/wins|spare/.test(await p.evaluate(() => localStorage.getItem("nh.k3d.save"))), await panel());
  await p.reload(); await p.waitForSelector("#sv-open", { timeout: 8000 }).catch(() => {});
  ok("in the game: after a reload the title card shows only that the runs are locked", /locked on this device/.test(await panel()) && !/Best:/.test(await panel()), await panel());
  await p.click("#sv-open").catch(() => {}); await p.waitForFunction(() => /1 win in 1 run/.test(document.getElementById("savepanel").textContent), null, { timeout: 8000 }).catch(() => {});
  ok("in the game: Unlock brings the record back", /1 win in 1 run/.test(await panel()) && pageErrors.length === 0, (await panel()) + pageErrors.join(" | "));
  await cx.close(); await b.close(); server.close();

  let failed = 0; for (const [pass, name, detail] of results) { console.log(`  ${pass ? "PASS" : "FAIL"}  ${name}${pass || !detail ? "" : " — " + detail}`); failed += !pass; }
  console.log(`\nLock browser checks (software authenticator): ${results.length - failed} passed, ${failed} failed (of ${results.length}).`); process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(2); });
