/* save: the small record of best runs, and the panel on the title card that shows it and its lock.
   The storage and the lock itself are in lock.js. Nothing here leaves the device.

   API (browser: NH.save, Node: module.exports):
     fresh()                 -> an empty record
     add(record, run)        -> a new record with one finished run counted (pure; used by the tests)
     line(record)            -> one sentence describing a record
     init(lock)              -> async; asks once whether the lock can be used here
     finish(S, tune)         -> async; counts a finished game (kept in memory if the save is locked)
     panel()                 -> HTML for the title card
     bind(box, redraw)       -> wires the panel's buttons; redraw() is called after anything changes */
(function (root) {
  "use strict";
  const fresh = () => ({ v: 1, runs: 0, wins: 0, best: null });
  const valid = (r) => !!r && r.v === 1 && Number.isInteger(r.runs) && r.runs >= 0 && Number.isInteger(r.wins) && r.wins >= 0 && r.wins <= r.runs;
  // run = { won, cheated, spare (minutes on the game clock left at the end), heat (0..100) }
  function add(record, run) {
    const r = valid(record) ? { v: 1, runs: record.runs, wins: record.wins, best: record.best ? { spare: record.best.spare, heat: record.best.heat } : null } : fresh();
    r.runs++;
    if (run.won && !run.cheated) { // a run with codes never counts as a win or a best
      r.wins++;
      const b = r.best, spare = Math.max(0, Math.round(run.spare)), heat = Math.max(0, Math.min(100, Math.round(run.heat)));
      if (!b || spare > b.spare || (spare === b.spare && heat < b.heat)) r.best = { spare, heat };
    }
    return r;
  }
  const line = (r) => !valid(r) || !r.runs ? "No runs saved yet." :
    (r.best ? `Best: doors open with ${r.best.spare} min to spare, heat ${r.best.heat}. ` : "") + `${r.wins} win${r.wins === 1 ? "" : "s"} in ${r.runs} run${r.runs === 1 ? "" : "s"}.`;

  let lock = null, can = { ok: false, why: "" }, waiting = [], note = "", busy = false;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  async function init(l) { lock = l; try { can = await lock.available(); } catch (e) { can = { ok: false, why: "The lock could not be checked on this device." }; } }
  async function finish(S, tune) {
    const run = { won: S.result === "won", cheated: !!S.cheated, spare: (tune.clockSeconds - S.t) / tune.clockSeconds * 180, heat: S.heat };
    if (!lock) return;
    if (lock.status() === "locked") { waiting.push(run); return; }          // counted when the save is next unlocked on this visit
    try { await lock.write(add(lock.read(), run)); } catch (e) { note = e.message; }
  }
  async function flush() { if (!waiting.length) return; let r = lock.read(); for (const run of waiting) r = add(r, run); waiting = []; await lock.write(r); }

  function panel() {
    if (!lock) return "";
    const st = lock.status(), btn = (id, label) => `<button class="ctl inline" id="${id}" type="button"${busy ? " disabled" : ""}>${label}</button>`;
    let body;
    if (st === "locked") body = `<p class="sub">Your saved runs are locked on this device.${waiting.length ? ` ${waiting.length} new run${waiting.length === 1 ? "" : "s"} will be added when you unlock.` : ""}</p><div class="row tight">${btn("sv-open", "Unlock")}${btn("sv-forget", "Forget saved runs")}</div>`;
    else {
      body = `<p class="sub">${esc(line(lock.read()))}</p>`;
      if (st === "open") body += `<p class="sub">Locked to this device, open for this visit.</p><div class="row tight">${btn("sv-off", "Turn lock off")}</div>`;
      else if (can.ok) body += `<div class="row tight">${btn("sv-lock", "Lock saved runs")}</div><p class="sub">Uses this device's own screen lock: fingerprint, face or PIN. Not yet tested on a real phone.</p>`;
      else if (can.why) body += `<p class="sub">Lock: ${esc(can.why)}</p>`;
    }
    return `<div id="savepanel"><h2>Saved runs</h2>${body}<p class="sub" id="sv-note" role="status">${esc(note)}</p></div>`;
  }
  function bind(box, redraw) {
    const on = (id, fn) => { const el = box.querySelector("#" + id); if (el) el.onclick = async () => { if (busy) return; busy = true; note = ""; redraw();
      try { await fn(); } catch (e) { note = e.message || "That did not work."; } busy = false; redraw(); }; };
    on("sv-lock", async () => { await lock.enable(lock.read() || fresh()); note = "Locked. From now on this device asks before showing your saved runs."; });
    on("sv-open", async () => { await lock.open(); await flush(); });
    on("sv-off", async () => { await lock.disable(); note = "Lock off."; });
    on("sv-forget", async () => { lock.forget(); waiting = []; note = "Saved runs deleted."; });
  }
  const save = { fresh, add, line, init, finish, panel, bind };
  if (typeof module !== "undefined" && module.exports) module.exports = save; else (root.NH = root.NH || {}).save = save;
})(typeof window !== "undefined" ? window : globalThis);
