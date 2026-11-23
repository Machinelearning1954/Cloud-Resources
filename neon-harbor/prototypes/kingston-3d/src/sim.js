/* Crate Run — the game rules. No drawing, no DOM, no clock: the same code runs in the
   browser and in Node (test/sim.test.js). Ported 1:1 from prototypes/crate-run so the 2D and
   3D views play the same game. Distances are in "sim pixels"; the 3D view scales them. */
(function (root) {
  "use strict";

  /* ---- tuning: every number the game uses lives here ---- */
  const config = {
    walkSpeed:    { value: 150, min: 80,  max: 260, step: 5 },    // px/s, empty-handed
    carrySpeed:   { value: 115, min: 60,  max: 220, step: 5 },    // px/s, carrying a crate
    cratesToWin:  { value: 5,   min: 1,   max: 8,   step: 1 },
    clockSeconds: { value: 210, min: 60,  max: 600, step: 10 },   // real seconds for 4 PM -> 7 PM
    heatRise:     { value: 34,  min: 5,   max: 100, step: 1 },    // heat per second while seen carrying
    heatFall:     { value: 4,   min: 0,   max: 30,  step: 1 },    // heat per second while unseen
    seeRange:     { value: 230, min: 80,  max: 500, step: 10 },   // patrol vision distance, px
    seeHalfAngle: { value: 32,  min: 10,  max: 80,  step: 1 },    // degrees either side of facing
    patrolSpeed:  { value: 95,  min: 30,  max: 220, step: 5 },    // px/s for the first patrol
    patrolRamp:   { value: 7,   min: 0,   max: 40,  step: 1 },    // extra px/s per crate delivered
    maxPatrols:   { value: 5,   min: 1,   max: 5,   step: 1 },
    pickRadius:   { value: 34,  min: 16,  max: 80,  step: 2 },
    slowSeconds:  { value: 30,  min: 5,   max: 120, step: 5 },    // how long the Slow Riddim code lasts
    bonusSeconds: { value: 30,  min: 5,   max: 120, step: 5 },    // clock time the Late Doors code gives back
    chasePercent: { value: 100, min: 100, max: 200, step: 5 },    // a hunting patrol moves at this % of its normal speed
    searchSeconds:{ value: 4,   min: 1,   max: 15,  step: 1 },    // how long a patrol looks around where it lost you
    changeSeconds:{ value: 1.5, min: 0.5, max: 5,   step: 0.5 },  // standing still at a clothesline this long changes your shirt
    graceSeconds: { value: 1.5, min: 0,   max: 5,   step: 0.5 },  // once spotted, heat waits this long before it climbs: time to break the sight line
  };
  const tune = new Proxy({}, { get: (_, k) => config[k].value });

  /* ---- the block, in sim pixels (x right, y toward the harbour) ---- */
  const W = 1200, H = 1600, QUAY = 1380, WATER = 1470, STEP = 1 / 60, PLAYER_RADIUS = 11;
  const PATROL_RADIUS = 10, HOLD = 30, STUCK = 90;   // patrols keep HOLD px from the carrier; STUCK steps without progress gives up
  const ROOF = ["#8fa3ad", "#a39a8b", "#7f9a8f", "#a98f7a", "#8c93a8", "#9aa58a"];
  const WALL = ["#d8c9a8", "#c9d6cf", "#e2b98f", "#b9c4d6", "#d6c2c0"];
  const B = [ // buildings: x, y, w, h. Gaps between them are alleys.
    [40,40,260,150],[40,230,110,150],[190,230,110,150],
    [380,40,200,340],[620,40,200,160],[620,240,200,140],
    [900,40,260,150],
    [40,460,260,200],[40,700,120,200],[200,700,100,200],
    [380,460,200,200],[620,460,200,200],[380,700,440,200],
    [900,460,260,130],[900,630,110,270],[1050,630,110,270],
    [40,980,260,140],[40,1160,260,140],
    [380,980,200,320],[620,980,200,150],[620,1170,200,130],
    [900,980,260,140],[900,1160,260,140],
    [40,1392,170,66],
  ].map(([x, y, w, h], i) => ({ i, x, y, w, h, c: ROOF[i % ROOF.length], tall: 1 + (i * 37 % 10) / 14, wall: WALL[i % WALL.length] }));
  const SOUNDHALL = B[6], BOATWORKS = B[23];
  const PICK = { x: 300, y: 1425 }, DROP = { x: 1030, y: 290 };
  const LINES = [[600, 968], [620, 410]];   // clotheslines where you can change your shirt, so patrols stop knowing you
  const LAMPS = [[340,420],[860,420],[340,940],[860,940],[340,1340],[860,1340],[600,420],[600,940],[600,1340],[1030,330]];
  const STREETS = { vertical: [340, 860], horizontal: [420, 940, 1340] };   // street centre lines
  const PALMS = [[380,1452],[555,1452],[800,1452],[900,1452],[1110,1452],[915,362],[1142,362]];
  const SKIFFS = [[470, "#c8573a"], [640, "#e8b23a"], [1010, "#3aa0c8"]];    // x, hull colour; moored at WATER + 44
  const ROUTES = [ // patrol loops along the street centre lines; `foot` walks instead of drives
    { pts: [[340,420],[860,420],[860,940],[340,940]] },
    { pts: [[860,940],[860,1340],[340,1340],[340,940]] },
    { pts: [[860,420],[340,420],[340,1340],[860,1340]] },
    { pts: [[120,1340],[1100,1340]], foot: true },
    { pts: [[860,940],[340,940],[340,420],[860,420]] },
  ];
  const CAST = [ // hair ids come from the project's hair library (ADR 0051)
    { id: "imani",  name: "Imani Vassell",          hair: "locs-updo",     skin: "#7a4a2b", shirt: "#e8b23a", note: "you" },
    { id: "omari",  name: "Omari 'Bassline' Grant", hair: "locs-freeform", skin: "#5c3620", shirt: "#3aa0c8", note: "sound engineer" },
    { id: "nadine", name: "Nadine 'Brass' Whyte",   hair: "locs-wrap",     skin: "#6b4026", shirt: "#c8573a", note: "boatwright" },
    { id: "ped-waves",    name: "Passer-by", hair: "waves",    skin: "#6a3f25", shirt: "#7fbf6a", note: "360 waves" },
    { id: "ped-afro",     name: "Passer-by", hair: "afro",     skin: "#8a5a36", shirt: "#b07ad0", note: "rounded afro" },
    { id: "ped-canerows", name: "Passer-by", hair: "canerows", skin: "#54311c", shirt: "#e07a9a", note: "canerows" },
    { id: "patrol", name: "Foot patrol",            hair: "cap",           skin: "#7a4a2b", shirt: "#2c3f66", note: "sees you carrying" },
    { id: "kemar",  name: "Kemar 'Tide' Bailey",    hair: "waves",         skin: "#4f2e1b", shirt: "#16161a", pattern: "#f1eee6", note: "you, if you choose: printed shirts" },
  ];
  const NPCS = [ // people who stand in one place
    { cast: 1, x: 1100, y: 250, ang: Math.PI },   // Omari at the rig
    { cast: 2, x: 232,  y: 1436, ang: 0 },        // Nadine at the yard
  ];
  // Demo codes for the prototypes only (ADR 0058): six directions, entered while paused.
  const CODES = [
    { seq: "UUDLRD", name: "Clean Slate", does: "heat cleared",          run: (S) => { S.heat = 0; } },
    { seq: "LRLRDD", name: "Slow Riddim", does: "patrols at half speed", run: (S) => { S.slowUntil = S.t + tune.slowSeconds; } },
    { seq: "DDURLU", name: "Late Doors",  does: "clock wound back",      run: (S) => { S.t = Math.max(0, S.t - tune.bonusSeconds); S.slowUntil = Math.max(0, S.slowUntil - tune.bonusSeconds); } },
  ];

  /* ---- helpers ---- */
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  function rnd(S) { // deterministic: the seed lives in the state
    S.seed |= 0; S.seed = S.seed + 0x6D2B79F5 | 0; let t = Math.imul(S.seed ^ S.seed >>> 15, 1 | S.seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
  function segHitsRect(x1, y1, x2, y2, r) { // does the sight line cross a building?
    let t0 = 0, t1 = 1; const dx = x2 - x1, dy = y2 - y1;
    for (const [p, q] of [[-dx, x1 - r.x], [dx, r.x + r.w - x1], [-dy, y1 - r.y], [dy, r.y + r.h - y1]]) {
      if (p === 0) { if (q < 0) return false; continue; }
      const t = q / p;
      if (p < 0) { if (t > t1) return false; if (t > t0) t0 = t; } else { if (t < t0) return false; if (t < t1) t1 = t; }
    }
    return true;
  }
  const clearSight = (a, b) => !B.some((r) => segHitsRect(a.x, a.y, b.x, b.y, r));
  function blocked(x, y, rad) {
    if (x < rad || x > W - rad || y < rad || y > WATER - rad) return true;
    return B.some((r) => x > r.x - rad && x < r.x + r.w + rad && y > r.y - rad && y < r.y + r.h + rad);
  }
  function stepToward(g, tx, ty, speed) { // move a patrol toward a point, sliding along walls; true once it arrives
    const d = Math.hypot(tx - g.x, ty - g.y);
    if (d < 4) return true;
    const s = Math.min(d, speed * STEP), nx = g.x + (tx - g.x) / d * s, ny = g.y + (ty - g.y) / d * s, ox = g.x, oy = g.y;
    if (!blocked(nx, g.y, PATROL_RADIUS)) g.x = nx;
    if (!blocked(g.x, ny, PATROL_RADIUS)) g.y = ny;
    g.ang = Math.atan2(ty - oy, tx - ox);
    g.stuck = Math.hypot(g.x - ox, g.y - oy) < 0.05 ? (g.stuck || 0) + 1 : 0;
    return false;
  }
  function nearestPoint(pts, x, y) {
    let best = 0;
    for (let i = 1; i < pts.length; i++) if (Math.hypot(pts[i][0] - x, pts[i][1] - y) < Math.hypot(pts[best][0] - x, pts[best][1] - y)) best = i;
    return best;
  }
  const TIERS = [0, 34, 67];                            // heat at which the street turns calm -> alert -> lockdown
  function heatTier(S) { return S.heat >= TIERS[2] ? 2 : S.heat >= TIERS[1] ? 1 : 0; }
  function patrolSpeed(S, g) {
    return (1 + 0.1 * heatTier(S)) * (tune.patrolSpeed + tune.patrolRamp * S.delivered) * (g.foot ? 0.55 : 1) * (S.t < S.slowUntil ? 0.5 : 1);
  }

  /* ---- state ---- */
  function newPatrol(i) {
    const r = ROUTES[i % ROUTES.length];
    return { id: i, route: r, foot: !!r.foot, leg: 0, dir: 1, x: r.pts[0][0], y: r.pts[0][1], ang: 0, sees: false,
             mode: "route", timer: 0, memory: null, homing: false, stuck: 0 };
  }
  function create() {
    const S = { seed: 20261114, t: 0, heat: 0, delivered: 0, carrying: false, stash: null, seen: false, result: "", over: false,
                cheated: false, slowUntil: 0, seenFor: 0, tier: 0, known: false, outfit: 0, changing: 0, events: [],
                player: { x: 240, y: 1425, ang: -Math.PI / 2, walk: 0, moving: false },
                patrols: [newPatrol(0)], peds: [] };
    for (let i = 0; i < 8; i++) {
      const l = LAMPS[Math.floor(rnd(S) * 9)];
      S.peds.push({ id: i, x: l[0] + 28, y: l[1] + 28, tx: l[0] + 28, ty: l[1] + 28, ang: 0, walk: 0, cast: 3 + i % 3, sp: 40 + rnd(S) * 25 });
    }
    return S;
  }

  /* ---- actions ---- */
  function toggleCrate(S) { // put the crate down, or pick a stashed one back up; returns what happened
    if (S.over) return null;
    if (S.carrying) { S.carrying = false; S.stash = { x: S.player.x, y: S.player.y, armed: false }; S.events.push({ type: "dropped" }); return "dropped"; }
    if (S.stash && dist(S.player, S.stash) < tune.pickRadius + 10) { S.carrying = true; S.stash = null; S.events.push({ type: "recovered" }); return "recovered"; }
    return null;
  }
  function nearStash(S) { return !!(S.stash && dist(S.player, S.stash) < tune.pickRadius + 10); }
  function enterCode(S, typed) { // `typed` is the last six directions; applies a code if it matches
    const hit = CODES.find((c) => c.seq === typed);
    if (hit) { hit.run(S); S.cheated = true; S.events.push({ type: "code", name: hit.name }); }
    return hit || null;
  }
  function end(S, result) { S.result = result; S.over = true; S.events.push({ type: "end", result }); }

  /* ---- one fixed step. `move` is {x, y} with length <= 1. ---- */
  function step(S, move) {
    if (S.over) return;
    const p = S.player, v = move || { x: 0, y: 0 }, sp = S.carrying ? tune.carrySpeed : tune.walkSpeed;
    S.t += STEP;
    const nx = p.x + v.x * sp * STEP, ny = p.y + v.y * sp * STEP;
    if (!blocked(nx, p.y, PLAYER_RADIUS)) p.x = nx;
    if (!blocked(p.x, ny, PLAYER_RADIUS)) p.y = ny;
    p.moving = !!(v.x || v.y);
    if (p.moving) { p.ang = Math.atan2(v.y, v.x); p.walk += STEP * sp / 14; }

    // pick up, recover a stashed crate, deliver
    if (!S.carrying && !S.stash && dist(p, PICK) < tune.pickRadius) { S.carrying = true; S.events.push({ type: "picked", first: S.delivered === 0 }); }
    // a crate you put down stays down until you walk away and come back (or press Pick up)
    if (S.stash && !S.stash.armed && dist(p, S.stash) > tune.pickRadius + 6) S.stash.armed = true;
    if (!S.carrying && S.stash && S.stash.armed && dist(p, S.stash) < tune.pickRadius - 12) { S.carrying = true; S.stash = null; S.events.push({ type: "recovered" }); }
    if (S.carrying && dist(p, DROP) < tune.pickRadius + 14) {
      S.carrying = false; S.delivered++;
      while (S.patrols.length < Math.min(tune.maxPatrols, 1 + S.delivered)) S.patrols.push(newPatrol(S.patrols.length));
      S.events.push({ type: "delivered", count: S.delivered });
      if (S.delivered >= tune.cratesToWin) return end(S, "won");
    }

    // patrols are small state machines. route: walk the loop. hunt: a carrier in view is chased (at chasePercent of
    // normal speed) and held HOLD px away; a carrier out of view is looked for where last seen. search: turn on the
    // spot for searchSeconds. return: walk back to the nearest point of the loop. Nobody carrying: they stop hunting.
    const cosHalf = Math.cos(tune.seeHalfAngle * Math.PI / 180), wasSeen = S.seen; S.seen = false;
    for (const g of S.patrols) {
      const pts = g.route.pts, speed = patrolSpeed(S, g), chase = speed * tune.chasePercent / 100;
      const dx = p.x - g.x, dy = p.y - g.y, dd = Math.hypot(dx, dy);
      g.sees = S.carrying && dd < tune.seeRange * (1 + 0.2 * Math.min(1, heatTier(S))) && dd > 0 &&
               (dx * Math.cos(g.ang) + dy * Math.sin(g.ang)) / dd > cosHalf && clearSight(g, p);
      if (g.sees) { g.memory = { x: p.x, y: p.y }; if (g.mode !== "hunt") { g.mode = "hunt"; g.stuck = 0; } }
      else if (!S.carrying && g.mode !== "route" && g.mode !== "return") g.mode = "return";
      if (g.mode === "hunt") {
        if (g.sees) {
          if (g.stuck > STUCK) { g.mode = "search"; g.timer = tune.searchSeconds; g.stuck = 0; }   // walled in: give up on this carrier
          else if (dd > HOLD) stepToward(g, p.x, p.y, chase);
          else { g.ang = Math.atan2(dy, dx); g.stuck = 0; }
        } else if (stepToward(g, g.memory.x, g.memory.y, chase) || g.stuck > STUCK) {
          g.mode = "search"; g.timer = tune.searchSeconds; g.stuck = 0;
        }
      } else if (g.mode === "search") {
        g.ang += 1.2 * STEP; g.timer -= STEP;                 // turns slowly, as if looking about
        if (g.timer <= 0) g.mode = "return";
      } else if (g.mode === "return") {
        if (!g.homing) { g.leg = nearestPoint(pts, g.x, g.y); g.homing = true; }
        if (stepToward(g, pts[g.leg][0], pts[g.leg][1], speed) || g.stuck > STUCK) { g.mode = "route"; g.homing = false; g.stuck = 0; }
      } else {
        let next = g.foot ? g.leg + g.dir : (g.leg + 1) % pts.length;
        if (g.foot && (next < 0 || next >= pts.length)) { g.dir = -g.dir; next = g.leg + g.dir; }
        const tx = pts[next][0], ty = pts[next][1], d = Math.hypot(tx - g.x, ty - g.y);
        if (d < speed * STEP) { g.x = tx; g.y = ty; g.leg = next; }
        else { g.x += (tx - g.x) / d * speed * STEP; g.y += (ty - g.y) / d * speed * STEP; g.ang = Math.atan2(ty - g.y, tx - g.x); }
      }
      if (g.sees) S.seen = true;
    }
    S.seenFor = S.seen ? S.seenFor + STEP : 0;
    if (S.seen && !wasSeen) S.grace = tune.graceSeconds * (S.known ? 0.5 : 1);         // a sighting gets half the grace only if they already knew your look
    if (S.seen && !S.known) { S.known = true; S.events.push({ type: "known" }); }      // from now on, patrols know your look
    const rising = S.seen && S.seenFor > (S.grace || tune.graceSeconds);
    // change your shirt: stand still at a clothesline, out of sight, for changeSeconds
    const atLine = LINES.some(([x, y]) => Math.hypot(p.x - x, p.y - y) < 26);
    S.changing = atLine && !p.moving && !S.seen ? S.changing + STEP : 0;
    if (S.changing >= tune.changeSeconds) { S.changing = 0; S.outfit ^= 1; const was = S.known; S.known = false; S.events.push({ type: "changed", outfit: S.outfit, wasKnown: was }); }
    S.heat = clamp(S.heat + (rising ? tune.heatRise : S.seen ? 0 : -tune.heatFall) * STEP, 0, 100);
    if (S.seen && !wasSeen) S.events.push({ type: "seen" });
    const tier = heatTier(S); if (tier !== S.tier) { S.events.push({ type: "tier", tier, up: tier > S.tier }); S.tier = tier; }

    // passers-by: decoration only
    for (const q of S.peds) {
      const d = Math.hypot(q.tx - q.x, q.ty - q.y);
      if (d < 3) { const l = LAMPS[Math.floor(rnd(S) * 9)], m = LAMPS[Math.floor(rnd(S) * 9)];
        if (rnd(S) < 0.5) { q.tx = l[0] + 28; q.ty = q.y; } else { q.ty = m[1] + 28; q.tx = q.x; } q.moving = false; }
      else { q.x += (q.tx - q.x) / d * q.sp * STEP; q.y += (q.ty - q.y) / d * q.sp * STEP; q.ang = Math.atan2(q.ty - q.y, q.tx - q.x); q.walk += STEP * 4; q.moving = true; }
    }
    if (S.heat >= 100) return end(S, "lost-heat");
    if (S.t >= tune.clockSeconds) return end(S, "lost-time");
  }

  const api = { config, tune, W, H, QUAY, WATER, STEP, PLAYER_RADIUS, ROOF, WALL, B, SOUNDHALL, BOATWORKS, PICK, DROP, LAMPS, STREETS,
                PALMS, SKIFFS, ROUTES, CAST, NPCS, CODES, TIERS, LINES, heatTier, clamp, dist, clearSight, blocked, patrolSpeed, create, step, toggleCrate, nearStash, enterCode };
  root.NH = root.NH || {}; root.NH.sim = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
