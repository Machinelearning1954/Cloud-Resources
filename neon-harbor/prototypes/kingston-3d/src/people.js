/* people: every character in the game. Each one is a single skinned figure with its own hairstyle.
   API: NH.people.make(cast) -> { group, carry, update(dt, st) }
     cast  = an entry of NH.sim.CAST ({ id, name, hair, skin, shirt, note })
     group = THREE.Group, feet at y = 0, 1.8 m tall, facing +X (main.js scales it by NH.HERO)
     carry = THREE.Object3D inside the group where a carried crate is parented (in front of the chest). It rests at
             (0.55, 1.05, 0) in the group's space and rides on the chest bone, so the crate moves with the hands that hold it
     st    = { moving: bool, carrying: bool, walk: number (phase that grows as the person walks) }
   Extra fields on the returned object (optional, nothing depends on them): mesh (the THREE.SkinnedMesh), hair (the cast's hair id).
   cast.hair must be one of: locs-updo, locs-freeform, locs-wrap, waves, afro, canerows, cap. An unknown id throws:
   a hairstyle is part of who a person is (ADR 0051), so there is no generic shape to fall back to.

   How it is drawn: one THREE.SkinnedMesh per person with three material groups, so three draw calls each:
     skin  - a MeshStandardMaterial whose colour is cast.skin, untouched (shared by everyone with that skin)
     cloth - one shared vertex-coloured material: clothes, shoes, eyes, the head wrap, the patrol cap
     hair  - one shared vertex-coloured material, NH.PALETTE.hair throughout; only the 360 waves add a second,
             lighter tone of the same hair on the crests of the ripples so the pattern reads from above
   The geometry is merged by hand once per cast id and shared; the skeleton is per person. Everything stays inside
   the capsule main.js uses for its seen-through-walls shape (radius 0.42 m, top at 1.94 m), which is why tall hair stops there. */
(function () {
  "use strict";
  const NH = window.NH, P = NH.PALETTE, PI = Math.PI, TAU = PI * 2;
  const sin = Math.sin, cos = Math.cos, abs = Math.abs, max = Math.max, min = Math.min, sqrt = Math.sqrt;

  /* ---------------- build-time helpers (plain arrays as vectors; none of this runs per frame) ---------------- */
  const lin = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };   // vertex colours live in the renderer's linear working space
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (a) => mul(a, 1 / (Math.hypot(a[0], a[1], a[2]) || 1)), mix = (a, b, t) => add(mul(a, 1 - t), mul(b, t));
  const hash = (s) => { let h = 2166136261; for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619); return h >>> 0; };
  const rnd = (i, k) => { let h = Math.imul((i + 1) * 374761393 ^ (k + 1) * 668265263, 1274126177); h ^= h >>> 15; h = Math.imul(h, 2246822519); h ^= h >>> 13; return (h >>> 0) / 4294967296; };   // 0..1 from two indices
  const spline = (pts, m) => { // m + 1 points along a smooth curve through pts
    const n = pts.length - 1, out = [];
    for (let i = 0; i <= m; i++) {
      const u = i / m * n, k = min(n - 1, Math.floor(u)), t = u - k, a = pts[max(0, k - 1)], b = pts[k], c = pts[k + 1], d = pts[min(n, k + 2)];
      out.push([0, 1, 2].map((q) => 0.5 * (2 * b[q] + (c[q] - a[q]) * t + (2 * a[q] - 5 * b[q] + 4 * c[q] - d[q]) * t * t + (3 * b[q] - a[q] - 3 * c[q] + d[q]) * t * t * t)));
    }
    return out;
  };

  /* ---------------- skeleton: [parent, x, y, z] in the bind pose ---------------- */
  const HIPS = 0, CHEST = 1, HEAD = 2, ARM_L = 3, FORE_L = 4, ARM_R = 5, FORE_R = 6, LEG_L = 7, SHIN_L = 8, FOOT_L = 9, LEG_R = 10, SHIN_R = 11, FOOT_R = 12, SWAY = 13;
  const SH = 0.212, HP = 0.092, HIP_Y = 0.95;   // shoulder and hip joints sit this far either side of the centre line (left is -Z)
  const REST = [[-1, 0, HIP_Y, 0], [HIPS, 0, 1.13, 0], [CHEST, 0, 1.47, 0],
    [CHEST, 0, 1.40, -SH], [ARM_L, 0, 1.13, -SH], [CHEST, 0, 1.40, SH], [ARM_R, 0, 1.13, SH],
    [HIPS, 0, 0.90, -HP], [LEG_L, 0, 0.50, -HP], [SHIN_L, 0, 0.08, -HP], [HIPS, 0, 0.90, HP], [LEG_R, 0, 0.50, HP], [SHIN_R, 0, 0.08, HP],
    [HEAD, -0.03, 1.68, 0]];   // SWAY: loose ends of hair hang from here
  const INVERSE = REST.map((r) => new THREE.Matrix4().makeTranslation(-r[1], -r[2], -r[3])), IDENTITY = new THREE.Matrix4();
  const BOUNDS = new THREE.Sphere(new THREE.Vector3(0, 0.95, 0), 1.25);
  const CARRY = (() => { // both palms flat on the sides of the crate: solved once for the right arm, mirrored for the left
    const S = new THREE.Vector3(0, 1.40, SH), T = new THREE.Vector3(0.36, 1.0, 0.357), down = new THREE.Vector3(0, -1, 0), LU = 0.27, LF = 0.32;
    const d = T.clone().sub(S), len = d.length(), a = (LU * LU - LF * LF + len * len) / (2 * len), h = sqrt(max(0, LU * LU - a * a));
    const dir = d.clone().normalize(), pole = new THREE.Vector3(-0.3, -1, 0.6); pole.addScaledVector(dir, -pole.dot(dir)).normalize();
    const E = S.clone().addScaledVector(dir, a).addScaledVector(pole, h);
    const up = new THREE.Quaternion().setFromUnitVectors(down, E.clone().sub(S).normalize());
    const fore = new THREE.Quaternion().setFromUnitVectors(down, T.clone().sub(E).normalize().applyQuaternion(up.clone().invert()));
    const mirror = (q) => new THREE.Quaternion(-q.x, -q.y, q.z, q.w);
    return { armR: up, foreR: fore, armL: mirror(up), foreL: mirror(fore) };
  })();

  /* ---------------- geometry builder: positions, normals, colours and bone weights, merged by hand ---------------- */
  const SKIN = 0, CLOTH = 1, HAIR = 2;
  function builder() {
    const pos = [], nor = [], col = [], bone = [], wgt = [], tri = [[], [], []], t0 = [0, 0, 0]; let v0 = 0, print = null; const ink = [];
    return {
      setPrint(ref, alt, kind) { print = { ref, alt, kind }; },   // a printed shirt: its vertices carry the second colour and the kind of print; the cloth shader draws the pattern
      get n() { return pos.length / 3; },
      v(p, c, b) { const w = b[2] || 0; if (print && c[0] === print.ref[0] && c[1] === print.ref[1] && c[2] === print.ref[2]) ink.push(print.alt[0], print.alt[1], print.alt[2], print.kind); else ink.push(0, 0, 0, 0); pos.push(p[0], p[1], p[2]); nor.push(0, 0, 0); col.push(c[0], c[1], c[2]); bone.push(b[0], b[1] || 0, 0, 0); wgt.push(1 - w, w, 0, 0); return pos.length / 3 - 1; },
      ring(r, n, c, b) { // n points round p, along u and v; e < 1 squares the oval off
        const first = pos.length / 3, e = r.e || 1;
        for (let j = 0; j < n; j++) {
          const a = j / n * TAU + (r.a || 0), cs = cos(a), sn = sin(a), x = e === 1 ? cs : Math.sign(cs) * Math.pow(abs(cs), e), z = e === 1 ? sn : Math.sign(sn) * Math.pow(abs(sn), e);
          this.v([r.p[0] + r.u[0] * x + r.v[0] * z, r.p[1] + r.u[1] * x + r.v[1] * z, r.p[2] + r.u[2] * x + r.v[2] * z], r.c || c, r.legs ? [HIPS, sn < 0 ? LEG_L : LEG_R, r.legs * min(1, abs(sn) * 2.2)] : r.b || b);
        }
        return first;
      },
      strip(g, lo, hi, n, open) { const T = tri[g]; for (let j = 0; j < (open ? n - 1 : n); j++) { const k = (j + 1) % n; T.push(lo + j, hi + j, lo + k, lo + k, hi + j, hi + k); } },
      fan(g, apex, a, n, end) { const T = tri[g]; for (let j = 0; j < n; j++) { const k = (j + 1) % n; if (end) T.push(a + j, apex, a + k); else T.push(apex, a + j, a + k); } },
      done() { // smooth normals for everything added since the last call, so separate parts keep a crease between them
        for (let g = 0; g < 3; g++) {
          const T = tri[g];
          for (let i = t0[g]; i < T.length; i += 3) {
            const a = T[i] * 3, b = T[i + 1] * 3, c = T[i + 2] * 3, ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2], vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
            const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
            for (const k of [a, b, c]) { nor[k] += nx; nor[k + 1] += ny; nor[k + 2] += nz; }
          }
          t0[g] = T.length;
        }
        for (let i = v0 * 3; i < nor.length; i += 3) { const l = Math.hypot(nor[i], nor[i + 1], nor[i + 2]) || 1; nor[i] /= l; nor[i + 1] /= l; nor[i + 2] /= l; }
        v0 = pos.length / 3;
      },
      geometry() {
        const g = new THREE.BufferGeometry(), f = (a, n) => new THREE.Float32BufferAttribute(a, n); let at = 0;
        g.setAttribute("position", f(pos, 3)); g.setAttribute("normal", f(nor, 3)); g.setAttribute("color", f(col, 3)); g.setAttribute("ink", f(ink, 4));
        g.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(bone, 4)); g.setAttribute("skinWeight", f(wgt, 4));
        g.setIndex(tri[0].concat(tri[1], tri[2])); tri.forEach((T, i) => { g.addGroup(at, T.length, i); at += T.length; });
        return g;
      },
    };
  }
  /* Rings stitched into a skin. The faces point outward when the rings advance along v x u (for level rings: upward).
     o = { c, b, start, end }: default colour and bones; start / end are points that close the two ends. A ring with `cut` starts a new skin. */
  function loft(B, g, rings, n, o) {
    let prev = -1, first = -1;
    for (const r of rings) { const cur = B.ring(r, n, o.c, o.b); if (prev >= 0 && !r.cut) B.strip(g, prev, cur, n); if (first < 0) first = cur; prev = cur; }
    if (o.start) B.fan(g, B.v(o.start, rings[0].c || o.c, rings[0].b || o.b), first, n, false);
    if (o.end) { const r = rings[rings.length - 1]; B.fan(g, B.v(o.end, r.c || o.c, r.b || o.b), prev, n, true); }
    B.done();
  }
  const hr = (y, rx, rz, cx, cz, more) => Object.assign({ p: [cx || 0, y, cz || 0], u: [rx, 0, 0], v: [0, 0, rz] }, more);   // a level oval
  /* A tube along a path; rad[i] is its radius at pts[i]. o = { c, b, bs (bones per point), ref, flat, root } */
  function tube(B, g, pts, rad, n, o) {
    const m = pts.length - 1, rings = []; let u = null, t = null;
    for (let i = 0; i <= m; i++) {
      t = unit(sub(pts[min(m, i + 1)], pts[max(0, i - 1)]));
      u = unit(u ? sub(u, mul(t, dot(u, t))) : cross(o.ref || (abs(t[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]), t));
      rings.push({ p: pts[i], u: mul(u, rad[i]), v: mul(cross(u, t), rad[i] * (o.flat || 1)), b: o.bs && o.bs[i] });
    }
    loft(B, g, rings, n, { c: o.c, b: o.b, start: o.root ? add(pts[0], mul(unit(sub(pts[0], pts[1])), rad[0] * 0.7)) : null, end: add(pts[m], mul(t, rad[m] * 0.8)) });
  }
  /* A flat oval facing along nrm */
  function oval(B, g, p, nrm, ru, rv, n, c, b) {
    const u = unit(cross([0, 1, 0], nrm)), first = B.ring({ p, u: mul(u, ru), v: mul(cross(nrm, u), rv) }, n, c, b);
    B.fan(g, B.v(p, c, b), first, n, false); B.done();
  }
  /* Recolour a stack of level rings below height y, with a clean edge at y */
  function dip(rings, y, c) {
    const out = [];
    rings.forEach((r, i) => {
      const q = rings[i - 1];
      if (q && q.p[1] < y && r.p[1] >= y) { const t = (y - q.p[1]) / (r.p[1] - q.p[1]), m = { p: mix(q.p, r.p, t), u: mix(q.u, r.u, t), v: mix(q.v, r.v, t), e: r.e, b: t < 0.5 ? q.b : r.b }; out.push(Object.assign({}, m, { c }), Object.assign({}, m, { c: r.c, cut: true })); }
      out.push(r.p[1] < y ? Object.assign({}, r, { c }) : r);
    });
    return out;
  }

  /* ---------------- the head as an oval ball: hair is laid on this ---------------- */
  const HC = [-0.002, 1.632, 0], HR = [0.128, 0.136, 0.107];   // centre and radii of the skull: a head about a quarter larger than life
  const skull = (az, el, s) => [HC[0] + cos(el) * cos(az) * HR[0] * s, HC[1] + sin(el) * HR[1] * s, HC[2] + cos(el) * sin(az) * HR[2] * s];   // az 0 = face, el = up from the ears
  // where the hair stops, as a height on the skull: square across the forehead, a step down at the temples, over the ears, low at the nape
  const hairline = (az, nape) => { const a = abs(az > PI ? az - TAU : az); return a < 0.85 ? 0.6 : a < 1.0 ? 0.6 - (a - 0.85) / 0.15 * 0.42 : a < 2.0 ? 0.18 : a < 2.5 ? 0.18 - (a - 2.0) / 0.5 * (0.18 - nape) : nape; };
  /* A shell over the skull from the crown down to the hairline. fn(k, j) -> [scale, colour] for ring k (0 = crown .. K = hairline) and column j.
     With `level` the rings are true circles round the crown that the hairline cuts across; without it they follow the hairline's shape. */
  function shell(B, g, K, n, nape, fn, level) {
    const b = [HEAD], top = B.v(skull(0, PI / 2, fn(0, 0)[0]), fn(0, 0)[1], b); let prev = -1;
    for (let k = 1; k <= K + 1; k++) { // the last ring tucks the edge back under
      const cur = B.n, kk = min(k, K);
      for (let j = 0; j < n; j++) { const az = j / n * TAU, f = fn(kk, j), edge = hairline(az, nape), el = level && kk < K ? max(edge, PI / 2 - kk / K * (PI / 2 - nape)) : PI / 2 - kk / K * (PI / 2 - edge); B.v(skull(az, el, k > K ? 0.99 : f[0]), f[1], b); }
      if (prev < 0) B.fan(g, top, cur, n, true); else B.strip(g, cur, prev, n);
      prev = cur;
    }
    B.done();
  }

  /* ---------------- colours ---------------- */
  const WHITE = [1, 1, 1], HAIRC = lin(P.hair), WAVE = mix(HAIRC, lin(0x6a4a34), 0.55)   /* the crest of each wave: dark hair catching the light, never grey */, EYE = lin(0x0e0a09);
  const WRAP_A = lin(P.harbour), WRAP_B = lin(P.ink), CAP_TOP = lin(P.patrolBody), CAP_BAND = lin(P.patrolBand), PEAK = lin(0x0d1017), BELT = lin(0x101216), TRIM = lin(P.zinc), NAVY = 0x18233a;
  const BOTTOMS = ["skirt", "shorts", "longshorts", "trousers", "cropped"], HEMS = { shorts: 0.6, longshorts: 0.45, cropped: 0.27, skirt: 2 };   // hem height; 2 = no trouser legs at all
  const LEG_COLS = [0x55633a, 0x5f7fa8, 0xb09868, 0x30333c, 0xf0e8d6, 0x2a3a5c], SHOE_COLS = [0x1c1c20, 0xece8dd, 0x8c2a2e];

  /* ---------------- what each cast entry wears, and how they are built and walk: all from a hash of the id ---------------- */
  function lookFor(cast) {
    const h = hash(cast.look || cast.id), pick   /* cast.look: wear another entry's build and clothes, so a change of shirt changes only the shirt */ = (list, sh) => list[(h >>> sh) % list.length], f = (sh) => ((h >>> sh) % 8) / 7, patrol = cast.hair === "cap", boat = /boatwright/.test(cast.note || "");
    return {
      shirt: lin(cast.shirt), kind: patrol ? "uniform" : boat ? "overalls" : pick(BOTTOMS, 10),
      legs: lin(patrol ? NAVY : boat ? 0x3f5f86 : pick(LEG_COLS, 11)), shoes: lin(patrol || boat ? 0x1c1c20 : pick(SHOE_COLS, 11)),
      sw: 0.18 + f(14) * 0.016, hw: 0.152 + f(17) * 0.022, ww: 0.134 + f(20) * 0.02, cd: 0.116 + f(23) * 0.012,
      gait: patrol ? { stride: 0.42, bob: 0.02, twist: 0.04, lean: 0.02, arm: 0.4 } : { stride: 0.44 + f(5) * 0.1, bob: 0.024 + f(9) * 0.014, twist: 0.07 + f(12) * 0.06, lean: 0.04 + f(15) * 0.04, arm: 0.42 + f(18) * 0.13 },
      phase: f(26) * 6,
    };
  }

  /* ---------------- the body: one continuous figure, coloured per ring ---------------- */
  function body(B, L) {
    const S = L.shirt, e = 0.82, tucked = L.kind === "uniform", sw = L.sw, hw = L.hw, ww = L.ww, cd = L.cd;
    // head, neck, face
    const hd = (el) => ({ p: [HC[0], HC[1] + sin(el) * HR[1], 0], u: [cos(el) * HR[0], 0, 0], v: [0, 0, cos(el) * HR[2]] });
    loft(B, SKIN, [hr(1.497, 0.074, 0.06, 0.024), hr(1.532, 0.105, 0.087, 0.014), hr(1.585, 0.123, 0.102, 0.004), hd(0), hd(0.45), hd(0.85), hd(1.2)], 10,
      { c: WHITE, b: [HEAD], start: [0.04, 1.468, 0], end: [HC[0], HC[1] + HR[1], 0] });
    loft(B, SKIN, [hr(1.44, 0.054, 0.054, -0.004, 0, { b: [CHEST] }), hr(1.53, 0.05, 0.05, 0, 0, { b: [HEAD] })], 8, { c: WHITE });
    loft(B, SKIN, [{ p: [0.118, 1.592, 0], u: [0, 0.03, 0], v: [0, 0, -0.02] }], 4, { c: WHITE, b: [HEAD], end: [0.153, 1.58, 0] });   // nose
    for (const sd of [-1, 1]) {
      loft(B, SKIN, [{ p: [-0.012, 1.6, sd * 0.098], u: [0.021, 0, 0], v: [0, -sd * 0.031, 0] }, { p: [-0.02, 1.603, sd * 0.118], u: [0.017, 0, 0], v: [0, -sd * 0.026, 0] }], 6, { c: WHITE, b: [HEAD], end: [-0.022, 1.603, sd * 0.124] });   // ear
      const az = sd * 0.37, el = -0.1, out = (a, l) => unit([cos(l) * cos(a) / HR[0], sin(l) / HR[1], cos(l) * sin(a) / HR[2]]);
      oval(B, CLOTH, skull(az, el, 1.035), out(az, el), 0.017, 0.011, 8, EYE, [HEAD]); oval(B, HAIR, skull(az * 1.04, 0.17, 1.03), out(az, 0.17), 0.024, 0.0065, 6, HAIRC, [HEAD]);
    }
    // torso: shirt from hem to collar; a tucked shirt gets a belt, overalls come up to the waist
    let T = [hr(0.905, 0.113, hw + 0.014, 0, 0, { b: [HIPS], e }), hr(0.96, 0.108, hw + 0.008, 0, 0, { b: [HIPS], e }), hr(1.02, 0.103, ww + 0.008, 0, 0, { b: [HIPS, CHEST, 0.35], e }), hr(1.1, 0.1, ww, 0, 0, { b: [HIPS, CHEST, 0.75], e }),
      hr(1.24, cd - 0.008, sw - 0.024, 0.006, 0, { e }), hr(1.35, cd, sw - 0.004, 0.01, 0, { e }), hr(1.422, cd - 0.02, sw + 0.004, 0.004, 0, { e }), hr(1.455, 0.074, sw - 0.055), hr(1.48, 0.058, 0.064)];
    if (tucked) T = dip(dip(T, 1.02, BELT), 0.985, L.legs); else if (L.kind === "overalls") T = dip(T, 1.07, L.legs);
    loft(B, CLOTH, T, 12, { c: S, b: [CHEST] });
    if (L.kind === "skirt") loft(B, CLOTH, [hr(0.5, 0.176, hw + 0.05, 0, 0, { legs: 1 }), hr(0.7, 0.14, hw + 0.024, 0, 0, { legs: 0.6 }), hr(0.9, 0.11, hw + 0.002), hr(0.965, 0.105, hw - 0.006)], 12, { c: L.legs, b: [HIPS] });   // each side of the hem swings with its leg
    else loft(B, CLOTH, [hr(0.83, 0.09, hw - 0.03, 0, 0, { e }), hr(0.885, 0.11, hw + 0.009, 0, 0, { e }), hr(0.965, 0.106, hw, 0, 0, { e })], 10, { c: L.legs, b: [HIPS], start: [0, 0.8, 0] });   // fills the shirt's hem, so nothing shows through at the hip
    for (const sd of [-1, 1]) {
      const z = sd * HP, LEG = sd < 0 ? LEG_L : LEG_R, SHIN = LEG + 1, FOOT = LEG + 2, ARM = sd < 0 ? ARM_L : ARM_R, FORE = ARM + 1, az = sd * (SH + 0.002);
      // legs: bare below whatever the bottoms cover
      const hem = HEMS[L.kind] || 0.07, lr = (y, rx, rz, cx) => hr(y, rx, rz, cx, z, { b: [LEG, SHIN, min(1, max(0, (0.6 - y) / 0.2))] });   // the knee bends between 0.4 and 0.6
      const bare = [lr(0.085, 0.037, 0.035, 0), lr(0.2, 0.044, 0.042, -0.004), lr(0.34, 0.058, 0.054, -0.008), lr(0.5, 0.056, 0.054, 0.002), lr(0.62, 0.068, 0.064, 0), lr(0.78, 0.084, 0.078, 0)];
      const cloth = [[0.07, 0.064, 0.06], [0.29, 0.068, 0.064], [0.5, 0.073, 0.069], [0.72, 0.09, 0.084], [0.9, 0.094, 0.086]].filter((r) => hem <= 0.07 || r[0] > hem + 0.03).map((r) => lr(r[0], r[1], r[2], 0));
      if (hem > 0.07) { loft(B, SKIN, bare.filter((r) => r.p[1] < hem + 0.1), 8, { c: WHITE }); if (hem < 1) cloth.unshift(lr(hem, 0.094 - (0.9 - hem) * 0.036, 0.088 - (0.9 - hem) * 0.036, 0)); }   // a loose hem
      if (hem < 1) loft(B, CLOTH, cloth, 8, { c: L.legs });
      // shoe: a blunt wedge along +X with a flat sole
      const sh = (x, y, ry, rz) => ({ p: [x, y, z], u: [0, ry, 0], v: [0, 0, -rz], e: 0.75, a: TAU / 12 });
      loft(B, CLOTH, [sh(-0.062, 0.04, 0.04, 0.04), sh(-0.02, 0.054, 0.056, 0.05), sh(0.06, 0.046, 0.048, 0.054), sh(0.14, 0.034, 0.036, 0.05), sh(0.185, 0.027, 0.027, 0.038)], 6, { c: L.shoes, b: [FOOT], start: [-0.084, 0.036, z], end: [0.207, 0.022, z] });
      // arm: short sleeve, then bare to a flat hand, palm inward
      loft(B, CLOTH, [hr(1.25, 0.054, 0.051, 0, az), hr(1.32, 0.054, 0.05, 0, az), hr(1.38, 0.056, 0.05, 0, az * 0.98), hr(1.412, 0.048, 0.042, 0, az * 0.955), hr(1.43, 0.03, 0.027, 0, az * 0.93)], 7, { c: S, b: [ARM], end: [0, 1.435, az * 0.92] });
      loft(B, SKIN, [hr(0.742, 0.03, 0.017, 0, az), hr(0.785, 0.044, 0.023, 0, az), hr(0.84, 0.043, 0.025, 0, az), hr(0.895, 0.032, 0.028, 0, az), hr(1.02, 0.047, 0.043, 0, az), hr(1.13, 0.045, 0.042, 0, az, { b: [ARM, FORE, 0.5] }), hr(1.255, 0.05, 0.047, 0, az, { b: [ARM] })], 7,
        { c: WHITE, b: [FORE], start: [0, 0.724, az] });
      if (L.kind === "overalls") tube(B, CLOTH, spline([[0.128, 1.3, sd * 0.085], [0.1, 1.43, sd * 0.1], [0, 1.487, sd * 0.108], [-0.09, 1.43, sd * 0.09], [-0.118, 1.24, sd * 0.05], [-0.108, 1.07, sd * 0.02]], 7), [0.026, 0.026, 0.026, 0.026, 0.026, 0.026, 0.026, 0.026], 4, { c: L.legs, b: [CHEST], flat: 0.3, ref: [0, 0, 1], root: true });   // strap over the shoulder
      if (tucked) loft(B, CLOTH, [{ p: [0, 1.447, sd * 0.15], u: [0.036, 0, 0], v: [0, 0, 0.05], e: 0.5, a: PI / 4 }], 4, { c: TRIM, b: [CHEST], end: [0, 1.452, sd * 0.15] });   // plain shoulder tab, no badge
    }
    if (L.kind === "overalls") { // the bib: a panel standing a little proud of the shirt
      const rows = [1.07, 1.2, 1.32]; let prev = -1;
      for (const y of rows) { const cur = B.n, t = (y - 1.07) / 0.25, w = 0.72 - 0.2 * t, rx = (0.103 + 0.023 * t) * 1.06, rz = (ww + 0.004 + (sw - ww - 0.02) * t) * 1.06;
        for (let j = 0; j < 5; j++) { const a = (j / 4 - 0.5) * 2 * w, cs = cos(a), sn = sin(a); B.v([0.008 * t + rx * Math.pow(abs(cs), e), y, rz * Math.sign(sn) * Math.pow(abs(sn), e)], L.legs, [CHEST]); }
        if (prev >= 0) B.strip(CLOTH, prev, cur, 5, true); prev = cur; }
      B.done();
    }
  }

  /* ---------------- the seven hairstyles: each one its own geometry ---------------- */
  /* One loc: a tube along a smooth path, thicker in the middle, with one swollen stretch (locs are not even), its free end hung from the SWAY bone */
  const loc = (B, pts, r0, uneven, m) => { const p = spline(pts, m || 4), n = p.length - 1;
    tube(B, HAIR, p, p.map((_, i) => r0 * (0.85 + 0.9 * (i / n) * (1 - i / n) - 0.1 * i / n) * (i === uneven ? 1.22 : 1)), 5, { c: HAIRC, bs: p.map((_, i) => [HEAD, SWAY, min(1, max(0, (i / n - 0.3) / 0.5))]) }); };
  const STYLES = {
    /* locs gathered up into a high coiled bun, the swept-up locs showing as ridges, a few left falling at the back */
    "locs-updo"(B, L) {
      shell(B, HAIR, 3, 20, -0.25, (k, j) => [1.035 + (j % 2 ? 0.02 : 0), HAIRC]);
      const pts = [], rad = [], band = [[0.4, 1.03], [0.43, 1.1], [0.8, 1.1], [0.83, 1.03]].map(([el, sc]) => { const at = B.n; for (let j = 0; j < 20; j++) B.v(skull(j / 20 * TAU, el, sc), L.shirt, [HEAD]); return at; });
      for (let i = 0; i < 3; i++) B.strip(CLOTH, band[i], band[i + 1], 20);   // the band that holds the locs up, in the colour of the shirt
      B.done();
      for (let i = 0; i <= 24; i++) { const t = i / 24, a = t * 2.2 * TAU + 0.6, R = 0.014 + 0.11 * Math.pow(1 - t, 0.85); pts.push([-0.045 + cos(a) * R, 1.8 + 0.095 * Math.pow(t, 0.9), sin(a) * R]); rad.push(0.052 - 0.014 * t); }
      tube(B, HAIR, pts, rad, 5, { c: HAIRC, b: [HEAD], root: true });
      for (let i = 0; i < 5; i++) { const z = (i - 2) * 0.036; loc(B, [[-0.1, 1.745, z * 0.8], [-0.156, 1.67, z], [-0.16, 1.56, z * 1.1], [-0.15, 1.39 + rnd(i, 3) * 0.05, z * 1.15]], 0.025, 1 + i % 3); }
    },
    /* thick freeform locs of uneven thickness, falling to the shoulders all round */
    "locs-freeform"(B) {
      shell(B, HAIR, 3, 10, -0.5, () => [1.05, HAIRC]);
      for (let i = 0; i < 12; i++) {
        const az = 0.85 + i / 11 * (TAU - 1.7) + (rnd(i, 1) - 0.5) * 0.14, el = i % 2 ? 0.5 : 0.95, side = abs(sin(az)), out = 0.168 + 0.028 * rnd(i, 2) + 0.022 * side, end = side > 0.75 ? 1.45 + 0.02 * rnd(i, 4) : 1.36 + 0.09 * rnd(i, 4);
        loc(B, [skull(az, el, 0.98), skull(az, el * 0.55, 1.3), [cos(az) * out * 1.05, 1.555, sin(az) * out], [cos(az) * (out + 0.014) * 1.05, end, sin(az) * (out + 0.014)]], 0.025 + 0.015 * rnd(i, 5), 1 + i % 3);
      }
      for (let i = 0; i < 4; i++) { const az = 0.4 + i * 1.6 + rnd(i, 6) * 0.3; loc(B, [skull(az, 1.25, 1), skull(az, 0.9, 1.24), [cos(az) * 0.17, 1.665, sin(az) * 0.165], [cos(az) * 0.2, 1.565 + rnd(i, 7) * 0.04, sin(az) * 0.19]], 0.032 + 0.008 * rnd(i, 8), 2); }
    },
    /* long locs wound into a tall head wrap: two-tone bands, the locs coiled in the open top, a few ends falling at the back */
    "locs-wrap"(B) {
      const W = (cx, y, rx, rz, tilt, c, cut) => ({ p: [cx, y, 0], u: [rx, rx * tilt, 0], v: [0, 0, rz], c, cut });
      loft(B, CLOTH, [W(-0.004, 1.64, 0.124, 0.104, 0.48, WRAP_A), W(-0.004, 1.645, 0.137, 0.117, 0.48, WRAP_A), W(-0.006, 1.7, 0.143, 0.124, 0.36, WRAP_A),
        W(-0.006, 1.7, 0.145, 0.126, 0.36, WRAP_B, true), W(-0.008, 1.735, 0.151, 0.133, 0.22, WRAP_B),
        W(-0.008, 1.735, 0.151, 0.133, 0.22, WRAP_A, true), W(-0.01, 1.8, 0.158, 0.142, 0.1, WRAP_A),
        W(-0.01, 1.8, 0.16, 0.144, 0.1, WRAP_B, true), W(-0.01, 1.836, 0.16, 0.146, 0.04, WRAP_B),
        W(-0.01, 1.836, 0.16, 0.146, 0.04, WRAP_A, true), W(-0.01, 1.876, 0.153, 0.142, 0, WRAP_A), W(-0.01, 1.888, 0.13, 0.12, 0, WRAP_A),
        W(-0.01, 1.888, 0.13, 0.12, 0, WRAP_B, true), W(-0.01, 1.868, 0.096, 0.09, 0, WRAP_B)], 14, { b: [HEAD] });
      const pts = [], rad = [];
      for (let i = 0; i <= 14; i++) { const t = i / 14, a = t * 1.6 * TAU, R = 0.008 + 0.07 * (1 - t); pts.push([-0.01 + cos(a) * R, 1.866 + 0.012 * t, sin(a) * R]); rad.push(0.028 - 0.004 * t); }
      tube(B, HAIR, pts, rad, 5, { c: HAIRC, b: [HEAD], root: true });
      for (let i = 0; i < 3; i++) { const z = (i - 1) * 0.045; loc(B, [[-0.07, 1.878, z * 0.6], [-0.135, 1.897, z * 0.9], [-0.18, 1.872, z], [-0.2, 1.77, z * 1.1], [-0.195, 1.63, z * 1.2], [-0.18, 1.5 + rnd(i, 9) * 0.05, z * 1.25]], 0.017, 2 + i, 7); }
      const K0 = [0.132, 1.79, 0.05];   // the knot, tied at the front, and its two ends
      loft(B, CLOTH, [hr(K0[1] - 0.02, 0.024, 0.03, K0[0], K0[2]), hr(K0[1] + 0.004, 0.032, 0.04, K0[0], K0[2]), hr(K0[1] + 0.026, 0.02, 0.026, K0[0], K0[2])], 6, { c: WRAP_B, b: [HEAD], start: [K0[0], K0[1] - 0.03, K0[2]], end: [K0[0], K0[1] + 0.034, K0[2]] });
      for (const sd of [-1, 1]) tube(B, CLOTH, [[K0[0], K0[1], K0[2] + sd * 0.02], [K0[0] + 0.03, K0[1] - 0.02 + sd * 0.02, K0[2] + sd * 0.06], [K0[0] + 0.026, K0[1] - 0.07 + sd * 0.02, K0[2] + sd * 0.085]], [0.02, 0.024, 0.016], 4, { c: sd < 0 ? WRAP_A : WRAP_B, b: [HEAD], flat: 0.4 });
    },
    /* 360 waves: cut close, ripples in rings out from the crown, a squared-off hairline */
    waves(B) { shell(B, HAIR, 4, 18, -0.42, (k) => [1.024 + (k % 2 ? 0.03 : 0), k % 2 ? WAVE : HAIRC], true); },
    /* a rounded natural afro, clearly larger than the head, opening round the face */
    afro(B) {
      const C = [-0.03, 1.71, 0], R = 0.222, n = 20, K = 10, b = [HEAD];
      const pt = (k, j) => {
        const el = PI / 2 - k / K * PI, az = j / n * TAU, s = R * (1 + (rnd(k, j) - 0.5) * 0.07), p = [C[0] + cos(el) * cos(az) * s, C[1] + sin(el) * s * 0.95, C[2] + cos(el) * sin(az) * s];
        const d = sub(p, HC), fa = Math.atan2(d[2], d[0]), fe = Math.asin(d[1] / Math.hypot(d[0], d[1], d[2]));   // seen from the middle of the head:
        return abs(fa) < 0.95 && fe < 0.56 ? skull(fa, 0.56, 1) : fe < -0.5 ? skull(fa, fe, 0.96) : p;           // whatever would cover the face goes back to the hairline; the underside tucks in
      };
      const top = B.v(pt(0, 0), HAIRC, b); let prev = -1;
      for (let k = 1; k < K; k++) { const cur = B.n; for (let j = 0; j < n; j++) B.v(pt(k, j), HAIRC, b); if (prev < 0) B.fan(HAIR, top, cur, n, true); else B.strip(HAIR, cur, prev, n); prev = cur; }
      B.done();
    },
    /* straight-back canerows: raised plaited rows from the hairline to the nape, the scalp itself showing in the partings */
    canerows(B) {
      [0.036, -0.036, 0.096, -0.096].forEach((z, i) => {
        const side = i > 1, f = sqrt(1 - z * z / (HR[2] * HR[2])), g0 = side ? 0.35 : 0.62, g1 = side ? 2.9 : 3.85, m = side ? 6 : 9, pts = [], rad = [], bs = [];
        for (let k = 0; k <= m; k++) { const g = g0 + (g1 - g0) * k / m; pts.push([HC[0] + cos(g) * (HR[0] * f + 0.004), HC[1] + sin(g) * (HR[1] * f + 0.004), z]); rad.push(0.0245 * (k % 2 ? 1.1 : 0.94)); /* thick enough to read as rows from the game camera */ bs.push([HEAD]); }
        const tail = side ? [] : [[-0.118, 1.51, z * 1.05]];
        tail.forEach((p, k) => { pts.push(p); rad.push(0.015); bs.push([HEAD]); });
        tube(B, HAIR, pts, rad, 4, { c: HAIRC, bs, b: [HEAD], ref: [0, 0, 1], flat: 1.35, root: true });
      });
    },
    /* the foot patrol: a peaked cap with a pale flat top, short hair showing under it at the sides and back */
    cap(B) {
      shell(B, HAIR, 2, 12, -0.42, () => [1.025, HAIRC]);
      const W = (cx, y, rx, rz, tilt, c, cut) => ({ p: [cx, y, 0], u: [rx, rx * tilt, 0], v: [0, 0, rz], c, cut });
      loft(B, CLOTH, [W(0, 1.68, 0.126, 0.106, 0.1, CAP_BAND), W(0, 1.682, 0.136, 0.116, 0.1, CAP_BAND), W(0, 1.728, 0.138, 0.118, 0.1, CAP_BAND),
        W(0, 1.728, 0.138, 0.118, 0.1, CAP_TOP, true), W(0.006, 1.776, 0.172, 0.156, 0.16, CAP_TOP),
        W(0.006, 1.776, 0.172, 0.156, 0.16, CAP_TOP, true), W(0.004, 1.79, 0.11, 0.1, 0.12, CAP_TOP)], 14, { b: [HEAD], end: [0.004, 1.797, 0] });
      const rows = [];
      for (const drop of [0, -0.007]) for (const reach of [0, 1]) { rows.push(B.n); for (let j = 0; j < 7; j++) { const a = (0.5 - j / 6) * 2.1, k = cos(a * 0.75); B.v([cos(a) * 0.134 + reach * 0.105 * k * k, 1.69 + cos(a) * 0.012 - reach * 0.024 + drop, sin(a) * (0.115 + reach * 0.012)], PEAK, [HEAD]); } }
      B.strip(CLOTH, rows[0], rows[1], 7, true); B.strip(CLOTH, rows[3], rows[2], 7, true); B.done();
    },
  };

  /* ---------------- one kit (geometry + materials) per cast id, shared by everyone made from it ---------------- */
  const KITS = {}, SKINS = {};
  const CLOTH_MAT = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.85 }), HAIR_MAT = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55 }), HAIR_MATTE = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 });
  /* printed shirts: where a vertex carries ink (rgb = second colour, a = 1 floral, 2 leopard) the pattern is drawn in the figure's own
     rest-pose space, so it stays on the cloth as the body moves. Everyone shares this material; plain clothes carry no ink. */
  CLOTH_MAT.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace("#include <common>", "#include <common>\nattribute vec4 ink;\nvarying vec4 vInk;\nvarying vec3 vRest;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvInk = ink; vRest = position;");
    sh.fragmentShader = sh.fragmentShader.replace("#include <common>", "#include <common>\nvarying vec4 vInk;\nvarying vec3 vRest;")
      .replace("#include <color_fragment>", `#include <color_fragment>
      if (vInk.a > 0.5) {
        float n;
        if (vInk.a > 1.5) { vec3 q = vRest * 24.0, id = floor(q), c = fract(q) - 0.5; float h = fract(sin(dot(id, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
          n = step(length(c + (h - 0.5) * 0.3), 0.2 + 0.14 * h) * step(0.25, h); }                                   // leopard: uneven spots, some cells left bare
        else { vec3 q = vRest * 30.0; n = step(0.12, sin(q.x * 1.3 + sin(q.y * 1.7) * 1.5) * sin(q.y * 1.1 + sin(q.z * 1.9 + q.x) * 1.5) + 0.3 * sin(q.z * 2.3)); }   // floral: big overlapping leaf shapes
        diffuseColor.rgb = mix(diffuseColor.rgb, vInk.rgb, n);
      }`);
  };
  function kitFor(cast) {
    if (KITS[cast.id]) return KITS[cast.id];
    if (!STYLES[cast.hair]) throw new Error("people: no model for the hairstyle '" + cast.hair + "'");
    const B = builder(), look = lookFor(cast); if (cast.pattern) B.setPrint(look.shirt, lin(cast.pattern), cast.print === "leopard" ? 2 : 1); body(B, look); STYLES[cast.hair](B, look);
    const skin = SKINS[cast.skin] || (SKINS[cast.skin] = new THREE.MeshStandardMaterial({ color: cast.skin, roughness: 0.62 }));   // the cast's skin colour as-is
    return (KITS[cast.id] = { geo: B.geometry(), mats: [skin, CLOTH_MAT, cast.hair === "afro" ? HAIR_MATTE : HAIR_MAT], look });   // coils scatter light: no sheen on the afro
  }

  NH.people = {
    make(cast) {
      const kit = kitFor(cast), G = kit.look.gait, group = new THREE.Group(), bones = REST.map(() => new THREE.Bone());
      REST.forEach((r, i) => { const q = REST[r[0]]; bones[i].position.set(r[1] - (q ? q[1] : 0), r[2] - (q ? q[2] : 0), r[3] - (q ? q[3] : 0)); if (q) bones[r[0]].add(bones[i]); });
      const mesh = new THREE.SkinnedMesh(kit.geo, kit.mats); mesh.castShadow = true; mesh.receiveShadow = true; mesh.boundingSphere = BOUNDS;
      mesh.add(bones[HIPS]); mesh.bind(new THREE.Skeleton(bones, INVERSE), IDENTITY); group.add(mesh);
      const carry = new THREE.Object3D(); carry.position.set(0.55, 1.05 - REST[CHEST][2], 0); bones[CHEST].add(carry);
      const hips = bones[HIPS], chest = bones[CHEST], head = bones[HEAD], armL = bones[ARM_L], foreL = bones[FORE_L], armR = bones[ARM_R], foreR = bones[FORE_R];
      const legL = bones[LEG_L], shinL = bones[SHIN_L], footL = bones[FOOT_L], legR = bones[LEG_R], shinR = bones[SHIN_R], footR = bones[FOOT_R], sway = bones[SWAY];
      let t = kit.look.phase, go = 0, hold = 0;
      return {
        group, carry, mesh, hair: cast.hair,
        update(dt, st) {
          t += dt;
          go = st.moving ? 1 : max(0, go - dt * 6);            // into a stride or a carry at once (the crate appears at once), out of it with an ease
          hold = st.carrying ? 1 : max(0, hold - dt * 5);
          const s = sin(st.walk), c = cos(st.walk), still = 1 - go, breath = sin(t * 1.9), shift = sin(t * 0.63);
          const A = G.stride * (1 - 0.3 * hold) * go, tw = G.twist * (1 - 0.6 * hold) * s * go, lean = G.lean * go * (1 - hold);
          hips.position.y = HIP_Y - (G.bob * s * s + 0.012 * hold) * go - 0.012 * hold - 0.004 * still * (1 + shift);
          hips.rotation.set(0.025 * shift * still, -tw, -lean - 0.02 * hold);
          chest.rotation.set(-0.02 * shift * still, tw * 1.9, 0.012 * breath * still + 0.06 * hold - lean * 0.4);
          head.rotation.set(0, -tw * 0.9, lean * 0.9 - 0.03 * hold - 0.008 * breath * still);
          sway.rotation.set(0.03 * shift * still, 0, (0.07 * sin(st.walk * 2 + 0.9) - 0.05) * go + 0.012 * breath * still);
          // legs: the thigh swings, the knee folds as the leg comes through, the foot stays near level
          const kL = (0.2 + 0.8 * max(0, c)) * A / 0.55 + 0.06 * hold + 0.03 * still, kR = (0.2 + 0.8 * max(0, -c)) * A / 0.55 + 0.06 * hold + 0.05 * still * (1 + shift);
          const tL = s * A + 0.1 * go + 0.03 * hold, tR = -s * A + 0.1 * go + 0.03 * hold + 0.03 * still * (1 + shift);
          legL.rotation.set(0.02, 0, tL); shinL.rotation.z = -kL; footL.rotation.z = (kL - tL) * 0.75;
          legR.rotation.set(-0.02, 0, tR); shinR.rotation.z = -kR; footR.rotation.z = (kR - tR) * 0.75;
          // arms: swing against the legs, or come forward to hold the crate
          if (hold < 1) {
            const aw = s * A * G.arm, out = 0.05 + (0.04 + 0.012 * breath) * still;   // a small swing: the hands stay inside the seen-through-walls capsule
            armL.rotation.set(out, 0, -aw + 0.04 * still); foreL.rotation.set(0, 0, 0.14 + 0.1 * go);
            armR.rotation.set(-out, 0, aw + 0.04 * still); foreR.rotation.set(0, 0, 0.14 + 0.1 * go);
          }
          if (hold > 0) {
            if (hold < 1) { armL.quaternion.slerp(CARRY.armL, hold); foreL.quaternion.slerp(CARRY.foreL, hold); armR.quaternion.slerp(CARRY.armR, hold); foreR.quaternion.slerp(CARRY.foreR, hold); }
            else { armL.quaternion.copy(CARRY.armL); foreL.quaternion.copy(CARRY.foreL); armR.quaternion.copy(CARRY.armR); foreR.quaternion.copy(CARRY.foreR); }
          }
        },
      };
    },
  };
})();
