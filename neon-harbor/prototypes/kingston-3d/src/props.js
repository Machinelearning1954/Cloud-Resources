/* props: vehicles and objects. Everything is built facing +X, at real size, on y = 0.
   API (main.js calls exactly this; every name, argument and returned field below is kept):
     NH.props.car()                      -> { group, update(dt, st) }   st = { sees: bool, night: 0..1, t: seconds }
                                            patrol car, 4.4 m long, 1.8 m wide (2 m over the mirrors), 1.55 m tall. The roof bar is a calm
                                            blue and flashes red and blue while st.sees; the glow of its lights on the road follows st.night
     NH.props.crate()                    -> THREE.Object3D   (1 m cube, origin at its base centre)
     NH.props.speaker()                  -> THREE.Object3D   (1.5 m wide, 2 m tall, 1.2 m deep, origin at base centre, drivers on the +X face)
     NH.props.palm(i)                    -> THREE.Object3D   (i = index, for variety: lean, height, 7 to 9 fronds; 6.1 to 6.5 m tall)
     NH.props.cart(i)                    -> THREE.Object3D   (i = index, for variety: a street vendor's handcart heaped with green
                                            coconuts under a striped shade, 1.7 m long, handles toward +X, origin at base centre; 1 draw call)
     NH.props.pickup(i)                  -> THREE.Object3D   (i = index, for variety: a parked, sun-faded pickup truck, 5.2 m long,
                                            1.9 m wide, cab toward +X, origin at base centre; 1 draw call)
     NH.props.mural(i)                   -> THREE.Object3D   (i = 0 or 1: a free-standing block wall, 6 m long and 2.4 m tall, its +X face
                                            hand-painted: harbour sunset, palms, a speaker stack and lettering for the cast's own venues;
                                            2 draw calls)
     NH.props.clothesline()              -> THREE.Object3D   (two posts 2.4 m apart along X with a line of three shirts pegged out;
                                            where the player changes shirt; 1 draw call)
     NH.props.plate()                    -> THREE.Object3D   (a dub plate: a 12-inch record standing on edge, 0.5 m across so it reads
                                            from the game camera, with a yellow label; a collectible; 1 draw call)
     NH.props.skiff(colorHex)            -> THREE.Object3D   (open boat, 5.4 m hull plus a small outboard, bow toward +X, origin at waterline)
     NH.props.lampPost()                 -> { group, setGlow(k) }   k = 0..1; post at the origin, an arm reaching NH.LAMP_ARM along +X,
                                            the lamp head about 6 m up at the arm's end, and a soft pool of light on the ground under the head
     NH.props.stringLights(a, b, count)  -> { group, setGlow(k) }   a, b = THREE.Vector3 ends of the string; k may pass 1 a little
     NH.props.visionCone(range, half)    -> { mesh, set(sees) }     flat sector on the ground, +X forward; range in metres, half in radians
     NH.props.dropRing(radius)           -> THREE.Object3D   (dashed ring on the ground, radius in metres)
     NH.props.arrow()                    -> THREE.Object3D   (flat arrow on the ground pointing +X, 1.4 m long) with setColor(hex)

   How it is drawn. Every prop is merged by hand into one mesh per material, and geometries, textures and materials are
   built once and shared, so a second crate, palm or car costs no new geometry. Draw calls: crate 1, speaker 1, palm 1,
   skiff 1, lamp post 2 (1 by day), string of lights 2, vision cone 1, ring 1, arrow 1, car 3.
     solid   one vertex-coloured MeshStandardMaterial for almost everything. Each vertex also carries a column of a tiny
             palette texture that sets its roughness, so paint, glass, rubber and steel share a draw call.
     light   nothing here is a THREE.Light. Lamps, bulbs and car lights are unlit bright meshes plus flat "glow" quads on
             the ground and around the lamp. A glow quad's colour adds light and its alpha takes light away, so one mesh
             holds both a car's headlight pools and the soft contact shadow under it. The strengths were set by eye
             against the placeholder sun and exposure: lampGlow, carGlow and stringLights hold the numbers to retune.
             Every glow mesh is tagged userData.overlay = true, the tag main.js gives to things that are drawn on the
             place without being part of it.
     marks   the vision cone, drop ring and arrow are plain MeshBasicMaterial meshes with per-vertex alpha, not tone
             mapped, so they keep their palette colour at any hour. Each bright edge carries a dark hairline: that is what
             makes a pale shape read on pale daylight concrete. visionCone.set() changes the material's colour and
             opacity and swaps in a bolder copy of the sector (thicker edge, stronger fill) while a patrol sees. */
(function () {
  "use strict";
  const NH = window.NH, P = NH.PALETTE, PI = Math.PI, TAU = PI * 2;
  const sin = Math.sin, cos = Math.cos, sqrt = Math.sqrt, min = Math.min, max = Math.max, floor = Math.floor;

  /* ---------------- build-time helpers (plain arrays as vectors; none of this runs per frame) ---------------- */
  const lin = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };   // vertex colours are in the renderer's linear space
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const crs = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]], dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const nrm = (a) => mul(a, 1 / (Math.hypot(a[0], a[1], a[2]) || 1));
  const lerp = (a, b, t) => a + (b - a) * t, clamp01 = (x) => max(0, min(1, x)), ease = (t) => t * t * (3 - 2 * t);
  const rnd = (i, k) => { let h = Math.imul(i + 11 | 0, 374761393) ^ Math.imul(k + 5 | 0, 668265263); h = Math.imul(h ^ h >>> 13, 1274126177); return ((h ^ h >>> 16) >>> 0) / 4294967296; };   // 0..1 from two indices
  const light = (c, k) => [c[0] * k, c[1] * k, c[2] * k, 0], dark = (a) => [0, 0, 0, a];   // glow colours: light added, light taken away
  const cache = {}, once = (key, make) => cache[key] || (cache[key] = make());
  const WHITE = [1, 1, 1];

  /* A bag is a triangle soup: positions, colours (r, g, b, a) and uvs. Normals are worked out at the end. */
  const ROUGHNESS = [0.92, 0.34, 0.1, 0.36, 0.6, 0.85, 0.48, 0.78];   // the roughness of each column of the palette texture
  const MATTE = 0, PAINT = 1, GLASS = 2, METAL = 3, SATIN = 4, GLOW = 5, VARNISH = 6, RUBBER = 7;
  const bag = () => ({ p: [], c: [], u: [] });
  function tri(b, A, B, C, cA, cB, cC, u) { // u = a palette column, or six numbers (a real uv per corner)
    b.p.push(A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2]);
    b.c.push(cA[0], cA[1], cA[2], cA.length > 3 ? cA[3] : 1, cB[0], cB[1], cB[2], cB.length > 3 ? cB[3] : 1, cC[0], cC[1], cC[2], cC.length > 3 ? cC[3] : 1);
    if (u && u.length) b.u.push(u[0], u[1], u[2], u[3], u[4], u[5]); else { const k = ((u || 0) + 0.5) / ROUGHNESS.length; b.u.push(k, 0.5, k, 0.5, k, 0.5); }
  }
  function quad(b, A, B, C, D, col, u) { // corners in order round the face, counter-clockwise from the front; col = one colour or one per corner; u = column or [u0, v0, u1, v1]
    const c = Array.isArray(col[0]) ? col : [col, col, col, col], r = u && u.length;
    tri(b, A, B, C, c[0], c[1], c[2], r ? [u[0], u[1], u[2], u[1], u[2], u[3]] : u); tri(b, A, C, D, c[0], c[2], c[3], r ? [u[0], u[1], u[2], u[3], u[0], u[3]] : u);
  }
  function upQuad(b, A, B, C, D, col, u, down) { // the same, turned so that it faces up (or down) whatever order the corners came in
    if ((B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2]) >= 0 !== !!down) quad(b, A, B, C, D, col, u);
    else quad(b, A, D, C, B, Array.isArray(col[0]) ? [col[0], col[3], col[2], col[1]] : col, u);
  }
  function upTri(b, A, B, C, cA, cB, cC, u) { if ((B[2] - A[2]) * (C[0] - A[0]) - (B[0] - A[0]) * (C[2] - A[2]) >= 0) tri(b, A, B, C, cA, cB, cC, u); else tri(b, A, C, B, cA, cC, cB, u); }
  const _e = new THREE.Euler(), _q = new THREE.Quaternion(), _t = new THREE.Vector3(), _s = new THREE.Vector3(), _v = new THREE.Vector3();
  const at = (x, y, z, sx, sy, sz, rx, ry, rz) => new THREE.Matrix4().compose(_t.set(x, y, z), _q.setFromEuler(_e.set(rx || 0, ry || 0, rz || 0)), _s.set(sx, sy, sz));
  const BOX = new THREE.BoxGeometry(1, 1, 1), cyl = (n) => once("cyl" + n, () => new THREE.CylinderGeometry(0.5, 0.5, 1, n)), ball = (w, h) => once("ball" + w + "/" + h, () => new THREE.SphereGeometry(1, w, h));
  function solid(b, geo, m, col, u) { // a three.js primitive, moved by matrix m. col = a colour or (point) => colour; u = column or a uv rectangle [u0, v0, u1, v1]
    const pos = geo.attributes.position, uv = geo.attributes.uv, idx = geo.index, n = idx ? idx.count : pos.count, flip = m.determinant() < 0, pts = [], fn = typeof col === "function", r = u && u.length;
    for (let i = 0; i < pos.count; i++) { _v.fromBufferAttribute(pos, i).applyMatrix4(m); pts.push([_v.x, _v.y, _v.z]); }
    for (let i = 0; i < n; i += 3) {
      const a = idx ? idx.getX(i) : i, j = idx ? idx.getX(i + (flip ? 2 : 1)) : i + (flip ? 2 : 1), k = idx ? idx.getX(i + (flip ? 1 : 2)) : i + (flip ? 1 : 2);
      tri(b, pts[a], pts[j], pts[k], fn ? col(pts[a]) : col, fn ? col(pts[j]) : col, fn ? col(pts[k]) : col,
        r ? [lerp(u[0], u[2], uv.getX(a)), lerp(u[1], u[3], uv.getY(a)), lerp(u[0], u[2], uv.getX(j)), lerp(u[1], u[3], uv.getY(j)), lerp(u[0], u[2], uv.getX(k)), lerp(u[1], u[3], uv.getY(k))] : u);
    }
  }
  /* A skin over a row of rings (each a list of points). It faces away from the rings' common centre, or along
     o.toward, or toward the centre with o.inward. col and u may be functions of (ring, point). */
  function loft(b, rings, col, u, o) {
    o = o || {}; const n = rings[0].length, m = o.closed ? n : n - 1; let cx = 0, cy = 0, cz = 0, k = 0, vote = 0;
    for (const r of rings) for (const q of r) { cx += q[0]; cy += q[1]; cz += q[2]; k++; } cx /= k; cy /= k; cz /= k;
    const each = (fn) => { for (let i = 0; i + 1 < rings.length; i++) for (let j = 0; j < m; j++) fn(rings[i][j], rings[i][(j + 1) % n], rings[i + 1][(j + 1) % n], rings[i + 1][j], i, j); };
    each((A, B, C, D) => { const N = crs(sub(C, A), sub(D, B)); vote += o.toward ? dot(N, o.toward) : N[0] * ((A[0] + C[0]) / 2 - cx) + N[1] * ((A[1] + C[1]) / 2 - cy) + N[2] * ((A[2] + C[2]) / 2 - cz); });
    const flip = (vote < 0) !== !!o.inward;
    each((A, B, C, D, i, j) => { const c = typeof col === "function" ? col(i, j) : col, uu = typeof u === "function" ? u(i, j) : u;
      if (flip) quad(b, A, D, C, B, Array.isArray(c[0]) ? [c[0], c[3], c[2], c[1]] : c, uu); else quad(b, A, B, C, D, c, uu); });
  }
  function cap(b, ring, dir, col, u) { // closes a ring with a fan that faces dir
    const n = ring.length, c = [0, 0, 0]; for (const q of ring) { c[0] += q[0] / n; c[1] += q[1] / n; c[2] += q[2] / n; }
    for (let j = 0; j < n; j++) { const A = ring[j], B = ring[(j + 1) % n]; if (dot(crs(sub(A, c), sub(B, c)), dir) >= 0) tri(b, c, A, B, col, col, col, u); else tri(b, c, B, A, col, col, col, u); }
  }
  function tube(b, pts, rad, sides, col, u, ref) { // a pipe along pts; rad = a radius or (k) => radius; ref = any direction the pipe never runs along
    loft(b, pts.map((p, k) => {
      const t = nrm(sub(pts[min(k + 1, pts.length - 1)], pts[max(k - 1, 0)])), n1 = nrm(crs(t, ref)), n2 = crs(t, n1), r = typeof rad === "function" ? rad(k) : rad, ring = [];
      for (let j = 0; j < sides; j++) { const a = j / sides * TAU; ring.push(add(p, add(mul(n1, cos(a) * r), mul(n2, sin(a) * r)))); }
      return ring;
    }), col, u, { closed: true });
  }
  function rev(b, cy, cz, prof, seg, col, u, toward) { // turned on a lathe about an axis along X through (cy, cz); prof = [[x, radius], ...]
    loft(b, prof.map(([x, r]) => { const ring = []; for (let j = 0; j < seg; j++) { const a = j / seg * TAU; ring.push([x, cy + r * cos(a), cz + r * sin(a)]); } return ring; }), col, u, { closed: true, toward });
  }
  function ann(b, r0, r1, a0, a1, seg, c0, c1, y, down) { // a flat ring sector on the ground, facing up (or down); colour c0 at r0, c1 at r1
    for (let k = 0; k < seg; k++) { const a = lerp(a0, a1, k / seg), e = lerp(a0, a1, (k + 1) / seg);
      upQuad(b, [r0 * cos(a), y || 0, r0 * sin(a)], [r0 * cos(e), y || 0, r0 * sin(e)], [r1 * cos(e), y || 0, r1 * sin(e)], [r1 * cos(a), y || 0, r1 * sin(a)], [c0, c0, c1 || c0, c1 || c0], 0, down); }
  }
  function glow(b, x, y, z, hx, hz, c, rot) { // a flat quad carrying the whole glow texture: a soft round spot hx by hz
    const cr = cos(rot || 0), sr = sin(rot || 0), pt = (dx, dz) => [x + dx * cr - dz * sr, y, z + dx * sr + dz * cr];
    quad(b, pt(-hx, hz), pt(hx, hz), pt(hx, -hz), pt(-hx, -hz), c, [0, 0, 1, 1]);
  }
  /* Normals: every corner takes the area-weighted normals of the faces that meet at its point and lie within `crease`
     degrees of its own face, so gentle bends shade smooth and hard edges stay hard. */
  function normals(p, crease) {
    const nt = p.length / 9, fn = new Float32Array(nt * 3), fl = new Float32Array(nt), out = new Float32Array(p.length), spot = new Map(), keys = new Array(nt * 3), cc = cos(crease * PI / 180);
    for (let t = 0; t < nt; t++) {
      const o = t * 9, ux = p[o + 3] - p[o], uy = p[o + 4] - p[o + 1], uz = p[o + 5] - p[o + 2], vx = p[o + 6] - p[o], vy = p[o + 7] - p[o + 1], vz = p[o + 8] - p[o + 2];
      const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx; fn[t * 3] = x; fn[t * 3 + 1] = y; fn[t * 3 + 2] = z; fl[t] = Math.hypot(x, y, z);
      for (let k = 0; k < 3; k++) { const q = o + k * 3, key = Math.round(p[q] * 4096) + "/" + Math.round(p[q + 1] * 4096) + "/" + Math.round(p[q + 2] * 4096), l = spot.get(key); keys[t * 3 + k] = key; if (l) l.push(t); else spot.set(key, [t]); }
    }
    for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) {
      let x = 0, y = 0, z = 0;
      for (const s of spot.get(keys[t * 3 + k])) if (s === t || fn[t * 3] * fn[s * 3] + fn[t * 3 + 1] * fn[s * 3 + 1] + fn[t * 3 + 2] * fn[s * 3 + 2] >= cc * fl[t] * fl[s] && fl[s] > 0) { x += fn[s * 3]; y += fn[s * 3 + 1]; z += fn[s * 3 + 2]; }
      const l = Math.hypot(x, y, z) || 1, q = t * 9 + k * 3; out[q] = x / l; out[q + 1] = y / l; out[q + 2] = z / l;
    }
    return out;
  }
  function finish(b, crease, alpha) { // crease < 0: no normals (an unlit mesh). alpha: keep the colours' fourth number
    const g = new THREE.BufferGeometry(), nv = b.p.length / 3;
    g.setAttribute("position", new THREE.Float32BufferAttribute(b.p, 3));
    if (alpha) g.setAttribute("color", new THREE.Float32BufferAttribute(b.c, 4));
    else { const c = new Float32Array(nv * 3); for (let i = 0; i < nv; i++) { c[i * 3] = b.c[i * 4]; c[i * 3 + 1] = b.c[i * 4 + 1]; c[i * 3 + 2] = b.c[i * 4 + 2]; } g.setAttribute("color", new THREE.BufferAttribute(c, 3)); }
    g.setAttribute("uv", new THREE.Float32BufferAttribute(b.u, 2));
    if (crease >= 0) g.setAttribute("normal", new THREE.BufferAttribute(normals(b.p, crease), 3));
    g.computeBoundingSphere(); return g;
  }

  /* ---------------- textures and materials: made once, shared by every instance ---------------- */
  const canvas = (w, h, draw) => { const c = document.createElement("canvas"); c.width = w; c.height = h; draw(c.getContext("2d")); return c; };
  const css = (hex, k) => "rgb(" + [hex >> 16 & 255, hex >> 8 & 255, hex & 255].map((v) => Math.round(min(255, v * k))).join(",") + ")";
  const strip = (cols) => { const t = new THREE.CanvasTexture(canvas(cols.length, 1, (g) => cols.forEach((c, i) => { g.fillStyle = c; g.fillRect(i, 0, 1, 1); }))); t.magFilter = t.minFilter = THREE.NearestFilter; t.generateMipmaps = false; return t; };
  const glowTex = () => once("glowTex", () => {
    const t = new THREE.CanvasTexture(canvas(128, 128, (g) => {
      const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
      for (let k = 0; k <= 20; k++) grd.addColorStop(k / 20, "rgba(255,255,255," + Math.pow(1 - k * k / 400, 2.2).toFixed(4) + ")");
      g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
    }));
    t.premultiplyAlpha = true; return t;   // the texels arrive as (g, g, g, g): the same soft spot scales the light added and the light taken away
  });
  const crateTex = () => once("crateTex", () => {
    const t = new THREE.CanvasTexture(canvas(256, 256, (g) => {
      for (let k = 0; k < 5; k++) { // five boards, each its own tone, with grain
        const y0 = k * 51.2; g.fillStyle = css(P.crate, 0.88 + 0.24 * rnd(k, 1)); g.fillRect(0, y0, 256, 52);
        for (let s = 0; s < 34; s++) { const y = y0 + 4 + rnd(k * 40 + s, 2) * 44, x = rnd(k * 40 + s, 3) * 300 - 40, len = 30 + rnd(k * 40 + s, 4) * 130;
          g.strokeStyle = rnd(k * 40 + s, 5) < 0.55 ? "rgba(70,40,10,0.2)" : "rgba(255,236,190,0.16)"; g.lineWidth = 1 + rnd(k * 40 + s, 6); g.beginPath(); g.moveTo(x, y); g.lineTo(x + len, y + (rnd(k * 40 + s, 7) - 0.5) * 4); g.stroke(); }
        g.fillStyle = "rgba(34,20,6,0.85)"; g.fillRect(0, y0 - 1.5, 256, 3); g.fillStyle = "rgba(255,238,200,0.22)"; g.fillRect(0, y0 + 1.5, 256, 1.5);
        for (const x of [34, 222]) for (const dy of [14, 38]) { g.fillStyle = "rgba(30,22,16,0.8)"; g.beginPath(); g.arc(x + (rnd(k, 8 + dy) - 0.5) * 3, y0 + dy, 2.2, 0, TAU); g.fill(); }
      }
      g.strokeStyle = g.fillStyle = "rgba(26,20,22,0.9)";   // the mark: a speaker cone, stencilled, so the ring is broken by four bridges
      for (let q = 0; q < 4; q++) { g.lineWidth = 15; g.beginPath(); g.arc(128, 128, 60, q * PI / 2 + 0.09, (q + 1) * PI / 2 - 0.09); g.stroke(); g.lineWidth = 5; g.beginPath(); g.arc(128, 128, 39, q * PI / 2 + 0.87, (q + 1) * PI / 2 + 0.7); g.stroke(); }
      g.beginPath(); g.arc(128, 128, 23, 0, TAU); g.fill();
      g.fillStyle = "#fff"; g.fillRect(0, 0, 256, 17); g.fillRect(0, 238, 18, 18);                                          // under the frame: pale grain for the frame boards, plain white for steel and rope
      for (let s = 0; s < 40; s++) { g.fillStyle = "rgba(40,20,0," + (0.06 + 0.12 * rnd(s, 9)).toFixed(3) + ")"; g.fillRect(rnd(s, 10) * 230, 1 + rnd(s, 11) * 14, 20 + rnd(s, 12) * 60, 1.2); }
    }));
    t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
  });
  const MURALS = [
    { words: ["BASSLINE", "SUNDAYS"], sky: ["#2a1b4a", "#c8466a", "#f6a24a"], sea: "#1f6f86", accent: "#f2c230" },
    { words: ["BRASS & KEEL", "BOATWORKS"], sky: ["#123a5a", "#2f8f9a", "#f2d28a"], sea: "#0f4f63", accent: "#e8573a" },
  ];
  const muralTex = (i) => once("muralTex" + i, () => { // painted in code, like everything else here: no image files
    const m = MURALS[i % MURALS.length];
    const t = new THREE.CanvasTexture(canvas(640, 256, (g) => {
      const sky = g.createLinearGradient(0, 0, 0, 170); m.sky.forEach((c, k) => sky.addColorStop(k / (m.sky.length - 1), c)); g.fillStyle = sky; g.fillRect(0, 0, 640, 256);
      g.fillStyle = "#ffd36b"; g.beginPath(); g.arc(470, 150, 46, 0, TAU); g.fill();                                   // the sun going down
      g.fillStyle = m.sea; g.fillRect(0, 160, 640, 96);
      g.fillStyle = "rgba(255,220,140,0.55)"; for (let k = 0; k < 9; k++) g.fillRect(430 + (k % 3) * 9 - k * 2, 168 + k * 9, 70 - k * 6, 3);   // its glitter on the water
      g.fillStyle = "#14141a"; for (const [x, h] of [[70, 150], [150, 120], [585, 140]]) {                             // palms in silhouette
        g.fillRect(x - 3, 200 - h, 6, h); for (let f = 0; f < 7; f++) { const a = -PI + f * PI / 6; g.beginPath(); g.ellipse(x + cos(a) * 26, 200 - h + sin(a) * 10, 30, 6, a, 0, TAU); g.fill(); }
      }
      g.fillStyle = "#14141a"; g.fillRect(250, 120, 60, 80); g.fillRect(316, 120, 60, 80); g.fillRect(283, 64, 60, 54);   // a sound-system stack
      g.fillStyle = m.accent; for (const [x, y, r] of [[280, 160, 20], [346, 160, 20], [313, 91, 15]]) { g.beginPath(); g.arc(x, y, r, 0, TAU); g.fill(); g.fillStyle = "#14141a"; g.beginPath(); g.arc(x, y, r * 0.45, 0, TAU); g.fill(); g.fillStyle = m.accent; }
      g.textAlign = "center"; g.lineJoin = "round";
      m.words.forEach((w, k) => { const y = 54 + k * 46, size = k ? 34 : 46; g.font = "900 " + size + "px Impact, 'Arial Black', sans-serif";
        g.lineWidth = 9; g.strokeStyle = "#14141a"; g.strokeText(w, 320, y + (k ? 160 : 0)); g.fillStyle = k ? "#f4efe2" : m.accent; g.fillText(w, 320, y + (k ? 160 : 0)); });
      g.fillStyle = "rgba(0,0,0,0.18)"; for (let k = 0; k < 220; k++) g.fillRect((k * 97) % 640, (k * 53) % 256, 2 + (k % 3), 1);   // block joints and chips showing through
    }));
    t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
  });
  const mats = () => once("mats", () => {
    const pal = strip(ROUGHNESS.map((r) => "rgb(0," + Math.round(r * 255) + ",0)")), emis = strip(ROUGHNESS.map((_, i) => i === GLOW ? "#fff" : "#000"));
    const std = (o) => new THREE.MeshStandardMaterial(Object.assign({ vertexColors: true, roughness: 1, metalness: 0, roughnessMap: pal }, o));   // nothing is a true metal: with no sky to mirror, one would only turn dark
    const mark = (o) => new THREE.MeshBasicMaterial(Object.assign({ vertexColors: true, transparent: true, depthWrite: false, fog: false, toneMapped: false }, o));
    return {
      solid: std({ name: "props" }), leaf: std({ name: "palm", side: THREE.DoubleSide }), speaker: std({ name: "speaker", emissive: P.sodium, emissiveMap: emis, emissiveIntensity: 0.9 }),
      crate: new THREE.MeshStandardMaterial({ name: "crate", map: crateTex(), vertexColors: true, roughness: 0.82, metalness: 0 }),
      lit: new THREE.MeshBasicMaterial({ name: "car lights", vertexColors: true }), ring: mark({ color: P.go }), mark,
      // colour adds, alpha darkens: out = rgb + (1 - alpha) * what is behind
      glow: () => new THREE.MeshBasicMaterial({ name: "glow", map: glowTex(), vertexColors: true, transparent: true, depthWrite: false, fog: false, blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor }),
    };
  });
  const mesh = (geo, mat, shadow) => { const m = new THREE.Mesh(geo, mat); if (shadow) m.castShadow = m.receiveShadow = true; return m; };
  // A glow mesh is a painted stand-in for light, not a thing in the place. main.js tags such objects userData.overlay, and
  // whatever reads the scene as geometry (the scene exporter in dev/) leaves them out; the glows carry the same tag.
  const glowMesh = (geo, mat) => { const m = mesh(geo, mat); m.userData.overlay = true; m.renderOrder = 1; return m; };

  /* ---------------- patrol car ---------------- */
  const XF = 2.13, XR = -2.17, WX = 1.32;   // nose, tail, axles
  function carShell() {
    const b = bag(), BODY = lin(P.patrolBody), BAND = lin(P.patrolBand), TRIM = lin(0x22262d), TYRE = lin(0x131417), STEEL = lin(0xaab1b7), UNDER = lin(0x090a0c), G0 = lin(0x0e1620), G1 = lin(0x3a546c), SILL = mul(BODY, 0.7), LOW = mul(BODY, 0.9);
    const keys = [[XR, 0.82], [XR + 0.2, 0.93], [-1.48, 0.965], [0.98, 0.955], [1.85, 0.87], [XF, 0.78]];
    const deck = (x) => { for (let k = 1; k < keys.length; k++) if (x <= keys[k][0]) return lerp(keys[k - 1][1], keys[k][1], ease(clamp01((x - keys[k - 1][0]) / (keys[k][0] - keys[k - 1][0])))); return 0.78; };
    const tip = (x) => clamp01(x > 0 ? (x - XF + 0.42) / 0.42 : (XR + 0.42 - x) / 0.42), half = (x) => 0.9 - 0.25 * (1 - sqrt(1 - 0.96 * tip(x) * tip(x)));
    const st = [], flat = (x) => st.push([x, 0.25 + 0.07 * tip(x)]), arch = (xc) => { for (let k = 0; k <= 10; k++) { const a = PI + 0.2 - k * (PI + 0.4) / 10; st.push([xc + 0.4 * cos(a), 0.33 + 0.4 * sin(a)]); } };
    [XR, XR + 0.015, XR + 0.05, XR + 0.11, XR + 0.2, XR + 0.3, XR + 0.42].forEach(flat); arch(-WX); [-0.8, -0.3, 0.3, 0.8].forEach(flat); arch(WX); [XF - 0.36, XF - 0.28, XF - 0.19, XF - 0.11, XF - 0.05, XF - 0.015, XF].forEach(flat);
    // body: an upside-down U at every station, from the sill up over the deck and down again; the sill rises into the wheel arches
    const rings = st.map(([x, yb]) => { const w = half(x), d = deck(x), R = [[yb, w - 0.03], [max(yb, 0.47), w], [max(yb, 0.7), w], [d - 0.1, w - 0.015], [d, w - 0.11], [d + 0.03, (w - 0.11) * 0.5]], ring = R.map(([y, z]) => [x, y, z]);
      ring.push([x, d + 0.04, 0]); for (let k = R.length - 1; k >= 0; k--) ring.push([x, R[k][0], -R[k][1]]); return ring; });
    loft(b, rings, (i, j) => { const s = min(j, 11 - j), x = (st[i][0] + st[i + 1][0]) / 2; return s === 0 ? (tip(x) > 0.02 ? TRIM : j ? [LOW, SILL, SILL, LOW] : [SILL, LOW, LOW, SILL]) : s === 1 && x < 1.78 || s > 3 && x < -1.53 ? BAND : BODY; }, (i, j) => min(j, 11 - j) === 0 && tip((st[i][0] + st[i + 1][0]) / 2) > 0.02 ? SATIN : PAINT);
    cap(b, rings[rings.length - 1], [1, 0, 0], BODY, PAINT); cap(b, rings[0], [-1, 0, 0], BODY, PAINT);
    solid(b, BOX, at(-0.03, 0.5, 0, 4.05, 0.6, 1.3), UNDER, MATTE);
    // cabin: glass all round, pillars and roof in body colour, a navy hoop over the roof under the light bar
    const cab = [[0.98, 0, 0.8], [0.22, 1, 0.63], [-0.02, 1, 0.63], [-0.14, 1, 0.63], [-0.24, 1, 0.63], [-0.36, 1, 0.63], [-0.8, 1, 0.63], [-1.48, 0, 0.79]];
    const cabin = cab.map(([x, up, wr]) => { const d = deck(x), R = up ? [[d - 0.03, 0.81], [1.4, wr], [1.406, wr - 0.07]] : [[d - 0.006, wr], [d - 0.006, wr], [d + 0.014, wr - 0.07]], ring = R.map(([y, z]) => [x, y, z]);   // the two flat ends follow the crown of the bonnet and the boot
      ring.push([x, up ? 1.43 : d + 0.05, 0]); for (let k = 2; k >= 0; k--) ring.push([x, R[k][0], -R[k][1]]); return ring; });
    const glassAt = (i, s) => s === 0 ? i !== 3 && i !== 6 : s === 2 && (i === 0 || i === 6);
    loft(b, cabin, (i, j) => { const s = min(j, 5 - j);
      if (!glassAt(i, s)) return i >= 2 && i <= 4 && (s > 0 || i === 3) ? BAND : BODY;
      if (s === 2) return i === 0 ? [G1, G1, G0, G0] : [G0, G0, G1, G1];
      return j === 0 ? [G1, G0, G0, G1] : [G0, G1, G1, G0]; }, (i, j) => glassAt(i, min(j, 5 - j)) ? GLASS : PAINT);
    solid(b, BOX, at(1.03, deck(1.03) + 0.034, 0, 0.07, 0.02, 1.5), TRIM, SATIN);                                         // cowl
    solid(b, BOX, at(-1.53, deck(-1.53) + 0.036, 0, 0.018, 0.012, 1.42), TRIM, SATIN);                                    // boot-lid seam
    solid(b, BOX, at(XF + 0.004, 0.61, 0, 0.02, 0.17, 0.8), UNDER, SATIN);                                                // grille
    for (const s of [-1, 1]) {
      solid(b, BOX, at(XF + 0.07, 0.56, s * 0.27, 0.05, 0.46, 0.05), TRIM, SATIN);                                         // push bar
      solid(b, BOX, at(0.84, 0.99, s * 0.925, 0.1, 0.075, 0.14), TRIM, SATIN);                                              // mirror
      for (const x of [0.8, -0.19]) solid(b, BOX, at(x, 0.57, s * 0.9, 0.014, 0.48, 0.012), TRIM, SATIN);                  // door seams
      for (const x of [-0.06, -0.8]) solid(b, BOX, at(x, 0.77, s * 0.9, 0.14, 0.028, 0.03), TRIM, SATIN);                  // door handles
      for (const x of [WX, -WX]) {
        solid(b, cyl(16), at(x, 0.33, s * 0.775, 0.66, 0.22, 0.66, PI / 2), TYRE, RUBBER);
        solid(b, cyl(16), at(x, 0.33, s * 0.875, 0.42, 0.04, 0.42, PI / 2), STEEL, METAL); solid(b, cyl(8), at(x, 0.33, s * 0.885, 0.15, 0.036, 0.15, PI / 2), BAND, PAINT);
      }
    }
    for (const y of [0.45, 0.67]) solid(b, BOX, at(XF + 0.07, y, 0, 0.05, 0.05, 0.9), TRIM, SATIN);
    solid(b, BOX, at(XR - 0.006, 0.55, 0, 0.012, 0.12, 0.44), lin(0xe6e2d2), MATTE);                                              // a blank plate
    solid(b, cyl(8), at(0.9, 1.07, -0.87, 0.12, 0.13, 0.12, 0, 0, PI / 2), TRIM, SATIN);                                          // spot lamp on the pillar
    solid(b, BOX, at(-0.19, 1.45, 0, 0.32, 0.05, 1.22), TRIM, SATIN); solid(b, BOX, at(-0.19, 1.5, 0, 0.26, 0.08, 0.12), STEEL, METAL);   // light bar: base and centre block
    return finish(b, 28);
  }
  const BAR_BLUE = lin(0x4a90ff), HEAD = [1, 0.93, 0.78], TAIL = lin(0xe0262f), FLASH_RED = lin(P.heat), FLASH_BLUE = lin(0x3f7bff);
  const barSet = (k) => k === 0 ? [BAR_BLUE, BAR_BLUE] : k === 1 ? [mul(FLASH_RED, 2.6), mul(FLASH_BLUE, 0.3)] : [mul(FLASH_RED, 0.3), mul(FLASH_BLUE, 2.8)];
  // The lit parts and the glow come in three versions with the same shape and different colours: calm, flash left, flash right.
  const carLit = (k) => once("carLit" + k, () => {
    const b = bag(), bar = barSet(k);
    for (const s of [-1, 1]) {
      solid(b, BOX, at(-0.19, 1.505, s * 0.34, 0.26, 0.09, 0.5), bar[s < 0 ? 0 : 1], 0);
      solid(b, BOX, at(XF + 0.01, 0.68, s * 0.55, 0.1, 0.13, 0.3, 0, -s * 0.55, 0), mul(HEAD, 1.15), 0);
      solid(b, BOX, at(XR - 0.01, 0.76, s * 0.57, 0.08, 0.12, 0.32, 0, s * 0.5, 0), mul(TAIL, 0.85), 0);
      solid(b, BOX, at(XF - 0.2, 0.5, s * 0.775, 0.13, 0.06, 0.05, 0, -s * 1.0, 0), [1, 0.5, 0.08], 0);
    }
    solid(b, BOX, at(XR - 0.004, 0.745, 0, 0.02, 0.045, 0.82), mul(TAIL, 0.5), 0);
    return finish(b, -1);
  });
  const carGlow = (k) => once("carGlow" + k, () => {
    const b = bag(), bar = barSet(k), wash = k === 0 ? 0.07 : 0.2;
    glow(b, -0.03, 0.03, 0, 3.15, 1.8, dark(0.62));                                                                         // contact shadow
    glow(b, 6.2, 0.05, 0, 4.2, 2.2, light(HEAD, 0.07)); for (const s of [-1, 1]) glow(b, 3.9, 0.052, s * 0.55, 2.0, 0.9, light(HEAD, 0.1), s * 0.06);   // headlight pools
    glow(b, -3.2, 0.05, 0, 1.5, 1.5, light(TAIL, 0.1));
    for (const s of [-1, 1]) {
      glow(b, XF + 0.06, 0.86, s * 0.55, 0.42, 0.42, light(HEAD, 0.5)); glow(b, XR - 0.02, 0.96, s * 0.57, 0.3, 0.3, light(TAIL, 0.4));
      glow(b, -0.19, 0.054, s * (k ? 1.5 : 0.6), 4.2, 4.2, light(bar[s < 0 ? 0 : 1], wash)); glow(b, -0.19, 1.62, s * 0.34, 0.75, 0.75, light(bar[s < 0 ? 0 : 1], 0.3));
    }
    return finish(b, -1, true);
  });

  /* ---------------- crate, speaker ---------------- */
  function crateGeo() {
    const b = bag(), EDGE = lin(P.crateEdge), STEEL = lin(0x8f989e), ROPE = lin(0xdccba0), FRAME = [0.06, 0.945, 0.94, 0.99], PLAIN = [0.012, 0.012, 0.06, 0.06], loop = new THREE.TorusGeometry(0.13, 0.024, 5, 9, PI);
    solid(b, BOX, at(0, 0.49, 0, 0.93, 0.94, 0.93), WHITE, [0, 0, 1, 1]);                                                  // five boarded faces, each carrying the whole texture
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      solid(b, BOX, at(sx * 0.44, 0.5, sz * 0.44, 0.12, 1, 0.12), EDGE, FRAME);
      for (const y of [0.075, 0.925]) solid(b, BOX, at(sx * 0.43, y, sz * 0.43, 0.152, 0.152, 0.152), STEEL, PLAIN);
    }
    for (const y of [0.06, 0.94]) for (const s of [-1, 1]) { solid(b, BOX, at(0, y, s * 0.44, 0.76, 0.12, 0.12), mul(EDGE, 1.12), FRAME); solid(b, BOX, at(s * 0.44, y, 0, 0.12, 0.12, 0.76), mul(EDGE, 0.94), FRAME); }
    for (const s of [-1, 1]) solid(b, loop, at(0, 0.72, s * 0.468, 1, 1, 1, PI - s * 0.32), ROPE, PLAIN);
    return finish(b, 30);
  }
  function speakerGeo() {
    const b = bag(), CAB = lin(P.speaker), LIP = mul(CAB, 1.5), STEEL = lin(0xb8c0c6), DARK = lin(0x08080a), FRAME = lin(0x3a3b45), CONE0 = lin(0x5c5f6e), CONE1 = lin(0x202127), DOME = lin(0x8b8fa0), RIM = lin(0x8a5318), BAFFLE = lin(0x2c2d36), INK = mul(lin(P.ink), 0.62);
    solid(b, BOX, at(-0.065, 1, 0, 1.05, 1.96, 1.46), (p) => mul(CAB, 0.8 + 0.35 * p[1]), SATIN);                           // the box; its front is a recessed baffle
    quad(b, [0.462, 0.02, 0.73], [0.462, 0.02, -0.73], [0.462, 1.98, -0.73], [0.462, 1.98, 0.73], [mul(BAFFLE, 0.7), mul(BAFFLE, 0.7), BAFFLE, BAFFLE], MATTE);
    for (const y of [0.045, 1.955]) solid(b, BOX, at(0.525, y, 0, 0.13, 0.06, 1.46), LIP, SATIN);
    for (const s of [-1, 1]) solid(b, BOX, at(0.525, 1, s * 0.705, 0.13, 1.96, 0.05), LIP, SATIN);
    for (const y of [0.02, 1.98]) for (const s of [-1, 1]) { solid(b, BOX, at(0, y, s * 0.73, 1.2, 0.04, 0.04), STEEL, METAL); solid(b, BOX, at(s * 0.58, y, 0, 0.04, 0.04, 1.5), STEEL, METAL); }   // steel edging
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      solid(b, BOX, at(sx * 0.58, 1, sz * 0.73, 0.04, 2, 0.04), STEEL, METAL);
      for (const y of [0.05, 1.95]) solid(b, BOX, at(sx * 0.553, y, sz * 0.703, 0.104, 0.104, 0.104), mul(STEEL, 1.1), METAL);
      solid(b, cyl(10), at(sx * 0.36, 1.985, sz * 0.5, 0.1, 0.02, 0.1), mul(STEEL, 0.5), METAL);                        // stacking cups
    }
    for (const s of [-1, 1]) { solid(b, BOX, at(0, 1.25, s * 0.733, 0.44, 0.2, 0.012), DARK, SATIN); solid(b, BOX, at(0, 1.25, s * 0.75, 0.3, 0.032, 0.032), STEEL, METAL); }   // side handles
    for (let q = 0; q < 4; q++) ann(b, 0.17, 0.235, q * PI / 2 + 0.1, (q + 1) * PI / 2 - 0.1, 5, INK, INK, 1.982); ann(b, 0, 0.075, 0, TAU, 12, INK, INK, 1.982);   // the mark, stencilled on the lid
    const prof = [[0.46, 0.385], [0.575, 0.385], [0.579, 0.358], [0.579, 0.315], [0.591, 0.294], [0.577, 0.272], [0.494, 0.1], [0.522, 0.066], [0.533, 0]];
    const tone = [[LIP, LIP], [RIM, RIM], [FRAME, FRAME], [DARK, mul(FRAME, 0.7)], [mul(FRAME, 0.7), DARK], [CONE0, CONE1], [mul(DOME, 0.7), DOME], [DOME, mul(DOME, 1.3)]], feel = [SATIN, GLOW, SATIN, RUBBER, RUBBER, MATTE, SATIN, SATIN];
    for (const cy of [0.475, 1.265]) {
      rev(b, cy, 0, prof, 24, (i) => [tone[i][0], tone[i][0], tone[i][1], tone[i][1]], (i) => feel[i], [1, 0, 0]);
      for (const dy of [-0.22, -0.075, 0.075, 0.22]) solid(b, BOX, at(0.588, cy + dy, 0, 0.012, 0.016, 2 * sqrt(0.352 * 0.352 - dy * dy)), STEEL, METAL);   // grille bars
    }
    const rect = (x, y0, y1, hz) => [[x, y0, hz], [x, y1, hz], [x, y1, -hz], [x, y0, -hz]], HORN = lin(0x74778c);         // the horn: a flared slot
    const horn = [rect(0.46, 1.675, 1.925, 0.56), rect(0.58, 1.675, 1.925, 0.56), rect(0.58, 1.695, 1.905, 0.54), rect(0.478, 1.785, 1.815, 0.1)];
    loft(b, horn, (i) => i === 0 ? LIP : i === 1 ? FRAME : [HORN, HORN, DARK, DARK], SATIN, { closed: true, toward: [1, 0, 0] }); cap(b, horn[3], [1, 0, 0], DARK, MATTE);
    return finish(b, 32);
  }

  /* ---------------- coconut palm, skiff ---------------- */
  function palmGeo(i) {
    const b = bag(), TRUNK = lin(P.trunk), LIME = lin(0xe6dfce), F1 = lin(P.frond), F0 = lin(P.frondDark), STEM = lin(0x9aa658), DRY = lin(0xa98d52), OLIVE = lin(0x8d9a46);
    const lean = 0.45 + 0.8 * rnd(i, 1), dir = rnd(i, 2) * TAU, H = 5.12 + 0.45 * rnd(i, 3), n = 7 + floor(rnd(i, 4) * 3), N = 16;
    const pts = []; for (let k = 0; k <= N; k++) { const t = k / N, o = lean * (1 - (1 - t) * (1 - t)); pts.push([cos(dir) * o, H * t, sin(dir) * o]); }
    // trunk: leans out of the ground and straightens; ringed with leaf scars, whitewashed to shoulder height
    tube(b, pts, (k) => (0.2 - 0.075 * k / N + 0.1 * Math.exp(-k / N * 16)) * (k & 1 ? 1.07 : 1), 7, (k) => mul(pts[k][1] < 0.8 ? LIME : TRUNK, (k & 1 ? 0.78 : 1.06) * (pts[k][1] < 0.8 ? 0.92 : 1)), MATTE, [0.31, 0.02, 0.95]);
    const top = pts[N]; solid(b, ball(7, 5), at(top[0], top[1] + 0.02, top[2], 0.21, 0.34, 0.21), mix(TRUNK, F0, 0.55), MATTE);
    for (let f = 0; f < n; f++) {
      const dry = f === n - 1 && rnd(i, 7) < 0.45, tier = f % 3, az = f / n * TAU + (rnd(i, 20 + f) - 0.5) * 0.45, hx = cos(az), hz = sin(az), sd = [-hz, 0, hx];
      const e0 = dry ? -0.55 : [0.95, 0.5, 0.1][tier] + (rnd(i, 40 + f) - 0.5) * 0.24, L = dry ? 2.0 : 2.5 + 0.5 * rnd(i, 60 + f), bend = dry ? 0.8 : 1.3 + 0.35 * rnd(i, 80 + f) + (tier ? 0 : 0.3), S = 8;
      const tone = 0.82 + 0.36 * rnd(i, 100 + f), cB = dry ? DRY : mul(mix(F1, OLIVE, tier === 2 ? 0.35 : 0.05), tone), cT = dry ? mul(DRY, 0.7) : mul(F0, tone), cM = mix(cB, cT, 0.5);
      const R = [[top[0] + hx * 0.08, top[1] + 0.14, top[2] + hz * 0.08]], T = [];
      for (let k = 0; k < S; k++) { const th = e0 - bend * Math.pow((k + 0.5) / S, 1.2), d = [hx * cos(th), sin(th), hz * cos(th)]; T.push(d); R.push(add(R[k], mul(d, L / S))); }
      for (let k = 0; k < S; k++) { const w0 = 0.04 * (1 - 0.7 * k / S), w1 = 0.04 * (1 - 0.7 * (k + 1) / S), c0 = mix(STEM, cB, k / S), c1 = mix(STEM, cB, (k + 1) / S);   // the rib of the frond
        upQuad(b, add(R[k], mul(sd, w0)), add(R[k + 1], mul(sd, w1)), sub(R[k + 1], mul(sd, w1)), sub(R[k], mul(sd, w0)), dry ? DRY : [c0, c1, c1, c0], MATTE); }
      for (let m = 0, M = 15; m < M; m++) for (const sg of [-1, 1]) { // leaflets: separate blades, so the ground shows between them
        const s = 0.1 + 0.88 * (m + 0.5) / M, k = min(S - 1, floor(s * S)), Pm = add(R[k], mul(T[k], (s * S - k) * L / S)), len = (dry ? 0.5 : 0.84) * Math.pow(sin(PI * (0.12 + 0.8 * s)), 0.7), droop = 0.35 + 0.55 * s + (dry ? 0.4 : 0);
        const out = nrm(add(mul(sd, sg * cos(0.5)), mul(T[k], sin(0.5)))), d1 = nrm(add(mul(out, cos(droop * 0.45)), [0, -sin(droop * 0.45), 0])), d2 = nrm(add(mul(out, cos(droop)), [0, -sin(droop), 0]));
        const mid = add(Pm, mul(d1, len * 0.55)), end = add(mid, mul(d2, len * 0.45));
        upQuad(b, sub(Pm, mul(T[k], 0.065)), add(Pm, mul(T[k], 0.065)), add(mid, mul(T[k], 0.04)), sub(mid, mul(T[k], 0.04)), [cB, cB, cM, cM], MATTE);
        upTri(b, sub(mid, mul(T[k], 0.04)), add(mid, mul(T[k], 0.04)), end, cM, cM, cT, MATTE);
      }
    }
    for (let c = 0, nc = 3 + floor(rnd(i, 9) * 3); c < nc; c++) { const a = c / nc * TAU + rnd(i, 200 + c), r = 0.13 + 0.03 * rnd(i, 210 + c);
      solid(b, ball(7, 5), at(top[0] + cos(a) * 0.21, top[1] - 0.2 - 0.14 * rnd(i, 220 + c), top[2] + sin(a) * 0.21, r, r * 1.15, r), rnd(i, 230 + c) < 0.5 ? lin(0x7f9440) : lin(0x8a6a34), SATIN); }
    return finish(b, 62);
  }
  function cartGeo(i) { // a vendor's handcart: plank bed on two cart wheels, push handles, a heap of coconuts and a sun shade
    const b = bag(), WOOD = lin(0x9a6a3c), DARK = lin(0x5a3a22), TYRE = lin(0x1c1c1e), HUB = lin(0x9aa0a6), SHADES = [[0xd8452e, 0xf2c230], [0x2f8f5a, 0xf2c230], [0x2b7fb8, 0xf4efe2]][i % 3];
    const r = (k) => { const x = Math.sin((i + 1) * 127.1 + k * 311.7) * 43758.5453; return x - Math.floor(x); };
    solid(b, BOX, at(0, 0.82, 0, 1.6, 0.08, 0.92), WOOD, VARNISH);                                                      // the bed
    for (const z of [-0.47, 0.47]) solid(b, BOX, at(0, 0.95, z, 1.6, 0.22, 0.04), mul(WOOD, 0.85), VARNISH);          // side boards
    for (const x of [-0.8, 0.8]) solid(b, BOX, at(x, 0.95, 0, 0.04, 0.22, 0.92), mul(WOOD, 0.8), VARNISH);
    for (const z of [-0.55, 0.55]) {
      solid(b, cyl(14), at(-0.25, 0.4, z, 0.8, 0.09, 0.8, PI / 2), TYRE, RUBBER);                                       // cart wheels
      solid(b, cyl(8), at(-0.25, 0.4, z * 1.02, 0.16, 0.1, 0.16, PI / 2), HUB, METAL);
      solid(b, BOX, at(1.15, 0.86, z * 0.62, 0.75, 0.05, 0.05, 0, 0, -0.18), DARK, VARNISH);                             // push handles
      solid(b, BOX, at(0.68, 0.39, z * 0.75, 0.05, 0.78, 0.05), DARK, VARNISH);                                         // front legs
    }
    solid(b, BOX, at(-0.25, 0.42, 0, 0.06, 0.06, 1.12), HUB, METAL);                                                    // axle
    for (let k = 0; k < 34; k++) {                                                                                        // the heap: a low mound of coconuts
      const a = r(k) * TAU, d = Math.sqrt(r(k + 50)) * 0.62, x = cos(a) * d * 1.15, z = sin(a) * d * 0.62, h = 0.93 + (1 - d / 0.62) * 0.28 + r(k + 90) * 0.05;
      const green = mix(lin(0x4f8a2a), lin(0x9cbf3a), r(k + 70)), cut = r(k + 30) < 0.12;
      solid(b, ball(7, 5), at(x * 0.6, h, z * 0.85, 0.2, 0.24, 0.2), cut ? lin(0xf3eedf) : green, MATTE);
    }
    solid(b, cyl(6), at(-0.78, 1.45, 0.44, 0.04, 1.3, 0.04), HUB, METAL);                                              // shade pole at the back corner, so the heap shows from above
    for (let k = 0; k < 8; k++) { const q = k * PI / 4 + PI / 8; solid(b, BOX, at(-0.78 + cos(q) * 0.24, 2.12, 0.44 + sin(q) * 0.24, 0.3, 0.035, 0.2, 0, -q, 0), lin(SHADES[k % 2]), PAINT); }
    return finish(b, 30);
  }
  function pickupGeo(i) { // a working pickup: hood, cab and open bed on four wheels, paint faded by the sun
    const b = bag(), PAINT_C = lin([0xe8e4da, 0xa8452f, 0x2f7f86][i % 3]), DARK = lin(0x1c1f22), GLASSC = lin(0x26313a), CHROME = lin(0xb9bec2), TYRE = lin(0x18181a), LAMP = lin(0xfff1c8), TAIL = lin(0xb0201a);
    const body = (p) => mul(PAINT_C, 0.86 + 0.18 * clamp01(p[1] / 1.9));                                                 // paint a shade lighter toward the roof
    solid(b, BOX, at(0, 0.72, 0, 5.2, 0.5, 1.86), body, PAINT);                                                             // lower body, end to end
    solid(b, BOX, at(1.95, 1.12, 0, 1.3, 0.32, 1.8), body, PAINT);                                                          // hood
    solid(b, BOX, at(0.55, 1.4, 0, 1.55, 0.86, 1.8), body, PAINT);                                                          // cab
    solid(b, BOX, at(1.36, 1.5, 0, 0.06, 0.6, 1.6, 0, 0, 0.5), GLASSC, GLASS);                                             // windscreen, leaning back
    for (const z of [-0.91, 0.91]) solid(b, BOX, at(0.55, 1.55, z, 1.2, 0.48, 0.03), GLASSC, GLASS);                       // side windows
    solid(b, BOX, at(-1.25, 0.99, 0, 2.7, 0.06, 1.7), DARK, MATTE);                                                         // bed floor
    for (const z of [-0.9, 0.9]) solid(b, BOX, at(-1.25, 1.2, z, 2.7, 0.45, 0.07), body, PAINT);                           // bed sides
    solid(b, BOX, at(-2.57, 1.2, 0, 0.07, 0.45, 1.86), body, PAINT);                                                        // tailgate
    solid(b, BOX, at(2.62, 0.62, 0, 0.12, 0.2, 1.9), CHROME, METAL); solid(b, BOX, at(-2.62, 0.62, 0, 0.12, 0.2, 1.9), DARK, RUBBER);   // bumpers
    for (const z of [-0.7, 0.7]) { solid(b, BOX, at(2.6, 0.95, z, 0.05, 0.16, 0.3), LAMP, GLOW); solid(b, BOX, at(-2.6, 1.05, z, 0.05, 0.22, 0.14), TAIL, GLOW); }
    for (const x of [-1.65, 1.7]) for (const z of [-0.86, 0.86]) {
      solid(b, cyl(16), at(x, 0.4, z, 0.8, 0.26, 0.8, PI / 2), TYRE, RUBBER);
      solid(b, cyl(8), at(x, 0.4, z * 1.08, 0.36, 0.06, 0.36, PI / 2), CHROME, METAL);
    }
    if (i % 3 === 1) for (let k = 0; k < 3; k++) solid(b, BOX, at(-0.7 - k * 0.75, 1.3, (k % 2 ? 0.35 : -0.3), 0.6, 0.55, 0.6, 0, k * 0.4, 0), lin(0x9a6a3c), VARNISH);   // a few boxes of load in one bed
    return finish(b, 30);
  }
  function lineGeo() {
    const b = bag(), WOOD = lin(0x7a5a3a), ROPE = lin(0xdccba0);
    for (const x of [-1.2, 1.2]) solid(b, BOX, at(x, 0.9, 0, 0.08, 1.8, 0.08), WOOD, VARNISH);
    solid(b, BOX, at(0, 1.72, 0, 2.4, 0.02, 0.02), ROPE, MATTE);
    [[0xe8b23a, -0.7], [0x2f8f5a, 0], [0xc8573a, 0.7]].forEach(([c, x]) => { solid(b, BOX, at(x, 1.42, 0, 0.5, 0.56, 0.03), lin(c), MATTE); solid(b, BOX, at(x, 1.62, 0, 0.62, 0.14, 0.03), lin(c), MATTE); });
    return finish(b, 30);
  }
  function skiffGeo(hex) {
    const b = bag(), HULL = lin(hex), CREAM = lin(0xf2ead6), INNER = mix(HULL, lin(0xe9e0c8), 0.08), WOOD = lin(0xa07c4c), FLOOR = lin(0x7c5d38), RAIL = lin(0x70492a), GREY = lin(0x2c3239), ROPE = lin(0xdccba0);
    const xs = [-2.7, -2.25, -1.6, -0.8, 0, 0.8, 1.5, 2.05, 2.45, 2.7], beams = [0.6, 0.72, 0.81, 0.85, 0.85, 0.79, 0.66, 0.46, 0.22, 0.03], sheers = [0.47, 0.44, 0.42, 0.41, 0.41, 0.43, 0.47, 0.53, 0.59, 0.64], drafts = [0.1, 0.16, 0.19, 0.2, 0.2, 0.19, 0.17, 0.13, 0.07, 0];
    const tab = (t, x) => { for (let k = 1; k < xs.length; k++) if (x <= xs[k]) return lerp(t[k - 1], t[k], clamp01((x - xs[k - 1]) / (xs[k] - xs[k - 1]))); return t[t.length - 1]; };
    const mirror = (x, R) => { const ring = R.map(([y, z]) => [x, y, z]); for (let k = R.length - 2; k >= 0; k--) ring.push([x, R[k][0], -R[k][1]]); return ring; };
    const outer = xs.map((x, k) => { const w = beams[k], s = sheers[k], d = drafts[k]; return mirror(x, [[s, w], [s - 0.11, w * 0.99], [s * 0.5, w * 0.95], [-d * 0.2, w * 0.78], [-d * 0.75, w * 0.42], [-d, 0]]); });
    const inner = xs.map((x, k) => { const w = beams[k], s = sheers[k], f = 0.05; return mirror(k ? x : x + 0.05, [[s, max(0.004, w - 0.05)], [s * 0.55, max(0.003, w * 0.95 - 0.05)], [f + 0.05, max(0.002, w * 0.78 - 0.03)], [f, w * 0.45], [f, 0]]); });   // the floor sits just above the waterline: the harbour is one flat sheet and would show through a deeper hull
    loft(b, outer, (i, j) => { const s = min(j, 9 - j); return s === 0 ? CREAM : mul(HULL, [1, 1, 0.82, 0.6, 0.5][s]); }, PAINT);
    loft(b, inner, (i, j) => { const s = min(j, 7 - j); return s < 3 ? mul(INNER, [1, 0.88, 0.74][s]) : mul(FLOOR, 0.85 + 0.3 * rnd(i, j)); }, (i, j) => min(j, 7 - j) < 3 ? PAINT : MATTE, { inward: true });   // painted inside as well: seen from above, the inside is most of the boat
    for (let k = 0; k + 1 < xs.length; k++) for (const j of [0, 1]) { const o = j ? outer[k].length - 1 : 0, q = j ? inner[k].length - 1 : 0; upQuad(b, outer[k][o], outer[k + 1][o], inner[k + 1][q], inner[k][q], CREAM, PAINT); }   // gunwale
    cap(b, outer[0], [-1, 0, 0], mul(HULL, 0.9), PAINT); cap(b, inner[0], [1, 0, 0], INNER, PAINT);
    upQuad(b, outer[0][0], inner[0][0], inner[0][inner[0].length - 1], outer[0][outer[0].length - 1], CREAM, PAINT);
    for (const [x, w] of [[-1.85, 0.34], [-0.35, 0.26], [1.0, 0.26]]) solid(b, BOX, at(x, tab(sheers, x) - 0.13, 0, w, 0.045, 2 * tab(beams, x) - 0.08), mul(WOOD, 0.9 + 0.2 * rnd(floor(x * 10), 3)), VARNISH);   // thwarts
    const fx0 = 1.9, y0 = tab(sheers, fx0) - 0.035, w0 = tab(beams, fx0) - 0.05; tri(b, [fx0, y0, w0], [2.64, 0.6, 0], [fx0, y0, -w0], HULL, mul(HULL, 1.15), HULL, PAINT);      // foredeck
    solid(b, BOX, at(fx0 + 0.01, y0 - 0.03, 0, 0.04, 0.07, 2 * w0), RAIL, VARNISH);
    for (const s of [-1, 1]) { solid(b, BOX, at(0.05, 0.325, s * 0.5, 2.5, 0.04, 0.05, 0, s * 0.05, 0), WOOD, VARNISH); solid(b, BOX, at(1.5, 0.33, s * 0.435, 0.56, 0.022, 0.14, 0, s * 0.05, 0), mul(WOOD, 1.15), VARNISH); }   // oars across the thwarts
    solid(b, cyl(10), at(-1.05, 0.19, 0.32, 0.3, 0.28, 0.3), lin(0xe8e4d8), SATIN); solid(b, cyl(10), at(-1.05, 0.335, 0.32, 0.26, 0.01, 0.26), lin(0x25303a), MATTE);           // bait bucket
    solid(b, ball(8, 5), at(0.35, 0.07, -0.22, 0.42, 0.13, 0.3), lin(0x1f5560), MATTE); solid(b, ball(7, 5), at(0.62, 0.17, -0.34, 0.12, 0.12, 0.12), lin(0xff7a2a), SATIN);   // net and float
    solid(b, new THREE.TorusGeometry(0.17, 0.04, 5, 10), at(1.45, 0.08, 0.05, 1, 1, 1, PI / 2), ROPE, MATTE);
    solid(b, BOX, at(-2.74, 0.42, 0, 0.1, 0.2, 0.24), GREY, SATIN); solid(b, BOX, at(-2.87, 0.22, 0, 0.09, 0.76, 0.08), GREY, SATIN);                                          // outboard: clamp, leg, cowl, tiller
    solid(b, ball(9, 6), at(-2.86, 0.7, 0, 0.26, 0.17, 0.17), (p) => p[1] > 0.74 ? lin(0xdadfe2) : GREY, PAINT);
    solid(b, BOX, at(-2.5, 0.71, 0.09, 0.56, 0.03, 0.03, 0, 0.22, 0), lin(0x8d949a), METAL);
    return finish(b, 40);
  }

  /* ---------------- street lamp, string of lights ---------------- */
  function lampGeo() {
    const b = bag(), POLE = lin(0x2e3a3c), A = NH.LAMP_ARM;
    solid(b, cyl(10), at(0, 0.17, 0, 0.46, 0.34, 0.46), lin(0x9a958a), MATTE); solid(b, cyl(8), at(0, 0.37, 0, 0.3, 0.06, 0.3), POLE, SATIN);
    tube(b, [[0, 0.3, 0], [0, 2.6, 0], [0, 5.75, 0]], (k) => [0.088, 0.074, 0.056][k], 8, POLE, SATIN, [0, 0, 1]);
    tube(b, [[0, 5.5, 0], [0, 5.86, 0], [0.1, 6.04, 0], [0.36, 6.13, 0], [0.8, 6.15, 0], [A - 0.3, 6.13, 0]], 0.04, 6, POLE, SATIN, [0, 0, 1]);
    solid(b, ball(10, 6), at(A - 0.02, 6.08, 0, 0.42, 0.085, 0.17), lin(0xaab2b6), SATIN); solid(b, ball(10, 6), at(A + 0.03, 6.01, 0, 0.3, 0.075, 0.21), lin(0xf3e6c4), GLASS);
    return finish(b, 45);
  }
  function lampGlow() {
    const b = bag(), A = NH.LAMP_ARM, SODIUM = [1, 0.43, 0.09], WARM = [1, 0.6, 0.28];   // sodium, a touch yellower than the palette swatch: added over blue night ground the swatch itself turns pink
    glow(b, A, 0.04, 0, 6.5, 6.5, light(SODIUM, 0.15)); glow(b, A, 0.044, 0, 3, 3, light(WARM, 0.045));     // the pool: wide sodium, a warmer heart
    glow(b, A, 5.93, 0, 0.95, 0.95, light(WARM, 0.16)); glow(b, A + 0.02, 6.22, 0, 0.34, 0.34, light([1, 0.82, 0.55], 1.0));   // haze round the head and the glare of the lamp itself
    return finish(b, -1, true);
  }

  /* ---------------- marks on the ground ---------------- */
  /* The cone. main.js draws a second copy of this mesh, with its own one-sided material, wherever a building covers it.
     Everything that is only there to help the eye (dark fringe, hairline, range ticks) is wound to face DOWN: this
     module's two-sided material shows it, the copy does not, so the copy stays exactly the sector a patrol can see. */
  function coneGeo(R, half, alert) {
    const b = bag(), seg = max(8, Math.round(half * 28)), w = alert ? 0.2 : 0.14, lo = alert ? 0.16 : 0.24, hi = alert ? 0.56 : 0.44, TINT = [1, 0.78, 0.46];   // the fill is a shade deeper than the edge, so it also shows on pale concrete
    const radii = [0.02, 0.3, 0.55, 0.78, 1].map((t) => t * (R - w)), fill = (r) => [TINT[0], TINT[1], TINT[2], lo + (hi - lo) * Math.pow(r / R, 1.5)];
    const edge = (s, wide, c0, c1, down) => { const d = [cos(half), 0, s * sin(half)], n = [sin(half), 0, -s * cos(half)], A = mul(d, 0.4), B = mul(d, R + (wide < 0 ? 0.05 : 0)); upQuad(b, A, B, add(B, mul(n, wide)), add(A, mul(n, wide)), [c0, c0, c1, c1], 0, down); };
    ann(b, R, R + 0.24, -half - 0.02, half + 0.02, seg, dark(0.3), dark(0), 0, true); for (const s of [-1, 1]) edge(s, -0.24, dark(0.3), dark(0), true);        // a soft dark fringe lifts the pale shape off pale ground
    ann(b, R, R + 0.13, -half - 0.01, half + 0.01, seg, dark(0.55), dark(0.55), 0, true); for (const s of [-1, 1]) edge(s, -0.13, dark(0.55), dark(0.55), true);   // and a dark hairline keeps its edge crisp
    for (let k = 0; k + 1 < radii.length; k++) ann(b, radii[k], radii[k + 1], -half, half, seg, fill(radii[k]), fill(radii[k + 1]));                              // the fill, stronger toward the rim
    for (const t of [1 / 3, 2 / 3]) ann(b, R * t - 0.03, R * t + 0.03, -half, half, seg, [1, 1, 1, 0.3], [1, 1, 1, 0.3], 0, true);                                // range ticks
    const E = alert ? [1, 1, 1, 1] : [1, 0.7, 0.32, 1]; ann(b, R - w, R, -half, half, seg, E); for (const s of [-1, 1]) edge(s, w * 0.8, E, E);   // calm edge is amber: brightness contrast on pale ground                                      // the bright edge
    return finish(b, -1, true);
  }
  function ringGeo(R) {
    const b = bag(), n = 14;
    ann(b, R * 0.3, R - 0.17, 0, TAU, 48, [1, 1, 1, 0], [1, 1, 1, 0.3]);
    ann(b, R - 0.37, R - 0.325, 0, TAU, 48, [1, 1, 1, 0.6]);
    for (let k = 0; k < n; k++) { const a = k / n * TAU, e = a + TAU / n * 0.62; ann(b, R - 0.2, R + 0.045, a - 0.02, e + 0.02, 5, dark(0.5)); ann(b, R - 0.16, R, a, e, 5, [1, 1, 1, 1]); }
    return finish(b, -1, true);
  }
  function arrowGeo() {
    const b = bag(), shape = (tipX, wingX, wingZ, notchX, c0, c1) => { upTri(b, [tipX, 0, 0], [wingX, 0, wingZ], [notchX, 0, 0], c0, c1, c1); upTri(b, [tipX, 0, 0], [notchX, 0, 0], [wingX, 0, -wingZ], c0, c1, c1); };
    shape(0.98, -0.66, 0.68, -0.3, dark(0.6), dark(0.6)); shape(0.85, -0.55, 0.55, -0.18, [1, 1, 1, 1], [0.86, 0.86, 0.86, 1]);
    return finish(b, -1, true);
  }

  /* ---------------- the API ---------------- */
  NH.props = {
    car() {
      const M = mats(), group = new THREE.Group(), glowMat = M.glow(), LIT = [carLit(0), carLit(1), carLit(2)], GLOWS = [carGlow(0), carGlow(1), carGlow(2)];
      const lit = mesh(LIT[0], M.lit), fx = glowMesh(GLOWS[0], glowMat); glowMat.color.setScalar(0);
      group.add(mesh(once("carShell", carShell), M.solid, true), lit, fx);
      return { group, update(dt, st) {
        const k = st.sees ? 1 + (floor(st.t * 5) & 1) : 0, n = st.night || 0;
        lit.geometry = LIT[k]; fx.geometry = GLOWS[k]; glowMat.color.setScalar(max(n, st.sees ? 0.3 : 0));
      } };
    },
    crate() { return mesh(once("crate", crateGeo), mats().crate, true); },
    speaker() { return mesh(once("speaker", speakerGeo), mats().speaker, true); },
    palm(i) { return mesh(once("palm" + i, () => palmGeo(i)), mats().leaf, true); },
    pickup(i) { return mesh(once("pickup" + (i % 3), () => pickupGeo(i % 3)), mats().solid, true); },
    mural(i) {
      const g = new THREE.Group(), face = new THREE.Mesh(once("muralFace", () => new THREE.PlaneGeometry(6, 2.4)), once("muralMat" + (i % 2), () => new THREE.MeshStandardMaterial({ name: "mural", map: muralTex(i), roughness: 0.9, metalness: 0 })));
      face.rotation.y = PI / 2; face.position.set(0.11, 1.32, 0); face.receiveShadow = true;
      const wall = mesh(once("muralWall", () => { const b = bag(), BLOCK = lin(0xbdb6a8);
        solid(b, BOX, at(0, 1.3, 0, 0.2, 2.6, 6.3), BLOCK, MATTE); solid(b, BOX, at(0, 2.64, 0, 0.3, 0.08, 6.4), mul(BLOCK, 0.85), MATTE);   // the block wall and its capping
        return finish(b, 30); }), mats().solid, true);
      g.add(wall, face); return g;
    },
    clothesline() { return mesh(once("line", lineGeo), mats().solid, true); },
    plate() { return mesh(once("plate", () => { const b = bag();
      solid(b, cyl(20), at(0, 0.45, 0, 0.5, 0.03, 0.5, PI / 2), lin(0x121214), GLASS); solid(b, cyl(12), at(0, 0.45, 0, 0.18, 0.04, 0.18, PI / 2), lin(0xf2c230), PAINT);
      return finish(b, 30); }), mats().solid, true); },
    cart(i) { return mesh(once("cart" + (i % 3), () => cartGeo(i % 3)), mats().solid, true); },
    skiff(color) { return mesh(once("skiff" + color, () => skiffGeo(color)), mats().solid, true); },
    lampPost() {
      const M = mats(), group = new THREE.Group(), glowMat = M.glow(), fx = glowMesh(once("lampGlow", lampGlow), glowMat); fx.visible = false; glowMat.color.setScalar(0);
      group.add(mesh(once("lamp", lampGeo), M.solid, true), fx);
      return { group, setGlow(k) { glowMat.color.setScalar(k); fx.visible = k > 0.004; } };
    },
    stringLights(a, b, count) {
      const M = mats(), group = new THREE.Group(), bulbs = bag(), fxb = bag(), COLS = [P.sodium, P.go, P.heat, 0xf2c14e].map(lin), WIRE = [0.012, 0.012, 0.016], w = 0.018;
      const len = a.distanceTo(b), sag = 0.045 * len, hang = (t) => [lerp(a.x, b.x, t), lerp(a.y, b.y, t) - sag * 4 * t * (1 - t), lerp(a.z, b.z, t)], side = nrm([a.z - b.z, 0, b.x - a.x]);
      for (let k = 0; k < 24; k++) { const p = hang(k / 24), q = hang((k + 1) / 24);
        quad(bulbs, add(p, mul(side, w)), add(q, mul(side, w)), sub(q, mul(side, w)), sub(p, mul(side, w)), WIRE); quad(bulbs, add(p, [0, w, 0]), add(q, [0, w, 0]), sub(q, [0, w, 0]), sub(p, [0, w, 0]), WIRE); }
      for (let i = 0; i < count; i++) { const p = hang((i + 0.5) / count), c = COLS[i % 4], mid = [p[0], p[1] - 0.2, p[2]];
        solid(bulbs, cyl(6), at(p[0], p[1] - 0.05, p[2], 0.075, 0.1, 0.075), WIRE);
        solid(bulbs, ball(8, 6), at(mid[0], mid[1], mid[2], 0.095, 0.12, 0.095), (q) => mix(mul(c, 0.8), mix(c, WHITE, 0.55), clamp01((q[1] - mid[1]) * 5.5 + (q[2] - mid[2]) * 5.5)));
        glow(fxb, mid[0], mid[1] + 0.06, mid[2], 0.55, 0.55, light(c, 0.4)); glow(fxb, mid[0], 0.046, mid[2], 2.3, 2.3, light(c, 0.045)); }
      const bulbMat = new THREE.MeshBasicMaterial({ name: "bulbs", vertexColors: true, side: THREE.DoubleSide }), glowMat = M.glow(), fx = glowMesh(finish(fxb, -1, true), glowMat);
      group.add(mesh(finish(bulbs, -1), bulbMat), fx);
      const api = { group, setGlow(k) { bulbMat.color.setScalar(0.55 + 1.15 * k); glowMat.color.setScalar(k); fx.visible = k > 0.004; } }; api.setGlow(0); return api;
    },
    visionCone(range, half) {
      const key = range.toFixed(3) + "/" + half.toFixed(4), calm = once("cone" + key, () => coneGeo(range, half, false)), alert = once("coneAlert" + key, () => coneGeo(range, half, true));
      const mat = mats().mark({ color: 0xfff0be, opacity: 0.85, side: THREE.DoubleSide }), fill = mesh(calm, mat), holder = new THREE.Group(); fill.position.y = 0.08; fill.renderOrder = 2; holder.add(fill);
      return { mesh: holder, set(sees) { mat.color.setHex(sees ? P.heat : 0xfff0be); mat.opacity = sees ? 1 : 0.85; fill.geometry = sees ? alert : calm; } };
    },
    dropRing(radius) { const m = mesh(once("ring" + radius.toFixed(3), () => ringGeo(radius)), mats().ring), g = new THREE.Group(); m.position.y = 0.06; m.renderOrder = 3; g.add(m); return g; },
    arrow() {
      const mat = mats().mark({ color: P.go }), m = mesh(once("arrow", arrowGeo), mat), g = new THREE.Group(); m.position.y = 0.09; m.renderOrder = 3; g.add(m);
      return Object.assign(g, { setColor(hex) { mat.color.setHex(hex); } });
    },
  };
})();
