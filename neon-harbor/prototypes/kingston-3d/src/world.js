/* world: everything static about the block — streets, lots (fence, yard, house), the two venues, quay, harbour,
   and the neighbouring roofs and hills past the edge of the map.
   API: NH.world.build(env) -> { group, update(S, tod), stats }
     env    = { scene, camera, renderer, sim, quality }
     group  = THREE.Group holding the whole world; main.js adds it to the scene
     update = called once a frame with the rules' state S and tod = NH.tod(dusk): drifts and tints the water,
              lights the windows, the signs and the pools of window light as tod.night rises. It allocates nothing.
     stats  = { meshes, triangles } of what was built (optional extra, for whoever checks the budget)
   Static geometry is merged by hand into one mesh per material. Walkable ground is flat at y = 0 and nothing solid
   stands on it; overlays lie under 0.03 m. The harbour surface is at y = -0.35, where main.js floats the skiffs. */
(function () {
  "use strict";
  const NH = window.NH, sim = NH.sim, U = NH.U, P = NH.PALETTE;
  const X = (sx) => (sx - sim.W / 2) * U, Z = (sy) => (sy - sim.H / 2) * U;
  const hash = (i, k) => { let h = Math.imul(i + 7 | 0, 374761393) + Math.imul(k + 13 | 0, 668265263) | 0; h = Math.imul(h ^ h >>> 13, 1274126177); return ((h ^ h >>> 16) >>> 0) / 4294967296; };
  const lin = (c) => { const k = new THREE.Color(c); return [k.r, k.g, k.b]; };
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
  const rich = (a, k) => { const g = a[0] * 0.3 + a[1] * 0.55 + a[2] * 0.15; return [g + (a[0] - g) * k, g + (a[1] - g) * k, g + (a[2] - g) * k]; };   // the same colour, more or less saturated
  const smooth = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

  /* ---- merging by hand: a bag collects the vertices of everything that shares one material ---- */
  const bag = (uv) => ({ p: [], n: [], c: [], u: [], i: [], count: 0, uv: uv || [0, 0, 0, 0] });
  const vtx = (b, x, y, z, nx, ny, nz, c, u, v) => { b.p.push(x, y, z); b.n.push(nx, ny, nz); b.c.push(c[0], c[1], c[2]); b.u.push(u, v); return b.count++; };
  // a quad listed counter-clockwise as seen from its front. uv = [u0, v0, u1, v1] puts u0,v0 at a and u1,v1 at c, or gives all four corners. cTop colours c and d.
  function quad(b, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, col, uv, cTop) {
    let nx = (cy - ay) * (dz - bz) - (cz - az) * (dy - by), ny = (cz - az) * (dx - bx) - (cx - ax) * (dz - bz), nz = (cx - ax) * (dy - by) - (cy - ay) * (dx - bx);
    const l = Math.hypot(nx, ny, nz) || 1, q = uv || b.uv, k = b.count; nx /= l; ny /= l; nz /= l;
    if (q.length === 8) { vtx(b, ax, ay, az, nx, ny, nz, col, q[0], q[1]); vtx(b, bx, by, bz, nx, ny, nz, col, q[2], q[3]); vtx(b, cx, cy, cz, nx, ny, nz, cTop || col, q[4], q[5]); vtx(b, dx, dy, dz, nx, ny, nz, cTop || col, q[6], q[7]); }
    else { vtx(b, ax, ay, az, nx, ny, nz, col, q[0], q[1]); vtx(b, bx, by, bz, nx, ny, nz, col, q[2], q[1]); vtx(b, cx, cy, cz, nx, ny, nz, cTop || col, q[2], q[3]); vtx(b, dx, dy, dz, nx, ny, nz, cTop || col, q[0], q[3]); }
    b.i.push(k, k + 1, k + 2, k, k + 2, k + 3);
  }
  function tri(b, ax, ay, az, bx, by, bz, cx, cy, cz, col) {
    let nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay), ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az), nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const l = Math.hypot(nx, ny, nz) || 1, k = b.count, q = b.uv; nx /= l; ny /= l; nz /= l;
    vtx(b, ax, ay, az, nx, ny, nz, col, q[0], q[1]); vtx(b, bx, by, bz, nx, ny, nz, col, q[0], q[1]); vtx(b, cx, cy, cz, nx, ny, nz, col, q[0], q[1]); b.i.push(k, k + 1, k + 2);
  }
  // an upright box without its bottom; cTop, if given, colours the top face
  function box(b, x0, y0, z0, x1, y1, z1, col, cTop) {
    quad(b, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1, col); quad(b, x1, y0, z0, x0, y0, z0, x0, y1, z0, x1, y1, z0, col);
    quad(b, x1, y0, z1, x1, y0, z0, x1, y1, z0, x1, y1, z1, col); quad(b, x0, y0, z0, x0, y0, z1, x0, y1, z1, x0, y1, z0, col);
    quad(b, x0, y1, z1, x1, y1, z1, x1, y1, z0, x0, y1, z0, cTop || col);
  }
  // a flat strip from a to b, w wide, facing up
  function ribbon(b, ax, ay, az, bx, by, bz, w, col, uv) {
    const dx = bx - ax, dz = bz - az, l = Math.hypot(dx, dz) || 1, px = -dz / l * w / 2, pz = dx / l * w / 2;
    quad(b, ax + px, ay, az + pz, bx + px, by, bz + pz, bx - px, by, bz - pz, ax - px, ay, az - pz, col, uv);
  }
  // a three.js geometry, moved by matrix m, in one colour
  const _v = new THREE.Vector3(), _n = new THREE.Vector3(), _m3 = new THREE.Matrix3(), _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _e = new THREE.Euler(), _s = new THREE.Vector3();
  function solid(b, g, m, col) {
    const pos = g.attributes.position, nor = g.attributes.normal, idx = g.index, base = b.count; _m3.getNormalMatrix(m);
    for (let i = 0; i < pos.count; i++) { _v.fromBufferAttribute(pos, i).applyMatrix4(m); _n.fromBufferAttribute(nor, i).applyMatrix3(_m3).normalize(); vtx(b, _v.x, _v.y, _v.z, _n.x, _n.y, _n.z, col, b.uv[0], b.uv[1]); }
    if (idx) for (let i = 0; i < idx.count; i++) b.i.push(base + idx.getX(i)); else for (let i = 0; i < pos.count; i++) b.i.push(base + i);
  }
  const at = (x, y, z, sx, sy, sz, rx, ry, rz) => _m.compose(_v.set(x, y, z), _q.setFromEuler(_e.set(rx || 0, ry || 0, rz || 0)), _s.set(sx, sy === undefined ? sx : sy, sz === undefined ? sx : sz));
  function toMesh(b, mat, cast, order) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(b.p, 3)); g.setAttribute("normal", new THREE.Float32BufferAttribute(b.n, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(b.c, 3)); g.setAttribute("uv", new THREE.Float32BufferAttribute(b.u, 2)); g.setIndex(b.i);
    const m = new THREE.Mesh(g, mat); m.castShadow = !!cast; m.receiveShadow = true; m.matrixAutoUpdate = false; if (order) m.renderOrder = order; return m;
  }

  /* ---- textures: each drawn once on a canvas of 256 px ---- */
  const lat = (x, y, px, py, s) => hash(((x % px) + px) % px + (((y % py) + py) % py) * 131, s);
  function noise(x, y, px, py, s) { // value noise that repeats every px by py cells, so the textures tile
    const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy, ax = fx * fx * (3 - 2 * fx), ay = fy * fy * (3 - 2 * fy);
    const a = lat(ix, iy, px, py, s), b = lat(ix + 1, iy, px, py, s), c = lat(ix, iy + 1, px, py, s), d = lat(ix + 1, iy + 1, px, py, s);
    return a + (b - a) * ax + (c - a + (a - b + d - c) * ax) * ay;
  }
  const fbm = (u, v, per, s) => (noise(u * per, v * per, per, per, s) * 4 + noise(u * per * 2, v * per * 2, per * 2, per * 2, s + 1) * 2 + noise(u * per * 4, v * per * 4, per * 4, per * 4, s + 2)) / 7;
  const canvas = () => { const c = document.createElement("canvas"); c.width = c.height = 256; return c; };
  function pixels(f) { // paint every pixel from f(x, y, out)
    const c = canvas(), g = c.getContext("2d"), im = g.createImageData(256, 256), d = im.data, o = [0, 0, 0, 255];
    for (let y = 0, k = 0; y < 256; y++) for (let x = 0; x < 256; x++, k += 4) { f(x, y, o); d[k] = o[0]; d[k + 1] = o[1]; d[k + 2] = o[2]; d[k + 3] = 255; }
    g.putImageData(im, 0, 0); return c;
  }
  function texture(c, tile, data) {
    const t = new THREE.CanvasTexture(c); if (tile) t.wrapS = t.wrapT = THREE.RepeatWrapping; if (!data) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t;
  }
  const cell = (x, y, w, h) => [(x + 0.5) / 256, 1 - (y + h - 0.5) / 256, (x + w - 0.5) / 256, 1 - (y + 0.5) / 256];   // a rectangle of an atlas as a uv box

  const TW = 5.28;   // the zinc tile is 16 corrugations of 0.33 m
  const lapOf = (x, y) => (y + (x >> 6 & 1 ? 0 : 128)) & 255;   // sheets are four corrugations wide; each has one end lap per tile
  function zincPixel(x, y, o) {
    const u = x / 256, v = y / 256, crest = 0.5 + 0.5 * Math.cos((x % 16) / 16 * 6.2832), lap = lapOf(x, y);
    let s = 0.8 + 0.17 * Math.pow(crest, 0.8) + (hash(x >> 6, 3) - 0.5) * 0.04 + (noise(u * 6, v * 6, 6, 6, 41) - 0.5) * 0.06;
    if (lap < 2) s *= 0.7; if (x % 64 === 8) s *= 0.88;
    const old = 0.12 + 0.88 * (smooth(118, 140, y) - smooth(244, 256, y) + 1 - smooth(0, 10, y));   // the lower half of the tile is the rusty one
    let r = smooth(0.56, 0.95, noise(u * 9, v * 3, 9, 3, 42) * 0.6 + noise(u * 32, v * 5, 32, 5, 43) * 0.4) * old;
    if (lap >= 2 && lap < 30) r = Math.max(r, (1 - (lap - 2) / 28) * 0.45 * hash(x >> 2, 44) * (0.4 + 0.6 * old));   // rust bleeding down the slope from a lap
    o[0] = 255 * s * (1 - r * 0.2); o[1] = 255 * s * (1 - r * 0.46); o[2] = 255 * s * (1 - r * 0.62);
  }
  function zincNormalPixel(x, y, o) {
    const nx = 0.5 * Math.sin((x % 16) / 16 * 6.2832), ny = lapOf(x, y) < 3 ? 0.5 : 0, l = Math.hypot(nx, ny, 1);
    o[0] = 128 + 127 * nx / l; o[1] = 128 + 127 * ny / l; o[2] = 128 + 127 / l;
  }
  function grainPixel(x, y, o) { // asphalt, concrete and packed earth all take their speckle from this
    const sp = hash(x + y * 256, 51); let s = 0.9 + (fbm(x / 256, y / 256, 5, 52) - 0.5) * 0.22 + (sp - 0.5) * 0.09;
    if (sp > 0.986) s += 0.08; else if (sp < 0.014) s -= 0.2; o[0] = 255 * s; o[1] = 253 * s; o[2] = 248 * s;
  }
  function plasterPixel(x, y, o) { // 6 m by 6 m of rendered wall; the bottom row is the ground. The top rows stay white for untextured things.
    if (y < 12) { o[0] = o[1] = o[2] = 255; return; }
    const u = x / 256, v = y / 256, m = (255 - y) / 256 * 6, g = 1 - smooth(0, 0.3 + 0.45 * noise(u * 16, 0, 16, 1, 22), m);
    const s = (0.95 + (fbm(u, v, 4, 21) - 0.5) * 0.11 - smooth(0.7, 0.92, noise(u * 48, v * 3, 48, 3, 23)) * 0.1) * (1 - 0.3 * g);
    o[0] = 255 * s; o[1] = 255 * s * (1 - 0.06 * g); o[2] = 255 * s * (1 - 0.16 * g);
  }
  function waterPixel(x, y, o) { // long thin glints lying east-west on a calm harbour
    const u = x / 256, v = y / 256, a = noise(u * 7, v * 40, 7, 40, 31), b = noise(u * 13, v * 64, 13, 64, 32), c = noise(u * 3, v * 6, 3, 6, 33);
    const s = 0.82 + 0.18 * smooth(0.7, 0.95, a) * smooth(0.25, 0.7, c) + 0.1 * smooth(0.74, 0.95, b) + (c - 0.5) * 0.1; o[0] = 255 * s * 0.96; o[1] = 255 * s; o[2] = 255 * s;
  }

  // the atlas of fronts: glass louvres, shutters, breeze blocks, doors, a gate, flyers. Light greys take the colour of the vertex.
  const GLASS = cell(0, 0, 64, 64), SHUT = cell(64, 0, 64, 64), VENT = cell(128, 0, 64, 64), CURTAIN = cell(192, 0, 64, 64), DOOR = cell(0, 64, 48, 96), PLANK = cell(48, 64, 48, 96),
        INSIDE = cell(96, 64, 64, 96), GATE = cell(160, 64, 96, 96), FLYER = [0, 1, 2, 3, 4, 5].map((k) => cell(k * 42, 160, 42, 56));
  function frontAtlas(glow) {
    const c = canvas(), g = c.getContext("2d"), R = (col, x, y, w, h) => { g.fillStyle = col; g.fillRect(x, y, w, h); };
    const grad = (x, y, w, h, a, b) => { const q = g.createLinearGradient(0, y, 0, y + h); q.addColorStop(0, a); q.addColorStop(1, b); g.fillStyle = q; g.fillRect(x, y, w, h); };
    R(glow ? "#000" : "#888", 0, 0, 256, 256);
    for (const x of [0, 192]) { // louvre windows: the second has a curtain behind the glass
      R(glow ? "#ffb45a" : "#1a2730", x, 0, 64, 64);
      if (glow) { const q = g.createRadialGradient(x + 32, 30, 4, x + 32, 30, 44); q.addColorStop(0, "#ffe2a8"); q.addColorStop(1, "#d9812e"); g.fillStyle = q; g.fillRect(x, 0, 64, 64); }
      if (x) { g.fillStyle = glow ? "#ff9d5a" : "#7d5a52"; g.beginPath(); g.moveTo(x, 0); g.lineTo(x + 30, 0); g.quadraticCurveTo(x + 22, 40, x, 56); g.fill(); g.beginPath(); g.moveTo(x + 64, 0); g.lineTo(x + 34, 0); g.quadraticCurveTo(x + 42, 40, x + 64, 56); g.fill(); }
      for (let k = 0; k < 8; k++) { if (glow) R("rgba(60,20,0,0.45)", x, k * 8 + 6, 64, 2); else { grad(x, k * 8, 64, 6, "rgba(205,226,236,0.95)", "rgba(92,124,142,0.8)"); } }
      R(glow ? "#3a1c08" : "#e9e4d6", x + 30, 0, 4, 64);
    }
    if (!glow) {
      for (let k = 0; k < 10; k++) grad(64, k * 6.4, 64, 6.4, "#f4f4f4", "#a9a9a9"); R("#6a6a6a", 95, 0, 2, 64);                 // shutters
      R("#e9e9e9", 128, 0, 64, 64); g.fillStyle = "#3d3d3d";                                                                // breeze blocks
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { g.beginPath(); g.moveTo(136 + i * 16, 2 + j * 16); g.lineTo(142 + i * 16, 8 + j * 16); g.lineTo(136 + i * 16, 14 + j * 16); g.lineTo(130 + i * 16, 8 + j * 16); g.fill(); }
      R("#dedede", 0, 64, 48, 96); for (const [x, y, w, h] of [[6, 70, 15, 34], [27, 70, 15, 34], [6, 112, 15, 40], [27, 112, 15, 40]]) { R("#a8a8a8", x, y, w, h); R("#cfcfcf", x + 2, y + 3, w - 3, h - 4); }
      R("#3a3a3a", 40, 108, 4, 6);                                                                                          // a panelled door
      R("#d6d6d6", 48, 64, 48, 96); for (let k = 1; k < 6; k++) R("#9c9c9c", 48 + k * 8, 64, 1, 96); g.strokeStyle = "#b4b4b4"; g.lineWidth = 5; g.beginPath(); g.moveTo(50, 74); g.lineTo(94, 74); g.lineTo(50, 150); g.lineTo(94, 150); g.stroke();   // a plank door
      for (let k = 0; k < 12; k++) grad(160 + k * 8, 64, 8, 96, "#f0f0f0", "#f0f0f0"), R("#b6b6b6", 160 + k * 8, 64, 3, 96);   // a zinc gate in a frame
      g.strokeStyle = "#5c5c5c"; g.lineWidth = 5; g.strokeRect(163, 67, 90, 90); g.beginPath(); g.moveTo(208, 64); g.lineTo(208, 160); g.moveTo(163, 112); g.lineTo(253, 112); g.stroke();
      const ink = ["#ff9d3c", "#0b1420", "#f3ead8", "#ff4d5e", "#3ed6a4", "#f2c14e", "#1f6f86"];   // flyers: shapes only, a speaker, a sun, bars
      for (let k = 0; k < 6; k++) { const x = k * 42, a = ink[k % 7], b = ink[(k + 2) % 7], d = ink[(k + 4) % 7]; R(a, x, 160, 42, 56); g.fillStyle = b;
        if (k % 3 === 0) { g.beginPath(); g.arc(x + 21, 182, 13, 0, 7); g.fill(); g.fillStyle = d; g.beginPath(); g.arc(x + 21, 182, 6, 0, 7); g.fill(); }
        else if (k % 3 === 1) for (let j = 0; j < 5; j++) g.fillRect(x + 5 + j * 7, 196 - j * 6, 5, 8 + j * 6);
        else { g.beginPath(); for (let j = 0; j < 10; j++) { const r = j % 2 ? 6 : 15, t = j * 0.6283; g.lineTo(x + 21 + Math.sin(t) * r, 182 - Math.cos(t) * r); } g.fill(); }
        R(d, x + 4, 204, 34, 4); R(b, x + 4, 210, 22, 3); }
    }
    grad(96, 64, 64, 96, glow ? "#ffd28a" : "#15181c", glow ? "#e06a3a" : "#2c2620");                                           // an open doorway
    if (glow) for (const [x, w, h] of [[104, 9, 30], [120, 12, 44], [140, 10, 36]]) { g.fillStyle = "rgba(40,10,10,0.75)"; g.beginPath(); g.arc(x + w / 2, 160 - h, w * 0.42, 0, 7); g.fill(); g.fillRect(x, 160 - h + 3, w, h); }
    return c;
  }

  // decals for the ground: patches, cracks, ironwork, worn paint. Drawn in white and grey, with alpha, and tinted by the vertex.
  const PATCH = cell(0, 0, 64, 64), CRACK = cell(64, 0, 64, 64), CRACK2 = cell(128, 0, 64, 64), STAIN = cell(192, 0, 64, 64), GRATE = cell(0, 64, 64, 64), COVER = cell(64, 64, 64, 64),
        PIT = cell(128, 64, 64, 64), BLOB = cell(192, 64, 64, 64), BAR = cell(0, 128, 128, 32), DASH = cell(0, 160, 128, 32), HOP = cell(128, 128, 64, 64), JOINT = cell(192, 128, 64, 64),
        SOLID = cell(4, 196, 24, 24), FADE = cell(36, 196, 24, 56), RUBBLE = cell(128, 192, 64, 64);
  function decalAtlas() {
    const c = canvas(), g = c.getContext("2d"); let k = 0; const r = () => hash(k++, 61);
    const rough = (x, y, w, h, col, n) => { g.fillStyle = col; g.beginPath(); for (let i = 0; i < n; i++) { const t = i / n * 6.2832, e = 0.82 + 0.18 * r(), cx = Math.cos(t), cy = Math.sin(t), q = Math.max(Math.abs(cx), Math.abs(cy)); g.lineTo(x + w / 2 + cx / q * w / 2 * e, y + h / 2 + cy / q * h / 2 * e); } g.fill(); };
    const crack = (x, y, a, len, w) => { g.lineWidth = w; g.beginPath(); g.moveTo(x, y); for (let i = 0; i < len; i++) { a += (r() - 0.5) * 1.1; x += Math.cos(a) * 5; y += Math.sin(a) * 5; g.lineTo(x, y); if (w > 1.2 && r() < 0.22) { g.stroke(); crack(x, y, a + (r() < 0.5 ? 1 : -1), len * 0.45 | 0, w * 0.6); g.lineWidth = w; g.beginPath(); g.moveTo(x, y); } } g.stroke(); };
    const soft = (x, y, rad, col) => { const q = g.createRadialGradient(x, y, 0, x, y, rad); q.addColorStop(0, col); q.addColorStop(1, "rgba(255,255,255,0)"); g.fillStyle = q; g.fillRect(x - rad, y - rad, rad * 2, rad * 2); };
    rough(5, 7, 54, 50, "rgba(255,255,255,0.8)", 40);                                                                                                // a resurfaced patch
    g.strokeStyle = "rgba(255,255,255,0.9)"; crack(68, 30, 0.1, 11, 2.2); crack(132, 6, 1.3, 12, 2.4); crack(150, 60, -1.2, 6, 1.6);                  // cracks
    for (let i = 0; i < 5; i++) soft(224 + (r() - 0.5) * 22, 32 + (r() - 0.5) * 22, 12 + r() * 14, "rgba(255,255,255,0.5)");                          // a stain
    g.fillStyle = "#e4e4e4"; g.fillRect(6, 70, 52, 52); g.fillStyle = "#9a9a9a"; g.fillRect(11, 75, 42, 42); g.fillStyle = "#3a3a3a"; for (let i = 0; i < 6; i++) g.fillRect(14 + i * 7, 78, 3, 36);   // a drain grate
    g.fillStyle = "#b9b9b9"; g.beginPath(); g.arc(96, 96, 27, 0, 7); g.fill(); g.strokeStyle = "#6a6a6a"; g.lineWidth = 3; g.beginPath(); g.arc(96, 96, 24, 0, 7); g.stroke();
    g.lineWidth = 2; for (let i = -2; i < 3; i++) { g.beginPath(); g.moveTo(78, 96 + i * 8); g.lineTo(114, 96 + i * 8); g.moveTo(96 + i * 8, 78); g.lineTo(96 + i * 8, 114); g.stroke(); }       // a manhole cover
    g.fillStyle = "#6a5a48"; g.beginPath(); g.arc(160, 96, 24, 0, 7); g.fill(); for (let i = 0; i < 13; i++) { const t = i / 13 * 6.2832; g.fillStyle = i % 2 ? "#f1ece0" : "#d9d2c2"; g.beginPath(); g.ellipse(160 + Math.cos(t) * 25, 96 + Math.sin(t) * 25, 6.2, 4.6, t + 1.57, 0, 7); g.fill(); }   // a tree pit ringed with whitewashed stones
    soft(224, 96, 31, "rgba(255,255,255,1)");
    g.fillStyle = "rgba(255,255,255,0.92)"; g.fillRect(4, 135, 120, 18); g.globalCompositeOperation = "destination-out";                                                                   // a worn stop bar
    for (let i = 0; i < 34; i++) { g.fillStyle = `rgba(0,0,0,${0.3 + 0.6 * r()})`; g.fillRect(2 + r() * 122, 133 + r() * 20, 2 + r() * 9, 1 + r() * 4); } g.globalCompositeOperation = "source-over";
    g.fillStyle = "rgba(255,255,255,0.85)"; for (let j = 0; j < 3; j++) for (let i = 0; i < 28; i += 2) if (r() > 0.2) g.fillRect(8 + j * 42 + i, 170 + r() * 2, 2, 9 + r() * 3);                                                      // worn dashes
    g.strokeStyle = "rgba(255,255,255,0.8)"; g.lineWidth = 1.6; for (const [x, y] of [[152, 130], [152, 142], [144, 154], [160, 154], [152, 166], [144, 178], [160, 178]]) g.strokeRect(x, y, 12, 12);   // chalk squares
    g.fillStyle = "rgba(255,255,255,0.9)"; g.fillRect(222, 128, 3, 64); g.fillRect(192, 158, 64, 2);                                                                                           // slab joints
    g.fillStyle = "#fff"; g.fillRect(0, 192, 32, 32); const q = g.createLinearGradient(0, 192, 0, 256); q.addColorStop(0.1, "#fff"); q.addColorStop(0.9, "rgba(255,255,255,0)"); g.fillStyle = q; g.fillRect(32, 192, 32, 64);
    for (let i = 0; i < 60; i++) { const s = 120 + 135 * r() | 0; g.fillStyle = `rgba(${s},${s},${s},0.95)`; const a = r() * 6.2832, d = 24 * Math.sqrt(r()); g.fillRect(160 + Math.cos(a) * d, 224 + Math.sin(a) * d, 2 + r() * 4, 2 + r() * 3); }   // rubble and leaves
    return c;
  }

  // the two painted signs, in the project's own hand: capitals built from brush strokes on a 4 by 6 grid
  const GLYPH = { Z: "0,6 4,6 0,0 4,0", I: "2,6 2,0|1,6 3,6|1,0 3,0", N: "0,0 0,6 4,0 4,6", C: "4,5 3,6 1,6 0,5 0,1 1,0 3,0 4,1", R: "0,0 0,6 3,6 4,5 4,4 3,3 0,3|2,3 4,0", O: "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0",
    F: "0,0 0,6 4,6|0,3 3,3", S: "4,5 3,6 1,6 0,5 0,4 1,3 3,3 4,2 4,1 3,0 1,0 0,1", U: "0,6 0,1 1,0 3,0 4,1 4,6", D: "0,0 0,6 3,6 4,5 4,1 3,0 0,0", H: "0,0 0,6|4,0 4,6|0,3 4,3", A: "0,0 0,4 2,6 4,4 4,0|0,2.4 4,2.4",
    L: "0,6 0,0 4,0", K: "0,0 0,6|4,6 0,2.6|1.5,3.7 4,0", E: "4,6 0,6 0,0 4,0|0,3 3,3", W: "0,6 1,0 2,4 3,0 4,6", G: "4,5 3,6 1,6 0,5 0,1 1,0 3,0 4,1 4,3 2.2,3", T: "0,6 4,6|2,6 2,0", B: "0,0 0,6 3,6 4,5 4,4 3,3 0,3|3,3 4,2 4,1 3,0 0,0" };
  const WORDS = ["ZINC ROOF", "SOUNDHALL", "KEELWRIGHT", "BOATWORKS"], WORD = WORDS.map((w, i) => cell(0, i * 64, 256, 64));
  function signAtlas() {
    const c = canvas(), g = c.getContext("2d"); g.lineCap = g.lineJoin = "round";
    WORDS.forEach((text, row) => {
      const adv = 5.7, total = text.length * adv - 1.7, k = Math.min(246 / total, 50 / 6.6), ox = (256 - total * k) / 2, oy = row * 64 + 32 + 3 * k;
      for (let pass = 0; pass < 2; pass++) {
        g.strokeStyle = pass ? "#fff" : "#10141c"; g.lineWidth = k * (pass ? 1.2 : 2.3);
        for (let i = 0; i < text.length; i++) for (const stroke of (GLYPH[text[i]] || "").split("|")) {
          if (!stroke) continue; const lean = (hash(row * 20 + i, 8) - 0.5) * 0.5, jy = (hash(row * 20 + i, 7) - 0.5) * k * 0.5; g.beginPath();
          stroke.split(" ").forEach((pt, j) => { const xy = pt.split(","), px = ox + (i * adv + +xy[0] + +xy[1] * lean * 0.12) * k, py = oy + jy - +xy[1] * k; if (j) g.lineTo(px, py); else g.moveTo(px, py); }); g.stroke();
        }
      }
    });
    return c;
  }

  /* ---- colours ---- */
  const ASPHALT = mul(mix(lin(P.asphalt), lin(P.laneMark), 0.4), 1.36), GUTTER = lin(0xb9b4a6), LANE = lin(0xb3a78f), EARTH = lin(0x9a8468), QUAY = mul(mix(lin(P.quay), lin(P.laneMark), 0.3), 1.2), CONCRETE = lin(0xc4bdac);
  const RUST = lin(0x8a4f2c), NEWZINC = lin(0xd3dadc), TIMBER = lin(0x7a5c3e), WHITE = [1, 1, 1], DARK = lin(0x23262b), ROPE = lin(0xb9a27a), INK = lin(P.ink), SODIUM = lin(P.sodium);
  const TRIMS = [0xf3ead8, 0xe9edf2, 0x5a4030, 0xf3ead8, 0x2f5f66, 0xe9edf2].map(lin);
  const ACCENTS = [0x1f7f86, 0xc8573a, 0xd6a23a, 0x3f8a48, 0x2c5fa8, 0x8a2f3a, 0x3aa0c8].map(lin);
  const ZPAINT = [0xb7c3c8, 0xa06a45, 0x3f8f96, 0x9c4a38, 0x4f7fb5, 0xc3cdd0, 0x5f9c68, 0xb58a5a].map(lin);
  const GREENS = [0x3f8a48, 0x2f6f3c, 0x5c9a4a, 0x7a8f3a, 0x8f4a3a, 0xb58f3a].map(lin), CLOTH = [0xf3ead8, 0xc8573a, 0x3aa0c8, 0xe8b23a, 0xe07a9a, 0x7fbf6a, 0xf3ead8, 0x2c3f66].map(lin);

  NH.world = {
    build(env) {
      const group = new THREE.Group();
      const zincTex = texture(pixels(zincPixel), true), zincNormal = texture(pixels(zincNormalPixel), true, true), grainTex = texture(pixels(grainPixel), true), plasterTex = texture(pixels(plasterPixel), true),
            waterTex = texture(pixels(waterPixel), true), frontTex = texture(frontAtlas(false)), glowTex = texture(frontAtlas(true)), decalTex = texture(decalAtlas()), signTex = texture(signAtlas());
      const G = bag(), M = bag([0.5, 0.985, 0.5, 0.985]), ZN = bag(), F = bag(), LIT = [bag(), bag()], D = bag(), SG = bag(), W = bag(), PL = bag();   // ground, matte solids, zinc, fronts, lit windows, decals, signs, water, pools of light
      const GT = 6.4;   // metres of ground per repeat of the grain

      // flat things on the ground, given in sim pixels
      const flat = (sx0, sy0, sx1, sy1, y, col) => { const x0 = X(sx0), x1 = X(sx1), z0 = Z(sy0), z1 = Z(sy1); quad(G, x0, y, z1, x1, y, z1, x1, y, z0, x0, y, z0, col, [x0 / GT, z1 / GT, x1 / GT, z0 / GT]); };
      // a decal lying on the ground: centre and size in metres, turned by rot
      function decal(uv, x, z, w, d, rot, col, y) {
        const c = Math.cos(rot || 0) / 2, s = Math.sin(rot || 0) / 2, ax = w * c, az = w * s, bx = -d * s, bz = d * c; y = y === undefined ? 0.014 : y;
        quad(D, x - ax + bx, y, z - az + bz, x + ax + bx, y, z + az + bz, x + ax - bx, y, z + az - bz, x - ax - bx, y, z - az - bz, col, uv);
      }
      // a wall face from (x0, z0) to (x1, z1), left to right as seen from its front, with the plaster laid on at 6 m a repeat
      const wallQuad = (x0, z0, x1, z1, y0, y1, col, cTop) => quad(M, x0, y0, z0, x1, y0, z1, x1, y1, z1, x0, y1, z0, col, [(x0 + z0) / 6, y0 / 6, (x1 + z1) / 6, y1 / 6], cTop);
      function wallBox(x0, y0, z0, x1, y1, z1, col, cTop) {
        wallQuad(x0, z1, x1, z1, y0, y1, col); wallQuad(x1, z0, x0, z0, y0, y1, col); wallQuad(x1, z1, x1, z0, y0, y1, col); wallQuad(x0, z0, x0, z1, y0, y1, col);
        quad(M, x0, y1, z1, x1, y1, z1, x1, y1, z0, x0, y1, z0, cTop || col);
      }
      // an upright zinc sheet with the corrugations running up it
      function zincBox(x0, y0, z0, x1, y1, z1, col) {
        const ew = x1 - x0 > z1 - z0, a0 = (ew ? x0 : z0) / TW, a1 = (ew ? x1 : z1) / TW, v0 = y0 / TW, v1 = y1 / TW;
        quad(ZN, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1, col, ew ? [a0, v0, a1, v1] : null); quad(ZN, x1, y0, z0, x0, y0, z0, x0, y1, z0, x1, y1, z0, col, ew ? [a1, v0, a0, v1] : null);
        quad(ZN, x1, y0, z1, x1, y0, z0, x1, y1, z0, x1, y1, z1, col, ew ? null : [a1, v0, a0, v1]); quad(ZN, x0, y0, z0, x0, y0, z1, x0, y1, z1, x0, y1, z0, col, ew ? null : [a0, v0, a1, v1]);
        quad(ZN, x0, y1, z1, x1, y1, z1, x1, y1, z0, x0, y1, z0, col);
      }
      // a roof sheet and, just under it, its underside: the shadow pass draws only faces that look away from the sun
      const UNDER = lin(0x4a443c);
      function roofQuad(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, col, uv) {
        quad(ZN, ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz, col, uv);
        quad(ZN, dx, dy - 0.08, dz, cx, cy - 0.08, cz, bx, by - 0.08, bz, ax, ay - 0.08, az, UNDER);
      }
      const startV = (seed, k, rusty) => (hash(seed, 400 + k) < rusty ? 0.02 : 0.52) + 0.1 * hash(seed, 440 + k);   // where on the zinc tile a sheet starts: its rusty half or its clean half
      function sheetColour(base, seed, k) { // every sheet a little different; now and then a rusty, a new or a painted one
        const r = hash(seed, 300 + k), c = mul(base, 0.93 + 0.14 * hash(seed, 200 + k));
        return r < 0.07 ? mix(c, RUST, 0.45) : r < 0.13 ? mix(c, NEWZINC, 0.5) : r < 0.16 ? mix(c, ZPAINT[2 + k % 5], 0.4) : c;
      }

      /* ---- ground: streets, lanes, the land under the lots ---- */
      const XS = [[-936, -676, 0], [-676, -596, 1], [-596, -336, 0], [-336, -256, 1], [-256, 0, 0], [0, 40, 2], [40, 300, 0], [300, 380, 1], [380, 820, 0], [820, 900, 1], [900, 1160, 0],
                  [1160, 1200, 2], [1200, 1456, 0], [1456, 1536, 1], [1536, 1796, 0], [1796, 1876, 1], [1876, 2136, 0]];   // 0 lots, 1 street, 2 edge lane
      const YS = [[-640, -380, 0], [-380, -300, 1], [-300, 0, 0], [0, 40, 2], [40, 380, 0], [380, 460, 1], [460, 900, 0], [900, 980, 1], [980, 1300, 0], [1300, 1380, 1]];
      function street(xa, xb, ya, yb, vertical, gutter, seed) { // asphalt in pieces of about 4 m, each a slightly different grey, with a concrete gutter on each lot line
        const s0 = vertical ? ya : xa, len = (vertical ? yb : xb) - s0, n = Math.ceil(len / 80), g = gutter ? 6 : 0;
        for (let j = 0; j < n; j++) {
          const a = s0 + len * j / n, b = s0 + len * (j + 1) / n, col = mul(ASPHALT, 0.94 + 0.12 * hash(seed, j)), gc = mul(GUTTER, 0.92 + 0.12 * hash(seed, j + 50));
          if (vertical) { flat(xa + g, a, xb - g, b, 0, col); if (g) { flat(xa, a, xa + g, b, 0, gc); flat(xb - g, a, xb, b, 0, gc); } }
          else { flat(a, ya + g, b, yb - g, 0, col); if (g) { flat(a, ya, b, ya + g, 0, gc); flat(a, yb - g, b, yb, 0, gc); } }
          if (xa > -400 && xb < 1600 && ya > -400) { // what years of patching leave behind
            const r = hash(seed, j + 100), mx = X(vertical ? xa + 16 + 48 * hash(seed, j + 110) : (a + b) / 2), mz = Z(vertical ? (a + b) / 2 : ya + 16 + 48 * hash(seed, j + 110)), rot = (vertical ? 1.5708 : 0) + (hash(seed, j + 120) - 0.5) * 0.5;
            if (r < 0.2) decal(PATCH, mx, mz, 1.4 + 1.6 * hash(seed, j + 130), 1.1 + hash(seed, j + 140), rot, mul(ASPHALT, r < 0.1 ? 0.72 : 1.16));
            else if (r < 0.5) decal(r < 0.35 ? CRACK : CRACK2, mx, mz, 3.2, 3.2, rot + r * 40, mul(ASPHALT, 0.5));
            else if (r < 0.6) decal(STAIN, mx, mz, 2.2, 1.8, rot, mul(ASPHALT, 0.6));
            else if (r < 0.68) decal(COVER, mx, mz, 0.8, 0.8, r * 50, lin(0x8f8d86));
            if (g && hash(seed, j + 150) < 0.22) decal(GRATE, vertical ? X(xa + 3.6) : X((a + b) / 2), vertical ? Z((a + b) / 2) : Z(yb - 3.6), 0.46, 0.34, vertical ? 1.5708 : 0, lin(0xb4b2aa));
          }
        }
      }
      flat(-2200, -720, 3400, sim.WATER, -0.03, EARTH);
      XS.forEach(([xa, xb, tx], ix) => YS.forEach(([ya, yb, ty], iy) => {
        const seed = ix * 31 + iy, mapX = xa >= 0 && xb <= sim.W, mapY = ya >= 0;
        if (tx === 1 && ty === 1) street(xa, xb, ya, yb, true, false, seed);
        else if (tx === 1) street(xa, xb, ya, yb, true, ty === 0 || !mapX, seed);
        else if (ty === 1) street(xa, xb, ya, yb, false, tx === 0 || !mapY, seed);
        else if (mapX && mapY) {
          if (tx === 2 || ty === 2) flat(xa, ya, xb, yb, 0, mul(LANE, 0.95));
          else if (xa === 900 && ya === 40) flat(xa, ya, xb, 190, 0, LANE);   // the dance yard takes the rest of this one
          else flat(xa, ya, xb, yb, 0, LANE);
        }
      }));
      for (const [jx, jy] of [[340, 420], [860, 420], [340, 940], [860, 940], [340, 1340], [860, 1340]]) { // worn stop bars where the cross streets meet the two long ones
        for (const side of [-1, 1]) decal(BAR, X(jx + side * 47), Z(jy), 3.3, 0.45, 1.5708, mul(lin(P.laneMark), 0.9));
        decal(DASH, X(jx), Z(jy - 62), 3.0, 0.5, 1.5708, mul(lin(P.laneMark), 0.8)); decal(DASH, X(jx), Z(jy + 62), 3.0, 0.5, 1.5708, mul(lin(P.laneMark), 0.8));
      }
      for (const [sx, sy, rot] of [[20, 610, 0], [600, 800, 1.5708], [170, 300, 0], [1180, 760, 0]]) decal(HOP, X(sx), Z(sy), 1.5, 1.5, rot, lin(0xf3ead8));   // chalk games in the lanes
      for (let k = 0; k < 44; k++) { const sx = 40 + hash(k, 71) * 1120, sy = 40 + hash(k, 72) * 1300; if (sx < 880 || sy > 400) decal(k % 3 ? RUBBLE : STAIN, X(sx), Z(sy), 1.2 + hash(k, 73), 1.2 + hash(k, 74), k, k % 3 === 1 ? mul(GREENS[k % 4], 0.9) : mul(LANE, k % 3 ? 0.78 : 0.7)); }   // litter and damp: it shows where it lands in a lane or on a street

      /* ---- one roofed volume: x0..x1 west to east, z0..z1 north to south (z1 is the front), walls h metres high ---- */
      function unit(x0, z0, x1, z1, h, o) {
        const tp = Math.tan(o.pitch), cp = Math.cos(o.pitch), wall = o.wall, low = mul(wall, 0.84);
        if (o.dado) { wallQuad(x0, z1, x1, z1, 0, 0.85, mul(o.dado, 0.8), o.dado); wallQuad(x0, z1, x1, z1, 0.85, h, wall); } else wallQuad(x0, z1, x1, z1, 0, h, low, wall);
        wallQuad(x1, z0, x0, z0, 0, h, low, wall); wallQuad(x1, z1, x1, z0, 0, h, low, wall); wallQuad(x0, z0, x0, z1, 0, h, low, wall);
        const xa = x0 - o.oW, xb = x1 + o.oE, zs = z1 + o.oS, zn = z0 - o.oN, yw = h + 0.05, hip = o.hip;
        const zr = hip ? (zs + zn) / 2 : (z0 + z1) / 2, ys = yw - o.oS * tp, yn = hip ? ys : yw - o.oN * tp, R = zs - zr, RN = zr - zn, yr = ys + R * tp;
        const cuts = [xa], n = Math.max(1, Math.round((xb - xa) / o.sheet)); if (hip) cuts.push(xa + R);
        for (let k = 1; k < n; k++) { const x = xa + (xb - xa) * k / n; if (!hip || (x > xa + R + 0.2 && x < xb - R - 0.2)) cuts.push(x); }
        if (hip) cuts.push(xb - R); cuts.push(xb);
        const up = (x) => hip ? Math.min(R, x - xa, xb - x) : 1e9;   // how far up the slope the roof reaches at x, measured on plan
        for (let k = 0; k + 1 < cuts.length; k++) {
          const xs = cuts[k], xe = cuts[k + 1], ds = Math.min(R, up(xs)), de = Math.min(R, up(xe)), dn = Math.min(RN, up(xs)), den = Math.min(RN, up(xe)), f = 1 / cp / TW;
          const v = startV(o.seed, k, o.rusty), w = startV(o.seed, k + 40, o.rusty);
          roofQuad(xs, ys, zs, xe, ys, zs, xe, ys + de * tp, zs - de, xs, ys + ds * tp, zs - ds, sheetColour(o.roof, o.seed, k), [xs / TW, v, xe / TW, v, xe / TW, v + de * f, xs / TW, v + ds * f]);
          roofQuad(xe, yn, zn, xs, yn, zn, xs, yn + dn * tp, zn + dn, xe, yn + den * tp, zn + den, sheetColour(o.roof, o.seed, k + 40), [xe / TW, w, xs / TW, w, xs / TW, w + dn * f, xe / TW, w + den * f]);
        }
        const cap = mix(mul(o.roof, 0.6), NEWZINC, 0.3);
        if (hip) {
          const ve = R / cp / TW, un = zn / TW, us = zs / TW, um = zr / TW;
          roofQuad(xa, ys, zn, xa, ys, zs, xa + R, yr, zr, xa + R, yr, zr, sheetColour(o.roof, o.seed, 90), [un, 0, us, 0, um, ve, um, ve]);
          roofQuad(xb, ys, zs, xb, ys, zn, xb - R, yr, zr, xb - R, yr, zr, sheetColour(o.roof, o.seed, 91), [us, 0, un, 0, um, ve, um, ve]);
          for (const [ex, ez, rx] of [[xa, zs, xa + R], [xa, zn, xa + R], [xb, zs, xb - R], [xb, zn, xb - R]]) ribbon(M, ex, ys + 0.03, ez, rx, yr + 0.05, zr, 0.26, cap);
          ribbon(M, xa + R, yr + 0.05, zr, xb - R, yr + 0.05, zr, 0.3, cap);
        } else {
          tri(M, x1, h, z1, x1, h, z0, x1, yr - 0.05, zr, wall); tri(M, x0, h, z0, x0, h, z1, x0, yr - 0.05, zr, wall);
          ribbon(M, xa, yr + 0.04, zr, xb, yr + 0.04, zr, 0.32, cap);
        }
        if (o.oS > 0.05) box(M, xa, ys - 0.17, zs - 0.03, xb, ys + 0.015, zs + 0.03, o.trim);   // fascia board on the front eave
        if (o.lite) return;
        for (let k = 0, m = Math.floor(hash(o.seed, 800) * 3.4); k < m; k++) { // what ends up on a roof: blocks and old tyres holding the sheets down, now and then a water tank
          const px = xa + 0.7 + (xb - xa - 1.4) * hash(o.seed, 810 + k), south = hash(o.seed, 830 + k) < 0.6, d = 0.5 + (Math.min(R, RN) - 1) * hash(o.seed, 820 + k), t = hash(o.seed, 840 + k);
          if (hip && d > up(px) - 0.4) continue;
          const z = south ? zs - d : zn + d, y = (south ? ys : yn) + d * tp;
          if (t < 0.55 || (t < 0.86 && !(hash(o.seed, 850 + k) < 0.15 && (o.seed & 3) === 1))) box(M, px - 0.2, y - 0.05, z - 0.1, px + 0.2, y + 0.17, z + 0.1, mul(CONCRETE, 0.8 + 0.2 * t));
          else if (t < 0.86) solid(M, TYRE, at(px, y + 0.11, z, 1.3, 1.3, 1.3, south ? o.pitch - 1.5708 : -o.pitch - 1.5708), mul(CONCRETE, 0.42));
          else if (!hip && R > 2.2) { const tz = zr + 0.75, ty = ys + (zs - tz) * tp; box(M, px - 0.5, ty - 0.3, tz - 0.5, px + 0.5, ty + 0.12, tz + 0.5, mul(CONCRETE, 0.85)); solid(M, CYL, at(px, ty + 0.62, tz, 0.48, 1.0, 0.48), lin(0x1c2622)); solid(M, CYL, at(px, ty + 1.15, tz, 0.2, 0.08, 0.2), lin(0x2c3a34)); }
        }
      }

      /* ---- the front of a house: a door with a step, windows either side. z is the wall face, looking south ---- */
      function opening(uv, x0, x1, y0, y1, z, tint, trim, lit, pool) {
        quad(M, x0 - 0.08, y0 - 0.06, z + 0.012, x1 + 0.08, y0 - 0.06, z + 0.012, x1 + 0.08, y1 + 0.08, z + 0.012, x0 - 0.08, y1 + 0.08, z + 0.012, trim);
        quad(lit < 0 ? F : LIT[lit], x0, y0, z + 0.024, x1, y0, z + 0.024, x1, y1, z + 0.024, x0, y1, z + 0.024, tint, uv);
        if (lit >= 0) { const k = pool || 0.34, wm = [1.0 * k, 0.62 * k, 0.26 * k], e = 0.5 + (x1 - x0) * 0.2; quad(PL, x0 - e, 0.03, z + 1.7 + (x1 - x0) * 0.3, x1 + e, 0.03, z + 1.7 + (x1 - x0) * 0.3, x1, 0.03, z + 0.04, x0, 0.03, z + 0.04, [0, 0, 0], null, wm); }
      }
      function front(x0, x1, z, h, s, o) {
        const w = x1 - x0, n = Math.max(1, Math.floor((w - 0.4) / 1.75)), slot = (w - 0.4) / n, door = Math.floor(hash(s, 60) * n), top = Math.min(2.05, h - 0.85);
        for (let k = 0; k < n; k++) {
          const cx = x0 + 0.2 + slot * (k + 0.5), r = hash(s, 61 + k);
          if (k === door) {
            opening(o.style === 1 ? PLANK : DOOR, cx - 0.46, cx + 0.46, 0.14, 2.06, z, o.accent, o.trim, -1);
            if (!o.lite) box(M, cx - 0.7, 0, z, cx + 0.7, 0.14, z + 0.24, mul(CONCRETE, 0.9));
          } else if (r > 0.16) {
            const ww = Math.min(0.56, slot * 0.36), lit = o.style !== 1 && hash(s, 70 + k) < 0.7 ? (hash(s, 75 + k) < 0.5 ? 0 : 1) : -1;
            if (r > 0.9 && !o.lite) opening(VENT, cx - 0.4, cx + 0.4, 1.3, 2.0, z, mul(o.wall, 1.05), o.trim, -1);
            else opening(o.style === 1 ? SHUT : lit === 1 ? CURTAIN : GLASS, cx - ww, cx + ww, 0.95, top, z, o.style === 1 ? o.accent : WHITE, o.trim, lit);
            if (!o.lite) box(M, cx - ww - 0.12, 0.86, z, cx + ww + 0.12, 0.92, z + 0.09, o.trim);
          }
        }
      }
      // a lean-to verandah along a front: slab, posts, a low wall between them, its own roof
      function verandah(x0, x1, zw, zf, h, s, o) {
        const w = x1 - x0, ytop = h - 0.38, ylow = ytop - (zf + 0.2 - zw) * 0.16, n = Math.max(2, Math.round(w / 2.3) + 1), bay = Math.floor(hash(s, 80) * (n - 1)), col = mix(o.roof, hash(s, 81) < 0.5 ? RUST : NEWZINC, 0.3), sheets = Math.max(1, Math.round(w / 1.1));
        box(M, x0, 0, zw, x1, 0.18, zf, mul(CONCRETE, 0.92));
        for (let k = 0; k < sheets; k++) { const xs = x0 - 0.1 + (w + 0.2) * k / sheets, xe = x0 - 0.1 + (w + 0.2) * (k + 1) / sheets, v = startV(s, 90 + k, 0.5); roofQuad(xs, ylow, zf + 0.2, xe, ylow, zf + 0.2, xe, ytop, zw, xs, ytop, zw, sheetColour(col, s, 60 + k), [xs / TW, v, xe / TW, v + (zf + 0.2 - zw) / TW]); }
        box(M, x0 - 0.1, ylow - 0.15, zf + 0.17, x1 + 0.1, ylow + 0.015, zf + 0.23, o.trim);
        for (let k = 0; k < n; k++) {
          const px = x0 + 0.07 + (w - 0.14) * k / (n - 1); box(M, px - 0.06, 0.18, zf - 0.14, px + 0.06, ylow - 0.05, zf - 0.02, o.trim);
          if (k < n - 1 && k !== bay) wallBox(px + 0.06, 0.18, zf - 0.11, px + (w - 0.14) / (n - 1) - 0.06, 0.84, zf - 0.04, mul(o.wall, 0.8), o.trim);
          else if (k === bay) box(M, px + 0.3, 0, zf, px + (w - 0.14) / (n - 1) - 0.3, 0.09, zf + 0.24, mul(CONCRETE, 0.88));
        }
      }

      /* ---- a fence along a lot line from (ax, az) to (bx, bz); `inward` is +1 or -1: which way the lot lies ---- */
      function fence(ax, az, bx, bz, inward, kind, h, seed, lite) {
        const ew = az === bz, s0 = ew ? ax : az, len = (ew ? bx : bz) - s0, fixed = ew ? az : ax, body = lin(0xd6cdb9), paint = ZPAINT[(seed * 3 + 1) % ZPAINT.length];
        const run = (f, a, b, y0, y1, d0, d1, col, cTop) => { // a piece from a to b along the line, d0..d1 metres into the lot
          const lo = fixed + Math.min(d0 * inward, d1 * inward), hi = fixed + Math.max(d0 * inward, d1 * inward);
          if (ew) f(a, y0, lo, b, y1, hi, col, cTop); else f(lo, y0, a, hi, y1, b, col, cTop);
        };
        const mbox = (x0, y0, z0, x1, y1, z1, col, cTop) => box(M, x0, y0, z0, x1, y1, z1, col, cTop);
        const sheets = (y0, uneven, tone) => { const n = Math.max(1, Math.round(len / (lite ? 1.72 : 0.86)));
          for (let k = 0; k < n; k++) { const r = hash(seed, 500 + k), col = mul(r < tone ? mix(paint, RUST, 0.6) : r > 0.9 ? ZPAINT[(seed + k) % ZPAINT.length] : paint, 0.86 + 0.24 * hash(seed, 600 + k));
            run(zincBox, s0 + len * k / n, s0 + len * (k + 1) / n, y0, h - uneven * hash(seed, 700 + k), 0, 0.06, col); } };
        if (kind === 1) { // rendered block wall with a painted band and a cap
          const band = mix(body, ACCENTS[seed % ACCENTS.length], 0.7);
          run(wallBox, s0, s0 + len, 0, h * 0.36, 0, 0.18, band); run(wallBox, s0, s0 + len, h * 0.36, h - 0.07, 0, 0.18, body); run(mbox, s0, s0 + len, h - 0.07, h, -0.02, 0.21, mul(body, 0.8));
          if (!lite) for (let k = 0, n = Math.max(1, Math.round(len / 3.2)); k <= n; k++) { const a = Math.min(s0 + len - 0.26, s0 + len * k / n); run(wallBox, a, a + 0.26, 0, h + 0.06, -0.02, 0.24, body); }
        } else if (kind === 3) { // low block wall with zinc above it
          run(wallBox, s0, s0 + len, 0, 0.62, 0, 0.16, body, mul(body, 0.8)); sheets(0.62, 0.1, 0.12);
        } else { // zinc sheets: raw and hand-cut to uneven heights, or painted and neat
          sheets(0, kind === 2 ? 0.05 : 0.24, kind === 2 ? 0.06 : 0.3);
          if (!lite) { run(mbox, s0, s0 + len, h * 0.7, h * 0.7 + 0.07, 0.06, 0.11, TIMBER); run(mbox, s0, s0 + len, h * 0.2, h * 0.2 + 0.07, 0.06, 0.11, TIMBER);
            for (let k = 0, n = Math.max(1, Math.round(len / 1.72)); k <= n; k++) { const a = Math.min(s0 + len - 0.09, s0 + len * k / n); run(mbox, a, a + 0.09, 0, h - 0.2, 0.06, 0.15, TIMBER); } }
        }
        if (ew && !lite && hash(seed, 9) < 0.6) { // a gate that is only drawn: two pillars and a painted leaf, seen from the yard side
          const gx = ax + len * (0.25 + 0.5 * hash(seed, 10)), gw = 1.25, gh = Math.min(h, 1.5), t = kind === 1 ? 0.19 : 0.07;
          quad(F, gx - gw, 0.05, az + t * inward, gx + gw, 0.05, az + t * inward, gx + gw, gh, az + t * inward, gx - gw, gh, az + t * inward, ACCENTS[(seed + 3) % ACCENTS.length], GATE);
          for (const sx of [-1, 1]) wallBox(gx + sx * gw - 0.13, 0, az - 0.02, gx + sx * gw + 0.13, h + 0.04, az + 0.26, body, mul(body, 0.8));
        }
      }

      /* ---- what stands in a yard: nothing over 1.7 m ---- */
      const CYL = new THREE.CylinderGeometry(1, 1, 1, 10), ICO = new THREE.IcosahedronGeometry(1, 0), TYRE = new THREE.TorusGeometry(0.27, 0.11, 5, 10), POST = new THREE.CylinderGeometry(1, 1, 1, 6);
      function bush(x, z, r, s) { solid(M, ICO, at(x, r * 0.62, z, r, r * 0.8, r, 0, s), mul(GREENS[s % GREENS.length], 0.8 + 0.4 * hash(s, 91))); }
      function yardThings(x0, x1, z0, z1, s) {
        const n = Math.floor((x1 - x0) / 2.3), zc = (z0 + z1) / 2;
        for (let k = 0; k < n; k++) {
          const cx = x0 + (x1 - x0) * (k + 0.3 + 0.4 * hash(s, 100 + k)) / n, cz = zc + (hash(s, 110 + k) - 0.5) * (z1 - z0 - 1.1), t = Math.floor(hash(s, 120 + k) * 9), q = s * 17 + k;
          if (t === 0) solid(M, CYL, at(cx, 0.45, cz, 0.29, 0.9, 0.29), lin(hash(q, 1) < 0.6 ? 0x2f6fb5 : 0x8a4f2c));
          else if (t <= 2) { bush(cx, cz, 0.5 + 0.3 * hash(q, 2), q); if (t === 2) bush(cx + 0.6, cz + 0.2, 0.4, q + 1); }
          else if (t === 3 && k + 1 < n) { // a washing line
            const xe = cx + (x1 - x0) / n * 0.9; box(M, cx - 0.04, 0, cz - 0.04, cx + 0.04, 1.62, cz + 0.04, TIMBER); box(M, xe - 0.04, 0, cz - 0.04, xe + 0.04, 1.62, cz + 0.04, TIMBER); box(M, cx, 1.55, cz - 0.012, xe, 1.575, cz + 0.012, DARK);
            for (let j = 0, m = Math.floor((xe - cx - 0.3) / 0.55); j < m; j++) { const a = cx + 0.2 + j * 0.55, wd = 0.3 + 0.2 * hash(q, 20 + j), y0 = 0.7 + 0.4 * hash(q, 30 + j), col = CLOTH[Math.floor(hash(q, 40 + j) * CLOTH.length)];
              quad(M, a, y0, cz + 0.02, a + wd, y0, cz + 0.02, a + wd, 1.56, cz + 0.02, a, 1.56, cz + 0.02, col); quad(M, a + wd, y0, cz - 0.02, a, y0, cz - 0.02, a, 1.56, cz - 0.02, a + wd, 1.56, cz - 0.02, col); }
            k++;
          } else if (t === 4) { box(M, cx - 0.7, 0.38, cz - 0.16, cx + 0.7, 0.44, cz + 0.16, TIMBER); box(M, cx - 0.6, 0, cz - 0.12, cx - 0.45, 0.38, cz + 0.12, CONCRETE); box(M, cx + 0.45, 0, cz - 0.12, cx + 0.6, 0.38, cz + 0.12, CONCRETE); }
          else if (t === 5) { box(M, cx - 0.4, 0, cz - 0.2, cx + 0.4, 0.2, cz + 0.2, mul(CONCRETE, 0.8)); box(M, cx - 0.3, 0.2, cz - 0.2, cx + 0.3, 0.4, cz + 0.2, mul(CONCRETE, 0.72)); box(M, cx - 0.1, 0.4, cz - 0.2, cx + 0.3, 0.6, cz + 0.2, mul(CONCRETE, 0.8)); }
          else if (t === 6) { solid(M, TYRE, at(cx, 0.11, cz, 1, 1, 1, 1.5708), DARK); solid(M, TYRE, at(cx + 0.1, 0.33, cz + 0.05, 1, 1, 1, 1.5708), DARK); }
          else if (t === 7) { bush(cx, cz, 0.42, q + 4); solid(M, CYL, at(cx, 0.2, cz, 0.3, 0.4, 0.3), lin(0xa0553a)); }
        }
      }

      /* ---- an ordinary lot: yard strip and fence on the north side, house flush with the south edge ---- */
      function lot(b, lite) {
        const hs = NH.house(b), H = hs.height, s = b.i, x0 = X(b.x), x1 = X(b.x + b.w), zN = Z(b.y), zH = Z(hs.y) + (hs.yard && Z(b.y + b.h) - Z(hs.y) > 6 ? 1.4 : 0), zS = Z(b.y + b.h), w = x1 - x0, d = zS - zH;
        const roof = mul(rich(lin(b.c), 1.5), 1.22), wall = mul(lin(b.wall), 1.08), trim = TRIMS[(s * 5 + 2) % TRIMS.length], accent = ACCENTS[(s * 3 + 1) % ACCENTS.length], style = s % 3;
        const dado = hash(s, 6) < 0.62 ? mix(wall, accent, 0.5) : null;
        if (hs.yard) {
          flat(b.x, b.y, b.x + b.w, b.y + (zH - zN) / NH.U, 0.02, mul(EARTH, 0.85 + 0.25 * hash(s, 1)));
          const kind = [0, 1, 2, 3, 0, 2, 1, 0, 3][(s * 4 + (s >> 2)) % 9], fh = 1.15 + 0.3 * hash(s, 2);
          fence(x0, zN, x1, zN, 1, kind, fh, s, lite); fence(x0, zN, x0, zH, 1, kind, fh + 0.06, s + 77, lite); fence(x1, zN, x1, zH, -1, kind, fh + 0.06, s + 131, lite);
          if (!lite) yardThings(x0 + 0.5, x1 - 0.5, zN + 0.3, zH - 0.2, s); else if (hash(s, 3) < 0.6) bush(x0 + w * (0.2 + 0.6 * hash(s, 4)), (zN + zH) / 2, 0.7, s);
        }
        const nx = lite ? 1 : w >= 17 ? 3 : w >= 11 ? 2 : 1, nz = d >= 13 ? 3 : d >= 8.6 ? 2 : 1, lead = Math.floor(hash(s, 5) * nx), xb = [x0], zb = [zS];
        for (let k = 1; k < nx; k++) xb.push(x0 + w * (k + (hash(s, 10 + k) - 0.5) * 0.3) / nx);
        for (let k = 1; k < nz; k++) zb.push(zS - d * (k + (hash(s, 20 + k) - 0.5) * 0.25) / nz);
        xb.push(x1); zb.push(zH);
        for (let ix = 0; ix < nx; ix++) for (let iz = 0; iz < nz; iz++) {
          const ux0 = xb[ix], ux1 = xb[ix + 1], uz1 = zb[iz], uz0 = zb[iz + 1], q = s * 9 + ix * 3 + iz;
          const h = H - (ix === lead ? 0 : 0.2 + 0.35 * hash(s, 30 + ix)) - iz * (0.15 + 0.3 * hash(s, 40 + iz + ix * 3));
          const hip = hash(q, 50) < 0.3 && ux1 - ux0 > uz1 - uz0 + 2, ver = !iz && !lite && uz1 - uz0 > 4.6 && hash(q, 52) < 0.36, vd = ver ? 1.25 : 0;
          const o = { wall, dado, roof: iz ? mix(roof, RUST, 0.1 * iz) : roof, trim, accent, style, lite, seed: q, hip, rusty: 0.08 + 0.6 * hash(q, 53) * hash(s, 54) + 0.2 * iz, pitch: (12 + 7 * hash(q, 51)) * Math.PI / 180, sheet: lite ? 2 : 1,
            oS: ver ? 0.1 : hip ? 0.1 : iz ? 0 : 0.3, oN: hip ? 0.1 : iz === nz - 1 ? 0.05 : 0, oW: ix ? 0 : 0.22, oE: ix === nx - 1 ? 0.22 : 0 };
          unit(ux0, uz0, ux1, uz1 - vd, h, o);
          if (ver) verandah(ux0, ux1, uz1 - vd, uz1, h, q, o); else if (!iz) front(ux0, ux1, uz1, h, q, o);
        }
      }

      /* ---- Zinc Roof Soundhall: a hall with a painted false front, a wide doorway and the dance yard south of it ---- */
      function hall(b) {
        const hs = NH.house(b), H = hs.height, x0 = X(b.x), x1 = X(b.x + b.w), zN = Z(b.y), zH = Z(hs.y), zS = Z(b.y + b.h), xm = (x0 + x1) / 2, wall = mul(lin(b.wall), 1.08), trim = TRIMS[0];
        const top = H + 1.0, band = lin(0x16283c), teal = lin(P.harbour), dx0 = X(1052), dx1 = X(1132), paint = mul(SODIUM, 1.3);
        flat(b.x, b.y, b.x + b.w, hs.y, 0.02, EARTH);
        fence(x0, zN, x1, zN, 1, 2, 1.45, 6); fence(x0, zN, x0, zH, 1, 2, 1.5, 83); fence(x1, zN, x1, zH, -1, 2, 1.5, 137); yardThings(x0 + 0.5, x1 - 0.5, zN + 0.3, zH - 0.2, 606);
        unit(x0, zH, x1, zS - 0.2, H, { wall, roof: mul(rich(lin(b.c), 1.5), 1.22), trim, seed: 606, hip: false, rusty: 0.3, pitch: 0.26, sheet: 1, oS: 0, oN: 0.05, oW: 0.22, oE: 0.22 });
        wallBox(x0 - 0.12, 0, zS - 0.2, x1 + 0.12, 0.55, zS, mul(teal, 0.85)); wallBox(x0 - 0.12, 0.55, zS - 0.2, x1 + 0.12, 2.5, zS, wall);
        box(M, x0 - 0.12, 2.5, zS - 0.2, x1 + 0.12, 2.62, zS + 0.01, SODIUM); box(M, x0 - 0.12, 2.62, zS - 0.2, x1 + 0.12, 2.74, zS + 0.01, teal);
        box(M, x0 - 0.12, 2.74, zS - 0.2, x1 + 0.12, top, zS, band); box(M, xm - 3.6, top, zS - 0.2, xm + 3.6, top + 0.34, zS, band);
        box(M, x0 - 0.16, top, zS - 0.24, xm - 3.6, top + 0.07, zS + 0.04, INK); box(M, xm + 3.6, top, zS - 0.24, x1 + 0.16, top + 0.07, zS + 0.04, INK); box(M, xm - 3.64, top + 0.34, zS - 0.24, xm + 3.64, top + 0.41, zS + 0.04, INK);
        [[xm - 5.75, xm - 0.4], [xm + 0.4, xm + 5.75]].forEach(([a, c], k) => quad(SG, a, 2.9, zS + 0.02, c, 2.9, zS + 0.02, c, 4.24, zS + 0.02, a, 4.24, zS + 0.02, INK, WORD[k]));
        box(M, xm - 0.13, 3.44, zS, xm + 0.13, 3.7, zS + 0.02, SODIUM);
        opening(INSIDE, dx0, dx1, 0, 2.4, zS, WHITE, trim, 0, 0.9);                                                   // the wide doorway, its zinc doors slid open
        zincBox(dx0 - 1.5, 0.05, zS + 0.03, dx0 - 0.08, 2.42, zS + 0.09, paint); zincBox(dx1 + 0.08, 0.05, zS + 0.03, x1 + 0.1, 2.42, zS + 0.09, paint);
        box(M, dx0 - 1.55, 2.42, zS + 0.02, x1 + 0.12, 2.5, zS + 0.13, DARK);
        for (let k = 0; k < 4; k++) { const fx = x0 + 0.3 + k * 0.5, fy = 1.0 + 0.12 * hash(k, 95); quad(F, fx, fy, zS + 0.015, fx + 0.42, fy, zS + 0.015, fx + 0.42, fy + 0.56, zS + 0.015, fx, fy + 0.56, zS + 0.015, WHITE, FLYER[(k * 2 + 1) % 6]); }
        for (const sx of [912, 1146]) box(M, X(sx) - 0.03, 3.5, zS, X(sx) + 0.03, 3.58, zS + 0.54, DARK);            // brackets for the string of lights
        box(M, x0 + 0.25, 0.36, zS + 0.04, x0 + 1.75, 0.42, zS + 0.4, TIMBER); box(M, x0 + 0.35, 0, zS + 0.08, x0 + 0.5, 0.36, zS + 0.36, CONCRETE); box(M, x0 + 1.5, 0, zS + 0.08, x0 + 1.65, 0.36, zS + 0.36, CONCRETE);   // a bench, drinks crates, a cooler
        [[0x3aa0c8, 0], [0xc8573a, 0.3], [0xe8b23a, 0.6]].forEach(([c, y], j) => box(M, x1 - 0.62 + (j & 1) * 0.04, y, zS + 0.12, x1 - 0.18 + (j & 1) * 0.04, y + 0.28, zS + 0.44, lin(c)));
        box(M, x1 - 1.3, 0, zS + 0.12, x1 - 0.75, 0.4, zS + 0.44, lin(0x2c5fa8), INK);
        for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) flat(900 + i * 65, 190 + j * 190 / 3, 965 + i * 65, 190 + (j + 1) * 190 / 3, 0, mul(CONCRETE, 0.9 + 0.12 * hash(i * 3 + j, 77)));
        for (const [ax, ay, bx, by] of [[900, 374, 985, 380], [1085, 374, 1160, 380], [900, 196, 906, 250], [900, 320, 906, 374], [1154, 196, 1160, 374]]) box(M, X(ax), 0, Z(ay), X(bx), 0.05, Z(by), lin(0xe9e3d4));   // a whitewashed kerb, low enough to step over
        const cable = [[1070, 196], [1058, 226], [1030, 242], [1000, 246]]; for (let k = 0; k + 1 < cable.length; k++) ribbon(D, X(cable[k][0]), 0.016, Z(cable[k][1]), X(cable[k + 1][0]), 0.016, Z(cable[k + 1][1]), 0.07, DARK, SOLID);
        ribbon(D, X(1076), 0.016, Z(196), X(1106), 0.016, Z(236), 0.07, DARK, SOLID);
        decal(PATCH, X(979), Z(224), 5.4, 2.0, 0, mul(CONCRETE, 0.78));                                               // the stand the speakers go on
      }

      /* ---- Keelwright Boatworks: an open-sided shed, its front bay bare rafters over a hull on trestles, and the slip ---- */
      function boatworks(b) {
        const x0 = X(b.x), x1 = X(b.x + b.w), zN = Z(b.y), zS = Z(b.y + b.h), xm = (x0 + x1) / 2, w = x1 - x0, hS = 2.8, hN = 2.05, k = (hS - hN) / (zS - zN), zc = zN + 1.9, yc = hN + (zc - zN) * k;
        const roof = mul(rich(lin(b.c), 1.5), 1.22), hx = xm - 0.25, hz = zS - 1.12, zE = Z(1566), yE = -(zE - zS) * 0.24;
        flat(b.x, b.y, b.x + b.w, b.y + b.h, 0.02, mul(QUAY, 0.8));
        for (const px of [x0 + 0.09, xm, x1 - 0.09]) { box(M, px - 0.08, 0, zS - 0.16, px + 0.08, hS - 0.1, zS, TIMBER); box(M, px - 0.08, 0, zN, px + 0.08, hN - 0.05, zN + 0.16, TIMBER); }
        for (let j = 0; j < 9; j++) { const xs = x0 - 0.2 + (w + 0.4) * j / 9, xe = x0 - 0.2 + (w + 0.4) * (j + 1) / 9, v = startV(23, j, 0.45);
          roofQuad(xe, hN - 0.05 * k, zN - 0.05, xs, hN - 0.05 * k, zN - 0.05, xs, yc, zc, xe, yc, zc, sheetColour(roof, 23, j), [xe / TW, v, xs / TW, v, xs / TW, v + 1.95 / TW, xe / TW, v + 1.95 / TW]);
          ribbon(M, xs + 0.4, yc - 0.03, zc - 0.1, xs + 0.4, hS - 0.02, zS - 0.05, 0.1, mul(TIMBER, 1.15)); }
        box(M, x0 - 0.2, hS - 0.2, zS - 0.14, x1 + 0.2, hS, zS, TIMBER); box(M, x0 - 0.2, yc + 0.28 * k - 0.1, zc + 0.68, x1 + 0.2, yc + 0.28 * k, zc + 0.78, TIMBER);
        zincBox(x0, 0, zN, x0 + 0.06, hN, zS - 0.2, mix(ZPAINT[1], ZPAINT[0], 0.5)); zincBox(x0, 0, zN, x1, hN - 0.1, zN + 0.06, ZPAINT[0]);
        [[xm - 4.25, xm - 0.2], [xm + 0.2, xm + 4.25]].forEach(([a, c], j) => { const za = zc - 0.22, zb = zN + 0.2, ya = hN + (za - zN) * k + 0.03, yb = hN + (zb - zN) * k + 0.03;   // the name, painted on the roof where the harbour can read it
          quad(SG, a, ya, za, c, ya, za, c, yb, zb, a, yb, zb, lin(0xf6e7c0), WORD[2 + j]); });
        // the hull: hard-chined, stern to the west, built up from stations
        const N = 9, L = 5.6, st = [], wood = lin(0xc49a62), primer = lin(0x9c4a38), inside = lin(0x9a7448);
        for (let j = 0; j <= N; j++) { const t = j / N, hb = 0.85 * (t < 0.45 ? 0.7 + 0.3 * Math.sin(t / 0.45 * 1.5708) : Math.pow(Math.max(0, Math.cos((t - 0.45) / 0.55 * 1.5708)), 0.75)), keel = 0.56 + (t > 0.78 ? (t - 0.78) * 2.6 : 0), sheer = 1.36 + 0.3 * t * t;
          st.push([hx - L / 2 + t * L, hb, keel, keel + (sheer - keel) * 0.36, sheer]); }
        for (let j = 0; j < N; j++) for (const sd of [-1, 1]) { const a = st[j], c = st[j + 1], za = hz + sd * a[1], zb = hz + sd * c[1], zca = hz + sd * a[1] * 0.62, zcb = hz + sd * c[1] * 0.62, f = sd > 0;
          const pan = (p, q, r, t, col, flip) => f !== flip ? quad(M, p[0], p[1], p[2], q[0], q[1], q[2], r[0], r[1], r[2], t[0], t[1], t[2], col) : quad(M, t[0], t[1], t[2], r[0], r[1], r[2], q[0], q[1], q[2], p[0], p[1], p[2], col);
          const k0 = [a[0], a[2], hz], k1 = [c[0], c[2], hz], c0 = [a[0], a[3], zca], c1 = [c[0], c[3], zcb], s0 = [a[0], a[4], za], s1 = [c[0], c[4], zb];
          pan(k0, k1, c1, c0, primer, false); pan(c0, c1, s1, s0, mul(wood, 0.92 + 0.16 * hash(j, 96)), false); pan(k0, k1, c1, c0, inside, true); pan(c0, c1, s1, s0, mul(inside, 1.15), true); }
        { const a = st[0]; quad(M, a[0], a[2], hz, a[0], a[3], hz - a[1] * 0.62, a[0], a[4], hz - a[1], a[0], a[4], hz + a[1], wood); tri(M, a[0], a[2], hz, a[0], a[4], hz + a[1], a[0], a[3], hz + a[1] * 0.62, wood); }
        for (const t of [2, 4, 6]) box(M, st[t][0] - 0.1, st[t][4] - 0.2, hz - st[t][1] * 0.9, st[t][0] + 0.1, st[t][4] - 0.15, hz + st[t][1] * 0.9, mul(wood, 0.8));
        for (const tx of [hx - 1.6, hx + 1.2]) { box(M, tx - 0.07, 0.42, hz - 0.8, tx + 0.07, 0.56, hz + 0.8, TIMBER); for (const sd of [-1, 1]) box(M, tx - 0.06, 0, hz + sd * 0.7 - 0.06, tx + 0.06, 0.42, hz + sd * 0.7 + 0.06, TIMBER); }
        solid(M, CYL, at(x1 - 0.6, 0.45, zS - 0.55, 0.29, 0.9, 0.29), lin(0x2f6fb5)); solid(M, TYRE, at(x1 - 1.3, 0.1, zS - 0.5, 1.1, 1.1, 1.1, 1.5708), ROPE);
        // the slip: a ramp into the water with four timber ways
        quad(G, X(64), yE, zE, X(186), yE, zE, X(186), 0, zS, X(64), 0, zS, mul(QUAY, 0.62), null, mul(QUAY, 0.95));
        quad(G, X(64), yE, zE, X(64), 0, zS, X(64), -1.5, zS, X(64), -1.5, zE, mul(QUAY, 0.5)); quad(G, X(186), 0, zS, X(186), yE, zE, X(186), -1.5, zE, X(186), -1.5, zS, mul(QUAY, 0.5));
        for (const wx of [-2.2, -0.75, 0.75, 2.2]) { ribbon(M, hx + wx, 0.05, zS - 2.1, hx + wx, 0.05, zS, 0.16, mul(TIMBER, 0.8)); ribbon(M, hx + wx, 0.05, zS, hx + wx, yE + 0.05, zE, 0.16, mul(TIMBER, 0.8)); }
        decal(FADE, X(125), zS + 3.2, 6.1, 3.6, 0, lin(0x6fb0b0), -0.34);                                                // the ramp showing through the shallows
      }

      for (const b of sim.B) { if (b === sim.SOUNDHALL) hall(b); else if (b === sim.BOATWORKS) boatworks(b); else lot(b, false); }

      /* ---- quay and harbour ---- */
      const SLIP0 = 64, SLIP1 = 186, zQ = Z(sim.WATER);
      for (let x = -936; x < 2136; x += 64) for (const [ya, yb] of [[sim.QUAY, 1420], [1420, 1460]]) { // concrete slabs; the slip cuts through them
        const c = mul(QUAY, 0.92 + 0.14 * hash(x, ya)); if (ya > 1400 && x + 64 > SLIP0 && x < SLIP1 + 26) { if (x < SLIP0) flat(x, ya, SLIP0, yb, 0, c); else if (x + 64 > SLIP1) flat(SLIP1, ya, x + 64, yb, 0, c); flat(Math.max(x, SLIP0), ya, Math.min(x + 64, SLIP1), 1458, 0, c); }
        else flat(x, ya, x + 64, yb, 0, c); }
      for (let x = -936; x < 2136; x += 6) if (x + 6 <= SLIP0 || x >= SLIP1) flat(x, 1460, x + 6, sim.WATER, 0, mul(TIMBER, 0.95 + 0.5 * hash(x, 81)));   // a timber capping, plank by plank
      for (const [xa, xb] of [[-936, SLIP0], [SLIP1, 2136]]) { quad(G, X(xa), -1.5, zQ, X(xb), -1.5, zQ, X(xb), 0, zQ, X(xa), 0, zQ, mul(QUAY, 0.4), null, mul(TIMBER, 0.8)); decal(FADE, X((xa + xb) / 2), zQ + 1.1, (xb - xa) * U, 2.2, 0, lin(0x7fb0ac), -0.34); }
      for (let x = 216; x < 1196; x += 44) if (!sim.PALMS.some((q) => Math.abs(q[0] - x - 14) < 30)) decal(SOLID, X(x + 14), Z(1456), 1.4, 0.16, 0, mul(lin(0xe0b040), 0.85 + 0.3 * hash(x, 82)));   // the painted edge stripe, worn
      for (const [px, py] of sim.PALMS) decal(PIT, X(px), Z(py), py > 1000 ? 1.25 : 1.5, py > 1000 ? 1.25 : 1.5, px, WHITE);
      const BOLLARDS = [250, 420, 520, 590, 690, 800, 960, 1060, 1160, -60, 1290];
      for (const bx of BOLLARDS) { solid(M, CYL, at(X(bx), 0.2, Z(1465), 0.15, 0.4, 0.15), DARK); solid(M, CYL, at(X(bx), 0.43, Z(1465), 0.21, 0.08, 0.21), lin(0xd6a23a)); }
      for (let x = 206; x < 1300; x += 60) { // fender piles, with an old tyre hung between each pair
        solid(M, POST, at(X(x), -0.5, zQ + 0.15, 0.13, 1.7, 0.13), mul(TIMBER, 0.7)); if (x % 120 === 86) solid(M, TYRE, at(X(x + 30), -0.1, zQ + 0.12, 1.15, 1.15, 1.15), DARK); }
      sim.SKIFFS.forEach(([sx], i) => { // mooring lines from the bollards to where main.js floats the skiffs
        for (const sd of [-1, 1]) { const bx = BOLLARDS.reduce((a, c) => Math.abs(c - sx - sd * 50) < Math.abs(a - sx - sd * 50) ? c : a); ribbon(M, X(bx), 0.36, Z(1465), X(sx) + sd * 2.25, -0.08, Z(sim.WATER + 44) + sd * 0.3 * (i - 1), 0.05, ROPE); }
      });
      for (let i = 0; i < 14; i++) for (let j = 0; j < 5; j++) { // the harbour: paler in the shallows under the quay, deeper further out
        const xa = X(-2200 + i * 400), xb = X(-1800 + i * 400), za = zQ + [0, 2.5, 7, 16, 30, 76][j], zb = zQ + [0, 2.5, 7, 16, 30, 76][j + 1], tone = (jj, ii) => mul(WHITE, [1.12, 1, 0.9, 0.8, 0.72, 0.66][jj] * (0.95 + 0.1 * hash(ii * 7 + jj, 83)));
        const a = W.count; quad(W, xa, -0.35, zb, xb, -0.35, zb, xb, -0.35, za, xa, -0.35, za, WHITE, [xa / 7, zb / 7, xb / 7, za / 7]);
        [tone(j + 1, i), tone(j + 1, i + 1), tone(j, i + 1), tone(j, i)].forEach((c, q) => { W.c[(a + q) * 3] = c[0]; W.c[(a + q) * 3 + 1] = c[1]; W.c[(a + q) * 3 + 2] = c[2]; });
      }

      for (let k = 0; k < 7; k++) solid(M, ICO, at(X(150 + k * 170 + 90 * hash(k, 97)), -0.3, zQ + 6.5 + 5 * hash(k, 98), 0.2 + 0.05 * (k % 2)), k % 2 ? INK : lin(0xe0703a));   // mooring buoys

      /* ---- beyond the map: neighbouring lots, the streets running on behind barricades, and low hills to the north ---- */
      let extra = 100;
      function neighbours(cx, cy, cw, ch) {
        const rows = Math.max(1, Math.round(ch / 215)), rh = (ch - 40 * (rows - 1)) / rows;
        for (let r = 0; r < rows; r++) { const cols = cw > 300 ? 2 + (hash(extra, 1) < 0.4 ? 1 : 0) : hash(extra, 2) < 0.45 ? 2 : 1, lw = (cw - 40 * (cols - 1)) / cols;
          for (let c = 0; c < cols; c++) { const i = extra++; lot({ i, x: cx + c * (lw + 40), y: cy + r * (rh + 40), w: lw, h: rh, c: sim.ROOF[i % 6], tall: 1 + (i * 37 % 10) / 14, wall: sim.WALL[i % 5] }, true); } }
      }
      for (const [xa, xb] of [[-936, -676], [-596, -336], [-256, -4], [1204, 1456], [1536, 1796], [1876, 2136]]) for (const [ya, yb] of [[-640, -380], [-300, 380], [460, 900], [980, 1300]]) neighbours(xa, ya, xb - xa, yb - ya);
      for (const [xa, xb] of [[4, 300], [380, 820], [900, 1196]]) for (const [ya, yb] of [[-640, -380], [-300, -4]]) neighbours(xa, ya, xb - xa, yb - ya);
      for (const [sx, i] of [[-250, 90], [1250, 91], [1560, 92], [-560, 93]]) lot({ i, x: sx, y: 1388, w: 200, h: 74, c: sim.ROOF[i % 6], tall: 1, wall: sim.WALL[i % 5] }, true);   // sheds along the quay
      fence(X(-1), Z(sim.QUAY + 4), X(-1), Z(sim.WATER - 2), -1, 0, 1.6, 201); fence(X(sim.W + 1), Z(sim.QUAY + 4), X(sim.W + 1), Z(sim.WATER - 2), 1, 0, 1.6, 202);
      function barricade(x, z, alongX) { // two drums and a striped board across a street nobody can walk down
        for (const sd of [-1, 1]) { const px = x + (alongX ? sd * 1.25 : 0), pz = z + (alongX ? 0 : sd * 1.25); solid(M, CYL, at(px, 0.45, pz, 0.3, 0.9, 0.3), lin(0xc8573a)); solid(M, CYL, at(px, 0.5, pz, 0.31, 0.22, 0.31), INK); }
        for (let j = 0; j < 7; j++) { const a = -1.75 + j * 0.5, col = j % 2 ? DARK : lin(0xe8b23a); if (alongX) box(M, x + a, 0.95, z - 0.03, x + a + 0.5, 1.15, z + 0.03, col); else box(M, x - 0.03, 0.95, z + a, x + 0.03, 1.15, z + a + 0.5, col); }
      }
      for (const sy of sim.STREETS.horizontal) { barricade(X(-16), Z(sy), false); barricade(X(sim.W + 16), Z(sy), false); }
      for (const sx of sim.STREETS.vertical) barricade(X(sx), Z(-16), true);
      const hillAt = (x, z) => smooth(-74, -150, z) * (9 + 6 * Math.sin(x * 0.045 + 1.3) + 4 * Math.sin(x * 0.11 + z * 0.06)), SCRUB = lin(0x7c8a55), SCRUB2 = lin(0x55703f);
      for (let i = 0; i < 16; i++) for (let j = 0; j < 8; j++) {
        const xa = -112 + i * 14, xb = xa + 14, za = -74 - j * 14, zb = za - 14, c = mix(SCRUB, SCRUB2, hash(i * 9 + j, 85)), c2 = mix(SCRUB, SCRUB2, hash(i * 9 + j, 86));
        tri(G, xa, hillAt(xa, za), za, xb, hillAt(xb, za), za, xb, hillAt(xb, zb), zb, c); tri(G, xa, hillAt(xa, za), za, xb, hillAt(xb, zb), zb, xa, hillAt(xa, zb), zb, c2);
      }
      for (let k = 0; k < 70; k++) { const x = -105 + 210 * hash(k, 87), z = -76 - 46 * hash(k, 88) * hash(k, 89), r = 1.1 + 1.3 * hash(k, 90); solid(M, ICO, at(x, hillAt(x, z) + r * 0.5, z, r, r * 0.8, r, 0, k), mul(GREENS[k % 3], 0.75 + 0.3 * hash(k, 91))); }

      /* ---- materials and meshes ---- */
      const std = (o) => new THREE.MeshStandardMaterial(Object.assign({ vertexColors: true, roughness: 0.92, metalness: 0 }, o));
      const lay = { polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 };
      const water = std({ map: waterTex, roughness: 0.32 });
      const lit = LIT.map(() => std(Object.assign({ map: frontTex, emissive: 0xffffff, emissiveMap: glowTex, emissiveIntensity: 0 }, lay)));
      const signs = std(Object.assign({ map: signTex, transparent: true, depthWrite: false, emissive: 0xffffff, emissiveMap: signTex, emissiveIntensity: 0 }, lay));
      const pools = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
      const meshes = [toMesh(G, std({ map: grainTex }), false), toMesh(M, std({ map: plasterTex }), true),
        toMesh(ZN, std({ map: zincTex, normalMap: zincNormal, roughness: 0.8, metalness: 0.12 }), true), toMesh(F, std(Object.assign({ map: frontTex }, lay)), false),
        toMesh(LIT[0], lit[0], false), toMesh(LIT[1], lit[1], false), toMesh(D, std(Object.assign({ map: decalTex, transparent: true, depthWrite: false }, lay)), false, -3),
        toMesh(W, water, false), toMesh(PL, pools, false, -1)];
      if (SG.count) meshes.push(toMesh(SG, signs, false, -2));
      let triangles = 0; for (const m of meshes) { triangles += m.geometry.index.count / 3; group.add(m); }
      const dayWater = new THREE.Color(0x2f93ab), goldWater = new THREE.Color(0xc9925a), nightWater = new THREE.Color(0x0d2436);
      return {
        group, stats: { meshes: meshes.length, triangles },
        update(S, tod) {
          const n = tod.night, t = S.t;
          waterTex.offset.set(t * 0.011 + Math.sin(t * 0.6) * 0.006, t * 0.017);
          water.color.copy(dayWater).lerp(goldWater, tod.warm * 0.45).lerp(nightWater, n);
          lit[0].emissiveIntensity = 2.4 * Math.min(1, n * 2.5); lit[1].emissiveIntensity = 2.4 * Math.max(0, Math.min(1, n * 2.5 - 0.9));
          signs.emissiveIntensity = 0.5 * n; pools.opacity = 0.9 * n;
        },
      };
    },
  };
})();
