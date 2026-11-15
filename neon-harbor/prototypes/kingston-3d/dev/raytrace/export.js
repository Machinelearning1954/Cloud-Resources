#!/usr/bin/env node
/* Export the prototype's 3D scene, exactly as the game builds it, to one JSON file that an
   offline ray tracer can load (see render.py). Needs Playwright.
     node dev/raytrace/export.js <page.html> <out.json> [--dusk 0.45] [--at 1010,345] [--delivered 3] [--carry] [--size 1080x1920]
   What is exported: every visible mesh (instances expanded, joined per material), materials
   (colour, roughness, metal, glow, textures drawn in code), lights, the game camera, the time of day.
   What is left out: game markers (vision cones, drop ring, arrow, see-through shapes), additive
   "pool of light" decals (a ray tracer makes real ones), sky domes, lines.
   Axes are converted from the game's (x east, y up, z south) to (x east, y north, z up). */
"use strict";
const fs = require("node:fs"), path = require("node:path");
const { chromium } = require("playwright");
const a = process.argv.slice(2), opt = (k, d) => { const i = a.indexOf(k); return i < 0 ? d : a[i + 1]; };

function inPage() { // runs inside the page
  const THREE = window.THREE, { scene, camera } = window.game.three;
  scene.updateMatrixWorld(true); camera.updateMatrixWorld(true);
  const b64 = (typed) => { const u = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength); let s = ""; for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768)); return btoa(s); };
  const rgb = (c, k) => [c.r * (k === undefined ? 1 : k), c.g * (k === undefined ? 1 : k), c.b * (k === undefined ? 1 : k)];
  const conv = (x, y, z) => [x, -z, y];
  const textures = [], texIndex = new Map(), materials = [], groups = new Map(), skipped = {};
  const skip = (why) => { skipped[why] = (skipped[why] || 0) + 1; };
  function tex(t) {
    if (!t || !t.image || !t.image.width) return -1;
    if (texIndex.has(t.uuid)) return texIndex.get(t.uuid);
    let c = t.image; if (!(c instanceof HTMLCanvasElement)) { const k = document.createElement("canvas"); k.width = c.width; k.height = c.height; k.getContext("2d").drawImage(c, 0, 0); c = k; }
    textures.push({ png: c.toDataURL("image/png").split(",")[1], w: c.width, h: c.height, srgb: t.colorSpace === THREE.SRGBColorSpace, repeat: [t.repeat.x, t.repeat.y], offset: [t.offset.x, t.offset.y], wrap: t.wrapS === THREE.RepeatWrapping });
    texIndex.set(t.uuid, textures.length - 1); return textures.length - 1;
  }
  function group(m, instanceColors) { // one bucket of triangles per material (and per "has instance colours")
    const key = m.uuid + (instanceColors ? "|ic" : "");
    if (groups.has(key)) return groups.get(key);
    const basic = !!m.isMeshBasicMaterial, one = new THREE.Color(1, 1, 1), zero = new THREE.Color(0, 0, 0);
    materials.push({ name: m.name || m.type, unlit: basic, color: rgb(m.color || one), rough: m.roughness === undefined ? 0.8 : m.roughness, metal: m.metalness || 0,
      emissive: rgb(m.emissive || zero, m.emissiveIntensity === undefined ? 1 : m.emissiveIntensity), map: tex(m.map), emissiveMap: tex(m.emissiveMap), roughnessMap: tex(m.roughnessMap), bumpMap: tex(m.bumpMap), bumpScale: m.bumpScale || 0,
      opacity: m.opacity, transparent: !!m.transparent, alphaTest: m.alphaTest || 0, vertexColors: !!m.vertexColors || !!instanceColors, doubleSided: m.side === THREE.DoubleSide, flat: !!m.flatShading });
    const g = { material: materials.length - 1, pos: [], nor: [], uv: [], col: [], idx: [], n: 0 };
    groups.set(key, g); return g;
  }
  const v = new THREE.Vector3(), nrm = new THREE.Vector3(), nm = new THREE.Matrix3(), mw = new THREE.Matrix4(), im = new THREE.Matrix4(), ic = new THREE.Color();
  function add(mesh, material, start, count, matrix, instColor) {
    const geo = mesh.geometry, P = geo.attributes.position, N = geo.attributes.normal, UV = geo.attributes.uv, C = geo.attributes.color, index = geo.index;
    const g = group(material, !!instColor), wantCol = materials[g.material].vertexColors, base = g.n, flip = matrix.determinant() < 0;
    nm.getNormalMatrix(matrix);
    for (let i = 0; i < P.count; i++) {
      v.fromBufferAttribute(P, i).applyMatrix4(matrix); g.pos.push(...conv(v.x, v.y, v.z));
      if (N) nrm.fromBufferAttribute(N, i).applyMatrix3(nm).normalize(); else nrm.set(0, 1, 0);
      g.nor.push(...conv(nrm.x, nrm.y, nrm.z));
      g.uv.push(UV ? UV.getX(i) : 0, UV ? UV.getY(i) : 0);
      if (wantCol) { let r = 1, gg = 1, b = 1; if (C && material.vertexColors) { r = C.getX(i); gg = C.getY(i); b = C.getZ(i); } if (instColor) { r *= instColor.r; gg *= instColor.g; b *= instColor.b; } g.col.push(r, gg, b); }
    }
    const end = Math.min(start + count, index ? index.count : P.count);
    for (let i = start; i + 2 < end; i += 3) { const i0 = index ? index.getX(i) : i, i1 = index ? index.getX(i + 1) : i + 1, i2 = index ? index.getX(i + 2) : i + 2; g.idx.push(base + i0, base + (flip ? i2 : i1), base + (flip ? i1 : i2)); }
    g.n += P.count;
  }
  const hidden = (o) => { for (let p = o; p; p = p.parent) if (!p.visible || p.userData.overlay) return true; return false; };
  scene.traverse((o) => {
    if (!o.isMesh) { if (o.isLine || o.isPoints || o.isSprite) skip("lines and sprites"); return; }
    if (hidden(o)) return skip("hidden or game markers");
    if (!o.geometry || !o.geometry.attributes.position) return skip("no geometry");
    const mats = Array.isArray(o.material) ? o.material : [o.material], geo = o.geometry, total = geo.index ? geo.index.count : geo.attributes.position.count;
    const parts = Array.isArray(o.material) && geo.groups.length ? geo.groups.map((gr) => [mats[gr.materialIndex], gr.start, gr.count]) : [[mats[0], geo.drawRange.start || 0, Math.min(total, geo.drawRange.count)]];
    for (const [m, start, count] of parts) {
      if (!m || m.visible === false) { skip("invisible material"); continue; }
      if (m.blending === THREE.AdditiveBlending) { skip("additive light decals"); continue; }
      if (m.depthFunc === THREE.GreaterDepth) { skip("see-through shapes"); continue; }
      if (m.side === THREE.BackSide && m.isMeshBasicMaterial) { skip("sky dome"); continue; }
      if (m.transparent && m.opacity <= 0.01) { skip("fully transparent"); continue; }
      if (o.isInstancedMesh) for (let k = 0; k < o.count; k++) { o.getMatrixAt(k, im); mw.multiplyMatrices(o.matrixWorld, im); if (o.instanceColor) o.getColorAt(k, ic); add(o, m, start, count, mw, o.instanceColor ? ic : null); }
      else add(o, m, start, count, o.matrixWorld, null);
    }
  });
  const lights = [];
  scene.traverse((o) => {
    if (!o.isLight || hidden(o)) return;
    const wp = o.getWorldPosition(new THREE.Vector3()), L = { type: o.type, color: rgb(o.color), intensity: o.intensity, position: conv(wp.x, wp.y, wp.z) };
    if (o.isDirectionalLight) { const t = o.target.getWorldPosition(new THREE.Vector3()), d = wp.clone().sub(t).normalize(); L.toward = conv(d.x, d.y, d.z); }   // unit vector from the scene toward the light
    if (o.isHemisphereLight) L.ground = rgb(o.groundColor);
    if (o.isPointLight || o.isSpotLight) { L.distance = o.distance; L.decay = o.decay; }
    lights.push(L);
  });
  const sim = window.NH.sim, S = window.game.state, dusk = Math.max(0, Math.min(1, S.t / sim.config.clockSeconds.value)), at = (x, y, h) => { const p = window.NH.at(x, y, h); return conv(p.x, p.y, p.z); };
  const e = camera.matrixWorld.elements, cam = []; // column-major 4x4; convert axes row by row: new rows are (x, -z, y, w)
  for (let col = 0; col < 4; col++) cam.push(e[col * 4], -e[col * 4 + 2], e[col * 4 + 1], e[col * 4 + 3]);
  const out = { version: 1, units: "metres, z up, y north", dusk, tod: window.NH.tod(dusk), background: scene.background && scene.background.isColor ? rgb(scene.background) : null,
    camera: { matrix: cam, fovY: camera.fov, aspect: camera.aspect }, focus: at(S.player.x, S.player.y, 1.2),
    lamps: sim.LAMPS.map(([x, y]) => at(x, y, 5.7)), drop: at(sim.DROP.x, sim.DROP.y, 0), pick: at(sim.PICK.x, sim.PICK.y, 0), block: { min: at(0, sim.H, 0), max: at(sim.W, 0, 0), waterY: at(0, sim.WATER, 0)[1] },
    lights, textures, materials, meshes: [], skipped };
  for (const g of groups.values()) { if (!g.idx.length) continue;
    out.meshes.push({ material: g.material, vertices: g.n, triangles: g.idx.length / 3, pos: b64(new Float32Array(g.pos)), nor: b64(new Float32Array(g.nor)), uv: b64(new Float32Array(g.uv)), col: g.col.length ? b64(new Float32Array(g.col)) : null, idx: b64(new Uint32Array(g.idx)) }); }
  return JSON.stringify(out);
}

(async () => {
  if (a.length < 2) { console.error("usage: node dev/raytrace/export.js <page.html> <out.json> [--dusk 0.45] [--at x,y] [--delivered n] [--carry] [--size WxH]"); process.exit(2); }
  const [w, h] = opt("--size", "1080x1920").split("x").map(Number);
  const b = await chromium.launch(), cx = await b.newContext({ viewport: { width: Math.round(w / 3), height: Math.round(h / 3) }, deviceScaleFactor: 1 }), p = await cx.newPage(), errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  await p.goto("file://" + path.resolve(a[0]));
  await p.evaluate((o) => { const g = window.game, sim = window.NH.sim; g.start(); const s = g.state, set = () => { s.t = sim.config.clockSeconds.value * o.dusk; s.player.x = o.at[0]; s.player.y = o.at[1]; s.carrying = o.carry; };
    for (let i = 0; i < o.delivered; i++) { s.carrying = true; s.player.x = sim.DROP.x; s.player.y = sim.DROP.y; g.step(1); }
    set(); g.step(40); set(); g.render(0.016); g.render(0.016); },
    { dusk: +opt("--dusk", "0.45"), at: opt("--at", "1010,345").split(",").map(Number), delivered: +opt("--delivered", "3"), carry: a.includes("--carry") });
  const json = await p.evaluate(inPage);
  await b.close();
  if (errors.length) { console.error("page errors: " + errors.join(" | ")); process.exit(1); }
  fs.mkdirSync(path.dirname(path.resolve(a[1])), { recursive: true }); fs.writeFileSync(a[1], json);
  const d = JSON.parse(json);
  console.log(JSON.stringify({ out: a[1], megabytes: +(json.length / 1e6).toFixed(1), meshes: d.meshes.length, triangles: d.meshes.reduce((n, m) => n + m.triangles, 0), vertices: d.meshes.reduce((n, m) => n + m.vertices, 0),
    materials: d.materials.length, textures: d.textures.length, lights: d.lights.map((l) => l.type), dusk: d.dusk, skipped: d.skipped }));
})().catch((e) => { console.error(e); process.exit(2); });
