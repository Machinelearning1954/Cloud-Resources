/* main: boot, renderer, camera, input, screens, sound, the loop and the test hooks.
   The rules live in sim.js. The look lives in world.js, people.js, props.js and lighting.js.
   This file only wires them together. */
(function () {
  "use strict";
  const NH = window.NH, sim = NH.sim, tune = sim.tune, STEP = sim.STEP, U = NH.U, P = NH.PALETTE;
  const $ = (id) => document.getElementById(id);
  const fail = (msg) => { $("failmsg").textContent = msg; $("fail").hidden = false; $("card").hidden = true; $("hud").hidden = true; $("drop").hidden = true; };
  if (!window.THREE) return fail("The 3D library did not load.");
  let booted = false;   // anything that goes wrong before the first frame is shown on the page, not left as a blank screen
  addEventListener("error", (e) => { if (!booted) fail("The 3D view could not start: " + (e.message || "unknown error")); });

  /* ---------------- renderer and scene ---------------- */
  const cv = $("c");
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, powerPreference: "high-performance" }); }
  catch (e) { return fail("This browser or viewer could not start WebGL, which the 3D view needs."); }
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  cv.addEventListener("webglcontextlost", (e) => { e.preventDefault(); fail("The 3D view was lost, usually because the device ran short of graphics memory. Reload to try again."); });

  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(35, 1, 1, 700);
  let quality = 1, vw = 1, vh = 1;
  const env = { scene, camera, renderer, sim, quality: () => quality };
  let world, light;
  try { world = NH.world.build(env); scene.add(world.group); light = NH.lighting.build(env); }
  catch (e) { return fail("The 3D scene could not be built: " + e.message); }

  /* ---------------- state ---------------- */
  let S = sim.create(), scn = "title", keys = {}, stick = null, muted = false, lastToast = "", toastUntil = 0, typed = "";

  /* Seen through walls: a tilted camera hides whatever stands right behind a building. Each actor gets a plain
     shape that is drawn only where something is in front of it, so nobody can hide from the player's eye.
     Draw order does the work: the block first (order 0), then these shapes (1), then the actors themselves (2).
     A shape is tested before its own actor is drawn, so an actor's crate or hair can never trigger its own shape. */
  const ghostMat = (color, opacity) => new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, depthFunc: THREE.GreaterDepth, depthWrite: false });
  const actor = (obj) => { obj.traverse((o) => { if (o.isMesh && !o.userData.ghost) o.renderOrder = 2; }); return obj; };
  const ghostGeo = { person: new THREE.CapsuleGeometry(0.42, 1.1, 3, 8).translate(0, 0.97, 0), car: new THREE.BoxGeometry(4.5, 1.7, 2.0).translate(0, 0.85, 0), crate: new THREE.BoxGeometry(1.1, 1.1, 1.1).translate(0, 0.55, 0) };
  const ghostMats = { you: ghostMat(0x2fb58a, 1), patrol: ghostMat(0xd8475a, 1), other: ghostMat(0x8f939c, 1), crate: ghostMat(0xd9872e, 1) };
  function ghost(parent, shape, kind) { const m = new THREE.Mesh(ghostGeo[shape], ghostMats[kind]); m.userData.ghost = m.userData.overlay = true; m.renderOrder = 1; actor(parent); parent.add(m); return m; }

  /* ---------------- things in the scene that follow the rules' state ---------------- */
  const person = (castIndex, kind) => { const p = NH.people.make(sim.CAST[castIndex]); p.group.scale.setScalar(NH.HERO); ghost(p.group, "person", kind || "other"); scene.add(p.group); return p; };
  // two outfits for you, built once: yellow and green shirts. Changing at a clothesline shows the other one.
  const outfits = [player0(sim.CAST[0]), player0(Object.assign({}, sim.CAST[0], { id: "imani-green", shirt: "#2f8f5a" }))];
  function player0(cast) { const p = NH.people.make(cast); p.group.scale.setScalar(NH.HERO); ghost(p.group, "person", "you"); scene.add(p.group); return p; }
  let player = outfits[0]; outfits[1].group.visible = false;
  const carried = NH.props.crate(); carried.scale.setScalar(1 / NH.HERO); carried.position.y -= 0.35; player.carry.add(actor(carried));
  function wear(i) { if (player === outfits[i]) return; player.group.visible = false; player = outfits[i]; player.group.visible = true; player.carry.add(carried); }
  sim.LINES.forEach(([x, y]) => { const l = NH.props.clothesline(); NH.put(l, x, y, 0); scene.add(l); });
  const peds = S.peds.map((q) => person(q.cast));
  const npcs = sim.NPCS.map((n) => { const p = person(n.cast); NH.put(p.group, n.x, n.y, 0); NH.face(p.group, n.ang); return p; });
  const patrolViews = [];                       // grows as the rules add patrols
  function patrolView(g) {
    const v = g.foot ? Object.assign(person(6, "patrol"), { foot: true }) : Object.assign(NH.props.car(), { foot: false });
    if (!g.foot) { ghost(v.group, "car", "patrol"); scene.add(v.group); }
    v.cone = NH.props.visionCone(tune.seeRange * U, tune.seeHalfAngle * Math.PI / 180); v.cone.mesh.userData.overlay = true; scene.add(v.cone.mesh);
    v.coneGhost = v.cone.mesh.clone(true); v.coneGhostMat = ghostMat(0xfff0be, 0.16);          // the same sector, shown faintly where a building covers it
    v.coneGhost.traverse((o) => { if (o.isMesh) { o.material = v.coneGhostMat; o.renderOrder = 8; } }); scene.add(v.coneGhost);   // see-through, so it is drawn late with the other see-through things
    return v;
  }
  const stash = NH.props.crate(); stash.visible = false; ghost(stash, "crate", "crate"); scene.add(stash);
  for (let i = 0; i < 4; i++) { const c = NH.props.crate(); NH.put(c, sim.PICK.x + (i % 2) * 26 - 13, sim.PICK.y + Math.floor(i / 2) * 26 - 13, 0); c.rotation.y = i * 0.3; scene.add(c); }
  const speakers = [];
  for (let i = 0; i < sim.config.cratesToWin.max; i++) { // a stack three wide against the soundhall's yard wall
    const s = NH.props.speaker(); NH.put(s, 945 + (i % 3) * 34, 222, Math.floor(i / 3) * 2.0); s.rotation.y = -Math.PI / 2; s.visible = false; scene.add(s); speakers.push(s);
  }
  const ring = NH.props.dropRing((tune.pickRadius + 14) * U); NH.put(ring, sim.DROP.x, sim.DROP.y, 0); scene.add(ring);
  const arrow = NH.props.arrow(); scene.add(arrow);
  ring.userData.overlay = arrow.userData.overlay = true;   // game markers, not part of the place: the scene exporter leaves "overlay" objects out
  sim.PALMS.forEach(([x, y], i) => { const t = NH.props.palm(i); NH.put(t, x, y, 0); t.rotation.y = i * 1.7; scene.add(t); });
  // vendors' coconut carts on the pavement, each with its seller standing by; scenery only, the rules do not see them
  [[372, 560, 0.0, 3], [828, 1210, Math.PI, 4], [372, 1120, 0.0, 5]].forEach(([x, y, a, c], i) => {
    const k = NH.props.cart(i); NH.put(k, x, y, 0); k.rotation.y = a + Math.PI / 2; scene.add(k);
    const v = person(c); NH.put(v.group, x + (a ? 24 : -24), y + 14, 0); NH.face(v.group, a); npcs.push(v);   // the seller stands on the street side, facing the cart
  });
  // two painted walls at the top of the quay, facing the harbour (and the camera); scenery only
  [[440, 1384, 0], [825, 1384, 1]].forEach(([x, y, i]) => { const w = NH.props.mural(i); NH.put(w, x, y, 0); w.rotation.y = -Math.PI / 2; scene.add(w); });
  // pickups parked on the quay; scenery only, the rules do not see them
  [[650, 1408, 0.02], [1000, 1404, -0.03], [1150, 1415, Math.PI + 0.04]].forEach(([x, y, a], i) => { const v = NH.props.pickup(i); NH.put(v, x, y, 0); v.rotation.y = a; scene.add(v); });
  sim.SKIFFS.forEach(([x, color], i) => { const b = NH.props.skiff(new THREE.Color(color).getHex()); NH.put(b, x, sim.WATER + 44, -0.35); b.rotation.y = 0.15 * (i - 1); scene.add(b); });
  const lamps = sim.LAMPS.map(([x, y], i) => { // the post stands on a corner; its arm reaches back so the lamp hangs over the lamp point
    const l = NH.props.lampPost(), yard = i === sim.LAMPS.length - 1, arm = NH.LAMP_ARM / U, a = yard ? Math.PI : (i % 2 ? 0.75 : 0.25) * Math.PI;
    NH.put(l.group, x - Math.cos(a) * arm, y - Math.sin(a) * arm, 0); NH.face(l.group, a); scene.add(l.group); return l;
  });
  const strings = NH.props.stringLights(NH.at(912, 200, 3.6), NH.at(1146, 200, 3.6), 14); scene.add(strings.group);

  /* ---------------- camera ---------------- */
  const calm = !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);   // players who ask for less motion get no shake
  const cam = { shake: 0, zoom: 1, pitch: NH.PITCH, half: 13, pan: new THREE.Vector3(), focus: new THREE.Vector3(), target: new THREE.Vector3() };

  function aim(snap) {
    cam.focus.lerp(cam.target, snap ? 1 : 0.14);
    const figures = scn === "figures", half = figures ? lineup.userData.half : cam.half, pitch = figures && cam.pitch === NH.PITCH ? LINEUP_PITCH : cam.pitch;
    const t = Math.tan(camera.fov * Math.PI / 360), d = half / Math.min(t, t * camera.aspect) * cam.zoom;
    camera.position.set(cam.focus.x, cam.focus.y + d * Math.sin(pitch), cam.focus.z + d * Math.cos(pitch));
    camera.lookAt(cam.focus);
    if (cam.shake > 0.01) { // a short jolt when a patrol spots you: the view slides a little, never turns, and settles in about half a second
      const w = S.t * 40; camera.position.x += Math.sin(w * 1.3) * cam.shake; camera.position.y += Math.sin(w * 1.7 + 1) * cam.shake * 0.6; cam.shake *= 0.9;
    } else cam.shake = 0;
  }
  function fit() {
    vw = Math.max(1, cv.clientWidth); vh = Math.max(1, cv.clientHeight);
    renderer.setPixelRatio(Math.max(0.6, Math.min(window.devicePixelRatio || 1, 2) * quality));
    renderer.setSize(vw, vh, false); camera.aspect = vw / vh; camera.updateProjectionMatrix(); if (layoutLineup.ready) layoutLineup();
  }
  // Dynamic resolution: if this device really draws slowly, draw fewer pixels. Frame rate first (ADR 0050).
  let slowFrames = 0, countedFrames = 0;
  function tally(ms) {
    if (ms <= 0 || ms > 250) return;
    countedFrames++; if (ms > 24) slowFrames++;
    if (countedFrames < 90) return;
    if (slowFrames > 45 && quality > 0.5) { quality -= 0.25; fit(); }
    countedFrames = 0; slowFrames = 0;
  }

  /* ---------------- the cast screen: every figure side by side, close up ---------------- */
  const lineup = new THREE.Group(), tags = [];
  (function buildLineup() {
    const items = sim.CAST.map((c) => ({ name: c.name, note: c.note, make: () => { const p = NH.people.make(c); p.group.scale.setScalar(NH.HERO); return p; } }))
      .concat([{ name: "Speaker crate", make: () => ({ group: NH.props.crate() }) }, { name: "Speaker", make: () => ({ group: NH.props.speaker() }) },
               { name: "Patrol car", make: () => NH.props.car() }]);
    const floor = new THREE.Mesh(new THREE.CircleGeometry(40, 48), new THREE.MeshStandardMaterial({ color: 0x4d5560, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; lineup.add(floor);
    items.forEach((it, i) => {
      const v = it.make();
      v.group.rotation.y = -Math.PI / 2 + 0.35;   // face the camera, turned a little
      lineup.add(v.group);
      const el = document.createElement("div"); el.className = "tag3d"; el.hidden = true; el.textContent = it.name;
      if (it.note) { const s = document.createElement("small"); s.textContent = it.note; el.appendChild(s); }
      $("tags").appendChild(el); tags.push({ el, v, carrying: i === 0 });
    });
    lineup.position.set(0, 0, -260); lineup.visible = false; scene.add(lineup);
  })();
  layoutLineup.ready = true;
  const tmp = new THREE.Vector3(), camAt = new THREE.Object3D(), LINEUP_PITCH = 0.75;
  function layoutLineup() { // as many columns as the screen shape allows, close enough to read a hairstyle
    const cols = camera.aspect < 0.8 ? 3 : camera.aspect < 1.5 ? 4 : 6, rows = Math.ceil(tags.length / cols), dx = 3.6, dz = 5.4;
    tags.forEach((t, i) => { const row = Math.floor(i / cols), inRow = Math.min(cols, tags.length - row * cols);   // a short last row is centred
      t.v.group.position.set((i % cols - (inRow - 1) / 2) * dx, 0.01, (row - (rows - 1) / 2) * dz); });
    const wide = cols * dx / 2 + 0.6, deep = (rows * dz / 2 + 2.4) * Math.sin(LINEUP_PITCH), a = camera.aspect;
    lineup.userData.half = a < 1 ? Math.max(wide, deep * a) : Math.max(deep, wide / a);
  }
  function syncLineup(dt) {
    cam.target.set(0, 1.2, -260).add(cam.pan); lineup.userData.t = (lineup.userData.t || 0) + dt;
    for (const t of tags) {
      if (t.v.update) t.v.update(dt, { moving: true, carrying: false, walk: lineup.userData.t * 7, sees: false, night: 0, t: lineup.userData.t });
      tmp.copy(t.v.group.position).add(lineup.position).project(camera);
      t.el.style.left = ((tmp.x + 1) / 2 * vw) + "px"; t.el.style.top = ((1 - tmp.y) / 2 * vh + 6) + "px";
    }
  }

  /* ---------------- copy the rules' state onto the scene ---------------- */
  function sync(dt, snap) {
    const figures = scn === "figures"; lineup.visible = figures; for (const t of tags) t.el.hidden = !figures;
    const tod = NH.tod(figures ? 0.1 : sim.clamp(S.t / tune.clockSeconds, 0, 1)), p = S.player, playing = scn === "play";
    NH.put(player.group, p.x, p.y, 0); NH.face(player.group, p.ang);
    player.update(dt, { moving: playing && p.moving, carrying: S.carrying, walk: p.walk });
    carried.visible = S.carrying;
    S.peds.forEach((q, i) => { const v = peds[i]; NH.put(v.group, q.x, q.y, 0); NH.face(v.group, q.ang); v.update(dt, { moving: playing && !!q.moving, carrying: false, walk: q.walk * 1.6 }); });
    for (const n of npcs) n.update(dt, { moving: false, carrying: false, walk: 0 });
    while (patrolViews.length < S.patrols.length) patrolViews.push(patrolView(S.patrols[patrolViews.length]));
    patrolViews.forEach((v, i) => {
      const g = S.patrols[i], on = !!g; v.group.visible = on; v.cone.mesh.visible = on; v.coneGhost.visible = on; if (!on) return;
      NH.put(v.group, g.x, g.y, 0); NH.face(v.group, g.ang); NH.put(v.cone.mesh, g.x, g.y, 0); NH.face(v.cone.mesh, g.ang); v.cone.set(g.sees);
      v.coneGhost.position.copy(v.cone.mesh.position); v.coneGhost.rotation.y = v.cone.mesh.rotation.y; v.coneGhostMat.color.setHex(g.sees ? P.heat : 0xfff0be); v.coneGhostMat.opacity = g.sees ? 0.3 : 0.16;
      if (v.foot) v.update(dt, { moving: playing, carrying: false, walk: S.t * 6 }); else v.update(dt, { sees: g.sees, night: tod.night, t: S.t });
    });
    stash.visible = !!S.stash; if (S.stash) NH.put(stash, S.stash.x, S.stash.y, 0);
    speakers.forEach((s, i) => { s.visible = i < S.delivered; });
    for (const l of lamps) l.setGlow(tod.night);
    strings.setGlow(Math.min(1, tod.night * (0.55 + 0.11 * S.delivered) + 0.05 * S.delivered));
    ring.rotation.y = S.t * 0.4;
    const tgt = S.carrying ? sim.DROP : (S.stash || sim.PICK), far = sim.dist(p, tgt) > 130, a = Math.atan2(tgt.y - p.y, tgt.x - p.x);
    arrow.visible = far && !figures; if (far) { NH.put(arrow, p.x + Math.cos(a) * 46, p.y + Math.sin(a) * 46, 0); NH.face(arrow, a); arrow.setColor(S.carrying ? P.go : P.sodium); }
    world.update(S, tod);
    if (figures) syncLineup(dt); else { NH.put(camAt, p.x, p.y, 0.8); cam.target.copy(camAt.position).add(cam.pan); }
    aim(snap); light.update(tod, cam.focus, S);
  }

  /* ---------------- one step of the rules, and what the player is told ---------------- */
  function input() {
    let x = (keys.ArrowRight || keys.d ? 1 : 0) - (keys.ArrowLeft || keys.a ? 1 : 0);
    let y = (keys.ArrowDown || keys.s ? 1 : 0) - (keys.ArrowUp || keys.w ? 1 : 0);
    if (stick) { x += stick.x; y += stick.y; }
    const m = Math.hypot(x, y); return m > 1 ? { x: x / m, y: y / m } : { x, y };
  }
  function say(msg) { lastToast = msg; $("toast").textContent = msg; toastUntil = S.t + 4.5; }
  const buzzes = [];                                   // the last few vibration patterns, for the checks
  function buzz(pattern) { // phone vibration on the moments that matter; silent with sound off, and a no-op where the browser has none
    if (muted) return; buzzes.push(pattern); if (buzzes.length > 8) buzzes.shift();
    try { if (navigator.vibrate) navigator.vibrate(pattern); } catch (e) { /* some browsers refuse outside a tap: ignore */ }
  }
  function tell() { // turn the rules' events into messages, sound layers and screens
    for (const e of S.events.splice(0)) {
      if (e.type === "picked") buzz([30]);
      if (e.type === "picked" && e.first) say("Follow the arrow to Zinc Roof Soundhall.");
      else if (e.type === "dropped") say("Crate down. Patrols ignore you now. Come back for it.");
      else if (e.type === "delivered") { buzz([60, 40, 60]); beat.layers = e.count; say(e.count === 1 ? "One in. Each crate brings another patrol. Alleys hide you." : `${e.count} of ${tune.cratesToWin} delivered.`); }
      else if (e.type === "seen") { buzz([140, 70, 140]); if (!calm) cam.shake = 0.35; say("Seen carrying! Break the sight line or drop the crate."); }
      else if (e.type === "known") say("They've seen your shirt. Patrols will spot you faster. Change at a clothesline.");
      else if (e.type === "changed") { wear(e.outfit); buzz([40, 30, 40]); say(e.wasKnown ? "New shirt. The patrols don't know you now." : "Shirt changed."); }
      else if (e.type === "tier") { say(e.up ? (e.tier === 2 ? "Lockdown! Patrols are faster. Get out of sight." : "The street is on alert: patrols see further.") : (e.tier === 0 ? "Things are calm again." : "Lockdown eased. Still on alert.")); if (e.up) buzz([90]); }
      else if (e.type === "end") { buzz(e.result === "won" ? [80, 50, 80, 50, 220] : [450]); scn = e.result === "won" ? "won" : "lost"; showCard(); if (NH.save) NH.save.finish(S, tune).then(() => { if (scn === "won" || scn === "lost") showCard(true); }); }
    }
  }
  function tick() { if (scn !== "play") return; sim.step(S, input()); tell(); }
  function toggleCrate() { if (scn !== "play") return; sim.toggleCrate(S); tell(); }
  function enterCode(d) {
    if (scn !== "pause") return;
    typed = (typed + d).slice(-6);
    const hit = sim.enterCode(S, typed), el = $("code"), msg = $("codemsg"); S.events.length = 0;
    if (hit) { typed = ""; if (msg) msg.textContent = `${hit.name}: ${hit.does}. This run no longer counts.`; }
    if (el) el.textContent = typed.split("").join(" ");
  }

  /* ---------------- sound: a generated riddim that gains a layer per crate ---------------- */
  const beat = { ac: null, next: 0, i: 0, layers: 0 };
  function tone(type, f, t, len, vol, slide) {
    const o = beat.ac.createOscillator(), g = beat.ac.createGain(); o.type = type; o.frequency.setValueAtTime(f, t);
    if (slide) o.frequency.exponentialRampToValueAtTime(slide, t + len);
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.001, t + len); o.connect(g).connect(bus()); o.start(t); o.stop(t + len + 0.02);
  }
  function audio() {
    if (muted || scn !== "play" || !beat.ac) return;
    const spb = 60 / 84 / 2; if (beat.next < beat.ac.currentTime) beat.next = beat.ac.currentTime + 0.05;
    while (beat.next < beat.ac.currentTime + 0.2) {
      const s = beat.i % 16, t = beat.next, L = beat.layers;
      if (s % 8 === 4) tone("sine", 120, t, 0.22, 0.5, 42);
      if (L >= 1 && [0, 3, 6, 8, 11, 14].includes(s)) tone("triangle", [55, 55, 65.4, 49, 49, 61.7][[0, 3, 6, 8, 11, 14].indexOf(s)], t, 0.26, 0.34);
      if (L >= 2 && s % 4 === 2) for (const f of [220, 261.6, 329.6]) tone("square", f, t, 0.07, 0.035);
      if (L >= 3 && s % 2 === 1) tone("square", 6200, t, 0.02, 0.012);
      if (L >= 4 && [0, 6, 10].includes(s)) tone("sawtooth", [440, 523.3, 392][[0, 6, 10].indexOf(s)], t, 0.3, 0.03);
      beat.next += spb; beat.i++;
    }
  }
  // 3D headphone sound: an HRTF panner places a source round the listener's head (front, behind, above), not just left and right
  function space(ac) {
    if (!ac.createPanner) return null;
    const pn = ac.createPanner(); pn.panningModel = "HRTF"; pn.distanceModel = "inverse"; pn.refDistance = 1; pn.rolloffFactor = 0; return pn;   // loudness is set by our own gains
  }
  function place(pn, x, y, z, t) {
    if (!pn) return;
    if (pn.positionX) { pn.positionX.setTargetAtTime(x, t, 0.05); pn.positionY.setTargetAtTime(y, t, 0.05); pn.positionZ.setTargetAtTime(z, t, 0.05); } else pn.setPosition(x, y, z);
  }
  // "8D" riddim: the music bus circles slowly round the listener (one turn every 8 s) with a short room echo, best on headphones
  function bus() {
    if (!beat.bus) {
      const ac = beat.ac, inp = ac.createGain(), pn = space(ac), d = ac.createDelay(1), fb = ac.createGain(), wet = ac.createGain();
      d.delayTime.value = 0.12; fb.gain.value = 0.28; wet.gain.value = 0.22;
      const out = pn || ac.destination; inp.connect(out); inp.connect(d); d.connect(fb).connect(d); d.connect(wet).connect(out); if (pn) pn.connect(ac.destination);
      beat.bus = inp; beat.panner = pn;
    }
    return beat.bus;
  }
  // each patrol hums from where it is: louder as it nears, panned left or right, higher while it hunts you
  const hums = [];
  function hum(i) {
    if (!hums[i]) {
      const ac = beat.ac, o = ac.createOscillator(), f = ac.createBiquadFilter(), g = ac.createGain(), pn = space(ac);
      o.type = "sawtooth"; o.frequency.value = 70; f.type = "lowpass"; f.frequency.value = 320; g.gain.value = 0;
      o.connect(f).connect(g); if (pn) g.connect(pn).connect(ac.destination); else g.connect(ac.destination); o.start();
      hums[i] = { o, g, pn, level: 0 };
    }
    return hums[i];
  }
  function patrolSound() {
    if (!beat.ac) return;
    const on = !muted && scn === "play", t = beat.ac.currentTime, p = S.player;
    S.patrols.forEach((q, i) => {
      const h = hum(i), d = Math.hypot(q.x - p.x, q.y - p.y), near = Math.max(0, 1 - d / 450), hunt = q.mode === "hunt";
      h.level = on ? (q.foot ? 0.03 : 0.07) * near * near * (hunt ? 1.6 : 1) : 0;
      h.g.gain.setTargetAtTime(h.level, t, 0.1);
      h.o.frequency.setTargetAtTime((q.foot ? 140 : 70) * (hunt ? 1.3 : 1), t, 0.2);
      place(h.pn, (q.x - p.x) / 100, 0, (q.y - p.y) / 100, t);   // the camera looks north, so a patrol to the south is behind you
    });
    if (beat.panner) { const a = t * Math.PI * 2 / 8; place(beat.panner, Math.sin(a) * 2, Math.sin(a * 0.5) * 0.6, -Math.cos(a) * 2, t); }   // the riddim's slow circle, rising and dipping a little
    for (let i = S.patrols.length; i < hums.length; i++) if (hums[i]) { hums[i].level = 0; hums[i].g.gain.setTargetAtTime(0, t, 0.1); }
  }
  function sound() { if (!beat.ac) { try { beat.ac = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { beat.ac = null; } } if (beat.ac && beat.ac.state === "suspended") beat.ac.resume(); }

  /* ---------------- screens and HUD ---------------- */
  function hud() {
    patrolSound();
    const playing = scn === "play", figures = scn === "figures";
    $("toast").classList.toggle("low", !playing); $("toast").hidden = !lastToast || !(playing || figures);
    $("hud").hidden = scn === "title" || figures; $("drop").hidden = !playing;
    const mins = Math.floor(sim.clamp(S.t / tune.clockSeconds, 0, 1) * 180);
    $("clock").textContent = `${4 + Math.floor(mins / 60)}:${String(mins % 60).padStart(2, "0")}`;
    $("crates").textContent = `${S.delivered}/${tune.cratesToWin}`;
    $("heat").style.width = S.heat + "%"; $("heatwrap").setAttribute("aria-valuenow", String(Math.round(S.heat)));
    $("drop").disabled = !(S.carrying || sim.nearStash(S)); $("drop").textContent = S.carrying ? "Drop" : "Pick up";
    if (lastToast && playing && S.t > toastUntil) say("");
    $("stick").hidden = !stick;
    if (stick) { $("stick").style.left = stick.ox + "px"; $("stick").style.top = stick.oy + "px"; $("knob").style.transform = `translate(${stick.x * 44}px, ${stick.y * 44}px)`; }
  }
  function restart() { S = sim.create(); beat.layers = 0; typed = ""; wear(0); }
  function showCard(keepFocus) {
    const box = $("box"), saved = NH.save ? NH.save.panel() : ""; $("card").hidden = scn === "play" || scn === "figures"; if (scn === "play") return;
    const btn = (id, label, alt) => `<button class="big${alt ? " alt" : ""}" id="${id}" type="button">${label}</button>`;
    if (scn === "title") box.innerHTML = `<h1>Crate Run <span>3D</span></h1>
      <p class="sub">A one-block prototype for Neon Harbor: Kingston, drawn in real-time 3D. Not the game.</p>
      <p>Doors open at 7:00 PM. Carry <b>5 speaker crates</b> from Keelwright Boatworks to Zinc Roof Soundhall.</p>
      <ul><li>Patrols only care while you are <b>carrying</b>.</li><li>Put the crate down to walk past clean, then come back for it.</li><li>Buildings block their view.</li></ul>
      <p class="sub">Phone: drag anywhere to move, tap Drop. Keyboard: arrows or WASD, Space to drop or pick up, P pause, M sound.</p>
      <div class="row">${btn("go", "Start")}${btn("cast", "Cast", true)}</div>${saved}`;
    else if (scn === "figures") { box.innerHTML = ""; say("Tap or press any key to go back"); return; }
    else if (scn === "pause") box.innerHTML = `<h1>Paused</h1><div class="row">${btn("go", "Resume")}${btn("again", "Restart", true)}</div>
      <p class="sub">Codes: tap six directions (or use the arrow keys). Try Up Up Down Left Right Down.</p>
      <div id="code"></div><p class="sub" id="codemsg" role="status"></p>
      <div class="pad"><span></span><button class="key" data-d="U" type="button">Up</button><span></span>
        <button class="key" data-d="L" type="button">Left</button><button class="key" data-d="D" type="button">Down</button><button class="key" data-d="R" type="button">Right</button></div>`;
    else if (scn === "won") box.innerHTML = `<h1>Doors <span>open</span></h1><p>All ${tune.cratesToWin} crates are stacked with ${Math.max(0, Math.round((tune.clockSeconds - S.t) / tune.clockSeconds * 180))} minutes to spare, heat at ${Math.round(S.heat)}.</p>${S.cheated ? '<p class="sub">Codes were used, so this run does not count.</p>' : ""}<div class="row">${btn("again", "Play again")}</div>${saved}`;
    else box.innerHTML = `<h1>${S.result === "lost-heat" ? "Too much <span>heat</span>" : "Doors <span>stayed shut</span>"}</h1>
      <p>${S.result === "lost-heat" ? "A patrol watched you carry for too long." : "Seven o'clock came with the stack unfinished."} You delivered ${S.delivered} of ${tune.cratesToWin}.</p>${S.cheated ? '<p class="sub">Codes were used on this run.</p>' : ""}<div class="row">${btn("again", "Try again")}</div>${saved}`;
    const go = $("go"), again = $("again"), cast = $("cast"); typed = "";
    for (const k of box.querySelectorAll(".key")) k.onclick = () => enterCode(k.dataset.d);
    if (go) go.onclick = () => { if (scn === "title") restart(); scn = "play"; sound(); showCard(); say(S.t ? "" : "Walk into the crate pile to pick one up."); };
    if (again) again.onclick = () => { restart(); scn = "play"; sound(); showCard(); say("Walk into the crate pile to pick one up."); };
    if (cast) cast.onclick = () => { scn = "figures"; showCard(); };
    if (NH.save) NH.save.bind(box, () => showCard(true));
    if (!keepFocus) (go || again).focus();
  }
  function pause() { if (scn === "play") { scn = "pause"; showCard(); } else if (scn === "pause") { scn = "play"; showCard(); } }
  function setMute() { muted = !muted; $("mute").textContent = muted ? "Muted" : "Sound"; $("mute").setAttribute("aria-pressed", String(muted)); }
  function leaveFigures() { if (scn === "figures") { scn = "title"; say(""); showCard(); return true; } return false; }
  function press(k) {
    if (leaveFigures()) return;
    if (scn === "pause" && k.indexOf("Arrow") === 0) return enterCode(k[5]);
    if (k === " " || k === "Space") toggleCrate(); else if (k === "p" || k === "Escape") pause(); else if (k === "m") setMute();
    else if (k === "Enter" && scn !== "play") { const b = $("go") || $("again"); if (b) b.click(); }
  }

  /* ---------------- input ---------------- */
  addEventListener("keydown", (e) => { const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(k)) e.preventDefault();
    if (scn === "play" && k === " ") { if (!e.repeat) toggleCrate(); keys[k] = true; return; }   // even with a button focused
    if (!e.repeat && e.target.tagName !== "BUTTON" || k !== "Enter" && k !== " " && !e.repeat) { if (!keys[k]) press(k); } keys[k] = true; });
  addEventListener("keyup", (e) => { keys[e.key.length === 1 ? e.key.toLowerCase() : e.key] = false; });
  addEventListener("blur", () => { keys = {}; stick = null; });
  cv.addEventListener("pointerdown", (e) => { if (leaveFigures()) return; if (scn !== "play") return; cv.setPointerCapture(e.pointerId); stick = { id: e.pointerId, ox: e.clientX, oy: e.clientY, x: 0, y: 0 }; });
  cv.addEventListener("pointermove", (e) => { if (!stick || e.pointerId !== stick.id) return; const dx = e.clientX - stick.ox, dy = e.clientY - stick.oy, m = Math.hypot(dx, dy), k = Math.min(1, m / 44); stick.x = m ? dx / m * k : 0; stick.y = m ? dy / m * k : 0; });
  for (const ev of ["pointerup", "pointercancel"]) cv.addEventListener(ev, (e) => { if (stick && e.pointerId === stick.id) stick = null; });
  $("drop").onclick = () => { toggleCrate(); $("drop").blur(); }; $("pause").onclick = pause; $("mute").onclick = () => { setMute(); $("mute").blur(); };
  addEventListener("resize", fit);

  /* ---------------- loop: the rules run on a fixed step, never on the wall clock ---------------- */
  let acc = 0, last = 0, manual = false;
  function draw(dt, snap) { sync(dt, snap); renderer.render(scene, camera); }
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000 || 0);
    if (!manual) { acc += dt; while (acc >= STEP) { tick(); acc -= STEP; } if (scn === "play" && last) tally(now - last); audio(); hud(); draw(dt, false); }
    last = now; requestAnimationFrame(frame);
  }
  // Check hooks: a script can drive the game and read its state. After the first step() the page
  // stops drawing by itself (software-rendered test machines are slow); call render() before a screenshot.
  window.game = {
    cam,
    buzzes,
    get circling() { return !!(beat.panner); }, get hums() { return hums.map((h) => (h ? +h.level.toFixed(4) : 0)); }, sound() { sound(); },
    get scene() { return scn; }, get state() { return S; }, config: sim.config, get three() { return { scene, camera }; },
    get quality() { return quality; }, frameTime(ms, n) { for (let i = 0; i < (n || 1); i++) tally(ms); },
    press(k) { press(k); }, hold(k, on) { keys[k] = on !== false; },
    step(n) { manual = true; for (let i = 0; i < (n || 1); i++) tick(); hud(); },
    start() { restart(); scn = "play"; showCard(); },
    code(seq) { if (scn === "play") pause(); for (const d of seq) enterCode(d); },
    view(o) { o = o || {}; for (const k of ["zoom", "pitch", "half"]) if (o[k]) cam[k] = o[k]; if (o.pan) cam.pan.fromArray(o.pan); },   // close inspection: zoom, pitch (radians), half (metres), pan [x, y, z] metres
    render(dt) { manual = true; hud(); draw(dt || 0, true); },
    info() { const r = renderer.info; return { calls: r.render.calls, triangles: r.render.triangles, geometries: r.memory.geometries, textures: r.memory.textures, pixelRatio: renderer.getPixelRatio(), shadowMap: renderer.shadowMap.enabled }; },
  };
  fit(); showCard(); draw(0, true); booted = true; requestAnimationFrame(frame);
  if (NH.save) NH.save.init(NH.lock).then(() => { if (scn === "title") showCard(true); });   // the saved-runs panel fills in once the device has answered
})();
