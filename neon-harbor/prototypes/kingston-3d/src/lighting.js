/* lighting: the sun and the moon (one shadow-casting beam between them), the sky fill, the sky, the haze and the night lights.
   This is the only module that creates THREE.Light objects: one DirectionalLight, one HemisphereLight, three PointLights and one
   SpotLight. It also owns scene.background, scene.fog and renderer.toneMappingExposure, and adds one mesh: the sky dome (one draw call).
   API: NH.lighting.build(env) -> { update(tod, focus, S), sun, fill, lamps, sky }
     env    = { scene, camera, renderer, sim, quality }
     tod    = NH.tod(dusk): { dusk, sun, warm, night } each 0..1
     focus  = THREE.Vector3 the camera looks at. Shadows and haze are fitted to what the camera sees around it.
     S      = the rules' state: S.player places the player's own light, S.delivered warms the soundhall yard. May be left out.
     update = called every frame, after the camera has been aimed. It allocates nothing, and what it shows depends only on
              its arguments and the camera, so the same moment always looks the same.
     sun, fill, lamps, sky = optional extras for whoever inspects the rig: the DirectionalLight, the HemisphereLight, the four
              night lights [yard, player, street lamp, street lamp] and the sky dome mesh. main.js does not use them.
   The day: at 4 PM the sun is high in the south-west; by golden hour it is low in the west-south-west and shadows are long.
   Its strength reaches zero when tod.sun does, and at that moment the same beam turns round and comes back as a still moon, high
   in the north. The sky fill carries the twilight in between and leans toward the afterglow while no beam is up.
   Shadows: the box is a square round the ground the camera can see (any screen shape, any zoom) and moves in whole texels.
   Night: the yard light grows with each crate delivered; the player's light is a soft spot from above on the camera's side; the
   two lamp lights sit on the two street lamps nearest the focus and fade so that a light only ever moves while it is dark.
   The four night lights stay in the scene at zero strength by day, so the number of lights never changes and no shader is
   rebuilt in the middle of a run. env.quality() is not read: every cheaper setting this module could switch to (a smaller
   shadow map, fewer lights) would rebuild a texture or shaders and show as a jump.
   Fog and sky are drawn after tone mapping: their colours below are the colours on screen, and exposure does not move them. */
(function () {
  "use strict";
  const NH = window.NH, sim = NH.sim, U = NH.U;
  const MAP = 2048, REACH = 120, PAD = 4, GROW = 2;   // shadow map size; metres from the middle of the shadow box up the beam to the light; metres added round what the camera sees; the box grows in steps of this
  const NIGHT_AT_FLIP = 0.374;                       // tod.night at the moment tod.sun reaches zero
  const RAD = Math.PI / 180;
  const smooth = (x) => { const t = x < 0 ? 0 : x > 1 ? 1 : x; return t * t * (3 - 2 * t); };
  const KEYS = [ // dusk, sunlight, its strength, sky fill, ground bounce, fill strength, horizon (and fog), zenith, exposure, metres of haze from clear to solid
    [0.00, 0xfff2dc, 3.60, 0xbdd8ff, 0x9c8a6e, 1.40, 0xcfe2ee, 0x4c8fd6, 1.00, 260],
    [0.30, 0xffe2b6, 3.80, 0xb6cef8, 0xa88a5e, 1.40, 0xdfe5de, 0x4f8ccf, 1.00, 220],
    [0.50, 0xffbe76, 4.80, 0xb6c4f0, 0xc08a52, 1.38, 0xf6d9a2, 0x6c98d2, 1.03, 140],
    [0.62, 0xff8e4a, 7.60, 0x98a8ee, 0xb4704c, 1.60, 0xf2a874, 0x5a78c0, 1.05, 125],   // the sun is strong here because tod.sun is already under a half: golden hour holds, then the sun goes quickly
    [0.72, 0xff6a3c, 7.60, 0x8ea2f0, 0x5c5684, 1.85, 0xa87c98, 0x3c4f9c, 1.08, 130],
    [0.84, 0xff6a3c, 7.60, 0x7890dc, 0x4a5880, 1.62, 0x4c5ea2, 0x1d2c6a, 1.10, 145],
    [1.00, 0xff6a3c, 7.60, 0x6e86d2, 0x525f8a, 1.30, 0x263a72, 0x0f1a44, 1.10, 155],
  ].map((k) => ({ at: k[0], sun: new THREE.Color(k[1]), power: k[2], sky: new THREE.Color(k[3]), ground: new THREE.Color(k[4]), fill: k[5],
                  horizon: new THREE.Color(k[6]), zenith: new THREE.Color(k[7]), exposure: k[8], haze: k[9] }));
  const MOON = new THREE.Color(0xb4c8ff), MOON_POWER = 0.42, MOON_EL = 66 * RAD, MOON_AZ = -82 * RAD;   // the moon stands still, high in the north, so night shadows never move and zinc and water shine back at the camera
  const GLOW = new THREE.Color(0xff9a4a);

  NH.lighting = {
    build(env) {
      const scene = env.scene, camera = env.camera, renderer = env.renderer;
      const sun = new THREE.DirectionalLight(0xffffff, 0); sun.castShadow = true; sun.shadow.mapSize.set(MAP, MAP);
      sun.shadow.bias = -0.00012; sun.shadow.normalBias = 0.045;
      const box = sun.shadow.camera; box.near = 1;
      const hemi = new THREE.HemisphereLight(0xffffff, 0xffffff, 1);
      const point = (color, reach) => new THREE.PointLight(color, 0, reach, 2);
      const yard = point(0xffb066, 26), lampA = point(0xffa04a, 17), lampB = point(0xffa04a, 17);
      const self = new THREE.SpotLight(0xffe9d2, 0, 18, 0.4, 1, 2);       // a soft follow-spot from high on the camera's side, so it lights the tops and fronts the camera sees and little else
      yard.position.set((sim.DROP.x - sim.W / 2) * U, 4.8, (sim.DROP.y + 10 - sim.H / 2) * U);
      scene.add(sun, sun.target, hemi, yard, self, self.target, lampA, lampB);
      const horizon = new THREE.Color(), zenith = new THREE.Color(); scene.background = horizon; scene.fog = new THREE.Fog(0x000000, 100, 300);

      // street lamps: every lamp but the last, which stands in the soundhall yard and is covered by the yard light
      const LN = sim.LAMPS.length - 1, LX = new Float64Array(LN), LZ = new Float64Array(LN), LAMP_Y = 5.3;
      for (let i = 0; i < LN; i++) { LX[i] = (sim.LAMPS[i][0] - sim.W / 2) * U; LZ[i] = (sim.LAMPS[i][1] - sim.H / 2) * U; }
      let atA = 0, atB = 1;

      // sky: rings of a dome that travels with the camera, so its horizon ring is always at eye level and always the fog colour
      const ELEV = [-5, 0, 3, 7, 13, 22, 36, 58, 90], SEG = 24, NV = ELEV.length * SEG, DOME = 500;
      const skyPos = new Float32Array(NV * 3), skyCol = new Float32Array(NV * 3), skyUp = new Float32Array(NV), skyBand = new Float32Array(NV), skyIdx = [];
      for (let r = 0; r < ELEV.length; r++) for (let s = 0; s < SEG; s++) {
        const e = ELEV[r] * RAD, a = s / SEG * Math.PI * 2, v = r * SEG + s;
        skyPos[v * 3] = Math.cos(a) * Math.cos(e) * DOME; skyPos[v * 3 + 1] = Math.sin(e) * DOME; skyPos[v * 3 + 2] = Math.sin(a) * Math.cos(e) * DOME;
        skyUp[v] = Math.pow(Math.max(0, Math.sin(e)), 0.6);                                   // how far from the horizon colour toward the zenith colour
        skyBand[v] = ELEV[r] <= 0 ? 0 : Math.exp(-Math.pow((ELEV[r] - 6) / 11, 2));           // where the low glow toward the sun sits: none on the horizon ring itself
        if (r < ELEV.length - 1) { const s1 = (s + 1) % SEG, b = v + SEG, c = r * SEG + s1; skyIdx.push(v, b, c, c, b, c + SEG); }
      }
      const skyGeo = new THREE.BufferGeometry(), skyAttr = new THREE.BufferAttribute(skyCol, 3); skyAttr.setUsage(THREE.DynamicDrawUsage);
      skyGeo.setAttribute("position", new THREE.BufferAttribute(skyPos, 3)); skyGeo.setAttribute("color", skyAttr); skyGeo.setIndex(skyIdx);
      const sky = new THREE.Mesh(skyGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, fog: false, toneMapped: false, depthWrite: false }));
      sky.renderOrder = -10; sky.frustumCulled = false; scene.add(sky);
      let skyDusk = -1;

      let boxR = 0;
      const dir = new THREE.Vector3();

      return {
        sun, fill: hemi, lamps: [yard, self, lampA, lampB], sky,
        update(tod, focus, S) {
          const dusk = tod.dusk, night = tod.night, day = tod.sun > 0;
          let i = 0; while (i < KEYS.length - 2 && dusk >= KEYS[i + 1].at) i++;
          const a = KEYS[i], b = KEYS[i + 1], t = Math.min(1, Math.max(0, (dusk - a.at) / (b.at - a.at)));

          /* the one beam: sun, then moon. It turns round only while its strength is zero. */
          const west = Math.PI - (56 - 50 * dusk) * RAD;     // where the sun stands, or last stood: it goes down in the west-south-west
          let el, az;
          if (day) { el = Math.max(7, 57 - 52 * dusk - 18 * dusk * dusk) * RAD; az = west; sun.color.lerpColors(a.sun, b.sun, t); sun.intensity = (a.power + (b.power - a.power) * t) * tod.sun; }
          else { el = MOON_EL; az = MOON_AZ; sun.color.copy(MOON); sun.intensity = MOON_POWER * Math.min(1, Math.max(0, (night - NIGHT_AT_FLIP) / 0.6)); }
          dir.set(Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el));        // from the ground toward the light; az runs from east (+X) round through south (+Z)

          /* sky fill, sky, haze, exposure */
          hemi.color.lerpColors(a.sky, b.sky, t); hemi.groundColor.lerpColors(a.ground, b.ground, t); hemi.intensity = a.fill + (b.fill - a.fill) * t;
          const lean = 0.85 * smooth((dusk - 0.56) / 0.16) * (1 - smooth((dusk - 0.76) / 0.2));        // at twilight the fill leans toward the afterglow in the west, so roofs and people keep their shape while no beam is up
          hemi.position.set(Math.cos(west) * Math.sin(lean), Math.cos(lean), Math.sin(west) * Math.sin(lean));
          horizon.lerpColors(a.horizon, b.horizon, t); zenith.lerpColors(a.zenith, b.zenith, t); scene.fog.color.copy(horizon);
          renderer.toneMappingExposure = a.exposure + (b.exposure - a.exposure) * t;
          const cx = camera.position.x - focus.x, cy = camera.position.y - focus.y, cz = camera.position.z - focus.z, flat = Math.hypot(cx, cz), away = Math.hypot(flat, cy);
          scene.fog.near = away; scene.fog.far = away + a.haze + (b.haze - a.haze) * t;      // clear up to the point the camera looks at, wherever the camera stands
          sky.position.copy(camera.position);
          if (Math.abs(dusk - skyDusk) > 0.0015) { // the sky changes slowly: repaint it about twice a second of play
            skyDusk = dusk;
            const g = tod.warm * tod.warm * 0.55, sx = Math.cos(west), sz = Math.sin(west);
            for (let v = 0, s = 0; v < NV; v++, s = v % SEG) {
              const u = skyUp[v], ang = s / SEG * Math.PI * 2, toward = 0.5 + 0.5 * (Math.cos(ang) * sx + Math.sin(ang) * sz), w = g * skyBand[v] * toward * toward;
              skyCol[v * 3] = horizon.r + (zenith.r - horizon.r) * u + GLOW.r * w; skyCol[v * 3 + 1] = horizon.g + (zenith.g - horizon.g) * u + GLOW.g * w; skyCol[v * 3 + 2] = horizon.b + (zenith.b - horizon.b) * u + GLOW.b * w;
            }
            skyAttr.needsUpdate = true;
          }

          /* shadow box: a square round the circle that holds the piece of ground on screen, so its size does not change as the light turns */
          const pitch = Math.atan2(cy, flat), sp = Math.sin(pitch), cp = Math.cos(pitch), tv = Math.tan(camera.fov * Math.PI / 360), th = tv * camera.aspect, h = Math.max(1, camera.position.y);
          const tTop = Math.min(h / Math.max(0.05, sp - tv * cp), 2.2 * away), tBot = h / (sp + tv * cp);                     // lengths of the rays through the top and bottom edges of the screen
          const fTop = tTop * (cp + tv * sp), fBot = tBot * (cp - tv * sp), wTop = tTop * th, wBot = tBot * th;                 // how far ahead of the camera they land, and half the width of the view there
          const mid = Math.min(fTop, Math.max(fBot, (fTop * fTop + wTop * wTop - fBot * fBot - wBot * wBot) / (2 * (fTop - fBot))));
          const R = Math.ceil((Math.sqrt(Math.max(wTop * wTop + (fTop - mid) * (fTop - mid), wBot * wBot + (fBot - mid) * (fBot - mid))) + PAD) / GROW) * GROW;
          if (R !== boxR) { boxR = R; box.left = -R; box.right = R; box.top = R; box.bottom = -R; box.far = REACH + R + 30; box.updateProjectionMatrix(); }
          const fx = flat > 0.001 ? -cx / flat : 0, fz = flat > 0.001 ? -cz / flat : -1;
          let px = camera.position.x + fx * mid, pz = camera.position.z + fz * mid, py = 0;
          // move the box in whole texels along its own axes (the same axes three.js derives: x = up x beam, y = beam x x), so shadow edges stay put as the camera travels
          const xl = Math.hypot(dir.z, dir.x), xx = dir.z / xl, xz = -dir.x / xl, yx = dir.y * xz, yy = dir.z * xx - dir.x * xz, yz = -dir.y * xx, texel = 2 * R / MAP;
          const ox = px * xx + pz * xz, oy = px * yx + pz * yz, dx = Math.round(ox / texel) * texel - ox, dy = Math.round(oy / texel) * texel - oy;
          px += xx * dx + yx * dy; py += yy * dy; pz += xz * dx + yz * dy;
          sun.target.position.set(px, py, pz); sun.position.set(px + dir.x * REACH, py + dir.y * REACH, pz + dir.z * REACH);

          /* night lights */
          const lit = smooth(night), crates = S ? S.delivered : 0;
          yard.intensity = lit * (26 + 16 * crates);
          const ux = S ? (S.player.x - sim.W / 2) * U : focus.x, uz = S ? (S.player.y - sim.H / 2) * U : focus.z;
          self.position.set(ux, 9, uz + 2.6); self.target.position.set(ux, 1.2, uz); self.intensity = 60 * lit;
          let n1 = 0, n2 = 0, d1 = 1e9, d2 = 1e9, d3 = 1e9;
          for (let k = 0; k < LN; k++) {
            const d = Math.hypot(LX[k] - focus.x, LZ[k] - focus.z);
            if (d < d1) { d3 = d2; d2 = d1; n2 = n1; d1 = d; n1 = k; } else if (d < d2) { d3 = d2; d2 = d; n2 = k; } else if (d < d3) d3 = d;
          }
          // each light stays on its lamp while that lamp is one of the nearest two; its strength is zero when its lamp ties with the third nearest, so a move never shows
          if (atA !== n1 && atA !== n2) atA = atB === n1 ? n2 : n1;
          if (atB !== n1 && atB !== n2 || atB === atA) atB = atA === n1 ? n2 : n1;
          lampA.position.set(LX[atA], LAMP_Y, LZ[atA]); lampA.intensity = 38 * lit * smooth((d3 - (atA === n1 ? d1 : d2)) / 8);
          lampB.position.set(LX[atB], LAMP_Y, LZ[atB]); lampB.intensity = 38 * lit * smooth((d3 - (atB === n1 ? d1 : d2)) / 8);
        },
      };
    },
  };
})();
