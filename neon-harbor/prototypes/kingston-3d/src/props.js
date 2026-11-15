/* props (BASELINE PLACEHOLDER): vehicles and objects. Everything is built facing +X, at real size, on y = 0.
   API (each returns what is listed; replace the insides, keep the API):
     NH.props.car()                      -> { group, update(dt, st) }   st = { sees: bool, night: 0..1, t: seconds }
     NH.props.crate()                    -> THREE.Object3D   (about 1 m cube, origin at its base centre)
     NH.props.speaker()                  -> THREE.Object3D   (1.5 m wide, 2 m tall, 1.2 m deep, origin at base centre)
     NH.props.palm(i)                    -> THREE.Object3D   (i = index, for variety; about 8 m tall)
     NH.props.skiff(colorHex)            -> THREE.Object3D   (small open boat, about 5.4 m long, origin at waterline)
     NH.props.lampPost()                 -> { group, setGlow(k) }   k = 0..1; post at the origin, an arm reaching NH.LAMP_ARM along +X,
                                            the lamp head about 6 m up at the arm's end, and a soft pool of light on the ground under the head
     NH.props.stringLights(a, b, count)  -> { group, setGlow(k) }   a, b = THREE.Vector3 ends of the string
     NH.props.visionCone(range, half)    -> { mesh, set(sees) }     flat sector on the ground, +X forward; range in metres, half in radians
     NH.props.dropRing(radius)           -> THREE.Object3D   (dashed ring on the ground, radius in metres)
     NH.props.arrow()                    -> THREE.Object3D   (flat arrow on the ground pointing +X, about 1.4 m long) */
(function () {
  "use strict";
  const NH = window.NH, P = NH.PALETTE;
  const std = (color, o) => new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.8 }, o || {}));
  const box = (w, h, d, mat, y) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.y = (y || 0) + h / 2; m.castShadow = true; return m; };
  NH.props = {
    car() {
      const group = new THREE.Group(), barMat = std(0x6fb7ff, { emissive: 0x6fb7ff, emissiveIntensity: 1 });
      group.add(box(4.2, 0.8, 1.8, std(P.patrolBody), 0.35), box(2.0, 0.6, 1.6, std(0x18222e), 1.15));
      const bar = box(0.3, 0.14, 1.2, barMat, 1.75); group.add(bar);
      return { group, update(dt, st) { const red = st.sees && Math.floor(st.t * 4) % 2; barMat.color.setHex(red ? P.heat : 0x6fb7ff); barMat.emissive.setHex(red ? P.heat : 0x6fb7ff); } };
    },
    crate() { return box(1, 1, 1, std(P.crate)); },
    speaker() { return box(1.5, 2, 1.2, std(P.speaker)); },
    palm(i) {
      const g = new THREE.Group(), trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.24, 7, 8), std(P.trunk)); trunk.position.y = 3.5; trunk.castShadow = true;
      const crown = new THREE.Mesh(new THREE.ConeGeometry(2.6, 1.6, 9), std(P.frond)); crown.position.y = 7.4; crown.castShadow = true; g.add(trunk, crown); return g;
    },
    skiff(color) { const m = new THREE.Mesh(new THREE.SphereGeometry(1, 14, 8), std(color)); m.scale.set(2.7, 0.45, 0.85); m.castShadow = true; return m; },
    lampPost() {
      const group = new THREE.Group(), headMat = std(0xffe2b0, { emissive: P.sodium, emissiveIntensity: 0 });
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 6, 8), std(0x2a2f36)); pole.position.y = 3; pole.castShadow = true;
      const arm = new THREE.Mesh(new THREE.BoxGeometry(NH.LAMP_ARM, 0.07, 0.07), std(0x2a2f36)); arm.position.set(NH.LAMP_ARM / 2, 6, 0);
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 10, 8), headMat); head.position.set(NH.LAMP_ARM, 5.85, 0);
      const poolMat = new THREE.MeshBasicMaterial({ color: P.sodium, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending });
      const pool = new THREE.Mesh(new THREE.CircleGeometry(6.5, 28), poolMat); pool.rotation.x = -Math.PI / 2; pool.position.set(NH.LAMP_ARM, 0.05, 0); pool.renderOrder = 1;
      group.add(pole, arm, head, pool);
      return { group, setGlow(k) { headMat.emissiveIntensity = 3 * k; poolMat.opacity = 0.16 * k; } };
    },
    stringLights(a, b, count) {
      const group = new THREE.Group(), mats = [P.sodium, P.go, P.heat, 0xf2c14e].map((c) => std(c, { emissive: c, emissiveIntensity: 0.4 }));
      for (let i = 0; i < count; i++) { const t = i / (count - 1), m = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 6), mats[i % 4]);
        m.position.lerpVectors(a, b, t); m.position.y -= Math.sin(t * Math.PI) * 0.5; group.add(m); }
      return { group, setGlow(k) { for (const m of mats) m.emissiveIntensity = 0.4 + 2.6 * k; } };
    },
    visionCone(range, half) {
      const mat = new THREE.MeshBasicMaterial({ color: 0xfff0be, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(new THREE.CircleGeometry(range, 24, -half, half * 2), mat); mesh.rotation.x = -Math.PI / 2; mesh.position.y = 0.08; mesh.renderOrder = 2;
      const holder = new THREE.Group(); holder.add(mesh);
      return { mesh: holder, set(sees) { mat.color.setHex(sees ? P.heat : 0xfff0be); mat.opacity = sees ? 0.34 : 0.2; } };
    },
    dropRing(radius) {
      const m = new THREE.Mesh(new THREE.RingGeometry(radius - 0.12, radius, 48), new THREE.MeshBasicMaterial({ color: P.go, transparent: true, opacity: 0.9, depthWrite: false, side: THREE.DoubleSide }));
      m.rotation.x = -Math.PI / 2; m.position.y = 0.07; const g = new THREE.Group(); g.add(m); return g;
    },
    arrow() {
      const s = new THREE.Shape(); s.moveTo(0.8, 0); s.lineTo(-0.5, 0.55); s.lineTo(-0.2, 0); s.lineTo(-0.5, -0.55); s.closePath();
      const m = new THREE.Mesh(new THREE.ShapeGeometry(s), new THREE.MeshBasicMaterial({ color: P.go, side: THREE.DoubleSide, depthWrite: false, transparent: true }));
      m.rotation.x = -Math.PI / 2; m.position.y = 0.09; const g = new THREE.Group(); g.add(m);
      return Object.assign(g, { setColor(hex) { m.material.color.setHex(hex); } });
    },
  };
})();
