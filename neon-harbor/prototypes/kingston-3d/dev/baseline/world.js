/* world (BASELINE PLACEHOLDER): the static block — ground, pavements, buildings, quay, water.
   API: NH.world.build(env) -> { group, update(S, tod) }.  Replace the insides, keep the API. */
(function () {
  "use strict";
  const NH = window.NH, sim = NH.sim, U = NH.U, P = NH.PALETTE;
  NH.world = {
    build(env) {
      const group = new THREE.Group();
      const slab = (x, y, w, h, top, thick, color, opts) => { // a box given in sim pixels; `top` and `thick` in metres
        const m = new THREE.Mesh(new THREE.BoxGeometry(w * U, thick, h * U), new THREE.MeshStandardMaterial(Object.assign({ color, roughness: 0.9 }, opts || {})));
        NH.put(m, x + w / 2, y + h / 2, top - thick / 2); m.receiveShadow = true; group.add(m); return m;
      };
      slab(-4000, -4000, 9200, 4000 + sim.WATER, -0.02, 0.4, 0x4a4d45);                        // land beyond the block
      slab(0, 0, sim.W, sim.WATER, 0, 0.2, 0x6a6c6e);                                          // streets
      slab(0, sim.QUAY, sim.W, sim.WATER - sim.QUAY, 0.06, 0.3, P.quay);                       // quay
      slab(-4000, sim.WATER, 9200, 4000, -0.5, 0.2, P.harbour, { roughness: 0.25, metalness: 0.1 }); // harbour
      for (const b of sim.B) {
        const hs = NH.house(b), hgt = hs.height;
        if (hs.yard) { const yard = slab(b.x, b.y, b.w, hs.yard, 0.9, 0.9, 0x7d8a8f); yard.castShadow = true; }   // stands in for the fenced yard
        const wall = slab(hs.x, hs.y, hs.w, hs.h, hgt, hgt, b.wall); wall.castShadow = true;
        const roof = slab(hs.x - 4, hs.y - 4, hs.w + 8, hs.h + 8, hgt + 0.18, 0.18, b.c, { roughness: 0.45, metalness: 0.55 }); roof.castShadow = true;
      }
      return { group, update(S, tod) {} };
    },
  };
})();
