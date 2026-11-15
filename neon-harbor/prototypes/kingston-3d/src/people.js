/* people (BASELINE PLACEHOLDER): one character built from primitives.
   API: NH.people.make(cast) -> { group, carry, update(dt, st) }
     cast  = an entry of NH.sim.CAST ({ id, name, hair, skin, shirt })
     group = THREE.Group, feet at y = 0, 1.8 m tall, facing +X
     carry = THREE.Object3D inside the group where a carried crate is parented (in front of the chest)
     st    = { moving: bool, carrying: bool, walk: number (phase that grows as the person walks) } */
(function () {
  "use strict";
  const NH = window.NH, P = NH.PALETTE;
  NH.people = {
    make(cast) {
      const group = new THREE.Group(), mat = (c) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.8 });
      const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.26, 0.85, 4, 10), mat(cast.shirt)); body.position.y = 0.95; body.castShadow = true;
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.19, 14, 10), mat(cast.skin)); head.position.y = 1.62; head.castShadow = true;
      const hair = new THREE.Mesh(new THREE.SphereGeometry(0.21, 14, 10, 0, Math.PI * 2, 0, Math.PI * 0.55), mat(P.hair)); hair.position.y = 1.64; hair.castShadow = true;
      const carry = new THREE.Object3D(); carry.position.set(0.55, 1.05, 0);
      group.add(body, head, hair, carry);
      return { group, carry, update(dt, st) { body.position.y = 0.95 + (st.moving ? Math.abs(Math.sin(st.walk)) * 0.04 : 0); } };
    },
  };
})();
