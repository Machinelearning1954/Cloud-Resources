/* lighting (BASELINE PLACEHOLDER): sun, sky colour, fog, exposure.
   API: NH.lighting.build(env) -> { update(tod, focus, S) }
     env   = { scene, camera, renderer, sim }
     tod   = NH.tod(dusk): { dusk, sun, warm, night } each 0..1
     focus = THREE.Vector3 the camera looks at (keep shadows centred on it) */
(function () {
  "use strict";
  const NH = window.NH;
  NH.lighting = {
    build(env) {
      const { scene, renderer } = env;
      const sun = new THREE.DirectionalLight(0xfff1dc, 3); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
      const c = sun.shadow.camera; c.left = -45; c.right = 45; c.top = 45; c.bottom = -45; c.near = 1; c.far = 220; sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.04;
      const hemi = new THREE.HemisphereLight(0xcfe3ff, 0x6b6257, 1.6);
      scene.add(sun, sun.target, hemi); scene.background = new THREE.Color(); scene.fog = new THREE.Fog(0x000000, 120, 420);
      const day = new THREE.Color(0x9fc6e8), dusk = new THREE.Color(0xf0a060), night = new THREE.Color(0x0e1638), sky = new THREE.Color();
      const dir = new THREE.Vector3();
      return {
        update(tod, focus, S) {
          const elev = (62 - 54 * Math.min(1, tod.dusk / 0.72)) * Math.PI / 180 * tod.sun + (1 - tod.sun) * 1.0, az = 0.55 + tod.dusk * 0.6;
          dir.set(-Math.cos(az) * Math.cos(elev), Math.sin(elev), -Math.sin(az) * Math.cos(elev)).normalize();     // from the ground toward the light
          sun.position.copy(focus).addScaledVector(dir, 110); sun.target.position.copy(focus);
          sun.color.setRGB(1, 0.95 - 0.25 * tod.warm, 0.86 - 0.45 * tod.warm).lerp(new THREE.Color(0x9db4ff), 1 - tod.sun);
          sun.intensity = 0.35 + 2.9 * tod.sun; hemi.intensity = 0.7 + 0.9 * (1 - tod.night);
          sky.copy(day).lerp(dusk, tod.warm).lerp(night, tod.night); scene.background.copy(sky); scene.fog.color.copy(sky);
          renderer.toneMappingExposure = 1.0 - 0.15 * tod.night;
        },
      };
    },
  };
})();
