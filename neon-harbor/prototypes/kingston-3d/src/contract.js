/* The contract every 3D module follows. Read this before touching any module.

   LIBRARY   three.js r160, global `THREE` (vendor/three.min.js). No add-ons, no modules, no network.
   UNITS     metres. One sim pixel = NH.U metres. Y is up.
   AXES      sim x -> world +X (east). sim y -> world +Z (south, toward the harbour and the camera).
   PLACING   NH.put(obj, simX, simY, heightMetres) and NH.at(simX, simY, h) -> THREE.Vector3.
   FACING    every model is built facing +X. NH.face(obj, simAngle) turns it (rotation.y = -angle).
   SIZE      people are modelled 1.8 m tall at the origin (feet on y = 0); the main loop scales them by
             NH.HERO so they read on a phone. Vehicles and props are modelled at real size.
   LOOK      stylised, clean shapes carried by real light and real shadows. Not photorealism.
             Colours come from NH.PALETTE and from the cast/building data in NH.sim. Skin tones are
             never tinted by a material trick: use the cast entry's `skin` as-is.
   CAMERA    fixed heading (north is up the screen), tilted NH.PITCH from the ground, about 45-90 m away,
             showing roughly 26 m across the short side of the screen. Things are seen from above and
             from the south: roofs, south walls, the tops of heads. Build for that view first.
   LOTS      each rectangle in NH.sim.B is a LOT, not a building: its edge is the line nobody can walk
             or see through (a zinc fence or a wall, 1.7 m at most). NH.house(lot) gives the footprint of
             the building inside it: flush with the lot's SOUTH edge (the front the camera sees) and set
             back from the NORTH edge by a yard, so a building never hides the street behind it. Nothing
             taller than 1.7 m may stand in the yard strip. Roofs: ridge east-west, pitch 20 degrees at most.
             The game is about seeing patrols; anything that hides a street from this camera is a bug.
   SHADOWS   meshes that should cast set castShadow = true; ground and roofs set receiveShadow = true.
   LIGHTS    only lighting.js creates THREE.Light objects. Everything else glows with emissive materials
             and flat additive "pool of light" shapes on the ground.
   TEXTURES  no image files. A texture may be drawn once at start-up on a small canvas (256 px or less)
             and used as a THREE.CanvasTexture. Set texture.colorSpace = THREE.SRGBColorSpace for colour maps.
   MERGING   BufferGeometryUtils is NOT in this build (it is an add-on). Merge by hand or use InstancedMesh.
   BUDGET    a phone must hold 60 fps: share geometries and materials, merge static meshes, use
             InstancedMesh for repeats, no per-frame allocation (no `new` inside update), no extra
             render passes, at most ONE shadow-casting light (the sun), one hemisphere fill, and at
             most 4 other real lights in the whole scene. Whole scene in play: under 220 draw calls
             and 120,000 triangles.
   TIME      NH.tod(dusk) -> { dusk, sun, warm, night }, each 0..1. dusk 0 = 4 PM, 1 = 7 PM.
             sun = how strong direct sunlight is; warm = orange dusk tint; night = how dark it is.
   NAMES     no real brands, people or places beyond the fictional ones already in NH.sim. */
(function () {
  "use strict";
  const NH = window.NH, sim = NH.sim;
  NH.THREE_VERSION = "160";
  NH.U = 0.05;        // metres per sim pixel: the block is 60 m x 80 m
  NH.HERO = 1.5;      // people are drawn 1.5x real size
  NH.PITCH = 64 * Math.PI / 180;   // camera tilt from the ground
  NH.FENCE_H = 1.7;   // metres; the most a fence, wall or yard object on a lot line may stand
  NH.LAMP_ARM = 1.6;  // metres; a lamp post's arm reaches this far along its own +X, and the lamp head hangs at its end
  NH.wallHeight = (b) => b === sim.BOATWORKS ? 2.8 : 3.0 + 1.9 * (b.tall - 1);   // metres: 3.0 to 4.2. The sim's `tall` runs 1 to about 1.64
  NH.house = (b) => { // the building inside a lot, in sim pixels, plus its wall height in metres and its yard depth in pixels
    const height = NH.wallHeight(b), yard = b === sim.BOATWORKS ? 0 : Math.min(Math.round(b.h * 0.4), Math.ceil((height * 0.5 + 0.4) / NH.U));
    return { x: b.x, y: b.y + yard, w: b.w, h: b.h - yard, yard, height };
  };
  NH.at = (x, y, h) => new THREE.Vector3((x - sim.W / 2) * NH.U, h || 0, (y - sim.H / 2) * NH.U);
  NH.put = (obj, x, y, h) => { obj.position.set((x - sim.W / 2) * NH.U, h || 0, (y - sim.H / 2) * NH.U); return obj; };
  NH.face = (obj, ang) => { obj.rotation.y = -ang; return obj; };
  NH.PALETTE = { // from docs/art/ART_BIBLE.md section 4
    sodium: 0xff9d3c, zinc: 0x9fb0b8, harbour: 0x1f6f86, go: 0x3ed6a4, heat: 0xff4d5e, ink: 0xf3ead8, night: 0x0b1420,
    asphalt: 0x3b4048, pavement: 0x575b62, quay: 0x6b6257, laneMark: 0xd9d9d0, trunk: 0x6e5236, frondDark: 0x1f5a34, frond: 0x3f8a48,
    crate: 0xb9853f, crateEdge: 0x6e4a1c, speaker: 0x1c1c22, patrolBody: 0xe9edf2, patrolBand: 0x2c3f66, hair: 0x1a1210,
  };
  const clamp = sim.clamp;
  NH.tod = (dusk) => ({ // the same curves the flat version uses
    dusk,
    sun: 1 - clamp((dusk - 0.5) / 0.22, 0, 1),
    warm: clamp(dusk * 1.6, 0, 1) * (1 - clamp((dusk - 0.62) * 2.6, 0, 1)),
    night: clamp((dusk - 0.55) * 2.2, 0, 1),
  });
})();
