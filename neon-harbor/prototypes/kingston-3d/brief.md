# Crate Run 3D — one-page brief

The same one-block prototype as `prototypes/crate-run`, drawn in real-time 3D in a
phone browser. Throwaway code, not the game. Ledger row 80.

- **Question this prototype answers:** does the Kingston block read, and does the
  carrying rule still play, when the block is a lit 3D place instead of a flat map?
- **What it is:** WebGL through three.js r160 (MIT, vendored and pinned by checksum).
  One self-contained page, no network, no asset files. Geometry and textures are made
  in code.
- **What it is not:** it is not the Unreal game and it tells nothing about how that
  game will look. There is no ray tracing here. Nothing in this folder is imported by
  game code.
- **Rules:** the same as Crate Run, plus the additions below. The rules live in `src/sim.js`,
  which has no drawing code and runs in Node, so they are tested without a browser. The values
  meant to be tuned (walking and patrol speeds, `chasePercent`, `searchSeconds`, `graceSeconds`,
  `changeSeconds` and the rest) are in its `config` table. The newer rules also use constants
  written beside them: the tier thresholds (34 and 67), the two 20% tier bonuses, the 30 px
  hold distance, the halved grace, and the 26 px clothesline and 22 px dub-plate reaches.
  - **Patrol AI** (ledger row 86). Each patrol is a small state machine. On a route it walks its
    loop. When it sees a carrier it hunts: it moves at `chasePercent` of its normal speed (100%)
    and holds 30 px away. If the carrier goes out of sight it walks to where it last saw them,
    looks around for `searchSeconds` (4), then walks back to its loop. Once you put the crate
    down, it stops hunting. The AI runs entirely on the device: no network, no cloud model (rule 6).
  - **Grace period** (row 90). Once spotted, heat waits `graceSeconds` (1.5) before it climbs,
    so a sighting gives a moment to break the sight line.
  - **Heat tiers** (row 90). Calm below 34. Alert from 34: patrols see 20% further, and the
    drawn vision cone grows to match. Lockdown from 67: patrols also move 20% faster. Cooling
    off drops back a tier; 100 still loses.
  - **Disguise** (row 89). Once a patrol has seen you carrying, patrols know your shirt and each
    later sighting gets half the grace. Stand still, unseen, within 26 px of one of the two
    clotheslines for `changeSeconds` (1.5) to swap shirts; then they no longer know you.
  - **Optional extras** (row 89). Three dub plates lie in the alleys, there every run, so nothing
    can be missed for good. The win card counts them and notes an unseen run if no patrol ever
    saw you carrying. That note is separate from the saved-run counter's "clean win", which
    means a win without codes: the note shows even on a run where codes were used. Neither the
    plates nor the note is saved between sessions.
  - **Two runners** (row 93). The title card's "Play as" button switches between Imani and
    Tavian 'Leaf' Wedderburn, a fictional character (the name check is in row 93). Each has two
    shirts for the clothesline change. The choice is made before the first Start: the title
    card is not shown again, so Restart, Play again and Try again keep the runner and put the
    first shirt back on. The choice is not saved.
- **Senses** (row 91). The phone vibrates on pickup, delivery, being seen, a rise in heat tier,
  a shirt change, a dub plate, a win and a loss (silent when sound is muted, nothing on
  browsers without vibration). The camera jolts for about half a second when you are seen,
  unless the device asks for reduced motion. Each patrol hums from where it is, louder as it
  nears and higher while it hunts; the riddim circles the listener once every 8 seconds. Both
  are placed with the browser's HRTF panner, so they are meant for headphones. None of this
  has been heard or felt on a real device.
- **Camera:** fixed heading, tilted 64 degrees, following the player. Every sim
  rectangle is a fenced lot with the house at its south edge and a low yard behind,
  so a building never hides the street behind it. People, patrols and the stash crate
  show as plain flat-colour shapes, and vision cones faintly, through anything that does
  cover them.
- **Look:** painted concrete, zinc roofs and fences, a working quay, and one sun that
  moves from afternoon through golden hour to night (ADR 0050). Every person wears a
  hairstyle from the hair library, modelled as itself (ADR 0051). Street life, scenery the
  rules do not see (rows 89, 92): three coconut handcarts with their sellers, three parked
  pickups, two painted mural walls and four hand-painted shop signs for made-up businesses.
  The two clotheslines are not scenery: the disguise rule uses their positions (`LINES` in
  `src/sim.js`). A soft patch of contact shading sits under people and
  standing props, and the far edge of the view is slightly out of focus (switched off once
  the page has lowered its drawing quality).
- **Frame rate first:** the page lowers its drawing resolution, never its frame-rate
  target, when a device is slow.
- **Saved runs and the lock:** wins and the best run are kept in the browser. They can
  be locked to the device: the record is sealed with AES-256-GCM and the key comes
  from a passkey, released by the device's own screen lock. Read "What the lock is
  not" below before describing it to anyone.
- **Controls and screens:** as Crate Run. Phone: drag to move, Drop button. Keyboard:
  arrows or WASD, Space, P, M. The title card has Start, Play as and Cast.

## How it is put together

```
src/sim.js        the rules (no drawing). Shared by browser and tests. Since row 86 it also exports
                  TIERS, LINES, PLATES, heatTier and sightRange; its state adds seenFor, grace, tier,
                  known, outfit, changing, plates and spotted; its events add known, changed, plate
                  and tier. CAST is read by position (0 Imani, 1 and 2 the two who stand at the rig
                  and the yard, 3 to 5 passers-by and cart sellers, 6 the foot patrol, 7 Tavian):
                  add new entries at the end.
src/contract.js   units, axes, camera, lots, light and budget rules every module follows
src/world.js      the block: streets, lots, houses, quay, water, what lies beyond
src/people.js     characters and their hairstyles, beards, and shirts printed by the cloth shader
src/props.js      patrol car, crates, speakers, palms, skiffs, lamps, vision cone, coconut carts,
                  pickups, mural walls, shop signs, clotheslines, dub plates, contact shading
src/lighting.js   sun, sky, haze, night
src/lock.js       sealing of the saved runs    src/save.js   the record and its panel
src/main.js       renderer, camera (and its jolt), input, screens, sound (riddim, patrol hums,
                  3D placement), vibration, the two runners and their outfits, where the street
                  scenery stands, the loop, and the test hooks on `window.game`. Hooks added since
                  row 86: cam, buzzes, choose(i), hero, cones, circling, hums, sound(). start() does
                  not start audio: call sound() before reading hums or circling.
src/shell.html    page shell and styles, the vignette and the far-edge blur
build.js          joins all of the above and the library into index.html (generated)
vendor/           three.min.js r160, its licence, checksums
dev/baseline/     the last integrated version of each visual module, used by --solo builds. Copy the
                  module over from src/ when its change is integrated: `node build.js --check` fails
                  while the two differ (before NH-111 it had not been refreshed since NH-85)
dev/              check.js (browser checks), sheet.js and shot.js (screenshots), lock-check.js
dev/raytrace/     Offline ray tracing of the game's own scene in Blender Cycles: pictures only, minutes
                  each on a CPU and seconds on a graphics card, not the game running.
                  export.js / export_scene.py (same export, from Node or Python) + inpage.js write the scene;
                  render.py renders it, on a graphics card if one is present, otherwise on the CPU;
                  cloud_render.sh does all of it in one command on a fresh machine; kaggle_render.ipynb
                  runs that on a free notebook service's graphics card. Run here on a CPU (five views at
                  256 and at 512 samples) and on a rented RTX PRO 6000 with OptiX (five views at 1024
                  samples, 3 to 5 seconds each; ledger row 87). On 4 CPU cores the five views took
                  1,361 seconds at 256 samples and 2,019 at 512, about 4.5 and 7 minutes a view. The
                  notebook has never been run. The export carries the painted murals and signs; shirts
                  printed by the shader come out plain. render.py has its own light rig: it sets the
                  strength of the sun, moon, sky and lamps itself from the time of day, and takes only
                  the sun's direction and daytime colour from the game. Changes to `src/lighting.js`
                  (NH-87 and NH-88, for example) therefore do not show in its pictures.
test/             Node tests: rules, saved-run counting, lock crypto. These run in CI.
```

Change something, then: `node build.js`, `node test/sim.test.js`, `node test/save.test.js`,
`node test/lock.test.js`. With Playwright installed: `node dev/check.js`,
`node dev/lock-check.js`, `node dev/sheet.js index.html <a folder outside the repo>`.
Keep screenshots out of the repository: the red-team scan fails on any file under
`prototypes/` that is not UTF-8 text (a PNG, for example), even in an untracked folder
such as `dev/out/`.

Budget, counted in a software renderer on 2026-11-28: the eleven standard views draw 27 to
118 draw calls and 15,700 to 83,800 triangles. The widest view tried (the quay at night, five
delivered, camera pulled back 1.6 times) draws 176 draw calls and about 111,000 triangles,
against the contract's 220 and 120,000. Shadows draw everything a second time. Each person
costs about 1,800 triangles and four draw calls before shadows (three for the figure, one
for the patch under the feet), so people are the next thing to lighten.

## How it was built

By the orchestrator-and-workers pattern from the post in ledger row 80: one session
planned and integrated; scouts looked facts up and a second agent re-checked each
claim; one builder per visual module worked from its own screenshots; a fresh
reviewer rebuilt and criticised each module; a fixer acted on confirmed findings.
Details and counts are in `evidence/checklist.md`.

## What the lock is not

- It is not a fingerprint lock. A web page can ask the device to verify the user; the
  device decides how (fingerprint, face, PIN, pattern or password) and does not say which.
- It is not post-quantum. AES-256 is a symmetric cipher of the kind not expected to be
  broken by quantum computers; the passkey is ordinary elliptic-curve cryptography.
- It has not been tried on a real phone. It passes with a software authenticator in a
  desktop test browser. Whether a given phone's passkeys can derive keys (the "prf"
  extension) varies by browser and passkey provider; the page checks and says so.
- It cannot work from a file or inside another site's frame. It needs the page served
  from its own https address.
- A locked record cannot be recovered if the passkey is deleted. "Forget saved runs"
  starts over.
