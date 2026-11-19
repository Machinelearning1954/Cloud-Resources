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
- **Rules:** the same as Crate Run, plus one addition: **patrol AI** (ledger row 86). Each patrol
  is a small state machine. On a route it walks its loop. When it sees a carrier it hunts: it
  chases at 130% of its normal speed and holds 30 px away. If the carrier goes out of sight it
  walks to where it last saw them, looks around for four seconds, then walks back to its loop.
  Once you put the crate down, it stops hunting. The AI runs entirely on the device: no network,
  no cloud model (rule 6). The rules live in `src/sim.js`, which has no drawing code and runs in
  Node, so they are tested without a browser. Tuning: `chasePercent` and `searchSeconds`. Once spotted, heat waits `graceSeconds` (1.5 s) before it climbs, so a sighting gives a moment to break the sight line instead of being an almost certain loss.
- **Camera:** fixed heading, tilted 64 degrees, following the player. Every sim
  rectangle is a fenced lot with the house at its south edge and a low yard behind,
  so a building never hides the street behind it. People, patrols, the stash crate
  and vision cones are drawn faintly through anything that does cover them.
- **Look:** painted concrete, zinc roofs and fences, a working quay, and one sun that
  moves from afternoon through golden hour to night (ADR 0050). Every person wears a
  hairstyle from the hair library, modelled as itself (ADR 0051).
- **Frame rate first:** the page lowers its drawing resolution, never its frame-rate
  target, when a device is slow.
- **Saved runs and the lock:** wins and the best run are kept in the browser. They can
  be locked to the device: the record is sealed with AES-256-GCM and the key comes
  from a passkey, released by the device's own screen lock. Read "What the lock is
  not" below before describing it to anyone.
- **Controls and screens:** as Crate Run. Phone: drag to move, Drop button. Keyboard:
  arrows or WASD, Space, P, M.

## How it is put together

```
src/sim.js        the rules (no drawing). Shared by browser and tests.
src/contract.js   units, axes, camera, lots, light and budget rules every module follows
src/world.js      the block: streets, lots, houses, quay, water, what lies beyond
src/people.js     characters and their hairstyles
src/props.js      patrol car, crates, speakers, palms, skiffs, lamps, vision cone
src/lighting.js   sun, sky, haze, night
src/lock.js       sealing of the saved runs    src/save.js   the record and its panel
src/main.js       renderer, camera, input, screens, sound, the loop, test hooks
src/shell.html    page shell and styles
build.js          joins all of the above and the library into index.html (generated)
vendor/           three.min.js r160, its licence, checksums
dev/baseline/     the last integrated version of each visual module, used by --solo builds
dev/              check.js (browser checks), sheet.js and shot.js (screenshots), lock-check.js
dev/raytrace/     Offline ray tracing of the game's own scene in Blender Cycles: pictures only, tens of
                  seconds each on a CPU, not the game running.
                  export.js / export_scene.py (same export, from Node or Python) + inpage.js write the scene;
                  render.py renders it, on a graphics card if one is present, otherwise on the CPU;
                  cloud_render.sh does all of it in one command on a fresh machine; kaggle_render.ipynb
                  runs that on a free notebook service's graphics card. Run and checked here on a CPU
                  only. Never run on a graphics card or on that service.
test/             Node tests: rules, saved-run counting, lock crypto. These run in CI.
```

Change something, then: `node build.js`, `node test/sim.test.js`, `node test/save.test.js`,
`node test/lock.test.js`. With Playwright installed: `node dev/check.js`,
`node dev/lock-check.js`, `node dev/sheet.js index.html <a folder outside the repo>`.
Keep screenshots out of the repository: the red-team scan refuses unreadable files
inside `prototypes/`.

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
