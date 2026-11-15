# Crate Run 3D — what was checked

Every line says how it was checked. "Headless" means a test browser on a server drawing
with software WebGL: the pictures are correct, its speed says nothing about a phone.
**No line here was checked on a real phone.**

## State at this commit (2026-11-15)

The four visual modules are being rebuilt by builder agents, and this commit is a
checkpoint of that work in progress: `src/world.js` and `src/people.js` are the builders'
current, unreviewed versions; `src/props.js` and `src/lighting.js` are still the first
crude versions. The page built from them was run through the browser checks below (25 of
25) but nobody has yet reviewed how it looks. This file is updated when the modules are
reviewed and integrated.

| What | How it was checked | Result |
|---|---|---|
| Game rules (pick up, carry, sight lines, heat, patrol ramp, win, both losses, codes) | `node test/sim.test.js`, no browser | 17 of 17 |
| Rules are the same as the flat prototype's | ported line for line; same scripted route delivers a crate in under 20 s | passes |
| Page opens from a file with no errors, title shows, WebGL draws with shadows | `node dev/check.js`, headless | passes |
| Real key events move the player; Space drops and picks up; P pauses; a code typed on the pause pad works | same | passes |
| Dragging on the street moves the player; the Drop button works | same, pointer events | passes |
| Win card, heat loss, time loss | same | passes |
| Slow frames lower the drawing resolution | same | passes |
| Narrow cover screen (344x882), unfolded (690x829), sideways (844x390): canvas fills, HUD and Drop stay on screen | same | passes |
| Streets are not hidden by buildings | lots with the house at the south edge and a yard behind; looked at `day-midblock` | visible fence to fence with the crude modules; to be re-checked after the rebuild |
| Saved-run counting (clean wins only, best run, damaged record repaired) | `node test/save.test.js` | 9 of 9 |
| Lock crypto: round trip, wrong key, tampering, fresh nonce every time, AES-256 confirmed by an independent implementation | `node test/lock.test.js` | 13 of 13 |
| Lock end to end: lock, reload, locked, unlock, refuse, passkey deleted, device without key derivation, no screen lock, opened from a file | `node dev/lock-check.js`, headless, **software authenticator** | 21 of 21 |
| Lock on a real phone, with a real fingerprint or face | not possible from here | **not assessed** |
| Frame rate on a phone | not possible from here | **not assessed** |
| How it looks | screenshots of the crude modules only | **not assessed** until the rebuild is in |
| Sound | carried over from the flat prototype, not listened to | **not assessed** |
| Played by a person | no | **not yet** |

## Guard rails

- `node build.js --check`: the page in the repository is exactly what the sources produce,
  and the vendored library matches its recorded checksum.
- `tools/red_team.py`: scan RT10 pins the library's hash a second time; ten attacks on this
  folder (swapped library, hand-edited page, changed rule, deleted tests, weakened lock,
  miscounted wins) are each caught. 62 of 62 on a clean copy of the tree.
