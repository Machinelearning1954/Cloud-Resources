# Crate Run 3D — what was checked

Every line says how it was checked. "Headless" means a test browser on a server drawing
with software WebGL: the pictures are correct, its speed says nothing about a phone.
**No line here was checked on a real phone.**

## State on 2026-11-28 (commits NH-87 to NH-110, ledger rows 87 to 93)

Twenty-three commits (NH-87 to NH-109) were added by one session, with no builder, reviewer
or fixer agents. NH-110 brought the documents up to date and corrected one rule.
Each change on screen was judged by that same session from its own before-and-after
screenshots. **None of this code has had an independent review.**

Closed since the review below: night roofs no longer read violet (NH-87; the bluest zinc
sheets still look cool), canerows read from the game camera (NH-101), and contact shading
sits under people and standing props (NH-108).

| What was added | How it was checked | Result |
|---|---|---|
| Grace period, heat tiers, chase at normal speed, disguise, dub plates, clean run | `node test/sim.test.js`: five new rule tests; three older ones updated (two heat tests for the grace period, the hunt test for the slower chase) | 26 of 26 |
| Alert tier changes sight only; lockdown is 20% faster | rule test added in NH-110, after the documents were checked against the code and the alert tier was found to be 10% faster than described | passes |
| Vibration | the patterns for a pickup and a delivery were recorded by a scripted run, headless; the others were not exercised | as written for those two; **not felt on a phone** |
| Camera jolt when seen | scripted sighting: starts at full strength, zero a second later | passes; **not seen on a phone** |
| Patrol hum | level read by script: 0.0009 far, 0.053 near, 0 when muted | as intended; **not heard** |
| Circling riddim and HRTF placement | the music bus exists after a real Enter key press, no page errors | starts; **not heard** |
| Far-edge blur gives way on a slow device | scripted slow frames: the blur is hidden once quality drops | passes |
| Second runner, printed shirts, beard | close screenshots of both shirts and of the face | reads as intended at close range |
| Carts, pickups, murals, signs, clotheslines, dub plates | a close screenshot of each | all read; the first cart and sign shots showed faults that were fixed and re-shot |
| Offline export after these changes | `node dev/raytrace/export.js`: 30 meshes, 17 textures, murals and signs carry their painted textures | runs; shader-printed shirts export plain |
| Lock end to end, again | `node dev/lock-check.js` on 2026-11-28, headless, software authenticator | 21 of 21 |
| The eleven standard views | `node dev/sheet.js` on 2026-11-28 | no page errors; 27 to 118 draw calls, 15,700 to 83,800 triangles |
| Widest view tried (quay at night, five delivered, camera pulled back 1.6 times) | `dev/shot.js` | 176 of 220 draw calls, about 111,000 of 120,000 triangles |
| Whether the new rules are fair or fun | a scratch bot was too weak to judge; nobody has played | **not assessed** |

## State on 2026-11-15 (kept as the record of the module reviews)

All four visual modules were written by builder agents, then each was reviewed by a fresh
agent from screenshots (against the art bible and the project's own ray-traced stills) and
a fixer acted on the confirmed findings. What the reviews found and what was done:

| Module | Found | Done |
|---|---|---|
| props | The calm vision cone was nearly invisible on sunlit concrete (must-fix); the yard lamp glow washed out the delivery ring and the yard at night; headlights were buried in the car body | All four fixed. The orchestrating session re-shot the daylight cone and a cone half behind a building: both read |
| lighting | The soundhall yard clipped to pale yellow at night because the yard light grew without limit; the sky dome was drawn although never on screen | Yard light capped and the follow light fades near it; dome hidden when out of view. The yard centre is still bright |
| world | Roofs turned one saturated blue at night; yards and fences were hidden under the eaves; roof tyres read as black holes | Roof made matte, yards widened on deep lots, roof clutter lightened. Roofs are still mostly blue at night: that is the moonlight, a lighting matter (closed in NH-87, see above) |
| people | Shoes floated below trouser legs (must-fix); 360 waves and canerows did not read at phone size; the updo read as a dot; tan shoes read as bare feet | Shoe gap fixed, updo enlarged, shoes recoloured, waves and canerows reworked. The waves fix came out pale grey, which read as grey hair; the orchestrating session darkened it and re-shot it. Canerows read close up and only faintly at game distance (closed in NH-101, see above) |

Hair, judged by the reviewer and then by the orchestrating session from close-ups: locs updo,
freeform locs, head wrap, afro and patrol cap read clearly; 360 waves reads as dark cropped
hair with a ripple; canerows is the weakest. No stylist has reviewed any of them (ADR 0051).

The independent review of the main loop, the lock and the tooling has still **not** run.

On 2026-11-15 the standard views drew 28 to 85 draw calls and 18,000 to 81,000 triangles
(contract budget: 220 and 120,000), counted in a software renderer. The table above has
the figures for 2026-11-28.

| What | How it was checked | Result |
|---|---|---|
| Game rules (pick up, carry, sight lines, heat, patrol ramp, win, both losses, codes, patrol AI and the additions above) | `node test/sim.test.js`, no browser | 26 of 26 on 2026-11-28 |
| Rules began as the flat prototype's | ported line for line on 2026-11-15; the same scripted route still delivers a crate in under 20 s. Rules added since (patrol AI, grace, tiers, disguise, plates) are in this prototype only | passes |
| Page opens from a file with no errors, title shows, WebGL draws with shadows | `node dev/check.js`, headless | passes |
| Real key events move the player; Space drops and picks up; P pauses; a code typed on the pause pad works | same | passes |
| Dragging on the street moves the player; the Drop button works | same, pointer events | passes |
| Win card, heat loss, time loss | same | passes |
| Slow frames lower the drawing resolution | same | passes |
| Narrow cover screen (344x882), unfolded (690x829), sideways (844x390): canvas fills, HUD and Drop stay on screen | same | passes |
| Streets are not hidden by buildings | lots with the house at the south edge and a yard behind; looked at `day-midblock`, `gold-street`, `night-street` | visible fence to fence |
| Saved-run counting (clean wins only, best run, damaged record repaired) | `node test/save.test.js` | 9 of 9 |
| Lock crypto: round trip, wrong key, tampering, fresh nonce every time, AES-256 confirmed by an independent implementation | `node test/lock.test.js` | 13 of 13 |
| Lock end to end: lock, reload, locked, unlock, refuse, passkey deleted, device without key derivation, no screen lock, opened from a file | `node dev/lock-check.js`, headless, **software authenticator** | 21 of 21 |
| Lock on a real phone, with a real fingerprint or face | not possible from here | **not assessed** |
| Frame rate on a phone | not possible from here | **not assessed** |
| How it looks | one independent review and fix pass per visual module, from screenshots; then re-shot by the orchestrating session | **assessed from screenshots only**; open items are in the table above |
| Sound | the riddim came from the flat prototype; patrol hums, the circling bus and HRTF placement were added in NH-95 to NH-97. None of it has been listened to | **not assessed** |
| Played by a person | no | **not yet** |

## Guard rails

- `node build.js --check`: the page in the repository is exactly what the sources produce,
  and the vendored library matches its recorded checksum.
- `tools/red_team.py`: scan RT10 pins the library's hash a second time; ten attacks on this
  folder (swapped library, hand-edited page, changed rule, deleted tests, weakened lock,
  miscounted wins) are each caught. 62 of 62 on a clean copy of the tree.
