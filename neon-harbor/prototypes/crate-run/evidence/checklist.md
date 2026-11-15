# Crate Run — brief checklist (2026-11-14)

Checked by a headless browser opening `index.html` straight from the file, at phone
(390×844, touch), foldable (720×840, touch) and desktop (1280×720, keyboard) sizes,
plus a look at the screenshots. The published page itself was not opened by the checks.
**Not yet played by a person.**

| Brief line | Status | How it was checked |
| --- | --- | --- |
| Opens with no errors, by double-click, no network | built | loaded from `file://`, no console or page errors |
| Title screen with the controls | built | screenshot |
| Phone: drag anywhere to move | built | real touch drag moved the player 93 px |
| Keyboard: arrows or WASD | built | real key press moved the player 91 px |
| Pick up a crate by walking into the pile | built | scripted and real input |
| Deliver to the soundhall; goal of 5 | built | scripted run delivered a crate in 16 s of game time; win screen at 5 |
| Patrols ignore you unless you carry | built | seen while carrying: yes; seen empty-handed in the same spot: no |
| Put the crate down, come back for it | built | stays down while you stand on it; picked up on return |
| Each delivery adds a patrol and speeds them up | built | 1 patrol at start, 2 after one delivery, 5 after four; a patrol covers 95 px/s at the start and 123 px/s after four |
| Lose on Heat 100 or at 7:00 PM | built | both end screens reached |
| Alleys and roofs break line of sight | built | same distance from a patrol: seen on the open street, not seen with a roof between |
| Afternoon-to-dusk light, lamps at night | built | screenshot at 6:45 PM with five patrols, phone and foldable; patrols and view cones stay readable |
| Every person has a hair-library hairstyle | built | cast screen screenshot: 7 figures, each distinct |
| Riddim that gains a layer per crate | not assessed | a headless browser has no audio output |
| Pause, mute, restart | built | Pause button freezes the clock, Resume continues, Restart clears crates and heat; Space still drops the crate after tapping Sound |
| Difficulty is fair and not trivial | not assessed | needs a person; all numbers are first guesses |

**Fixed by the checks:** overlays that never hid; a crate that jumped back into your
hands; the Space key doing nothing after a HUD button was clicked; view cones too dim
at night.

**Guessed (not in any ADR):** every number in `config`, Imani's hairstyle (locs pinned
up), the block's street layout, the patrol routes, and the look of the patrol car.
**No frame rate is claimed.** A headless test machine is not a phone.

## Change 1 — demo cheat codes (2026-11-14, ledger row 77)

| Line | Status | How it was checked |
| --- | --- | --- |
| Code pad in the pause menu, six directions | built | real taps on a phone-size screen entered Clean Slate; heat went 60 → 0 |
| Arrow keys enter codes while paused | built | real key presses |
| Codes do nothing during play | built | same six arrow keys while playing: no effect |
| A wrong sequence does nothing | built | scripted |
| Slow Riddim halves patrol speed | built | a patrol covered 95 px in a second before, 48 after |
| Late Doors winds the clock back 30 s | built | clock 100 → 70 |
| Using a code marks the run; restart clears it | built | win card shows the note; restart resets the flag |
| The codes are fun or fair | not assessed | needs a person |
| Pause card and code pad fit a landscape phone (844×390) | built | pad fully on the card, no overlap with the heading, no sideways scroll; screenshot looked at |
| The code's result is shown inside the pause card | built | screenshot; gameplay hints are hidden while a menu is open |

## Change 2 — visual pass (2026-11-15, ledger row 79)

Done without a play verdict, because it is the only part of the project anyone can see.
Still a flat 2D browser prototype: this is not the game's graphics.

| Line | Status | How it was checked |
| --- | --- | --- |
| Buildings have height: near walls show and roofs shift as you move | built | screenshots at phone and desktop sizes |
| Sun shadows: short in the afternoon, long at dusk, gone at night | built | screenshots at 4:15, 5:34 and 6:46 PM. The first version drew nothing (the shapes cancelled each other out); fixed |
| Kerbs and pavements, palms with shadows, deeper water with foam | built | street and harbour screenshots, day and night |
| Night: lamp pools, coloured string lights, a yard glow that grows with the stack, lit windows | built | night screenshots |
| Patrol view cones readable day and night; red and flashing when they see you | built | screenshots; the cone is two flat steps, not a gradient |
| Soft dark screen edges | built | done in CSS, so it costs nothing per frame |
| Gameplay unchanged | built | every earlier check re-run: movement, delivery, sight lines, heat, pause, restart, codes, landscape |
| Frame cost | measured, software renderer only | about 7–15 ms a frame after, against 5–12 ms before, on a headless browser with no graphics card. The first version of this pass cost 20–34 ms and was rejected. **This says nothing reliable about a phone** |
| Looks good on a real phone | not assessed | needs a person |

**What the measurement changed:** pre-drawing the world into big images made frames
slower here, not faster, so the world is drawn as flat shapes with off-screen parts skipped.
