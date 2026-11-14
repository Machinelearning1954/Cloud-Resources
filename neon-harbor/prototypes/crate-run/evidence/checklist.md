# Crate Run — brief checklist (2026-11-14)

Checked by a headless browser opening `index.html` straight from the file, at phone
(390×844, touch) and desktop (1280×720, keyboard) sizes, plus a look at the screenshots.
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
| Each delivery adds a patrol and speeds them up | built | 1 patrol at start, 2 after the first delivery |
| Lose on Heat 100 or at 7:00 PM | built | both end screens reached |
| Alleys break line of sight | built in code | sight line is tested against every roof; not separately exercised |
| Afternoon-to-dusk light, lamps at night | built in code | not looked at: the screenshots are from early in the clock |
| Every person has a hair-library hairstyle | built | cast screen screenshot: 7 figures, each distinct |
| Riddim that gains a layer per crate | not assessed | a headless browser has no audio output |
| Pause, mute, restart | built | buttons present; pause and restart exercised by script |
| Difficulty is fair and not trivial | not assessed | needs a person; all numbers are first guesses |

**Guessed (not in any ADR):** every number in `config`, Imani's hairstyle (locs pinned
up), the block's street layout, the patrol routes, and the look of the patrol car.
**No frame rate is claimed.** A headless test machine is not a phone.
