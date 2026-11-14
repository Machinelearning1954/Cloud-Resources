# Crate Run — one-page brief

Prototype of one mechanic for Neon Harbor: Kingston. Throwaway code, not the game.
Method adapted from the open-source Claude Code Game Studios template (MIT), ledger row 76.
No questions were asked: every answer below is taken from the project's existing ADRs.

- **Question this prototype answers:** is "you are only in trouble while carrying"
  an interesting core for the Tide Line missions of Act 1 (ADR 0056)?
- **Where:** one Kingston block between Keelwright Boatworks (NH-AM-25) and Zinc Roof
  Soundhall (NH-AM-04). This is the ADR 0050 vertical-slice block, drawn flat.
- **You are:** Imani Vassell, on foot.
- **Goal:** carry 5 speaker crates from the boatworks to the soundhall before doors
  open at 7:00 PM.
- **Lose:** Heat reaches 100, or the clock runs out.
- **Hook:** patrols ignore you unless you are carrying a crate. You can put the crate
  down, walk past clean, and come back for it.
- **First minute:** one patrol. Pick up a crate by walking into the pile, follow the
  arrow, deliver. The first delivery teaches the loop.
- **Challenge:** each delivery adds a patrol (up to 5) and makes them faster. Alleys
  break line of sight.
- **Look:** top-down, flat shapes drawn in code. Zinc roofs, a harbour edge, and light
  that moves from hot afternoon to sodium-orange dusk as the clock runs (ADR 0050).
  Every person is drawn with a hairstyle from the hair library (ADR 0051).
- **Sound:** a riddim generated in the browser that gains a layer per crate delivered.
  No recorded or licensed music (ADR 0054).
- **Controls:** phone: drag anywhere to move, Drop button. Keyboard: arrows or WASD,
  Space to drop or pick up, P pause, M mute.
- **Screens:** title with controls, cast, pause, win, lose, restart.
- **Out of scope:** driving, combat, dialogue, the story meters, saving, multiplayer,
  3D, any art or audio asset file.
