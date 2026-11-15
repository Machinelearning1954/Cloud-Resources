# ADR 0058 — Cheat Codes (Original Sequences, Single-Player Only)

**Status:** PROPOSED (design) — nothing built in the game; three demo codes exist in the
Crate Run prototype · **Date:** 2026-11-14 · **Directive:** ledger row 77

## Context

The user sent a social-media image titled as a complete cheat list for the genre
leader's unreleased game and asked to integrate it.

- **The image is not a real list** (ADR 0014 forensics, ledger row 77). The game it
  names is not out yet, no official codes have been published, the image has typos
  and duplicate row numbers, and its art looks AI-generated. It is closed as GOSSIP.
- **Its contents are another studio's.** The vehicle names, the title art and the
  button sequences are that studio's material, so none of them enter this repo
  (walls W2 and W3).
- **The category is fair game.** Button-sequence cheat codes are a genre staple, and
  the milestone map already lists "weapon cheat rows" as an M1 consumer. This ADR
  designs that system with original names, original effects and original sequences.

## Decision

Cheat codes live in `services.json → cheat_codes`: 21 codes in six groups.

| Group | Codes | What they do |
| --- | --- | --- |
| Gear | Yard Kit, Pro Kit, Loud Kit | Three gear sets (fictional equipment only) |
| Player | Second Wind, Deep Pockets, Flying Fish, Moon Step | Full health and armour, no ammunition limit, fast swim, high jump |
| Street heat | Bring Them, Cool Breeze, Clean Slate | Raise, lower or clear street heat |
| Vehicles | Call the Streak, Courier, Thumper, Hauler, Skiff, Spray Board, Zippy | Spawn one of the project's own registry vehicles |
| World | Blue Sky, Squall Line, Slow Riddim | Clear weather, tropical squall, slow motion |
| Flight | Sky Lane Pass | Spawn a personal flyer, locked behind the M3.5 flight gate |

**Sequences.** Each code is 8 inputs from 12 platform-neutral tokens: four
directions, four face buttons named by position (SOUTH, EAST, WEST, NORTH), two
bumpers and two triggers. Each platform shows its own glyphs. The sequences were
generated at random by a script, so they copy no other game's codes. The script was
not kept, so the registry itself is the record. All are the
same length, so no code is the start of another.

## Rules

1. **Single-player only.** Every cheat is refused in multiplayer, where the server
   decides outcomes (ADR 0057).
2. **No cheat grants currency.** The story tracks where Imani's money came from
   (ADR 0056), and property and the Dubai flagship are paid for with it (ADR 0055).
   A money cheat would cut through both. The source image's money code is therefore
   not translated.
3. **No cheat touches a story meter.** Clean, Loyal and the story's Heat decide the
   ending and are save data. The cheats act on **street heat**, a separate transient
   value for the patrol response in free roam.
4. **Using any cheat marks the save as cheated.** Achievements and leaderboards are
   off for that save. The mark is save data; the cheat's own effect is transient
   (ADR 0048) and ends with the session.
5. **Vehicle cheats spawn only registry vehicles,** and water craft only on water.
6. **Flight cheats are their own category and inherit the M3.5 gate.** No flyer exists in
   the vehicles registry yet (ADR 0055), so the flight code names no vehicle.
7. **Tier rules still apply.** A cheat cannot spawn content that is gated on the
   device's tier (ADR 0013).

## Entering a code

- Controller: the sequence, typed during play, with a 2-second timeout between inputs.
- Phone: a code pad in the pause menu, since there is no free hand for sequences
  during touch play.
- A recognised code shows its name once. An unrecognised sequence does nothing.

## Prototype

Crate Run (`prototypes/crate-run/`) carries three demo codes (Clean Slate, Slow Riddim,
Late Doors), entered on a directions-only pad in its pause menu. They are prototype codes, shorter than the
registry's, and exist to try how entering a code feels on a phone.

## Consequences

- The registry count stays at 7.
- `tools/red_team.py` checks that sequences are unique and well-formed, that vehicle
  cheats point at real registry vehicles, that flight cheats are gated, and that the
  no-currency and single-player rules are present.
- The doctor is unchanged at 26 checks.
