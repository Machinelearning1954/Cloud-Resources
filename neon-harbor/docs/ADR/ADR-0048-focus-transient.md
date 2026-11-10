# ADR 0048 — Focus System: Transient State ≠ Save Data

**Status:** RECONSTRUCTED (core contract preserved) · **Rebuilt:** 2026-11-10

## Decision

Systems like the focus mode are **transient**: they never enter the save codec.
The design idiom is a static assertion — `static_assert(!bHasSleepMeter)`-style — i.e.
"if this state needs saving, the design is wrong by definition." Transient systems
reset to a deterministic default on load; anything the player would miss across a
session belongs in codec v2, not in focus.

## Consequences

- In-game TV channels (amenity NH-AM-23, ADR 0049) are transient by the same rule:
  what's airing is a function of world time, not of save data.
- The save codec stays small and testable; no amenity may smuggle state into it.
