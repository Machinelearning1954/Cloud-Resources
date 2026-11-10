# ADR 0019 — Mission Schema: 48 Rows, Twentieth-Anniversary Scar List

**Status:** RECONSTRUCTED (core contract preserved) · **Rebuilt:** 2026-11-10

## Decision

The campaign is **48 mission rows**, schema-encoded with the project's
"twentieth-anniversary scar list" — the genre's classic failure modes we refuse to
repeat. Schema fields include: turning points, checkpoints, escort phases, cleanup
phases, gold-medal criteria — every one explicit, none implicit.

## Scar list (recovered highlights)

- Escorts that fail on AI stupidity → escort phases declare AI promise in-schema.
- Cleanup phases without bounds → every cleanup row carries its radius and exit state.
- Unwinnable gold medals → gold criteria are data, testable in isolation.
- Saves that resurrect dead state → missions write codec v2 compatible rows only.

## Consequences

- 48 rows validate against schema before content build; schema is authority.
- Mission reward economy touches the **single-currency guard** (ledger row 0057) —
  one currency, no special tokens.
