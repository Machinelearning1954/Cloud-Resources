# ADR 0008 — Real Place Names + Fictional Depiction, Four Districts

**Status:** RECONSTRUCTED (core contract preserved) · **Rebuilt:** 2026-11-10

## Decision

Neon Harbor: Kingston uses **real place names with fictional depiction**:

1. **Kingston, Jamaica** — harbour, downtown, Blue Mountain fringe, sink-country interior.
2. **Silver Springs, Virginia** — wetlands, glass-bottom boat heritage, marina row.
3. **Alexandria, Virginia** — Old Town grid, industrial waterfront, metro corridor.
4. **Dubai** — fourth district (standing decision, retained; ledger row 62).

## Rules

- Real geography anchors **orientation**; every interactive venue, business, gang,
  brand and character is **fictional** and original (ADR 0049 naming rules).
- No real institution is simulated, defamed, or depicted operationally
  (institutions wall).
- District data lives in a single registry (`data/registries/districts.json`);
  POIs must reference an existing district id or fail the doctor.

## Consequences

- Marketing can say "set in Kingston and Northern Virginia" — ownable, legal, distinctive.
- Ledger row 62 (2026-11-10): user re-affirmed Jamaica / Silver Springs / Alexandria VA
  as the anchor districts; Dubai retained per standing decision.

> **Amended 2026-11-10 by ADR 0052:** the world now has seven districts (Washington, Ethiopia and Eritrea added). This reconstructed text is kept as-is for history.
