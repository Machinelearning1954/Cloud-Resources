# ADR 0013 — Four-Tier Performance Budgets, static_assert-Enforced

**Status:** RECONSTRUCTED (core contract preserved) · **Rebuilt:** 2026-11-10

## Decision

Every system slice declares which device tier it serves, and the declaration is
compile-time law:

| Tier | Device | Envelope |
| --- | --- | --- |
| T0 | RTX 5060 desktop | 1080p60, full amenity simulation |
| T1 | Low-end laptop | reduced density, same systems |
| T2 | iPhone 16 | capped agents/streaming |
| T3 | Galaxy Fold 6 | reduced footprint |

Budgets (agents, streaming windows, amenity sim ticks) are encoded as constants and
guarded with `static_assert` so a slice that would blow a tier **fails to compile**.

## Standing examples (recovered contracts)

- Traversal budgets 20 / 4 / 1 minutes guarded by static asserts.
- Flight content gated behind milestone **M3.5**: `bFlightEnabled=False` until streaming
  validation passes; base-jump/parachute amenities inherit the same gate (ADR 0049).

## Consequences

- "It runs on my machine" is not an argument; the tier table is.
- Mobile tiers receive **content gating, not degraded clones** — a gated amenity is
  hidden, never half-simulated.
