# ADR 0047 — Browser World Browser: No WASM Native Port

**Status:** RECONSTRUCTED (core contract preserved) · **Rebuilt:** 2026-11-10

## Decision

The world browser ships as plain **`docs/web/` HTML/JS** consuming registry JSON.
Porting it to WASM to "look native" is refused: the browser's job is to be the
lowest-friction, most portable view of registry truth — zero build step, zero
toolchain, readable by anyone who opens `index.html`.

## Consequences

- Registry JSON is the single source of truth; the browser renders, never edits.
- Build artifacts were lost with the tree (RECOVERY_LOG); regeneration is a task, not a
  redesign. The WASM refusal stands.
