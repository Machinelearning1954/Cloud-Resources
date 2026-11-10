# CLAUDE.md — Neon Harbor: Kingston

Agent contract for Claude (Fable 5.2 / Claude Code / any Claude agent) working in this
repository. Read fully before the first edit.

## What this project is

UE 5.5 C++ open-world action RPG in the GTA-style genre, on **original IP** built to be
marketable. Real place names + fictional depiction (ADR 0008): **Kingston, Jamaica ·
Silver Springs, VA · Alexandria, VA · Dubai** — expanded to seven districts by ADR 0052
(+ Washington D.C., Addis Ababa, Asmara; Ethiopia/Eritrea are civilian-life only). Four-tier device budgets (ADR 0013):
RTX 5060 → low-end laptop → iPhone 16 → Galaxy Fold 6, enforced as compile-time law.

**Current state: NH-REBUILD-1.** The original merged C++ tree was lost in a sandbox
recycle before it was ever pushed to GitHub. This tree is the faithful docs/policy/data
layer; `Source/` is an honest empty marker. Full incident record and the
restore-from-local protocol: `docs/RECOVERY_LOG.md`. Do not "helpfully" recreate source
files that claim to be the original — that is fabrication.

## Non-negotiable rules (doctor-enforced)

1. **Four walls** — content files (`data/`, `tools/`, `Source/`) must never contain
   real-person names, real brands, R* canon characters/trade dress, or real institutions.
   Discussion files (`docs/`, `README.md`, `CLAUDE.md`) may carry evidence context only.
2. **Single currency** (ledger row 0057 guard). No dual-currency economies, no tokens.
3. **Tier budgets are law** (ADR 0013): budgets live in `static_assert`s; on weaker
   tiers content is **gated, never degraded**.
4. **Flight gate M3.5**: `bFlightEnabled=False` until streaming validation passes;
   NH-AM-10 (Cockpit Sinkholes) and NH-AM-20 (Blue Mountain Jump Point) inherit the gate.
5. **Transient state ≠ save data** (ADR 0048): no amenity or focus state enters the save
   codec.
6. **§0 — no online services**: the game never consumes the web, APIs, or remote content
   at runtime.
7. **No fabricated artifacts**: no invented compiler output, no invented "original"
   source, no invented facts. Unverified = marked PENDING.
8. **Density promise** (ADR 0044): falsifiable claims only; never compare against
   R* numbers ("5× GTA 6" and friends are banned).

## Workflow (every change)

1. Make the change.
2. `bash tools/repo_doctor.sh` — must be green (26 checks) before any commit.
3. Commit with an `NH-` prefixed message.
4. Publish only via `NEON_REMOTE=git@github.com:<you>/neon-harbor.git bash tools/publish.sh`
   (it runs the doctor first; never publish red).

**The compile loop (closes the code story):** generate the VS project → compile →
paste the **full** error log back into the session. All follow-up work responds to real
compiler line numbers, not assumptions.

## Milestone map

- **M0** — save codec v2 (dual wallet pocket/stored) + v1→v2 migration + registries.
  Regenerate from ADR slices; each slice passes the doctor before the next begins.
- **M1** — GAS init; waiting consumers: BreathSet, Focus (ADR 0048), weapon cheat rows;
  amenity rows with `milestone: M1` in `data/registries/amenities.json`.
- **M2** — amenity content wave + 48 mission rows (ADR 0019 schema).
- **M3 / M3.5** — Dubai branch venues (NH-AM-28/29), flight-gate streaming validation,
  unlock of the two flight-gated amenities.

## Pending intake queue

- Ledger rows **54** and **59** are PENDING — awaiting pasted content (links cannot be
  opened). Row 59 needs a §0 review before any "decode and embed" interpretation.
- New external claims/links/screenshots → append a row to `docs/RUMOR_LEDGER.md` with a
  close category (ADOPT / DEDUPE / DEFER / GOSSIP / REFERENCE / POLICY). If the source
  cannot be opened, record PENDING and request pasted content. Unverifiable media never
  adopts into architecture (ADR 0014 forensics).

## Repo map

```
docs/RECOVERY_LOG.md        incident + restore protocol
docs/RUMOR_LEDGER.md        intake machine (rows 1–58 lossy, 60–65 full, 54/59 pending)
docs/ADR/INDEX.md           ADR index (7 reconstructed + ADR 0049 + ADR 0050 visual targets + ADR 0051 character appearance
                            + ADR 0052 seven districts + ADR 0053 geodata pipeline)
docs/web/index.html         world browser (ADR 0047), plain HTML/JS over registry JSON
docs/ADR/ADR-0049-*.md      amenity parity matrix (29 original venues)
data/registries/*.json      7 registries; amenities.json is authoritative
tools/repo_doctor.sh        26-check policy scanner (walls, integrity, cross-refs)
tools/publish.sh            GitHub publish (doctor-gated, needs NEON_REMOTE)
Source/README.md            honest NOT-RECOVERED marker
```
