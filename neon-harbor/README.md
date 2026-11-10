# Neon Harbor: Kingston

> **NH-REBUILD-1** — Docs-core reconstruction, 2026-11-10.
> The original merged C++ tree was lost when the build sandbox was recycled before
> `tools/publish.sh` ever ran (no GitHub remote copy exists). What you are reading is the
> **policy / docs / data layer**, faithfully rebuilt from the project log and re-verified by
> `tools/repo_doctor.sh`. See `docs/RECOVERY_LOG.md` for the honest loss inventory.
> **Nothing in this tree claims to be the original compiled code.**

---

## Identity

UE 5.5 C++ open-world action RPG in the GTA-style genre, on an **original IP** — built to be
marketable, trademark-scannable, and legally ownable. Real place names + fictional depiction
(ADR 0008), anchored to:

- **Kingston, Jamaica** (harbour, Blue Mountain fringes, sink-country interior)
- **Silver Spring, Maryland** (corrected by the user on 2026-11-10; earlier text said
  "Silver Springs, Virginia" — see ledger row 66 and the open water-venue item in ADR 0053)
- **Alexandria, Virginia** (Old Town grid, industrial waterfront, metro corridor)
- **Dubai** (fourth district per standing ADR 0008 — retained; see ledger row 62)
- **Washington, D.C.**, **Addis Ababa (Ethiopia)** and **Asmara (Eritrea)** — added by ADR 0052
  (seven districts in total; the Ethiopian and Eritrean districts are civilian-life only)

Four-tier performance budgets (ADR 0013): RTX 5060 1080p60 → low-end laptop → iPhone 16 →
Galaxy Fold 6. Every slice declares its tier in code with `static_assert` — budgets are
compile-time law, not documentation.

## What is in this tree

| Path | Contents | State |
| --- | --- | --- |
| `docs/ADR/` | Architecture decision records (INDEX + 14 ADRs) | 7 reconstructed, ADR 0049, **ADR 0050 (visual targets), 0051 (character appearance), 0052 (seven districts), 0053 (geodata pipeline), 0054 (radio), 0055 (near-future Dubai)** |
| `docs/web/` | World browser (ADR 0047): plain HTML/JS over registry JSON | **regenerated 2026-11-10** |
| `docs/RUMOR_LEDGER.md` | Intake machine — rows 1–58 lossy summary, **rows 60–72 full** | rows 54/59 PENDING |
| `docs/RECOVERY_LOG.md` | What was lost, what was rebuilt, restore-from-local protocol | current |
| `data/registries/` | Districts, **amenities (27 evidence-mapped + 2 Dubai branches = 29 rows)**, species, vehicles, services, bounties, crews | amenities authoritative |
| `tools/repo_doctor.sh` | Policy scanner — four walls, ledger integrity, registry cross-refs, amenity completeness | 26 checks |
| `CLAUDE.md` | Agent contract for Claude (Fable 5.2 / Claude Code) | current |
| `tools/publish.sh` | GitHub publish script (ready; needs remote URL) | ready |
| `Config/DefaultEngine.ini` | Unreal renderer settings (Lumen, hardware ray tracing, Virtual Shadow Maps) | names verified, **rendering untested**; checked by `tools/check_engine_config.py` (10 checks) |
| `Source/` | **NOT RECOVERED** — honest marker only | empty |

## The three commands that still close the loop

1. Generate VS project (UE 5.5: `GenerateProjectFiles` or open `.uproject`)
2. Compile
3. **Paste the full error log back**

Until a real compiler output exists, every claim in this repo is policy-checked, not
linker-checked. Doctor green ≠ code green. This is the project's standing honesty rule.

## New this revision

**ADR 0049 — GTA6 Amenity Parity Through Original Translation.** Every side activity /
amenity evidenced in Rockstar's GTA 6 marketing (Trailer 2, *An Extended Look*, Launch
Trailer era) is translated into an **original Neon Harbor equivalent** — original venue
names, district-anchored, tier-gated, four-walls-scanned — so the game delivers the amenity
density players expect from the genre while every brand, venue and character remains ours.
27 amenities, full matrix in `docs/ADR/ADR-0049-amenity-parity.md`, registry data in
`data/registries/amenities.json`.

## Added 2026-11-10 (after NH-REBUILD-1)

- **`docs/web/`** — the world browser (ADR 0047), regenerated. Run
  `python3 -m http.server 8000` in this folder, then open `http://localhost:8000/docs/web/`.
- **Vehicles registry 7 → 17** — 10 NEW original rows, each tagged
  `provenance: NEW-2026-11-10`. They are not recovered rows.
- **ADR 0050** — original art direction and per-tier visual targets. Every number is
  PENDING until a profiled capture from a real build replaces it.
- **Ledger row 63** — records the "GTA 6 graphics / 150 GB" request and how it was
  closed: no likeness target and no size padding.
- **Cast (ledger row 64, ADR 0051)** — 8 new original crew members:
  - characters with locs, Jamaicans with 360 waves, and three Ethiopian women
  - every role is skilled and tied to a venue
  - hairstyles are identity-locked on every tier
- **Rename:** `CR-PERF-02` was renamed because its rebuilt name matched a real artist.
- **Seven districts (ADR 0052, ledger row 65)** — Washington, Ethiopia (Addis Ababa) and
  Eritrea (Asmara) added, with 0 amenities each for now. The Silver Springs, VA pin proposed
  here was later superseded: the district is Silver Spring, Maryland (row 66).
- **Geodata pipeline (ADR 0053)** — a build-time-only satellite and elevation plan: free
  public lidar and aerial imagery for the US districts, and 30 cm commercial imagery as
  a purchase for the rest. No imagery has been acquired yet.
- **Radio (ADR 0054, ledger row 66)** — three original stations: reggae, dancehall and
  Afrobeats. They are offline and "simulated live" on a world-time schedule. 0 tracks
  are cleared; music licensing or commissioning is PENDING.
- **Hair library 8 → 28 styles** — Jamaican and Ethiopian/Eritrean styles, each tagged
  sourced, generic or pending review.
- **Silver Spring is Maryland** — corrected per the user.
- **60 fps on every tier (ledger row 67)** — ADR 0050 now targets 60 fps on all four
  device tiers, with resolution scaling to reach it. Not measured yet.
- **RTX Ultra mode (ledger row 68)** — an optional 4K ray-traced PC preset above the
  four tiers, defined in ADR 0050. Settings plan only; nothing is rendered or measured.
- **Near-future Dubai (ADR 0055, ledger row 72)** — Dubai becomes a smart-city district,
  with four proposed systems: destructible interiors, flying vehicles (behind the
  M3.5 gate), property renovation and smart homes. Design only.
