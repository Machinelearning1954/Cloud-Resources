# RECOVERY LOG — NH-REBUILD-1 (2026-11-10)

## Incident

The previous working session held the complete merged Neon Harbor repository
(Phase 1–4 reconstruction: tree/config, Core/Player/Save/UI, World/AI/Missions/Vehicles,
plugins/docs, 90-check doctor, publish.sh). The build sandbox was recycled between
sessions. The tree was **not** persisted. `tools/publish.sh` had not yet been run, so no
GitHub remote copy exists. Recovery from remote: impossible. Recovery from user's local
copy: possible — see protocol below.

## Recovered (from the project log, faithful)

- Method: RUMOR_LEDGER intake machine — six close categories (ADOPT / DEDUPE / DEFER /
  GOSSIP / REFERENCE / POLICY), four walls (real-person likeness / real brands / R* canon
  characters / real institutions), forensic tools per incident (AI-leak checklist,
  trademark scan, single-currency guard, DLL ban, political-figures scan,
  institutions scan).
- Architecture contracts: save codec v2 dual wallet (pocket/stored) + v1→v2 migration;
  48-row mission schema with twentieth-anniversary scar list; 10 registries; flight gate
  M3.5 with `bFlightEnabled=False` and static traversal budget assertions
  (20/4/1 minutes); focus system as transient state (ADR 0048); browser world browser
  (ADR 0047, WASM port refused); density promise ADR 0044 ("highest system density in
  its size class", falsifiable, no "5× GTA6" claims).
- Registry concepts incl. 17 vehicles with NoiseRating spectrum, species registry with
  unknown-species = protected fail-safe, single police registry with private security
  deferred.
- Three pre-fixed compile-error classes (UHT enum scoping,
  UPerlinNoiseCameraShakePattern, include elevation) — recorded, not re-applied to code
  (code not recovered).

## Not recovered (declared honestly)

- **All C++/UHT source files.** No source, no headers, no build files. The `Source/`
  directory in this tree is an honest marker, not code.
- The exact 90-check list of the original doctor. This rebuild ships a **26-check**
  doctor covering the same policy families; count is honest, not padded.
- Verbatim text of ledger rows 1–58 and ~40 ADR bodies not listed in the INDEX below.
  A lossy summary block stands in their place, marked per row.
- The `docs/web/` world browser build artifacts.

## Restore-from-local protocol

If you (the user) kept a local copy (downloaded during the previous session, or on a
machine that ran `publish.sh` late):

1. Drop it anywhere as `neon-harbor-local/`.
2. Say the word; we diff it against NH-REBUILD-1, keep the local copy as authoritative
   for code, merge this revision's docs/data layer (ledger rows 60–62, ADR 0049,
   amenities registry) on top.
3. Re-run `tools/repo_doctor.sh`, then proceed to the three commands
   (VS project → compile → paste log).

If no local copy exists: regenerate code **from the ADRs, slice by slice**, starting at
M0 (save codec v2 + registries), with every slice passing the doctor before the next
one starts. The compiler-loop commands above apply unchanged.
