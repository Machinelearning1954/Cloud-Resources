# ADR 0050 — Visual Fidelity Targets per Tier (Original Art Direction)

**Status:** PROPOSED — targets PENDING measurement · **Date:** 2026-11-10

## Context

The user asked for top-tier, current-generation open-world graphics and a full-size
install. This ADR turns that request into **targets the build can be measured against**.
It respects two standing rules:

- **ADR 0044 / CLAUDE.md rule 8:** no comparisons to Rockstar titles, and no
  "looks like X" promises. A visual claim counts only if a capture from our own build
  shows it.
- **CLAUDE.md rule 7:** nothing here is measured yet. Every number is a **target**
  marked PENDING until a profiled capture from a real build replaces it.

Install size is an output of the content, never an input. A 100+ GB footprint comes
from authored assets: photogrammetry, Nanite meshes, 4K–8K textures, audio and
cinematics. Padding the repo to a size target is refused. Those assets are made in
Unreal Editor and kept in Git LFS or Perforce, not in this docs/data repository.

## Art direction (original)

- **Light as identity.** Each district has a signature light:
  - Kingston: hard tropical noon and sodium-orange harbour nights, with sound-system
    LED spill.
  - Silver Springs: green-filtered wetland haze, with caustics under glass-bottom hulls.
  - Alexandria: cool brick-and-river dusk, with wet-asphalt reflections.
  - Dubai: dust-scattered white heat, with glass-tower specular at golden hour.
- **Material truth.** The street-level hero materials are:
  - Kingston: zinc roofing, hand-painted signage (original fonts and brands only),
    jerk-pan smoke, wet limestone.
  - Alexandria: Old Town brick.
  - Silver Springs: mangrove mud.
- **Weather as gameplay light.** Tropical squall fronts, harbour fog, and Virginia
  thunderstorms drive both visuals and the systems the density promise counts.

## Engine feature plan (UE 5.5)

| Feature | T0 RTX 5060 | T1 low-end laptop | T2 iPhone 16 | T3 Galaxy Fold 6 |
| --- | --- | --- | --- | --- |
| Global illumination | Lumen HW RT | Lumen SW | baked + SSGI-lite | baked |
| Reflections | Lumen RT reflections | Lumen SW | SSR | cubemaps |
| Geometry | Nanite (world + foliage) | Nanite | LOD chains | LOD chains |
| Shadows | Virtual Shadow Maps | VSM (reduced) | CSM 2 cascades | CSM 1 cascade |
| Upscaling | TSR / DLSS, 1080p out | TSR, 900p out | MetalFX, native-ish | FSR-mobile |
| Volumetrics | clouds + fog + god rays | fog + clouds (low) | height fog | height fog |
| Water | single-layer water + caustics | same, no caustics | simplified | simplified |
| Crowd/traffic draw | per ADR 0013 budgets | per ADR 0013 | per ADR 0013 | per ADR 0013 |

**Render settings scale per tier**, as the table shows: lower shadow resolution,
cheaper volumetrics, simplified water. ADR 0013's "gated, never degraded" rule applies
to **content**. An amenity, venue or system that a tier can't run properly is hidden on
that tier, never shipped as a half-simulated shell.

## Measurable targets (all PENDING)

| Metric | T0 | T1 | T2 | T3 |
| --- | --- | --- | --- | --- |
| Frame rate | 60 fps @1080p | 30 fps @900p | 30 fps | 30 fps |
| GPU frame time p95 | ≤ 16.6 ms | ≤ 33 ms | ≤ 33 ms | ≤ 33 ms |
| Streaming hitch p99 | ≤ 50 ms | ≤ 80 ms | ≤ 100 ms | ≤ 100 ms |
| VRAM / memory budget | ≤ 7 GB | ≤ 3.5 GB | ≤ 3 GB | ≤ 3 GB |
| Install size (output) | measured at content lock | — | — | — |

## How a target becomes a fact

1. Capture a profile in a packaged build (`stat unit`, `stat gpu`, Unreal Insights) on
   the actual tier device.
2. Add a row to `docs/RUMOR_LEDGER.md` with class REFERENCE, linking the capture.
3. Replace PENDING with the measured value and the capture id. Without a capture the
   value stays PENDING.

## Alternatives rejected

- **Matching a named competitor's look.** That is trade dress risk, it is unfalsifiable,
  and rule 8 already bans it.
- **Filling the repo to a size target.** That fabricates scale, and size is not quality.
