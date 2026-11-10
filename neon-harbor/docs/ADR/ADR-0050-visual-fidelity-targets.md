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
  - Silver Spring: green-filtered creek-valley haze (water look PENDING the open item
    in ADR 0053).
  - Alexandria: cool brick-and-river dusk, with wet-asphalt reflections.
  - Dubai: dust-scattered white heat, with glass-tower specular at golden hour.
- **Material truth.** The street-level hero materials are:
  - Kingston: zinc roofing, hand-painted signage (original fonts and brands only),
    jerk-pan smoke, wet limestone.
  - Alexandria: Old Town brick.
  - Silver Spring: creek stone and wet leaf litter.
- **Weather as gameplay light.** Tropical squall fronts, harbour fog, and Virginia
  thunderstorms drive both visuals and the systems the density promise counts.

## Engine feature plan (UE 5.5)

| Feature | T0 RTX 5060 | T1 low-end laptop | T2 iPhone 16 | T3 Galaxy Fold 6 |
| --- | --- | --- | --- | --- |
| Global illumination | Lumen HW RT | Lumen SW | baked + SSGI-lite | baked |
| Reflections | Lumen RT reflections | Lumen SW | SSR | cubemaps |
| Geometry | Nanite (world + foliage) | Nanite | LOD chains | LOD chains |
| Shadows | Virtual Shadow Maps | VSM (reduced) | CSM 2 cascades | CSM 1 cascade |
| Upscaling | TSR / DLSS, 1080p out | TSR, 720p–900p out (dynamic) | MetalFX, dynamic resolution | FSR-mobile, dynamic resolution |
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
| Frame rate | 60 fps @1080p | 60 fps (dynamic res) | 60 fps (dynamic res) | 60 fps (dynamic res) |
| GPU frame time p95 | ≤ 16.6 ms | ≤ 16.6 ms | ≤ 16.6 ms | ≤ 16.6 ms |
| Streaming hitch p99 | ≤ 50 ms | ≤ 80 ms | ≤ 100 ms | ≤ 100 ms |
| VRAM / memory budget | ≤ 7 GB | ≤ 3.5 GB | ≤ 3 GB | ≤ 3 GB |
| Install size (output) | measured at content lock | — | — | — |

**60 fps on every tier (user directive, ledger row 67).** Frame rate is the fixed
target; resolution and render settings are what give way to reach it.

- T1–T3 use dynamic resolution and the cheaper settings in the feature table.
- If a tier still can't hold 60 fps in a district, content is **gated** on that tier
  (ADR 0013). The frame rate is not dropped to 30.
- On phones, sustained 60 fps is also limited by heat and battery. A 30 fps
  battery-saver mode is allowed as a player option, never as the default.
- These were 30 fps targets for T1–T3 before row 67. Nothing is measured yet, so all
  four remain PENDING.

## RTX Ultra mode (added 2026-11-10, ledger row 68) — all PENDING

The user asked for "the highest quality RTX graphics", with a reference screenshot of
a high-end-card benchmark overlay (RTX 5090, 4K, upscaling + frame generation). This
adds an optional **Ultra mode above T0**. It is a PC graphics preset, not a fifth
budget tier: ADR 0013's four tiers and their gating rules are unchanged, and no
content exists only in Ultra.

| Setting | RTX Ultra (RTX 5080/5090 class) |
| --- | --- |
| Output | 4K |
| Global illumination + reflections | Lumen hardware ray tracing at its highest quality, with hit lighting |
| Direct lighting and shadows | MegaLights for many ray-traced area lights — **experimental in UE 5.5**, so behind a toggle until it is stable |
| Photo mode / cinematics | UE Path Tracer (production-ready in 5.5) for stills and cutscene renders, not real-time play |
| Geometry | Nanite at full detail; ray-traced Nanite where supported |
| Upscaling | DLSS Super Resolution, Quality mode |
| Denoising | DLSS Ray Reconstruction — **UE 5.5 plugin support PENDING verification** |
| Frame generation | Optional, off by default (see below) |
| Latency | Low-latency mode on whenever frame generation is on |

**Targets for Ultra:**

- **60 fps of rendered frames at 4K** with upscaling, GPU frame time p95 ≤ 16.6 ms.
- Frame generation may multiply that up to the display's refresh rate. Generated
  frames are reported separately and never counted toward the 60 fps target.
  Benchmark overlays like the one in the reference usually include generated frames,
  so their numbers are not comparable to rendered-frame targets.
- VRAM budget and install-size impact: PENDING.

**Not adopted yet:** the newest neural-rendering release from the same vendor
(launched autumn 2026, RTX 50 series only, in a handful of games). It needs vendor
integration work and an engine plugin that hasn't been verified for UE 5.5. Review it
again at M3.

**Other vendors:** equivalent upscalers and frame generation (FSR, XeSS) are offered
at the same preset, so Ultra is not locked to one GPU brand.

The reference screenshot shows another studio's game. It is evidence for the
**settings class** only. No art, character or look is taken from it (rule 8, wall W3).

## Where Neon Harbor aims to lead (added 2026-11-10, ledger row 71) — all PROPOSED

The user asked the project to try to outdo the biggest title in the genre on
graphics. Rule 8 bans any claim of that kind, and a small team cannot out-spend a
studio of that size on overall asset volume. The ambition is kept, but restated as
four areas where a focused game can be the best in its class. Each one is checked
against **our own build**, never against another title.

| Area | Why this game can lead here | Pass test (in our build) |
| --- | --- | --- |
| **Textured hair** | It is this game's identity (ADR 0051), and it is widely done poorly. | Every style in the hair library reads correctly at 2 m and at 20 m on all four tiers, signed off by the community stylists. On T0 and Ultra, hero characters use strand or mesh hair, not cards. |
| **Skin under real light** | Dark skin tones are often lit badly by default exposure and bounce settings. | A lighting test scene with the full cast passes review in all seven districts' signature light, day and night, with no crushed or ashen skin tones. |
| **District light identity** | Seven places with very different sun, haze and night light. | A blind test: reviewers shown unlabelled captures name the district from light and atmosphere alone, at 80% or better. |
| **Density per block** | ADR 0044's falsifiable promise. | Interacting systems per km² counted from the registries and shown working in one block per district at the tier's frame-rate target. |

**Order of work:** build one **vertical slice** first, a single Kingston block, at
Ultra quality, with two characters, one vehicle and one venue. It is far cheaper to
make one block look exceptional than a whole map look average, and the slice is
what every later district is measured against.

**What is not promised:** more total content, a bigger map, or higher overall
fidelity than any named title.

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
