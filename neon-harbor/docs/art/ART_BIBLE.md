# Neon Harbor: Kingston — Art Bible

**Status:** DRAFT v0.1 (2026-11-15) — written without an art director; needs sign-off
from the user and, before M2, from an art lead and the community consultants named in
ADR 0051. Structure adapted from the art-bible method in the open-source Claude Code
Game Studios template (MIT licence); no text is copied from it. Ledger row 78.

This document turns the art direction in ADR 0050, 0051 and 0055 into rules an artist
can build from. It is words and numbers, not pictures. No concept art exists yet.

## 1. Visual identity statement

> **Real light on real people in real places, heightened by sound.**

When a visual decision is unclear, apply these in order:

1. **People first.** If a choice makes skin, hair or clothing read worse, it is the
   wrong choice, however good the environment looks. *(Test: does the cast still read
   correctly at 2 m and at 20 m?)*
2. **Each place has its own light.** If two districts could be swapped without anyone
   noticing, one of them is wrong. *(Test: can a reviewer name the district from an
   unlabelled capture?)*
3. **Sound shows.** Music is part of the world's look: speaker stacks, string lights,
   crowds, bass you can see in a puddle. *(Test: is there something in frame that
   tells you what you would hear?)*

Grounded realism is the base. The one heightening is light and colour at night.

## 2. Mood and atmosphere by game state

| State | Feeling | Light | Carried by |
| --- | --- | --- | --- |
| Free roam, day (Kingston) | Open, hot, alive | Hard overhead sun, short shadows, warm bounce off zinc and limestone | Heat shimmer over roofs, saturated paint, deep blue shade |
| Free roam, night (Kingston) | Electric, close | Sodium orange street light, coloured string lights, cool moonlit sea | Wet asphalt reflections, moths at lamps, speaker glow |
| Carrying / under watch | Tight, exposed | Same scene, patrol light bars add cold blue-white | Long hard shadows from headlights, view cones in dust |
| Venue interior (soundhall, studio) | Warm, loud, safe | Low warm practicals, haze, coloured wash from the stage | Bodies in silhouette against light, sweat sheen |
| Story beats | Still, heavy | One motivated key light, deep fill | Faces; background falls to near black |
| Silver Spring / Alexandria / Washington | Temperate, brick-and-river | Soft overcast or low golden sun, cool dusk | Wet brick, bare or turning trees, long blue twilight |
| Addis Ababa / Asmara | High, clear, bright | Thin-air sun, crisp shadows, cool nights | Dust in light shafts, eucalyptus smoke, cafe terraces |
| Dubai (near-future) | Clean, vast | White heat by day; cool blue-white architectural lines at night on a warm dusk sky | Glass speculars, sea haze, sky-lane traffic lights |

## 3. Shape language

- **Characters:** every principal has a silhouette readable at thumbnail size, and the
  **hairstyle is the first read** (ADR 0051). Clothing shapes are the second read.
- **Kingston:** low, wide and patched. Corrugated lines, hand-cut edges, things leaning.
  Verticals come from speaker stacks, palms and utility poles.
- **US districts:** regular grids, brick rhythm, pitched roofs, street trees.
- **Addis Ababa and Asmara:** terraces and stairs on slopes; Asmara's curved modernist
  corners and wide boulevards.
- **Dubai:** long sweeping curves, cantilevers, pylons over water. No sharp, hostile
  shapes: the tone rule in ADR 0055 says the city serves daily life.
- **Hero versus supporting:** in any frame, one warm, rounded hero shape (a person, a
  speaker stack, a boat hull) against cooler, straighter supporting shapes.
- **UI:** flat, rectangular, quiet. It never imitates the world's textures.

## 4. Colour system

| Role | Colour | Meaning in this world |
| --- | --- | --- |
| Sodium | warm orange `#FF9D3C` | Night streets, home ground, the soundhall |
| Zinc | cool grey-blue `#9FB0B8` | Roofs, rain, the everyday |
| Harbour | deep teal `#1F6F86` | The sea, the routes, the Tide Line |
| Signal green | `#3ED6A4` | Go, safe, a goal you can reach |
| Heat red | `#FF4D5E` | Being watched, danger, the law |
| Ink | off-white `#F3EAD8` | Text and UI on dark |
| Night | blue-black `#0B1420` | UI ground, deep shadow |

- **Skin is never colour-graded as a side effect.** Grades and exposure are checked
  against the full range of the cast's skin tones in every district's light (ADR 0050).
- **District temperature:** Kingston warm-on-cool; US districts neutral to cool; the
  highland districts neutral and clear; Dubai cool-on-warm.
- **Colour-blind safety:** red (heat) versus green (go) never stands alone. Heat also
  shows as a filling bar and a pulsing edge; goals also show as a dashed ring and an
  arrow. The prototype already does this.

## 5. Character design direction

- **Hair is identity** and is locked on every tier (ADR 0051). The hair library in
  `crews.json` is the source list; no character gets a default style.
- **Skin:** physically based skin with subsurface scattering; darker tones get
  specular and rim-light care so faces keep shape at night.
- **Clothing:** specific and current to each place. No real brand marks (wall W2);
  original prints and labels only.
- **Posture and motion:** grounded and individual. Principals each have a walk.
- **Telling people apart:** allies and crew by silhouette and colour accent; patrols
  by uniform shape and the cold light they carry; crowds by variety, never by
  repeating one figure.
- **Detail by distance:** at 20 m the hairstyle, build and top colour must survive.
  Faces can simplify; hair shape cannot.

## 6. Environment design language

- **Kingston materials:** zinc sheet (new, weathered, rusted, painted), limestone,
  concrete block, hand-painted signage, timber, rope, sea-worn hulls.
- **Texture approach:** physically based, built from the project's own reference
  shoots or licensed scans. Real imagery is reference only and never ships (ADR 0053).
- **Prop density:** dense and personal near venues (crates, cables, chairs, coolers);
  sparse on through-roads. Density follows where people gather.
- **Storytelling in the world:** repairs on repairs, hand-lettered signs, sound-system
  flyers by fictional promoters, speaker crates stencilled with the Vassell mark.
- **Signage:** original names only, in the project's own lettering styles.
- **Weather:** squalls, harbour fog and Mid-Atlantic thunderstorms change both the
  look and the systems (ADR 0050).

## 7. UI and HUD direction

- **Art view:** screen-space, flat, minimal. Condensed display type for numbers and
  titles, a plain sans for text. Icons are simple outlines. Motion is short and
  functional.
- **Usability view:** the HUD must read on a phone in sunlight. Minimum text size
  12 px at phone scale; every state shown by shape as well as colour; touch targets
  at least 44 px.
- **Where they conflict:** usability wins. A diegetic element (a phone, a radio dial)
  is allowed only where it is as readable as the flat version.

## 8. Asset standards (PROPOSED — to be set with a technical artist)

- **Naming:** `<type>_<district>_<name>_<variant>`, lower case, e.g.
  `env_kgn_zincroof_a`, `chr_imani_body`, `prp_kgn_speakercrate_a`.
- **Source formats:** FBX or USD for meshes; 16-bit PNG or EXR for textures. Source
  files live in asset storage, not in this repo.
- **Geometry:** static world meshes are authored for Nanite on T0 and T1, with a
  generated low-detail fallback for T2 and T3. Characters use standard LOD chains.
- **Texture tiers (starting point):**

  | Use | T0 / Ultra | T1 | T2 / T3 |
  | --- | --- | --- | --- |
  | Hero character | 4K sets | 2K | 1K |
  | Hero prop / vehicle | 4K | 2K | 1K |
  | World surface (tiling) | 2K | 2K | 1K |
  | Small prop | 1K | 1K | 512 |

- **Texel density target:** 10.24 px/cm for hero surfaces, 5.12 px/cm for the world.
- **Art versus budget:** where the ideal exceeds a tier's memory budget (ADR 0050),
  the tier's content is gated, not thinned (ADR 0013). Texture resolution may scale.

All numbers in this section are proposals and are PENDING a technical artist and
measurement in a build.

## 9. Reference direction

- **Shoot or license our own reference.** Kingston streets and harbour, sound-system
  nights, zinc and paint up close; Silver Spring and Alexandria streets; highland
  light in Addis Ababa and Asmara. Model releases for any person photographed.
- **Reference categories, not titles.** Documentary photography, music-video
  lighting, architectural photography of modernist Asmara and of sea-platform
  concepts. No other game is used as a look target (rule 8).
- **The four user videos** (ledger row 72) are concept reference for systems only.

## Open items

- Sign-off on section 1 by the user. Everything else flows from it.
- Concept art for the vertical slice (one Kingston block).
- Asset specs: `docs/art/ASSET_SPECS_kingston_slice.md`.
