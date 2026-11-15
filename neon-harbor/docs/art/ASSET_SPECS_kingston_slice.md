# Asset Specs — Vertical Slice: One Kingston Block

**Status:** DRAFT v0.1 (2026-11-15) · every asset is **Needed** · budgets are PROPOSED and
PENDING a technical artist · format adapted from the asset-spec method in the Claude Code
Game Studios template (MIT); ledger row 78.

This is the build list for the vertical slice in ADR 0050: one block between Keelwright
Boatworks and Zinc Roof Soundhall, two principal characters, a crowd set, one vehicle and
one venue. It is the same block as the Crate Run prototype. Nothing here exists as art yet.

**16 assets.** Rules come from `docs/art/ART_BIBLE.md` (§ numbers below).

## ASSET-001 — Imani Vassell — body and head

| Field | Value |
| --- | --- |
| Category | Character |
| Naming | `chr_imani_body` |
| Geometry budget | ≤ 90k tris LOD0; LOD chain to 8k |
| Texture resolution | 4K sets (T0), 2K (T1), 1K (T2/T3) |

**Description:** Protagonist, late twenties, athletic build. Promoter's working clothes: fitted top, cargo trousers, trainers, one gold accent. Skin shader tuned for night rim light.

**Art bible anchors:** §5 skin and clothing rules; §4 sodium accent

**Status:** Needed

## ASSET-002 — Imani — hair: locs pinned up

| Field | Value |
| --- | --- |
| Category | Character hair |
| Naming | `chr_imani_hair_locsupdo` |
| Geometry budget | Mesh locs ≤ 40k tris LOD0; cards fallback |
| Texture resolution | 2K |

**Description:** Locs gathered into a crown with a bun at the back (hair library: locs pinned up). Individual locs readable at 2 m; crown and bun shape readable at 20 m.

**Art bible anchors:** §1 people first; §5 hair is identity

**Status:** Needed

## ASSET-003 — Omari 'Bassline' Grant — body, head, freeform locs

| Field | Value |
| --- | --- |
| Category | Character |
| Naming | `chr_omari_body, chr_omari_hair_locsfreeform` |
| Geometry budget | ≤ 90k + 45k tris LOD0 |
| Texture resolution | 4K / 2K / 1K |

**Description:** Sound engineer, forties. Work shirt, cable loops on the belt. Shoulder-length freeform locs of uneven thickness.

**Art bible anchors:** §5; hair library HAIR-LOCS-FREEFORM

**Status:** Needed

## ASSET-004 — Crowd set A — 6 bodies, 6 hairstyles

| Field | Value |
| --- | --- |
| Category | Character (crowd) |
| Naming | `chr_kgn_crowd_a01–a06` |
| Geometry budget | ≤ 25k tris each LOD0; LOD to 2k |
| Texture resolution | 2K shared atlas |

**Description:** Six Kingston residents of varied age and build. Hairstyles: 360 waves, rounded afro, canerows, microlocs, low fade, head wrap. Mix-and-match clothing colours.

**Art bible anchors:** §5 crowds by variety; hair library

**Status:** Needed

## ASSET-005 — Foot patrol officer

| Field | Value |
| --- | --- |
| Category | Character |
| Naming | `chr_patrol_foot_a` |
| Geometry budget | ≤ 40k tris LOD0 |
| Texture resolution | 2K |

**Description:** Fictional municipal patrol uniform: dark blue, peaked cap, no real insignia. Carries a cold white torch.

**Art bible anchors:** §2 under watch; §5 patrol read; wall W4

**Status:** Needed

## ASSET-006 — Patrol car

| Field | Value |
| --- | --- |
| Category | Vehicle |
| Naming | `veh_patrol_car_a` |
| Geometry budget | ≤ 150k tris LOD0 (Nanite body), wheels separate |
| Texture resolution | 4K body, 2K interior |

**Description:** Original boxy four-door, white with a blue band, roof light bar. Shape must not resemble any production car.

**Art bible anchors:** §3 straight supporting shapes; wall W2

**Status:** Needed

## ASSET-007 — Zinc-roof yard house — modular kit

| Field | Value |
| --- | --- |
| Category | Environment kit |
| Naming | `env_kgn_yardhouse_kit (walls ×6, roofs ×4, doors ×3, fences ×4)` |
| Geometry budget | Nanite source; ≤ 20k tris per piece fallback |
| Texture resolution | 2K tiling + 1K trims |

**Description:** Low block-and-zinc houses that snap on a 4 m grid. Roof sheets in four wear states: new, weathered, rusted, painted.

**Art bible anchors:** §3 low, wide, patched; §6 Kingston materials

**Status:** Needed

## ASSET-008 — Zinc Roof Soundhall — exterior and yard

| Field | Value |
| --- | --- |
| Category | Environment hero |
| Naming | `env_kgn_soundhall_ext` |
| Geometry budget | Nanite source; ≤ 120k tris fallback |
| Texture resolution | 4K hero trims, 2K tiling |

**Description:** Large zinc-roofed hall with an open yard, hand-painted name board, string lights on poles, a low stage edge.

**Art bible anchors:** §1 sound shows; §4 sodium + signal colours

**Status:** Needed

## ASSET-009 — Keelwright Boatworks — shed and slipway

| Field | Value |
| --- | --- |
| Category | Environment hero |
| Naming | `env_kgn_boatworks_ext` |
| Geometry budget | Nanite source; ≤ 100k tris fallback |
| Texture resolution | 4K / 2K |

**Description:** Timber-and-zinc boat shed on the quay, a slipway with rails, two hulls on stands in different stages of repair.

**Art bible anchors:** §6 timber, rope, sea-worn hulls

**Status:** Needed

## ASSET-010 — Speaker stack — 5 cabinet types

| Field | Value |
| --- | --- |
| Category | Prop hero |
| Naming | `prp_kgn_speaker_a–e` |
| Geometry budget | ≤ 12k tris each |
| Texture resolution | 2K shared |

**Description:** Hand-built plywood cabinets: sub, mid, horn, scoop, tweeter box. Scuffed paint, stencilled Vassell mark, steel corners.

**Art bible anchors:** §1 sound shows; §6 storytelling

**Status:** Needed

## ASSET-011 — Speaker crate

| Field | Value |
| --- | --- |
| Category | Prop (carried) |
| Naming | `prp_kgn_speakercrate_a` |
| Geometry budget | ≤ 4k tris |
| Texture resolution | 1K |

**Description:** Rope-handled timber shipping crate, stencilled mark, one cabinet's size. Must read clearly in the player's arms from behind.

**Art bible anchors:** §3 hero shape; Crate Run prototype

**Status:** Needed

## ASSET-012 — Street dressing set

| Field | Value |
| --- | --- |
| Category | Props |
| Naming | `prp_kgn_street_a (lamp, utility pole, cables, drum, cooler, plastic chairs, handcart)` |
| Geometry budget | ≤ 3k tris each |
| Texture resolution | 1K atlas |

**Description:** Everyday street objects for dressing near venues. Sodium street lamp is the key light source at night.

**Art bible anchors:** §6 prop density; §2 night

**Status:** Needed

## ASSET-013 — Road and quay surfaces

| Field | Value |
| --- | --- |
| Category | Surface materials |
| Naming | `mat_kgn_asphalt_a/b, mat_kgn_quayplank_a, mat_kgn_limestone_a` |
| Geometry budget | n/a (tiling) |
| Texture resolution | 2K, with wetness and puddle masks |

**Description:** Patched asphalt with faded lane marks, salt-bleached quay planks, limestone kerbs. Each has a wet variant for squalls.

**Art bible anchors:** §2 wet reflections; §6 weather

**Status:** Needed

## ASSET-014 — Harbour water

| Field | Value |
| --- | --- |
| Category | Water material |
| Naming | `mat_kgn_harbourwater` |
| Geometry budget | n/a |
| Texture resolution | — |

**Description:** Shallow teal harbour water: gentle chop, foam at the quay edge, caustics on T0 and Ultra only.

**Art bible anchors:** §4 harbour teal; ADR 0050 water row

**Status:** Needed

## ASSET-015 — Moored skiffs — 3 colourways

| Field | Value |
| --- | --- |
| Category | Vehicle (static in slice) |
| Naming | `veh_kgn_skiff_a` |
| Geometry budget | ≤ 30k tris |
| Texture resolution | 2K |

**Description:** Open wooden fishing skiff with an outboard, hand-painted in red, yellow or blue. Fictional names on the bow.

**Art bible anchors:** §3 rounded hero shape; registry VH-BOAT-01

**Status:** Needed

## ASSET-016 — Block lighting rig

| Field | Value |
| --- | --- |
| Category | Lighting |
| Naming | `lgt_kgn_block_day, lgt_kgn_block_dusk, lgt_kgn_block_night` |
| Geometry budget | n/a |
| Texture resolution | — |

**Description:** Three lighting states for the block: hard afternoon sun, sodium dusk, night with string lights and lamps. The night state is the showcase.

**Art bible anchors:** §2 all Kingston rows; ADR 0050 light identity

**Status:** Needed

## Not in this slice

Interiors, the flying vehicles, property renovation, the other six districts, and any
cinematic. They are specified after the slice is built and judged.

## How this becomes graphics

1. Sign off art bible section 1.
2. Commission concept art for ASSET-001, 002, 008 and 016 first: they set the look.
3. Build the block in Unreal with the settings in `Config/DefaultEngine.ini`.
4. Capture it on a T0 machine and run ADR 0050's four pass tests.
