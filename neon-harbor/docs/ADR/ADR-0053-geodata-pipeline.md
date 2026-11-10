# ADR 0053 — Satellite & Geodata Pipeline (Build-Time Only)

**Status:** PROPOSED — no data has been acquired; every licence detail is to be
re-verified before purchase or bake · **Date:** 2026-11-10 · **Directive:** ledger row 65

## Context

The user asked for "satellite imagery tech, top tier as of today" across all seven
districts. Three constraints shape the answer:

- **Rule §0, no online services:** the shipped game never streams map tiles. Imagery
  and elevation are acquired at **build time**, processed offline, and baked into
  Unreal assets.
- **The four walls:** real imagery contains real storefronts, signage, licence plates
  and private homes. Imagery is used as **reference** for terrain shape, land cover,
  road and block layout. Visible textures, signage and buildings are authored
  originals.
- **Rule 7, no invented facts:** every number below is cited in ledger row 65 or
  marked PENDING.

## What "top tier as of today" means (Nov 2026)

| Claim | Status |
| --- | --- |
| Finest commercially sold optical satellite imagery is the **30 cm class** (multiple constellations) | sourced, row 65 |
| A **25 cm** native product exists | **unverified**: one source only |
| A **10 cm** commercial provider | **not operating**: the provider lost contact with its satellite and pivoted away from selling imagery |
| Streamed photorealistic 3D city tiles | **not usable for the shipped game**: the provider's policy bars offline caching and storage (forum and secondary sources; **re-verify against the provider's current terms**) |

30 cm imagery is a **purchase**. Pricing, licence tier (internal use versus derived
works in a commercial game) and archive versus new tasking are PENDING quotes. Tasking
new captures over areas of active conflict is out of scope (ADR 0052).

## Data classes (ids used in `districts.json`)

| Class id | What it is | Licence posture | Used for |
| --- | --- | --- | --- |
| `GEO-ELEV-LIDAR-1M-PUBLIC` | US national lidar programme: 1 m DEM + point clouds | public domain (row 65) | terrain: Silver Springs, Alexandria, Washington |
| `GEO-IMG-AERIAL-60CM-PUBLIC` | US national agricultural aerial orthoimagery, 60 cm | public domain; **Virginia/DC vintage PENDING** | land cover and layout reference: US districts |
| `GEO-ELEV-30M-OPEN` | Global 30 m DEM (European programme) | free worldwide licence with attribution; **commercial-use clause to re-read in the licence PDF** | terrain: Kingston, Dubai, Addis Ababa, Asmara |
| `GEO-IMG-SAT-10M-OPEN` | European 10 m multispectral satellite imagery | free, full and open; the attribution notice "Contains modified Copernicus Sentinel data [year]" is required | land cover fallback: all districts |
| `GEO-IMG-SAT-30CM-PURCHASE` | Commercial 30 cm class optical imagery | **purchase + licence PENDING** | layout reference where no public aerial imagery exists |
| `GEO-VEC-OPEN-MAP` | Community open map data (roads, blocks, water) | ODbL. The game counts as a "produced work": attribution goes in credits or a splash screen, the source data is supplied on request, and raw data is never redistributed in bulk | road graph and block layout seed for all districts |

Names of vendors and agencies appear only here and in the ledger. `data/` uses only
the neutral class ids, which keeps content files clean under wall W2.

## Pipeline (to run on the developer's machine, not in this repo)

1. **Acquire.** Download each class per district into a local `geodata/` store,
   outside git, and record source, date and licence in a manifest. No rasters are
   committed to this repository.
2. **Process.** Reproject to a local metric grid. Convert DEMs to 16-bit heightmaps
   (GDAL). Classify land cover from the 10 m imagery.
3. **Import.** Use Unreal Landscape import with a 16-bit heightmap, plus the
   Georeferencing plugin for a world origin per district. Community reports say this
   is more reliable offline than tile-streaming plugins. The **exact UE 5.5 import
   route is PENDING** a test: the third-party GeoTIFF tools found target other engine
   versions.
4. **Author.**
   - Generate the road graph and block footprints from the open map vectors, then
     **re-author** them: fictional names, original buildings, no real businesses.
   - Use 30 cm imagery only as an artist's reference layer, never as a texture that
     ships.
5. **Bake.** Bake to Nanite meshes, landscape layers and World Partition cells, inside
   the per-tier budgets of ADR 0013 and ADR 0050.
6. **Credit.** Add every attribution the licences require to the credits screen.

## Fidelity by district (honest)

| District | Terrain | Layout reference |
| --- | --- | --- |
| Silver Springs, Alexandria, Washington | 1 m lidar (public) | 60 cm aerial (public) |
| Kingston, Dubai, Addis Ababa, Asmara | 30 m DEM (open), hand-sculpted detail | 10 m open satellite; 30 cm if purchased |

The US districts get much better free data. The other four reach the same quality only
if 30 cm imagery is bought, plus artist sculpting.

## Rejected

- **Streaming map tiles at runtime:** breaks rule §0 and the provider's caching terms.
- **Shipping raw satellite imagery as textures:** carries real brands, plates and
  homes (four walls) and licence risk.
- **Committing raster data to git:** size, and licence redistribution limits.
