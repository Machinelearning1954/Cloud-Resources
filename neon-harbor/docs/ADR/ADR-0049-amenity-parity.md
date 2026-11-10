# ADR 0049 — GTA6 Amenity Parity Through Original Translation

**Status:** ADOPTED · **Date:** 2026-11-10 · **Evidence:** ledger row 60 · **Directive:** ledger row 61

## Context

Rockstar's GTA 6 marketing has now revealed, across Trailer 2 (Nov 6 2025) and
*An Extended Look* (Aug 27 2026, ~30 min of gameplay), a large amenity/side-activity
layer. The user directive: **"add everything I'm missing … keep it original to make it
marketable."** The genre promise of this project (ADR 0044: highest system density in its
size class) requires that amenity layer — but copying R* venue names, characters, trade
dress, or content would be both illegal and self-defeating: un-ownable IP cannot be
marketed.

## Decision — translate the category, never the content

Every evidenced amenity **category** is adopted; every **name, venue, brand, character and
flavor** is original. Parity is measured in categories and density, never in content.

**Naming rules (binding for all amenity venues):**

1. Venue names are invented, with Jamaican / Chesapeake-Virginia flavor — never R* names,
   never real businesses.
2. Every venue anchors to exactly one district (ADR 0008): Kingston Jamaica, Silver
   Springs VA, Alexandria VA, Dubai. The three user-named anchors carry the density
   emphasis (row 62).
3. Every venue passes the four walls (real-person likeness / real brands / R* canon
   characters / real institutions) — enforced by the doctor on every run.
4. Every amenity declares a milestone and tier envelope (ADR 0013). Flight-dependent
   amenities inherit the M3.5 streaming gate (`bFlightEnabled=False` until validated).
5. Amenity progression may touch only the single currency (ledger row 0057 guard);
   amenity transient state never enters the save codec (ADR 0048).

## Evidence (ledger row 60)

| Source | Date | What it evidences |
| --- | --- | --- |
| [IGN — "Every GTA 6 Side Activity"](https://www.ign.com/articles/heres-every-gta-6-side-activity-weve-spotted-so-far) | 2026-08-27 | 18 activities from *Extended Look* gameplay |
| [Red Bull — "Every side activity in Trailer 2"](https://www.redbull.com/us-en/gta-6-side-activities) | 2025-11-06 | 17 activities incl. boat customisation, hunting, MMA, litter-picking |
| [Rockstar Newswire — launch date](https://www.rockstargames.com/newswire/article/ak3ak31a49a221/grand-theft-auto-vi-is-now-set-to-launch-november-19-2026) | 2025-11-06 | launch 2026-11-19 (pre-load began 2026-11-10) |
| Rockstar Newswire — Extended Look announcement | 2026-08-06 | gameplay showcase exists (evidence class for the above) |

R* venue names seen in evidence (Jack of Hearts, Ammu-Nation, Pendeyhos, Mount Kalaga,
Leonida …) are **wall terms**: banned from all content files. They appear only in
evidence context in docs and in the doctor's banned-term list.

## The Matrix — 27 original translations

Categories are the deduplicated union of both evidence sources. IDs feed
`data/registries/amenities.json` (authoritative).

| ID | Evidenced category | Neon Harbor translation (original) | District | Hook | Milestone | Tier |
| --- | --- | --- | --- | --- | --- | --- |
| NH-AM-01 | Fishing | Cane Bay Charters | Kingston | species registry (fail-safe) | M1 | T0–T3 |
| NH-AM-02 | Gym: weights, punching bag | Iron Dutty Fitness | Kingston | focus (transient, ADR 0048) | M1 | T0–T3 |
| NH-AM-03 | Strip club | The Silk Circuit | Kingston | age-gate; nightlife economy | M1 | T0–T3 |
| NH-AM-04 | Nightclub / dancing | Zinc Roof Soundhall | Kingston | crew registry performers | M1 | T0–T3 |
| NH-AM-05 | Bar drinking | Duppy Rum Bar · Jerk Pan Yard | Kingston | single currency | M1 | T0–T3 |
| NH-AM-06 | Pool / billiards | Rack Room (Duppy annex) | Kingston | minigame loop | M1 | T0–T3 |
| NH-AM-07 | Mini golf | Crazy Cane Links | Silver Springs | minigame loop | M2 | T0–T3 |
| NH-AM-08 | Basketball | Back-A-Wall Courts | Kingston | pickup 3v3 | M2 | T0–T3 |
| NH-AM-09 | Scuba diving | Cay Divers' Shack | Kingston cays | species registry | M2 | T0–T3 (shore variant T2/T3) |
| NH-AM-10 | Cave diving | Cockpit Sinkholes | Kingston interior | underwater streaming | M3.5 gate | T0 only |
| NH-AM-11 | Kayaking | Mangrove Run Tours | Kingston wetlands | watercraft class | M2 | T0–T3 |
| NH-AM-12 | Jet skiing | Spray Rentals | Silver Springs | vehicle registry (NoiseRating) | M2 | T0–T3 |
| NH-AM-13 | Airboating | Glassbottom & Airboat Co. | Silver Springs | wetlands streaming | M2 | T0–T1 |
| NH-AM-14 | Hunting / trapping | Backwood Bounty Board | Blue Mountain fringe | bounty registry; invasive-only | M2 | T0–T3 |
| NH-AM-15 | Dirt bike racing | Dirt Dutty Series | Blue Mountain fringe | vehicle registry | M2 | T0–T3 |
| NH-AM-16 | Rally racing | Bauxite Rally Circuit | Kingston interior | vehicle registry; 4-car grid on T2/T3 | M2 | T0–T3 |
| NH-AM-17 | Cage fighting (MMA) | Chain-Link Cage | Kingston waterfront | betting (single currency) | M2 | T0–T3 |
| NH-AM-18 | Backyard wrestling | Yard Brawl Circuit | Alexandria suburbs | schedule-driven events | M2 | T0–T3 |
| NH-AM-19 | Shooting range | Leadworks Range | Alexandria | fictional firearms only | M1 | T0–T3 |
| NH-AM-20 | Base-jump / parachute | Blue Mountain Jump Point | Blue Mountain fringe | flight gate M3.5 | M3.5 gate | T0–T1 |
| NH-AM-21 | Sunbathing | Two Fish Point | Kingston shore | passive; all tiers | M1 | T0–T3 |
| NH-AM-22 | Studying animals | Harbour Field Guide | all districts | species registry photography | M1 | T0–T3 |
| NH-AM-23 | Litter-picking / community service | Court-Ordered Cleanup | Kingston streets | probation mechanic | M1 | T0–T3 |
| NH-AM-24 | Watching TV | NH-Cast channels | all districts | ADR 0048 transient | M2 | T0–T2 |
| NH-AM-25 | Boat customisation | Keelwright Boatworks | Kingston slipway | vehicle registry watercraft | M2 | T0–T2 |
| NH-AM-26 | Music artist management | Riddim Ranch Studio | Kingston | crew registry | M2 | T0–T2 |
| NH-AM-27 | Electric scooters | Zippy Rentals | Alexandria · Dubai | micro-mobility vehicle class | M2 | T0–T3 |

**District weighting (row 62):** Kingston 21 amenities (incl. the three all-district
venues: Back-A-Wall Courts, Harbour Field Guide, NH-Cast), Silver Springs 3, Alexandria 3,
Dubai 2 branch venues (Zinc Roof DXB — M3, Leadworks Gulf annex — M3) so the fourth
district is not a shell; Zippy Rentals scooters additionally spawn on the Dubai promenade.
Branch venues are listed as registry rows `NH-AM-28`/`NH-AM-29` with milestone M3.

## Why "original" is the marketable choice

- **Ownable:** every venue name, logo, gang, brand and character is ours — nothing to
  license, nothing that can be pulled under trademark or trade-dress claims.
- **Distinct:** a Jamaica/Northern-Virginia identity is a shelf-visible difference from
  Vice City; a clone is not.
- **Safe to ship:** the four walls make every storefront and platform review a checklist,
  not a lawsuit.
- **Density is the promise** (ADR 0044): category parity delivers what players compare;
  originality is what lets us keep it.

## Budget impact (ADR 0013)

- Amenity simulation runs in a fixed tick budget per tier; venues above budget are
  **gated, never degraded** — mobile tiers hide NH-AM-13/20/10 instead of simulating
  shells.
- Underwater + wetlands + air content share one streaming validation pipeline
  (M3.5 gate) — one gate, three amenities.
- No amenity adds save-codec fields; progression is single-currency numbers only.

## Enforcement

Doctor (this tree, 26 checks) adds amenity checks: registry entry count = 27 + 2 branches;
every row has district ∈ districts registry, milestone ∈ {M1, M2, M3, M3.5}, tier envelope,
four-walls clean name; content-file brand scan includes all R* evidence terms.

## Alternatives rejected

1. **1:1 content clone** — illegal, unmarketable, and it forfeits the density claim's
   credibility (a clone is measured against its source forever).
2. **Ignore amenities, ship "story + driving"** — breaks ADR 0044's density promise in the
   genre players actually compare against.
3. **License R*-adjacent brands** — impossible (R* licenses nothing) and pointless: the
   differentiation *is* the originality.
