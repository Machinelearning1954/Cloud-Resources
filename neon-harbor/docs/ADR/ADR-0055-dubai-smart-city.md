# ADR 0055 — Dubai as a Near-Future Smart City, plus Four Systems from Reference Videos

**Status:** PROPOSED (design) — nothing built, no budgets measured · **Date:** 2026-11-10 ·
**Directive:** ledger row 72

## Context

The user supplied four short videos and asked for "futuristic smart cities in Dubai"
and for what the game is missing. The videos were reviewed frame by frame (ledger
row 72, ADR 0014 forensics). All four appear AI-generated, so they are **concept
reference only**: no frame, design, logo or likeness is taken from them.

| Video | What it shows | System it suggests |
| --- | --- | --- |
| 1 | A brawl in a phone shop; shelves and stock get smashed | Close-quarters fights in destructible interiors |
| 2 | A luxury flying car lifts off from a villa; cockpit view over a suburb | Personal flying vehicles |
| 3 | A crew turns a tiled floor into a glass-topped reef pool under a kitchen | Property renovation |
| 4 | Sea-platform homes, hologram kitchen, towers with sky traffic | The smart-city setting itself |

Video 2 shows a real carmaker's badge. That is a wall W2 term: the game's flying
vehicles get original names and original shapes.

## Decision 1 — Dubai becomes the near-future district

The other six districts stay present-day. Dubai is the one place where the game
looks ahead roughly a generation, as a fictional depiction (ADR 0008).

- **Sea-platform quarter.** Homes and promenades on pylons over shallow water, joined
  by bridges and water taxis.
- **Sky lanes.** Marked air corridors between towers, used by flying taxis and
  private flyers. Until the flight gate opens they are visible scenery only.
- **Responsive city.** Traffic lights, street lighting, shade canopies and transit
  that react to crowds, heat and time of day. These are simulated locally: rule §0
  still bans any real online service.
- **Smart interiors.** Voice-and-gesture homes: glass that tints, kitchen projections,
  walls that open to the view.
- **Light identity** (extends ADR 0050): white heat by day; at night, cool
  blue-white architectural lines on a warm dusk sky.

**Tone rule:** the technology serves daily life. It is not a surveillance dystopia
and makes no claim about the real city's government, companies or plans.

## Decision 2 — Four systems, each gated

| System | Design | Gate and rules |
| --- | --- | --- |
| **Destructible interiors** | Shop and venue interiors with breakable shelving, glass and stock. Fights leave lasting mess until the venue "resets" on a world-time schedule. | Debris is transient state (ADR 0048), never saved. Piece counts are capped per tier (ADR 0013). First built in the vertical slice (ADR 0050). |
| **Flying vehicles** | Original flyers with a cockpit view; take-off from pads at homes and towers; sky lanes in Dubai. | **Inherits the M3.5 flight gate**: `bFlightEnabled=False` until streaming validation passes. Dubai-first, then other districts. No registry rows yet: the vehicles registry stays at its 17-row contract until a later ADR extends it. |
| **Property renovation** | Buy a property, then order builds that complete over in-game days: pools, glass floors over water features, workshops, landing pads. Crews are seen working in stages. | Property state **is** save data (codec v2). Costs use the single currency (ledger row 0057). Reef and fish features follow the species registry's protected rule. |
| **Smart home** | Owned Dubai homes respond to the player: lighting scenes, tinting glass, a garage or pad that opens on approach. | Local simulation only (rule §0). Room state is transient; ownership and upgrades are saved. |

## What makes this stronger than a bigger map

Design notes for the team, offered as craft advice rather than insider knowledge:

1. **Depth beats area.** A block where every shop can be entered, fought in and
   changed is remembered longer than ten blocks of closed doors.
2. **Let the player leave a mark.** Renovation and lasting venue damage make the
   world feel owned. Most open worlds reset everything the moment you look away.
3. **Seven places, seven moods.** Travelling from a Kingston night to a near-future
   Dubai dawn is a contrast a single-region map cannot offer.
4. **Identity is a feature.** The cast, hair, music and food of this game are
   specific. Specific worlds age better than generic ones.
5. **Ship one perfect slice first.** One Kingston block at full quality proves the
   pipeline. Nothing else in this ADR starts before that slice exists.

## Budget and risk

- A dense sky city with flying traffic is the most expensive scene in the game.
  On T2 and T3 the sea-platform quarter and sky traffic are **gated**, not thinned.
- Destruction, flight and renovation are each large systems. Building all three
  before the vertical slice would stall the project; they are ordered after it.
- No real brand, real building or real person appears. Tower and vehicle designs
  must be original shapes, not recognisable landmarks or cars.

## Consequences

- `districts.json`: Dubai's character line describes the near-future setting.
- No amenity, vehicle or crew rows are added (their counts are fixed by earlier
  contracts and doctor check R5).
- ADR 0050's Dubai light identity is extended by this ADR.
