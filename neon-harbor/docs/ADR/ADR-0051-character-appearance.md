# ADR 0051 — Character Appearance: Hairstyles as Identity

**Status:** PROPOSED — tech choices and budgets PENDING measurement · **Date:** 2026-11-10 ·
**Directive:** ledger row 64

## Context

The user asked for characters with locs ("dread heads"), Ethiopian women, and Jamaicans
with 360 waves. The cast should reflect the game's real places. Kingston is a Jamaican
city, and Alexandria and the wider DC region have one of the largest Ethiopian
communities in the US (sources in ledger row 64). Textured hair is a well-known weak
spot in game rendering: locs, waves, coils and braids often end up flattened into a
generic default. This ADR makes them first-class.

## Decision

1. **Hairstyle is identity.** Each crew row names a `hair_style` from
   `crews.json → hair_styles`, and `identity_lock: true` is the default.
   - A weaker tier may render the **same** style more cheaply.
   - It must never swap the style for another one, never hide it under a hat to save
     budget, and never fall back to a generic default.
   - This follows ADR 0013's gated-not-degraded rule: identity is content.
2. **Representation rules for the crew registry:**
   - **Roles are varied and skilled.** The new cast includes a sound engineer, a master
     boatwright, a charter captain, a rally champion, a dive instructor, a fleet
     founder, a record producer and a range safety officer.
   - **Hair never signals criminality.** No archetype ties a hairstyle to gang or
     villain roles.
   - **Locs are not shorthand for religion.** No religious titles or sacred names are
     used as nicknames or decoration.
   - **Ethiopian names** follow the given name + father's given name (patronymic)
     pattern, not a Western family surname (ledger row 64 sources).
   - **Dialogue in Amharic and Jamaican Patois** is written or approved by native
     speakers. Until then it is marked PENDING; no machine-written dialect goes in.
3. **Characters are fictional.** Each new full name was web-searched against public
   figures, in grouped queries. Nicknames were searched in a Jamaican-music context.
   These checks run on top of the doctor's four-walls scan. Two collisions were found
   and fixed (ledger row 64). A web search lowers the risk of a collision but doesn't
   rule it out, so trademark/likeness clearance before ship stays PENDING.

## Rendering plan (UE 5.5) — proposal, all PENDING

| Style | T0 RTX 5060 | T1 laptop | T2 iPhone 16 | T3 Fold 6 |
| --- | --- | --- | --- | --- |
| Locs | mesh-based locs (geometry per loc) + groom flyaways | mesh locs, fewer flyaways | hair cards + normal detail | hair cards |
| 360 waves | scalp shader with a wave-pattern normal/height map + short groom | scalp shader + map | scalp shader + map | scalp shader + map |
| Coils / afro | strand groom | groom (reduced density) | cards | cards |
| Shuruba / box braids | mesh braids | mesh braids (lower LOD) | cards | cards |

- Waves are cheap on every tier because they are mostly a shader. They should look the
  same on T0 and T3.
- Head scans or likeness references only come from consenting, paid models with a
  release on file. No real public figure is scanned or referenced (wall W1).
- Use hair and cultural consultants from the communities being represented. Their
  sign-off is a milestone gate before M2 character content locks.

## Hair library (expanded 2026-11-10, ledger row 66)

`crews.json → hair_styles` now holds 29 styles, each tagged with a `heritage`
(`jamaican`, `ethiopian_eritrean` or `shared`) and a `source_status`:

- **sourced** — name and description are cited in the ledger.
- **generic** — a common barber or salon style, with no cultural claim made.
- **pending** — needs review by a stylist from that community before art is made.

The library is not a complete list of every Jamaican or Ethiopian hairstyle. It grows
by adding a row with a source. Styles tied to a specific people or region (gilbach,
mertu) are assigned only to characters of that background.

## Consequences

- `crews.json` gains `hair_styles`, per-row `appearance`, `district` and `provenance`.
  The registry count stays at 7.
- The doctor doesn't validate crew venue, district or hair references yet. They were
  checked with a one-off script on 2026-11-10. A doctor check is a candidate for a
  future revision, which would change the documented check count.
