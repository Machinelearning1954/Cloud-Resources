# ADR 0052 — Seven Districts

**Status:** ADOPTED (district list) · city anchors for Ethiopia and Eritrea PROPOSED ·
**Date:** 2026-11-10 · **Directive:** ledger row 65 · **Amends:** ADR 0008

## Decision

ADR 0008 had four districts. This ADR expands the world to the seven places the user
named:

| id | District | Real-world anchor | Status |
| --- | --- | --- | --- |
| `kingston` | Kingston | Kingston, Jamaica | existing |
| `silver-springs` | Silver Spring | Silver Spring, Montgomery County, **Maryland** | existing; state corrected by the user (ledger row 66) |
| `alexandria` | Alexandria | Alexandria, Virginia | existing |
| `dubai` | Dubai | Dubai, UAE | existing (user re-confirmed) |
| `washington-dc` | Washington | Washington, D.C. | **new** |
| `ethiopia` | Addis Ababa | Addis Ababa, Ethiopia | **new**, city PROPOSED |
| `eritrea` | Asmara | Asmara, Eritrea | **new**, city PROPOSED |

**Silver Spring correction (2026-11-10, ledger row 66):** the user confirmed they mean
Silver Spring, **Maryland**. Earlier text in this repo said "Silver Springs, Virginia",
and row 65 had proposed a Fairfax County pin; both are superseded. The district id
stays `silver-springs` so existing references hold.

The user named the countries Ethiopia and Eritrea. Their capitals were picked as
**proposed** anchors, and the district ids stay country-level so the anchor city can
change without breaking references.

## Rules that are new or stricter

1. **Ethiopia and Eritrea: civilian city life only.**
   - As of mid-October 2026 the two countries had severed diplomatic ties, amid
     escalating fighting in northern Ethiopia (ledger row 65, with dated sources).
   - The districts may show markets, cafes, music, transport, food and daily life.
   - They may **not** show the armed conflict, either military, the border, political
     figures or parties, or any mission, rivalry or faction that sets one district
     against the other.
   - Travel between the two districts goes through the world map, never over a
     depicted border.
   - Shipping these districts at all is the user's call. Review it again before any
     public reveal.
2. **Washington, D.C.: no government targets.**
   - No federal building, monument or government site is a venue, an interactive
     interior or a mission target.
   - The monumental core appears only as distant skyline.
   - Gameplay lives in the neighbourhoods. The institutions wall (W4) applies with
     extra care, because the doctor's regex covers only six terms.
3. **The amenity count stays fixed.** The three new districts ship with
   `amenity_ids: []`. ADR 0049's 29-row contract (doctor check R5) is unchanged; new
   venues need a future ADR and a doctor update.
4. **No real businesses, people or brands** in any district (ADR 0008, the four walls).

## Consequences

- `districts.json` holds 7 rows, each with a `geodata` block whose class ids are
  defined in ADR 0053, and a `content_policy` field where needed.
- The density emphasis (ADR 0044) is unchanged, since amenities are still counted
  from the registry.
- ADR 0008 is kept as reconstructed, with a dated note pointing here.
