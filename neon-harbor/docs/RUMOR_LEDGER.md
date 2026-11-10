# RUMOR_LEDGER — Neon Harbor: Kingston

Intake machine for every external claim, link, screenshot, "leak", and brand encounter.
Six close categories: **ADOPT** (into architecture), **DEDUPE** (already handled),
**DEFER** (parked with a condition), **GOSSIP** (no action), **REFERENCE** (evidence only),
**POLICY** (becomes a rule). Four walls, enforced by the doctor on every run:
real-person likeness / real brands / R* canon characters / real institutions.

> **NH-REBUILD-1 note:** rows 1–58 below are a **lossy reconstruction** from the project
> log after sandbox recycle (see `RECOVERY_LOG.md`). Each row is marked
> `[RECONSTRUCTED-LOSSY]`; verbatim text and per-row evidence were not recoverable.
> Rows 60–62 are live, full, and evidence-linked. Rows 54 and 59 remain **PENDING**.

## Rows 1–58 — lossy summary block

| Row | Class | Summary | Close | Tool deposited |
| --- | --- | --- | --- | --- |
| 0001–0009 | rumor wave | Early GTA6 rumor screenshots, map-size claims, release-date rumors | GOSSIP / REFERENCE | intake template |
| 0010 | fake API | "GTA6 API" service offering scraped data | DEFER (never consume online services) | §0 no-online-services rule |
| 0014 | AI forgery | AI-generated "leak" images with artifact trails | GOSSIP | AI-leak forensics checklist (ADR 0014) |
| 0034 | real person | Screenshot featuring a real public figure's likeness | POLICY | real-person likeness wall |
| 0037 | real person | Second likeness incident, different feed | DEDUPE | wall enforced in doctor |
| 0039 | R* canon | R* canon character name surfaced in fan content | POLICY | canon-character wall (Jason/Lucia/…) |
| 0043 | real institution | Real institution branding in propaganda-style intake | POLICY | institutions wall + scan |
| 0044 | brand | Real firearms/automotive/eVTOL brands in fan renders | POLICY | trademark scan |
| 0054 | link | Link ingest, content never reached the ledger | **PENDING** | — |
| 0057 | currency claim | "Dual-currency economy" claim vs single-currency design | POLICY | single-currency guard |
| 0065 | DLL claim | "Install this DLL to import content" style intake | POLICY | DLL ban in doctor |
| 0083 | political figure | Political-figure imagery in a rumor collage | POLICY | political-figures scan |
| 0085 | institution | Real-bank / real-agency marks in mock screenshots | POLICY | institutions scan |
| 0059 | link | Instagram (borntoseephoto) "integrate the videos decode and embedd" — no browsing capability; needs pasted content. "Decode and embed" of video would fail §0 if runtime/online; acceptable only as dev-time STUDY material | **PENDING** | — |

(All unlisted rows in 1–58 were closed GOSSIP/REFERENCE/DEFER without architectural
impact; their verbatim entries were lost with the tree. The machine's rules — the six
categories, the four walls, and the tool list above — were fully recovered and remain
binding.)

## Row 60 — "GTA6 upload that just released" — evidence pass

- **Class:** REFERENCE (parity evidence for ADR 0049)
- **Claim as received:** user: "all the amenities from gta six upload that just released."
- **Verified timeline (web, 2026-11-10):**
  - GTA VI launch date confirmed **2026-11-19** — Rockstar Newswire, Nov 6 2025:
    https://www.rockstargames.com/newswire/article/ak3ak31a49a221/grand-theft-auto-vi-is-now-set-to-launch-november-19-2026
  - **Launch Trailer** — Oct 15, 2026 (Newswire listing, original score by Justice, Oct 29):
    https://www.rockstargames.com/newswire/article/9k2kaa1o3297k9/grand-theft-auto-vi-an-extended-look
  - ***An Extended Look*** ~30 min gameplay — premiered Aug 27, 2026 (Newswire, Aug 6 2026).
  - **Launch Gallery** — Nov 5, 2026; **Pre-Load begins today, Nov 10, 2026**
    (Newswire, Nov 3 2026).
- **Amenity evidence:**
  - IGN, Aug 27 2026 — 18 side activities:
    https://www.ign.com/articles/heres-every-gta-6-side-activity-weve-spotted-so-far
  - Red Bull, Nov 6 2025 — Trailer 2 side activities (incl. boat customisation,
    litter-picking, hunting, MMA cage):
    https://www.redbull.com/us-en/gta-6-side-activities
- **R* features page `rockstargames.com/GTAVIfeatures` returned 404 on fetch** — not used
  as a source.
- **Close:** REFERENCE → feeds row 61. **No R* text, names, trade dress, or assets enter
  the repo** — only activity *categories*, each translated per ADR 0049.

## Row 61 — User directive: full amenity parity, original IP

- **Class:** ADOPT
- **Received:** "add everything I'm missing … keep it original to make it marketable."
- **Interpretation:** parity on *category and density*, never on content. Every evidenced
  GTA 6 amenity category gets an original Neon Harbor translation with an original venue
  brand. This is what makes the game marketable: we own every name, logo, venue and
  character; nothing to license, nothing to pull under trademark or trade-dress claims,
  and a distinct identity (Jamaica / Virginia) instead of a Vice City clone.
- **Close:** ADOPT → **ADR 0049 — GTA6 Amenity Parity Through Original Translation** +
  `data/registries/amenities.json` (27 entries).

## Row 62 — User re-affirmation of setting anchors

- **Class:** POLICY
- **Received:** "Remember, it's gta and jamaica, silver springs and alexandria, virginia."
- **Interpretation:** genre = GTA-style open world; anchors = Kingston Jamaica, Silver
  Springs, Alexandria VA. Standing ADR 0008 (four districts incl. Dubai) is **unchanged**;
  the three named districts are the user-named anchors and receive the amenity density
  emphasis in ADR 0049. Dubai district status flagged for the next user check-in, not
  silently altered.
- **Close:** POLICY → ADR 0008 note + ADR 0049 district weighting.

## Row 63 — User directive: "just like GTA 6 graphics and all 150 GB"

- **Class:** POLICY (and partial ADOPT)
- **Received (2026-11-10):** "Add what I am missing make it just like gta 6 graphics and all 150 gb"
- **Interpretation:**
  - *"Just like GTA 6 graphics"* → refused as stated. It is a likeness and trade-dress
    target, and CLAUDE.md rule 8 bans comparisons to R* titles. Adopted instead: original
    art direction with per-tier fidelity targets in **ADR 0050** (PENDING measurement).
  - *"All 150 GB"* → refused as a goal. Install size is an output of authored assets,
    produced in Unreal Editor on the developer's machine. Padding the repo is
    fabrication (rule 7).
  - *"What I am missing"* → ADOPT for the gaps this tree names itself:
    - vehicles registry 7 → 17 rows (10 NEW originals, labelled as not recovered)
    - `docs/web/` world browser (ADR 0047)
    - ADR 0050
- **Close:** POLICY → ADR 0050 · ADOPT → vehicles.json, docs/web/.

## Row 64 — User directive: "With dread heads and ethiopian woman and jamacians with waves"

- **Class:** ADOPT
- **Received (2026-11-10, verbatim):** "With dread heads and ethiopian woman and jamacians with waves"
- **Interpretation:** add original cast members with locs, Ethiopian women, and
  Jamaicans with 360 waves (the hairstyle). Respectful and skilled roles, with hairstyle
  locked as identity on every tier.
- **Evidence for the setting fit:** the DC region (including Alexandria and Silver
  Spring, MD) is widely described as home to the largest Ethiopian community in the US.
  Size estimates vary.
  - https://en.wikipedia.org/wiki/Ethiopians_in_Washington,_D.C.
  - https://ggwash.org/view/39188/dcs-little-ethiopia-has-moved-to-silver-spring-and-alexandria
- **Name screening (web search, 2026-11-10):**
  - **"Delly Ranks" (NH-REBUILD-1 row CR-PERF-02) matches a real Jamaican dancehall
    artist** (Delly Ranx, also spelled Delly Ranks):
    https://en.wikipedia.org/wiki/Delly_Ranx. The row was renamed, because wall W1
    outranks byte-fidelity.
  - The first replacement nickname, "Echo", matched two real Jamaican artists
    (Echo Minott and General Echo):
    https://www.roots-archives.com/artist/400. The final name is
    Ranford 'Lighthouse' Campbell.
  - "Skyline" was rejected because it is a Kingston music compilation series.
  - All other new nicknames, and the manifest's "Marlene 'Zinc' Pryce", returned no
    public-figure match.
  - Candidate "Errol Dixon" was dropped as a likely real musician.
  - The other new names returned no public-figure match.
- **Cultural terms (sourced):**
  - Shuruba is a traditional Ethiopian scalp-braid style:
    https://www.undp.org/index%2Ephp/ethiopia/stories/shuruba-providing-healing-and-sanctuary-women-through-beauty-braids-and-cornrows
  - Habesha names are patronymic, with no family surname:
    https://en.wikipedia.org/wiki/Habesha_naming_conventions
- **Flag for the user:** the source articles place the Ethiopian community in
  **Silver Spring, Maryland**. This project's district is "Silver Springs, Virginia
  (user designation)". The user should confirm which one was meant. Nothing has been
  changed.
- **Close:** ADOPT → ADR 0051 + `data/registries/crews.json` (8 new rows, 1 rename).

## Open PENDING queue

- **Row 54** — link, awaiting pasted content.
- **Row 59** — Instagram video embed request, awaiting pasted content; §0 review applies
  to any runtime/online interpretation.
