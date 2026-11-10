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

## Row 65 — User directive: seven places + top-tier satellite imagery

- **Class:** ADOPT (districts) + POLICY (sensitivity, build-time geodata)
- **Received (2026-11-10, verbatim):** "Silver springs va alexandria va washington d.c jamacia Ethiopia ERTRIEA AND DUBAI ADD ALL THESE PLACES USE SATILLITE IMAGERY TECH TOP TIER AS OF TODAY"
- **Silver Springs, VA confirmed:** a populated place in Fairfax County, inside the
  Rose Hill CDP, at about 38.784 N, 77.129 W. It is also a neighbourhood name in the
  City of Alexandria.
  - https://virginia.hometownlocator.com/va/fairfax/silver-springs.cfm
  - https://nesbittrealty.com/guide/alexandria/s/silver-springs
  - The row-64 question about Maryland is closed: the user meant Virginia.
- **Silver Springs pin and the glass-bottom boat heritage:** the glass-bottom boat
  heritage in the district's character is from **Silver Springs, Florida**:
  https://www.floridastateparks.org/park/Silver-Springs . The Virginia pin is
  PROPOSED. Terrain mismatch is an open item in ADR 0053.
- **District character lines (sourced):**
  - Asmara: inscribed 2017 as "Asmara: A Modernist African City", with wide
    boulevards and modernist cafes:
    https://whc.unesco.org/en/urban-heritage-atlas/asmara
  - Addis Ababa: about 2,355 m elevation, Merkato open-air market, coffee-ceremony
    culture (travel-guide sources, so moderate confidence):
    https://africamap.net/capitals/addis-ababa ,
    https://www.globaltravelerusa.com/mercato-addis-ababa/
- **Ethiopia–Eritrea situation (latest found: mid-October 2026):**
  - The two countries severed diplomatic ties, amid escalating fighting in northern
    Ethiopia: https://www.aljazeera.com/news/2026/10/2/ethiopia-eritrea-break-ties-does-the-conflict-risk-becoming-regional-war
  - Each side accused the other of launching war: https://www.aljazeera.com/news/2026/10/9/ethiopia-and-eritrea-accuse-one-another-of-launching-war
  - Policy: civilian city life only, no conflict or cross-district hostility content
    (ADR 0052).
- **Satellite / geodata evidence (ADR 0053):**
  - The 30 cm class is the finest commercial optical imagery; a 25 cm claim is
    unverified: https://skyfi.com/en/blog/high-resolution-satellite-imagery ,
    https://space-solutions.airbus.com/imagery/our-optical-and-radar-satellite-imagery/pleiades-neo/
  - The 10 cm provider is not operating (lost contact, pivoted):
    https://albedo.com/post/clarity-1-what-worked-and-where-we-go-next
  - Streamed photorealistic 3D tiles: offline caching is barred per the provider's
    policy, from forum and secondary sources, so **re-verify**:
    https://mapsplatform.google.com/resources/blog/commonly-asked-questions-about-our-recently-launched-photorealistic-3d-tiles/
  - US national lidar programme products are public domain:
    https://www.usgs.gov/media/images/washington-dc-3dep-lidar-derived-dsm-time-daylight
  - US agricultural aerial imagery is public domain at 60 cm GSD (2025 vintage shown
    for one state; **Virginia/DC vintage PENDING**):
    https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery
  - Global 30 m DEM licence (the commercial-use clause still needs a re-read):
    https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf
  - Sentinel data is free, full and open, with a modified-data attribution notice:
    https://ads.atmosphere.copernicus.eu/licences/ec-sentinel
  - OSM / ODbL: a game is a produced work, and attribution plus data-on-request apply:
    https://osmfoundation.org/wiki/Licence_and_Legal_FAQ
  - Unreal offline import is more reliable through Landscape heightmaps than tile
    streaming (community reports):
    https://community.cesium.com/t/custom-heightmaps-in-unreal-engine-5/20043
- **Close:**
  - ADOPT → ADR 0052 + `districts.json` (7 rows; the 3 new rows have 0 amenities,
    R5 is unchanged)
  - POLICY → ADR 0053 (build-time only, imagery as reference, purchase PENDING)
  - No imagery was acquired in this revision.

## Row 66 — User directives: radio, Silver Spring MD, graphics question, hairstyles

- **Class:** ADOPT (radio, hair, district) + POLICY (no streaming) + answer
- **Received (2026-11-10, verbatim):**
  - "Connect radio station to latest Jamacian reggae live dance hall livealso afro beats live also possibly use youtube"
  - "Did you mean Silver Spring, Maryland yes"
  - "Did you use gta 6 level graphics?"
  - "Add all jamacian and ethiopian hairstyles"
- **Silver Spring:** the user confirmed **Maryland**. This supersedes the Virginia pin
  in row 65.
  - The place is named for a mica-flecked spring found in 1840:
    https://boundarystones.weta.org/2022/03/22/whats-name-silver-spring
  - Centre about 38.996 N, 77.028 W: https://en.wikipedia.org/wiki/Silver_Spring,_Maryland
  - It fits the row-64 evidence that the region's Ethiopian businesses concentrate
    there.
  - The district id stays `silver-springs`.
- **Radio (ADR 0054):**
  - YouTube's developer policies bar separating audio from video and background
    play: https://developers.google.com/youtube/terms/developer-policies-guide
  - A commercial song needs a sync licence and a master-use licence, and expiring
    terms can pull a game from sale:
    https://www.carltonfields.com/insights/publications/2019/getting-creative-video-games-sync-rights-wrongs ,
    https://usethirdchair.com/blog/music-licensing-for-video-games-rights-and-deal-terms
  - Station names searched ("Blue Mountain Roots Radio", "Zinc Roof Live",
    "Afro Current"): no existing station found.
  - Close: three offline, simulated-live stations; 0 tracks cleared; music PENDING.
- **Graphics question:** answered **no**. This repository holds no rendered graphics
  at all: only docs, data and a registry browser. ADR 0050 holds targets, all PENDING.
  No comparison to any other title is made (rule 8).
- **Hairstyles (ADR 0051, `crews.json → hair_styles`, 8 → 28):**
  - Ethiopian/Eritrean: albaso, shuruba variants, asa shuruba, gilbach (Tigray),
    gofere, mertu (Oromo):
    https://ethiopianbusinessreview.net/returning-to-roots/ ,
    https://ethiopedia.blogspot.com/2008/07/ethiopian-hair-styles.html ,
    https://omirenstyles.com/habesha-hairstyling-traditions/
  - Asa shuruba and shuruba worn open at the shoulders:
    https://solomon-abate.blogspot.com/2020/09/hair-style-as-expression-of-ones-culture.html
  - Jamaican: canerows (with and without beads), traditional/freeform locs,
    and dancehall statement styles:
    https://en.wikipedia.org/wiki/Cornrows ,
    https://jamaica-star.com/article/entertainment/20180328/star-fashion-dancehall-style-%E2%80%98hair%E2%80%99-stay
  - A trademarked locking-method name was found in the first draft and replaced
    with the generic "microlocs":
    https://trademarks.justia.com/745/76/sisterlocks-74576996.html
  - Bantu knots have a local Jamaican name whose origin is contested, so the
    registry uses the neutral name and leaves in-game naming to a stylist:
    https://jamaicans.com/?p=152404
  - "All" is not claimed. Sources are blogs and press, so regional attributions are
    indicative. Two styles are marked `pending` for stylist review. No Hamar or
    Afar style was added, because no usable source was found.

## Row 67 — User directive: 60 fps (and a video link)

- **Class:** ADOPT (frame rate) + POLICY (DEDUPE of row 63)
- **Received (2026-11-10):** a video link, https://youtu.be/GLZP4bhvBrs , with a
  request to "train it on Gta 6 graphics for the absolute match"; then the
  clarification, verbatim: "I mean 60 fps".
- **Link:** not opened. **PENDING**: the user is asked to describe in words what in
  the video matters.
- **Interpretation:**
  - An exact match of another title's graphics stays refused (rule 8, row 63).
    Nothing in this repo can be "trained": it holds no graphics and no model.
  - **60 fps is adopted** as the frame-rate target on all four tiers.
- **Close:** ADOPT → ADR 0050 (60 fps on T0–T3, resolution scales, content gates).
  Targets remain PENDING measurement.

## Row 68 — User directive: "highest quality RTX graphics" (screenshot)

- **Class:** ADOPT (settings class) + REFERENCE (screenshot)
- **Received (2026-11-10):** a phone screenshot of a short video, with the verbatim
  request "Get it to the highest quality rtx graphics". The overlay reads
  "RTX 5090 · 236 FPS · 4K · DLSS 4". The footage is another studio's superhero game.
  This resolves row 67's PENDING link: the user has now shown what matters in it.
- **Forensics (ADR 0014):** a third-party benchmark short, not first-hand. The frame
  rate shown is not verified, and overlays of this kind usually include generated
  frames. Used as evidence of a **settings class** only.
- **Tech status (web, 2026-11-10):**
  - DLSS 4.5 multi frame generation and an updated ray reconstruction are broadly
    available; DLSS 5 launched in autumn 2026 for RTX 50 series only, in very few
    games: https://www.nvidia.com/en-us/geforce/news/dlss-5-faster-ai-model-hogwarts-legacy-rtx-spark-laptops/ ,
    https://en.wikipedia.org/wiki/Deep_Learning_Super_Sampling
  - UE 5.5: MegaLights is experimental, the Path Tracer is production-ready, and
    Lumen can run at 60 Hz with hardware support:
    https://www.pugetsystems.com/blog/2024/11/06/unreal-engine-5-5-from-smpte-2110-to-megalights-whats-new-and-noteworthy/ ,
    https://tomlooman.com/unreal-engine-5-5-performance-highlights/
  - Ray reconstruction plugin support for UE 5.5: not confirmed, so PENDING.
- **Close:** ADOPT → ADR 0050 "RTX Ultra mode" (optional PC preset above T0, all
  targets PENDING). No art or look is taken from the referenced game.

## Open PENDING queue

- **Row 54** — link, awaiting pasted content.
- **Row 68** — RTX Ultra: ray reconstruction plugin for UE 5.5 unverified; all Ultra targets unmeasured.
- **Row 66** — radio music catalogue (0 of 72 track slots cleared); two hairstyles await stylist review; Silver Spring water venues (see ADR 0053 open items).
- **Row 65** — Ethiopia/Eritrea city anchors (Addis Ababa, Asmara) await user confirmation; 30 cm imagery purchase and licence terms await quotes.
- **Row 59** — Instagram video embed request, awaiting pasted content; §0 review applies
  to any runtime/online interpretation.
