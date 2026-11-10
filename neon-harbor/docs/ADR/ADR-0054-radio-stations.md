# ADR 0054 — In-Game Radio: Reggae, Dancehall, Afrobeats (Offline, Simulated Live)

**Status:** ADOPTED (design) · music catalogue PENDING (0 tracks cleared) ·
**Date:** 2026-11-10 · **Directive:** ledger row 66

## Context

The user asked for radio stations connected to the latest live Jamaican reggae,
dancehall and Afrobeats, "possibly using YouTube". Two things block a literal live
connection:

- **Rule §0, no online services.** The shipped game never streams audio or video.
- **YouTube's developer policies** bar separating the audio from a video, and
  playing a video while its window is closed or minimised. A car radio needs audio
  with no video on screen, which developers generally read as covered by those
  rules. The sources are in ledger row 66, and the provider's current terms need a
  re-read before anyone revisits this.

So the game cannot play a real live stream. What it can do is feel live.

## Decision

1. **Three original stations**, stored under `radio_stations` in `services.json`:

   | id | Station | Genre | Host |
   | --- | --- | --- | --- |
   | `RS-REGGAE-01` | Blue Mountain Roots Radio | reggae | none yet |
   | `RS-DANCEHALL-01` | Zinc Roof Live | dancehall | Omari 'Bassline' Grant |
   | `RS-AFROBEATS-01` | Afro Current | Afrobeats | none yet |

2. **Simulated live.**
   - Each station runs a 24-hour show schedule driven by world time. This is the same
     rule as the in-game TV (ADR 0048): what is on air depends on the clock, never on
     save data.
   - The late-night dancehall slot is presented as a live set from Zinc Roof
     Soundhall (NH-AM-04), with crowd noise and DJ talk-over recorded offline.
3. **All audio ships inside the game**, baked and offline.
4. **"Latest" music arrives by game update**, not by stream. A content patch can add
   newly cleared tracks. Patching is not runtime web access, so rule §0 holds.
5. **Each station declares 24 track slots and 0 cleared tracks.** No real song title
   or artist name enters `data/` until its licences are signed (wall W1/W2).

## Getting real music (PENDING — this costs money)

- A commercial song needs **two licences**: a sync licence from the publisher for the
  composition, and a master-use licence from the label for the recording.
- Licences are negotiated per use, platform, territory and **time period**. A deal
  that expires can force a game off sale, so ask for perpetual terms.
- A game licence does not cover streamers. Add a "streamer mode" that mutes licensed
  tracks.
- **Cheaper route:** commission original reggae, dancehall and Afrobeats from working
  artists, with both rights assigned or licensed in one contract. This also fits the
  original-IP rule best.
- DJ and host lines in Patois or Amharic follow ADR 0051: written by native speakers.

## Consequences

- The registry count stays at 7, since the stations live inside `services.json`.
- The doctor does not validate station references. They were checked by a one-off
  script on 2026-11-10.
- Station names were web-searched for existing stations (ledger row 66). A search
  lowers the risk of a clash but is not trademark clearance. Show names and dial
  numbers were not screened.

## Rejected

- **Streaming a live radio or video feed at runtime:** breaks rule §0 and the
  provider's terms.
- **Listing real current hits in the registry now:** unlicensed names in content
  files break the walls and would be an invented catalogue (rule 7).
