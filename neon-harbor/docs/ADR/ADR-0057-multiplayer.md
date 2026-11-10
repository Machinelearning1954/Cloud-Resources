# ADR 0057 — Multiplayer: Opt-In, Server-Authoritative, Budgeted

**Status:** PROPOSED (design and budgets) — nothing built, nothing measured ·
**Date:** 2026-11-10 · **Directive:** ledger row 74 · **Amends:** rule §0

## Context

The user asked to "make it multi-player" and to train, test and debug it for "no
glitches, no lagging, using top tier tech".

- **Rule §0** said the game never goes online. Multiplayer needs a network, so the
  rule is amended, narrowly (below).
- **Nothing can be play-tested yet.** There is no game build, so no session, server
  or connection exists to test. What can be tested today is whether the design's
  numbers are consistent, and that is now automated.
- **"No lag" is not achievable for any online game.** Network delay is physics.
  What a game controls is whether the player *feels* it. The targets below are
  written that way.

## Decision 1 — Rule §0 amendment

> **Single-player never connects to anything.** Multiplayer is a separate, opt-in
> mode. In it, the game talks only to the game's own session servers and the
> platform's sign-in and matchmaking. It still never streams media, never fetches
> web content, and never loads remote code or assets.

The radio (ADR 0054) and geodata (ADR 0053) rules are unchanged.

## Decision 2 — Modes

| Mode | Players | Notes |
| --- | --- | --- |
| **Co-op story** | 2–4 | Play "Two Tides" (ADR 0056) together. The host's save owns the story meters; guests keep their own character progress and earnings. |
| **Free-roam session** | up to 16 (8 on phones) | One district per session: races, venue activities, property visits. |

- Phones (T2, T3) join sessions of up to 8. Content a tier cannot run is gated
  (ADR 0013), so cross-tier sessions use the lowest common content set.
- No mode frames one district against another, and the Ethiopia and Eritrea
  districts get co-operative activities only (ADR 0052).
- **Economy:** single currency only (ledger row 0057). Nothing is sold for real
  money, which also removes the main incentive for cheating and scams.

## Decision 3 — Technology: proven over newest

| Layer | Choice | Why |
| --- | --- | --- |
| Authority | **Dedicated servers**, server-authoritative | The server decides every outcome; clients cannot cheat by editing their own game. |
| Replication | Unreal's **standard replication** system | The newer Iris system is still labelled experimental (ledger row 74). |
| Player movement | **Character Movement Component** with built-in client prediction | The newer Mover system is experimental and has been reported incompatible with Iris. Revisit both at M3. |
| Abilities and combat | Gameplay Ability System prediction (M1 already plans GAS) | One prediction model for abilities, hits and items. |
| Hits | Server-side lag compensation (rewind) | A hit that looked right on your screen counts, within a capped window. |
| Other players | Interpolation buffer | Smooth motion from 20 updates a second. |
| Relevancy | Distance and district based | Each client only receives what is near it. |
| Crowds | Simulated on each client from a shared seed | Ambient pedestrians are not sent over the network at all. |
| Hosting | Regional server fleets, match players to the nearest region | Keeps round-trip time low. Provider and cost PENDING. |

"Top tier" here means the stack that large shipped games rely on. The newest systems
are not adopted until they leave experimental status.

## Decision 4 — Budgets (data in `services.json → multiplayer`)

| Budget | Value |
| --- | --- |
| Server tick | 30 Hz |
| Per-client download | 179 of 256 kbps on PC tiers; 88 of 128 kbps on phones |
| Target round trip (ping) | ≤ 80 ms |
| Worst playable ping | 150 ms |
| Felt delay on your own actions | 34 ms (predicted locally, so the network is not in it) |
| Delay before you see another player's action, worst case | ≈ 192 ms |
| Packet loss tolerated | 5% |

These are design budgets. Every one is PENDING until measured on real servers.

**Design history:** the first draft sent players at 30 updates a second and
replicated 60 pedestrians. The checker showed it needed 397 kbps against a 256 kbps
budget, and it failed. The second draft lowers update rates, shrinks the payloads,
and stops replicating ambient crowds.

## Decision 5 — Testing

**Today (automated, in CI):** `tools/check_net_budget.py`, 13 arithmetic checks on
the budgets above. It is not a network test.

**When a build exists (PENDING, in this order):**

1. **Two-client local test** in the editor, with network emulation set to the
   worst playable ping and 5% packet loss.
2. **Bot soak test:** 16 scripted players in one session for 8 hours on a dedicated
   server; no crash, no memory growth, server frame time stays inside 33 ms.
3. **Desync test:** bots repeat every mission's turning points (ADR 0019); client
   and server state must match at each checkpoint.
4. **Regional ping survey** to place servers so most players reach the 80 ms target.
5. **Closed player test** before any public release, with crash and lag reporting
   that the player opts into.

"Train" has no meaning for this game's netcode: nothing here is a learned model.

## Safety and operations

- Voice and text chat need moderation tools, mute, block and report from day one,
  plus age-appropriate defaults.
- Player accounts mean personal data: a privacy policy and regional compliance are
  required before launch (PENDING, needs legal review).
- Dedicated servers are a running cost for as long as the mode exists.

## Consequences

- `CLAUDE.md` rule 6 is amended to the wording above.
- `services.json` gains a `multiplayer` block; the registry count stays at 7.
- CI runs the net budget check alongside the doctor and the engine config check.
- Save codec v2 needs a clear split between host-owned story state and each
  player's own progress. That goes into the M0 design.
