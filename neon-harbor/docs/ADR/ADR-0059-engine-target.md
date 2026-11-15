# ADR 0059 — Move the Engine Target from Unreal 5.5 to 5.8

**Status:** PROPOSED — **awaiting the user's yes or no** · **Date:** 2026-11-15 ·
**Directive:** ledger row 78

## Context

This project has targeted Unreal Engine 5.5 since its first manifest. 5.5 was released
in November 2024. The current release is **5.8** (17 June 2026; hotfix 5.8.3 in
September 2026). The gap was noticed while reading the Unreal reference inside the
Claude Code Game Studios template. The headline facts were then checked through web
search summaries of Epic's announcement and release notes (ledger row 78). Epic's own
site is blocked in this session, so the pages themselves were not read.

No engine code exists in this repo (`Source/` is empty), so changing the target costs
nothing here. See Risks for the one exception: a 5.5 code tree may still exist on the
user's own machine.

## What 5.8 changes for this project

**Verified by search:** the 5.8 release date, MegaLights production-ready, Lumen Lite
in beta, Iris production-ready for licensees. **From the template only, UNVERIFIED:**
the Substrate row, the 5.6 open-world row, the mobile rendering row and the 5.8.3
hotfix. Check those against Epic's release notes before relying on them.

| Area | In 5.5 (current target) | In 5.8 | Effect here |
| --- | --- | --- | --- |
| Many ray-traced lights (MegaLights) | Experimental | **Production-ready**, aimed at 60 fps on current consoles | The night scenes in Kingston and Dubai rely on many lights. ADR 0050 could enable it for T0 and Ultra instead of hiding it behind a toggle. |
| Cheaper global illumination (Lumen Lite) | Not available | **Beta**: about twice as fast as high-quality Lumen | A candidate for the laptop tier (T1), which must hold 60 fps. Beta, so not a commitment. |
| Large open worlds at 60 Hz | — | Hardware ray-traced Lumen performance work landed in 5.6 | Directly serves ADR 0050's 60 fps targets. |
| Materials (Substrate) | Beta | Production-ready since 5.7, on by default for new projects | Better layered materials: wet zinc, car paint, skin. |
| Network replication (Iris) | Experimental | **Production-ready** (for licensees) | ADR 0057 chose the older replication system because Iris was experimental. That reason would no longer hold and the choice should be revisited. |
| Mobile rendering | Forward by default | Multi-pass deferred by default | Affects the phone tiers (T2, T3); needs its own look. |

One outlet reports that 5.8 is the last planned major UE5 release, with Unreal
Engine 6 in early access targeted for late 2027. That is a single secondary source,
UNVERIFIED. If true, 5.8 would be a long-lived base; do not plan on it yet.

## Decision (proposed)

Target **Unreal Engine 5.8** (latest 5.8.x hotfix) instead of 5.5.

If accepted, these follow, each as its own small change:

1. `CLAUDE.md` and `README.md`: "UE 5.5" becomes "UE 5.8".
2. ADR 0050: MegaLights moves from "experimental, behind a toggle" to on for T0 and
   Ultra; Lumen Lite is evaluated for T1; the table is re-checked line by line.
3. `Config/DefaultEngine.ini` and `tools/check_engine_config.py`: every setting is
   re-verified against 5.8, and the MegaLights line may be switched on.
4. ADR 0057: the replication choice (standard versus Iris) is re-decided.
5. ADR 0053: the import route is tested on 5.8, not 5.5.

## Risks

- **None of this has been run.** The claims above come from Epic's announcement and
  release notes and from the template's reference, not from an installed engine.
- Third-party plugins (upscalers, ray reconstruction) must have 5.8 builds. PENDING.
- **The lost source tree.** `docs/RECOVERY_LOG.md` says the original C++ tree may still
  exist on the user's machine, written against 5.5. If it does, moving to 5.8 means
  porting it, which is real work. Ask before deciding.
- If the user already has 5.5 installed and working, staying on it is a valid choice;
  the table above is then a list of features to do without.

## If the answer is no

The project stays on 5.5, and ADR 0050 and ADR 0057 stand as written.
