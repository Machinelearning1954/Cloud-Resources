#!/usr/bin/env python3
"""Arithmetic check for the multiplayer budgets in data/registries/services.json.

This is NOT a network test: no game, server or connection is involved. It checks
that the PROPOSED design numbers (ADR 0057) are consistent with each other, so a
design that cannot fit its own bandwidth or delay budget fails before anyone
builds it.

Usage: python3 tools/check_net_budget.py [path-to-services.json]
Exit 0 = pass, 1 = fail.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "data", "registries", "services.json")
HEADER_BYTES = 28          # IPv4 + UDP header per packet
SAFETY = 0.75              # use at most 75% of a tier's bandwidth budget
TIERS = ("T0", "T1", "T2", "T3")
CLASSES = ("NET-PLAYER", "NET-VEHICLE", "NET-NPC", "NET-WORLD")
# Hard ceilings. A design may be stricter than these, never looser, so nobody can
# pass the check by quietly raising a limit (red-team finding, ledger row 75).
MAX_BUDGET_KBPS = 1024
MAX_PLAYERS = 64
LIMITS_MS = {"felt_delay_limit_local": 50, "seen_delay_remote_limit": 300,
             "max_playable_round_trip": 200, "target_round_trip": 100}
LATENCY_KEYS = ("target_round_trip", "max_playable_round_trip", "input_sampling", "server_tick",
                "interpolation_buffer", "render_frame", "felt_delay_local_actions",
                "felt_delay_limit_local", "seen_delay_remote_limit")


def num(v, lo, hi, whole=False):
    """True if v is a real number (not bool/text) within [lo, hi]."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    if whole and not isinstance(v, int):
        return False
    return lo <= v <= hi


def validate(mp):
    """Return a list of problems with the shape and ranges of the input."""
    bad = []
    if not isinstance(mp, dict):
        return ["multiplayer block is not an object"]
    if not num(mp.get("server_tick_hz"), 10, 120, whole=True):
        bad.append("server_tick_hz must be a whole number from 10 to 120")
    tick = mp.get("server_tick_hz") if num(mp.get("server_tick_hz"), 10, 120, whole=True) else 120

    modes = mp.get("modes")
    if not isinstance(modes, list) or not modes:
        bad.append("modes must be a non-empty list")
    else:
        for m in modes:
            if not isinstance(m, dict) or not num(m.get("max_players"), 2, MAX_PLAYERS, whole=True):
                bad.append(f"mode max_players must be a whole number from 2 to {MAX_PLAYERS}")

    caps = mp.get("tier_caps")
    cap_by_tier = {}
    if not isinstance(caps, list):
        bad.append("tier_caps must be a list")
    else:
        names = [c.get("tier") if isinstance(c, dict) else None for c in caps]
        if sorted(map(str, names)) != sorted(TIERS):
            bad.append(f"tier_caps must list each of {list(TIERS)} exactly once (got {names})")
        for c in caps:
            if not isinstance(c, dict):
                continue
            t = c.get("tier")
            if not num(c.get("max_session_players"), 2, MAX_PLAYERS, whole=True):
                bad.append(f"{t}: max_session_players must be a whole number from 2 to {MAX_PLAYERS}")
            if not num(c.get("downstream_budget_kbps"), 32, MAX_BUDGET_KBPS):
                bad.append(f"{t}: downstream_budget_kbps must be from 32 to {MAX_BUDGET_KBPS}")
            if c.get("render_fps") not in (30, 60, 120):
                bad.append(f"{t}: render_fps must be 30, 60 or 120")
            cap_by_tier[t] = c

    classes = mp.get("replication_classes")
    if not isinstance(classes, list):
        bad.append("replication_classes must be a list")
    else:
        ids = [c.get("id") if isinstance(c, dict) else None for c in classes]
        if sorted(map(str, ids)) != sorted(CLASSES):
            bad.append(f"replication_classes must be exactly {list(CLASSES)} (got {ids})")
        for c in classes:
            if not isinstance(c, dict):
                continue
            cid = c.get("id")
            if not num(c.get("update_hz"), 1, tick):
                bad.append(f"{cid}: update_hz must be from 1 to the server tick")
            if not num(c.get("bytes_per_update"), 8, 512, whole=True):
                bad.append(f"{cid}: bytes_per_update must be a whole number from 8 to 512")
            rc = c.get("relevant_count")
            if not isinstance(rc, dict) or sorted(rc) != sorted(TIERS):
                bad.append(f"{cid}: relevant_count needs exactly {list(TIERS)}")
                continue
            for t in TIERS:
                if not num(rc[t], 1, 10000, whole=True):
                    bad.append(f"{cid}: relevant_count[{t}] must be a whole number of at least 1")
                cap = cap_by_tier.get(t, {}).get("max_session_players")
                if cid == "NET-PLAYER" and isinstance(cap, int) and rc[t] != cap - 1:
                    bad.append(f"NET-PLAYER: relevant_count[{t}] must be the worst case, "
                               f"{cap - 1} (session cap minus yourself), not {rc[t]}")

    lat = mp.get("latency_budget_ms")
    if not isinstance(lat, dict):
        bad.append("latency_budget_ms block is missing")
    else:
        for k in LATENCY_KEYS:
            if not num(lat.get(k), 1, 10000):
                bad.append(f"latency_budget_ms.{k} must be a positive number")
        for k, ceiling in LIMITS_MS.items():
            if num(lat.get(k), 1, 10000) and lat[k] > ceiling:
                bad.append(f"latency_budget_ms.{k} may not exceed {ceiling} ms (got {lat[k]})")
    if not isinstance(mp.get("status"), str) or not isinstance(mp.get("authority"), str):
        bad.append("status and authority must be text")
    return bad


def downstream_kbps(mp, tier):
    """Bandwidth one client receives: payload for every relevant object + packet headers."""
    payload = sum(c["relevant_count"][tier] * c["update_hz"] * c["bytes_per_update"]
                  for c in mp["replication_classes"])
    headers = mp["server_tick_hz"] * HEADER_BYTES   # one packet per server tick
    return (payload + headers) * 8 / 1000.0


def run(path):
    out = []

    def check(name, ok, detail=""):
        out.append((bool(ok), name, detail))

    def report():
        failed = sum(not ok for ok, _, _ in out)
        for ok, name, detail in out:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail and not ok else ""))
        return failed

    try:
        mp = json.load(open(path, encoding="utf-8")).get("multiplayer")
    except (OSError, ValueError, AttributeError) as exc:
        print(f"  FAIL  N0 file loads as JSON — {exc}")
        return 1
    problems = validate(mp)
    check("N0 inputs are well-formed and inside the hard ceilings", not problems, "; ".join(problems))
    if problems:
        report()
        print("\nNet budget check: stopped at N0 — fix the inputs before the arithmetic can run.")
        return 1

    caps = {t["tier"]: t for t in mp["tier_caps"]}
    check("N1 all four tiers have a cap", set(caps) == set(TIERS), str(sorted(caps)))

    over = []
    for t in TIERS:
        if t not in caps:
            continue
        used, budget = downstream_kbps(mp, t), caps[t]["downstream_budget_kbps"] * SAFETY
        if used > budget:
            over.append(f"{t}: {used:.0f} kbps > {budget:.0f} kbps")
    check("N2 per-client bandwidth fits 75% of each tier's budget", not over, "; ".join(over))

    tick = mp["server_tick_hz"]
    fast = [c["id"] for c in mp["replication_classes"] if c["update_hz"] > tick]
    check("N3 no class updates faster than the server tick", not fast, str(fast))

    biggest = max(m["max_players"] for m in mp["modes"])
    small = [t for t in TIERS if t in caps and caps[t]["max_session_players"] > biggest]
    check("N4 no tier allows more players than the largest mode", not small, str(small))

    seen = []
    for t in TIERS:
        if t not in caps:
            continue
        others = caps[t]["max_session_players"] - 1
        players = next(c for c in mp["replication_classes"] if c["id"] == "NET-PLAYER")
        if players["relevant_count"][t] > others:
            seen.append(f"{t}: {players['relevant_count'][t]} > {others}")
    check("N5 a client never tracks more players than can be in its session", not seen, "; ".join(seen))

    lat = mp["latency_budget_ms"]
    check("N6 server tick time matches the tick rate",
          abs(lat["server_tick"] - 1000 / tick) <= 1, f"{lat['server_tick']} vs {1000 / tick:.1f}")
    fps = min(c["render_fps"] for c in caps.values())
    check("N7 frame time matches the frame-rate target",
          abs(lat["render_frame"] - 1000 / fps) <= 1, f"{lat['render_frame']} vs {1000 / fps:.1f}")
    local = lat["input_sampling"] + lat["render_frame"]
    check("N8 own actions are predicted: felt delay excludes the network",
          lat["felt_delay_local_actions"] == local and local <= lat["felt_delay_limit_local"],
          f"{local} ms vs limit {lat['felt_delay_limit_local']}")
    remote = lat["max_playable_round_trip"] / 2 + lat["server_tick"] + lat["interpolation_buffer"] + lat["render_frame"]
    check("N9 other players are seen within the limit at the worst playable ping",
          remote <= lat["seen_delay_remote_limit"], f"{remote:.0f} ms vs {lat['seen_delay_remote_limit']}")
    slowest = min(c["update_hz"] for c in mp["replication_classes"] if c["id"] in ("NET-PLAYER", "NET-VEHICLE"))
    check("N10 interpolation buffer covers two updates of the slowest moving class",
          lat["interpolation_buffer"] >= 2 * 1000 / slowest * 0.66, f"{lat['interpolation_buffer']} ms, {slowest} Hz")
    check("N11 target ping is below the playable limit",
          lat["target_round_trip"] < lat["max_playable_round_trip"])
    check("N12 authority is the dedicated server", mp["authority"] == "dedicated_server_authoritative")
    check("N13 status says the numbers are proposed, not measured", "PROPOSED" in mp["status"])

    failed = report()
    for t in TIERS:
        if t in caps:
            print(f"  info  {t}: {downstream_kbps(mp, t):.0f} kbps used of {caps[t]['downstream_budget_kbps']} kbps")
    print(f"\nNet budget check: {len(out) - failed} passed, {failed} failed (of {len(out)}). "
          "Arithmetic only — not a network test.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
