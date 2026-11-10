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


def downstream_kbps(mp, tier):
    """Bandwidth one client receives: payload for every relevant object + packet headers."""
    payload = sum(c["relevant_count"][tier] * c["update_hz"] * c["bytes_per_update"]
                  for c in mp["replication_classes"])
    headers = mp["server_tick_hz"] * HEADER_BYTES   # one packet per server tick
    return (payload + headers) * 8 / 1000.0


def run(path):
    mp = json.load(open(path, encoding="utf-8"))["multiplayer"]
    out = []

    def check(name, ok, detail=""):
        out.append((bool(ok), name, detail))

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

    failed = 0
    for ok, name, detail in out:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail and not ok else ""))
        failed += not ok
    for t in TIERS:
        if t in caps:
            print(f"  info  {t}: {downstream_kbps(mp, t):.0f} kbps used of {caps[t]['downstream_budget_kbps']} kbps")
    print(f"\nNet budget check: {len(out) - failed} passed, {failed} failed (of {len(out)}). "
          "Arithmetic only — not a network test.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
