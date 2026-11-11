#!/usr/bin/env python3
"""Red-team harness for the Neon Harbor repository (ledger row 75).

Two parts:

  SCAN    Extra checks on the real tree that close holes found by attacking
          tools/repo_doctor.sh. The doctor itself is left unchanged at 26 checks.

  ATTACK  Copies the tree to a temp folder, sabotages the copy in one specific
          way, and runs every checker on it. An attack is "caught" when at least
          one checker fails cleanly. An attack that passes, or that only stops a
          checker by crashing it, is a hole and fails this script.

This is NOT a game, network or rendering test. It tests the project's own
guard rails.

Usage:
    python3 tools/red_team.py            scan + attack suite
    python3 tools/red_team.py --scan     scan only
Exit 0 = all good, 1 = a scan check failed or an attack got through.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = ("districts.json", "amenities.json", "species.json", "vehicles.json",
       "services.json", "bounties.json", "crews.json")
GEO_CLASSES = {"GEO-ELEV-LIDAR-1M-PUBLIC", "GEO-IMG-AERIAL-60CM-PUBLIC", "GEO-ELEV-30M-OPEN",
               "GEO-IMG-SAT-10M-OPEN", "GEO-IMG-SAT-30CM-PURCHASE", "GEO-VEC-OPEN-MAP"}
# Characters that can hide a banned word from a text search: invisible characters,
# and letters from scripts that look like Latin letters.
INVISIBLE = set(range(0x200B, 0x2010)) | set(range(0x202A, 0x202F)) | set(range(0x2060, 0x2065)) | {0xFEFF, 0x00AD}
LOOKALIKE_BLOCKS = ((0x0370, 0x03FF), (0x0400, 0x052F), (0xFF00, 0xFFEF))


def wall_patterns(root):
    """Read the wall regexes out of the doctor, so banned terms live in one place."""
    text = open(os.path.join(root, "tools", "repo_doctor.sh"), encoding="utf-8").read()
    found = re.findall(r'^scan_content\s+"([^"]+)"\s+\'(.+)\'\s*$', text, re.M)
    return [(label, re.compile(rx, re.I)) for label, rx in found]


def strings_in(node):
    """Every key and string value inside parsed JSON."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield str(k)
            yield from strings_in(v)
    elif isinstance(node, list):
        for v in node:
            yield from strings_in(v)
    elif isinstance(node, str):
        yield node


def lists_with_ids(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from lists_with_ids(v, f"{path}/{k}")
    elif isinstance(node, list):
        if node and all(isinstance(x, dict) and "id" in x for x in node):
            yield path, [x["id"] for x in node]
        for v in node:
            yield from lists_with_ids(v, path)


def scan(root):
    """Run the supplementary checks. Returns a list of (ok, name, detail)."""
    out = []

    def check(name, ok, detail=""):
        out.append((bool(ok), name, detail))

    regdir = os.path.join(root, "data", "registries")
    data, broken = {}, []
    for f in REG:
        try:
            data[f] = json.load(open(os.path.join(regdir, f), encoding="utf-8"))
        except (OSError, ValueError) as exc:
            broken.append(f"{f}: {exc}")
    check("RT1 all seven registries load", not broken, "; ".join(broken))
    if broken:
        return out

    walls = wall_patterns(root)
    check("RT2 wall patterns could be read from the doctor", len(walls) == 4, f"found {len(walls)}")

    hits = []
    for f, doc in data.items():
        for s in strings_in(doc):
            folded = unicodedata.normalize("NFKC", s)
            for label, rx in walls:
                if rx.search(s) or rx.search(folded):
                    hits.append(f"{f}: {label}")
    check("RT3 walls hold on decoded JSON text (escapes cannot hide a term)", not hits, "; ".join(sorted(set(hits))))

    raw_hits, odd = [], []
    doctor_rel = os.path.join("tools", "repo_doctor.sh")
    for base in ("data", "Config", "tools", "Source"):
        for dirpath, dirs, files in os.walk(os.path.join(root, base)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                p = os.path.join(dirpath, name)
                rel = os.path.relpath(p, root)
                if name.endswith(".pyc"):
                    continue
                try:
                    text = open(p, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    odd.append(f"{rel}: not readable as UTF-8 text")
                    continue
                # Only the doctor itself may hold the wall terms: exempt that exact
                # path, never a filename (a look-alike name elsewhere is not exempt).
                if rel != doctor_rel:
                    for label, rx in walls:
                        if rx.search(text):
                            raw_hits.append(f"{rel}: {label}")
                for ch in set(text):
                    cp = ord(ch)
                    if cp in INVISIBLE or any(lo <= cp <= hi for lo, hi in LOOKALIKE_BLOCKS):
                        odd.append(f"{rel}: U+{cp:04X}")
    # The same characters can also be written as JSON escapes, which the raw text
    # scan above cannot see, so check the decoded strings too.
    for f, doc in data.items():
        for s in strings_in(doc):
            for ch in set(s):
                cp = ord(ch)
                if cp in INVISIBLE or any(lo <= cp <= hi for lo, hi in LOOKALIKE_BLOCKS):
                    odd.append(f"data/registries/{f}: U+{cp:04X} (decoded)")
    check("RT4 walls hold on every file in data/, Config/, tools/ and Source/, whatever its type", not raw_hits,
          "; ".join(sorted(set(raw_hits))))
    check("RT5 no invisible or look-alike characters in any content folder", not odd, "; ".join(sorted(set(odd))))

    extra = sorted(set(os.listdir(regdir)) - set(REG))
    check("RT6 registries folder holds exactly the seven known files", not extra, ", ".join(extra))

    dupes = []
    for f, doc in data.items():
        for path, ids in lists_with_ids(doc):
            seen = set()
            for i in ids:
                if i in seen:
                    dupes.append(f"{f}{path}: {i}")
                seen.add(i)
    check("RT7 ids are unique inside every registry list", not dupes, "; ".join(dupes))

    refs = []
    try:
        districts = {d["id"] for d in data["districts.json"]["districts"]}
        amenities = {a["id"] for a in data["amenities.json"]["amenities"]}
        vehicles = {v["id"] for v in data["vehicles.json"]["vehicles"]}
        crews_doc = data["crews.json"]
        crews = {c["id"] for c in crews_doc["crews"]}
        hair = {h["id"] for h in crews_doc.get("hair_styles", [])}
        for c in crews_doc["crews"]:
            if c.get("venue") not in amenities:
                refs.append(f"{c['id']} venue {c.get('venue')}")
            if "district" in c and c["district"] not in districts:
                refs.append(f"{c['id']} district {c['district']}")
            style = c.get("appearance", {}).get("hair_style")
            if style is not None and style not in hair:
                refs.append(f"{c['id']} hair_style {style}")
        services = data["services.json"]
        for s in services.get("services", []):
            if s.get("district") not in districts:
                refs.append(f"{s['id']} district {s.get('district')}")
            for target in s.get("consumes", []):
                if target not in amenities | vehicles:
                    refs.append(f"{s['id']} consumes {target}")
        for r in services.get("radio_stations", []):
            if r.get("host_crew") is not None and r["host_crew"] not in crews:
                refs.append(f"{r['id']} host_crew {r['host_crew']}")
            for d in r.get("districts", []):
                if d not in districts:
                    refs.append(f"{r['id']} district {d}")
        for d in data["districts.json"]["districts"]:
            geo = d.get("geodata", {})
            for key in ("elevation", "imagery_primary", "imagery_fallback", "vectors"):
                if geo.get(key) not in GEO_CLASSES:
                    refs.append(f"{d['id']} geodata.{key} {geo.get(key)}")
    except (KeyError, TypeError, AttributeError) as exc:
        refs.append(f"registry shape is wrong: {exc!r}")
    check("RT8 crew, radio, service and geodata references all resolve", not refs, "; ".join(refs))
    return out


# --------------------------------------------------------------------------
# Attack suite
# --------------------------------------------------------------------------

def edit_json(tree, name, fn):
    p = os.path.join(tree, "data", "registries", name)
    doc = json.load(open(p, encoding="utf-8"))
    fn(doc)
    json.dump(doc, open(p, "w", encoding="utf-8"), indent=2)


def edit_text(tree, rel, fn):
    p = os.path.join(tree, rel)
    text = open(p, encoding="utf-8").read()
    open(p, "w", encoding="utf-8").write(fn(text))


def write(tree, rel, text):
    open(os.path.join(tree, rel), "w", encoding="utf-8").write(text)


def build_attacks(term):
    """`term` is one banned word, read from the doctor at run time."""
    mp = lambda d: d["multiplayer"]                     # noqa: E731
    veh0 = lambda d: d["vehicles"][0]                   # noqa: E731
    escaped = "".join(f"\\u{ord(c):04x}" for c in term)
    lookalike = term.replace("o", chr(0x043E)).replace("e", chr(0x0435)).replace("a", chr(0x0430))
    if lookalike == term:
        lookalike = chr(0x0430) + term
    ini = os.path.join("Config", "DefaultEngine.ini")
    veh = os.path.join("data", "registries", "vehicles.json")

    def duplicate_amenity(t):
        edit_json(t, "amenities.json", lambda d: d["amenities"][1].update(id=d["amenities"][0]["id"]))

        def fix_district(d):
            ids = d["districts"][0]["amenity_ids"]
            ids[ids.index("NH-AM-02")] = "NH-AM-01"
        edit_json(t, "districts.json", fix_district)

    def thousand_players(t):
        def f(d):
            mp(d)["modes"][1]["max_players"] = 1000
            for c in mp(d)["tier_caps"]:
                c["max_session_players"] = 1000
        edit_json(t, "services.json", f)

    A = [
        # --- content walls ---
        ("walls", "banned word in a vehicle name", lambda t: edit_json(t, "vehicles.json", lambda d: veh0(d).update(name=term.title() + " GT"))),
        ("walls", "banned word hidden as JSON unicode escapes", lambda t: edit_text(t, veh, lambda s: s.replace('"Spray Board 500"', '"' + escaped + '"'))),
        ("walls", "banned word spelled with look-alike letters", lambda t: edit_json(t, "vehicles.json", lambda d: veh0(d).update(name=lookalike))),
        ("walls", "banned word split by an invisible character", lambda t: edit_json(t, "vehicles.json", lambda d: veh0(d).update(name=term[:2] + chr(0x200B) + term[2:]))),
        ("walls", "banned word in a non-JSON file under data/", lambda t: write(t, os.path.join("data", "registries", "notes.txt"), term)),
        ("walls", "banned word in Config/", lambda t: edit_text(t, ini, lambda s: s + f"; partner: {term}\n")),
        ("walls", "banned word used as a JSON key", lambda t: edit_json(t, "vehicles.json", lambda d: veh0(d).update({term: 1}))),
        # --- registry integrity ---
        ("registry", "an eighth registry file", lambda t: write(t, os.path.join("data", "registries", "extra.json"), "{}")),
        ("registry", "duplicate amenity id", duplicate_amenity),
        ("registry", "crew at a venue that does not exist", lambda t: edit_json(t, "crews.json", lambda d: d["crews"][2].update(venue="NH-AM-99"))),
        ("registry", "crew with a hairstyle that does not exist", lambda t: edit_json(t, "crews.json", lambda d: d["crews"][2]["appearance"].update(hair_style="HAIR-NOPE"))),
        ("registry", "radio station in a district that does not exist", lambda t: edit_json(t, "services.json", lambda d: d["radio_stations"][0]["districts"].append("atlantis"))),
        ("registry", "district with an unknown geodata class", lambda t: edit_json(t, "districts.json", lambda d: d["districts"][0]["geodata"].update(elevation="GEO-MADE-UP"))),
        ("registry", "registry file that is not valid JSON", lambda t: write(t, os.path.join("data", "registries", "species.json"), "{not json")),
        ("registry", "amenity moved to an unknown district", lambda t: edit_json(t, "amenities.json", lambda d: d["amenities"][0].update(district="atlantis"))),
        ("registry", "thirtieth amenity added", lambda t: edit_json(t, "amenities.json", lambda d: d["amenities"].append(dict(d["amenities"][0], id="NH-AM-30", name="Extra")))),
        ("registry", "source code faked into Source/", lambda t: write(t, os.path.join("Source", "Fake.cpp"), "// original\n")),
        ("walls", "look-alike banned word in Source/README.md", lambda t: edit_text(t, os.path.join("Source", "README.md"), lambda s: s + lookalike + "\n")),
        ("walls", "banned word in Source/ in a file named like the doctor", lambda t: write(t, os.path.join("Source", "repo_doctor.sh"), term + "\n")),
        ("walls", "invisible-split banned word in a tools/ file", lambda t: write(t, os.path.join("tools", "notes.txt"), term[:2] + "\u200b" + term[2:] + "\n")),
        # --- multiplayer budget ---
        ("net", "negative object count to hide bandwidth", lambda t: edit_json(t, "services.json", lambda d: mp(d)["replication_classes"][0]["relevant_count"].update(T0=-500))),
        ("net", "zero-byte updates", lambda t: edit_json(t, "services.json", lambda d: [c.update(bytes_per_update=0) for c in mp(d)["replication_classes"]])),
        ("net", "delete the pedestrian class", lambda t: edit_json(t, "services.json", lambda d: mp(d).update(replication_classes=[c for c in mp(d)["replication_classes"] if c["id"] != "NET-NPC"]))),
        ("net", "raise the bandwidth budget to 1 Gbps", lambda t: edit_json(t, "services.json", lambda d: [c.update(downstream_budget_kbps=10 ** 6) for c in mp(d)["tier_caps"]])),
        ("net", "raise the allowed delay limits", lambda t: edit_json(t, "services.json", lambda d: mp(d)["latency_budget_ms"].update(felt_delay_limit_local=5000, seen_delay_remote_limit=60000, max_playable_round_trip=9000))),
        ("net", "server tick of zero", lambda t: edit_json(t, "services.json", lambda d: mp(d).update(server_tick_hz=0))),
        ("net", "player update rate of zero", lambda t: edit_json(t, "services.json", lambda d: mp(d)["replication_classes"][0].update(update_hz=0))),
        ("net", "budget given as text", lambda t: edit_json(t, "services.json", lambda d: mp(d)["tier_caps"][0].update(downstream_budget_kbps="lots"))),
        ("net", "latency block removed", lambda t: edit_json(t, "services.json", lambda d: mp(d).pop("latency_budget_ms"))),
        ("net", "duplicate tier row hiding a bad one", lambda t: edit_json(t, "services.json", lambda d: mp(d)["tier_caps"].insert(0, {"tier": "T0", "max_session_players": 16, "downstream_budget_kbps": 1, "render_fps": 60}))),
        ("net", "1000 players per session", thousand_players),
        ("net", "under-declare how many players a client must track", lambda t: edit_json(t, "services.json", lambda d: mp(d)["replication_classes"][0]["relevant_count"].update(T0=1, T1=1))),
        ("net", "double the player update size", lambda t: edit_json(t, "services.json", lambda d: mp(d)["replication_classes"][0].update(bytes_per_update=96))),
        ("net", "clients decide outcomes instead of the server", lambda t: edit_json(t, "services.json", lambda d: mp(d).update(authority="client_authoritative"))),
        ("net", "whole multiplayer block removed", lambda t: edit_json(t, "services.json", lambda d: d.pop("multiplayer"))),
        # --- engine config ---
        ("engine", "experimental setting in a second ini file", lambda t: write(t, os.path.join("Config", "DefaultGame.ini"), "[/Script/Engine.RendererSettings]\nr.MegaLights.EnableForProject=True\n")),
        ("engine", "unverified setting", lambda t: edit_text(t, ini, lambda s: s.replace("r.AntiAliasingMethod=4", "r.AntiAliasingMethod=4\nr.Invented.Setting=1"))),
        ("engine", "experimental setting in different letter case", lambda t: edit_text(t, ini, lambda s: s.replace(";r.MegaLights.EnableForProject=True", "R.MEGALIGHTS.ENABLEFORPROJECT=True"))),
        ("engine", "ray tracing on DirectX 11", lambda t: edit_text(t, ini, lambda s: s.replace("DefaultGraphicsRHI_DX12", "DefaultGraphicsRHI_DX11"))),
        ("engine", "untested banner replaced with 'fully tested'", lambda t: edit_text(t, ini, lambda s: s.replace("RENDERING UNTESTED", "FULLY TESTED"))),
        ("engine", "same setting twice with different values", lambda t: edit_text(t, ini, lambda s: s.replace("r.RayTracing=True", "r.RayTracing=True\nr.RayTracing=False"))),
        ("engine", "settings file deleted", lambda t: os.remove(os.path.join(t, ini))),
        ("engine", "invalid value for a verified setting", lambda t: edit_text(t, ini, lambda s: s.replace("r.AntiAliasingMethod=4", "r.AntiAliasingMethod=99"))),
    ]
    return A


CHECKERS = (
    ("doctor", ["bash", os.path.join("tools", "repo_doctor.sh")]),
    ("engine", [sys.executable, os.path.join("tools", "check_engine_config.py")]),
    ("net", [sys.executable, os.path.join("tools", "check_net_budget.py")]),
    ("scan", [sys.executable, os.path.join("tools", "red_team.py"), "--scan"]),
)


def run_checkers(tree):
    """Return (names that failed cleanly, names that crashed)."""
    failed, crashed = [], []
    for name, cmd in CHECKERS:
        p = subprocess.run(cmd, cwd=tree, capture_output=True, text=True)
        if "Traceback (most recent call last)" in p.stderr:
            crashed.append(name)
        elif p.returncode != 0:
            failed.append(name)
    return failed, crashed


def attack_suite(root):
    term = None
    for label, rx in wall_patterns(root):
        if label.startswith("W2"):
            m = re.search(r"\(([^)]*)\)", rx.pattern)
            term = m.group(1).split("|")[1] if m else None
    if not term:
        print("  FAIL  could not read a banned word from the doctor to attack with")
        return 1

    base_failed, base_crashed = run_checkers(root)
    if base_failed or base_crashed:
        print(f"  FAIL  the untouched tree does not pass: {base_failed + base_crashed}")
        return 1
    print("  PASS  baseline: the untouched tree passes every checker")

    attacks = build_attacks(term)
    holes = 0
    for group, name, sabotage in attacks:
        tmp = tempfile.mkdtemp(prefix="nh-redteam-")
        tree = os.path.join(tmp, "nh")
        try:
            shutil.copytree(root, tree)
            sabotage(tree)
            failed, crashed = run_checkers(tree)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if crashed:
            holes += 1
            print(f"  HOLE    [{group}] {name} — crashed {crashed} instead of a clean failure")
        elif failed:
            print(f"  caught  [{group}] {name} — by {', '.join(failed)}")
        else:
            holes += 1
            print(f"  HOLE    [{group}] {name} — every checker passed")
    print(f"\nRed-team attacks: {len(attacks) - holes} of {len(attacks)} caught, {holes} got through.")
    return 1 if holes else 0


def main():
    results = scan(ROOT)
    bad = 0
    for ok, name, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail and not ok else ""))
        bad += not ok
    print(f"\nRed-team scan: {len(results) - bad} passed, {bad} failed (of {len(results)}).")
    if "--scan" in sys.argv[1:]:
        return 1 if bad else 0
    print("\n== ATTACK SUITE (each attack runs on a throwaway copy) ==")
    return 1 if (attack_suite(ROOT) or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
