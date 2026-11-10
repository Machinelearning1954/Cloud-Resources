#!/usr/bin/env bash
# ============================================================================
# repo_doctor.sh — Neon Harbor: Kingston (NH-REBUILD-1)
#
# WALL SCOPE (binding): the scan applies to CONTENT files — data/registries/**.json,
# tools/**, Source/**. docs/** and README.md are DISCUSSION files: they carry
# evidence context (ledger, ADRs) by design (ADR 0049) and are scanned only by
# their own L/P/S checks, never by the walls.
#
# 26 checks. Exit 1 on any failure. Honest count — the original 90-check doctor
# was lost with the tree (see docs/RECOVERY_LOG.md).
# ============================================================================
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
PASS=0; FAIL=0; FAILED_IDS=""

say()  { printf '%s\n' "$*"; }
ok()   { PASS=$((PASS+1)); say "  PASS  $1"; }
bad()  { FAIL=$((FAIL+1)); FAILED_IDS="$FAILED_IDS $1"; say "  FAIL  $1 — $2"; }

# ---- wall scan helper -------------------------------------------------------
scan_content() { # $1=label $2=extended-regex
  # --exclude=repo_doctor.sh: this script contains the wall patterns themselves;
  # without the exclusion it would trip on its own law.
  if grep -rqiE --include='*.json' --exclude=repo_doctor.sh -- "$2" data/ 2>/dev/null \
     || grep -rqiE --exclude=repo_doctor.sh -- "$2" tools/ Source/ 2>/dev/null; then
    bad "$1" "banned term found in content files: /$2/"
  else
    ok "$1"
  fi
}

say "== WALLS (content files: data/, tools/, Source/) =="
scan_content "W1 real-person likeness"  '\b(trump|rihanna|drake|usain|vybz kartel|andrew holness|sean paul|lebron)\b'
scan_content "W2 real/R* brands"        '\b(rockstar|glock|beretta|sig sauer|remington|bmw|mercedes|toyota|honda|nissan|audi|ferrari|lamborghini|porsche|tesla|joby|lilium|scotiabank|digicel|jutc)\b'
scan_content "W3 R* canon & trade dress" '(grand theft auto|gta[ 0-9]|vice city|leonida|ammu-nation|jack of hearts|pendeyhos|mount kalaga|jason duval|lucia caminos|cal hammons)'
scan_content "W4 real institutions"     '\b(fbi|cia|dea|atf|pentagon|homeland security)\b'

say "== LEDGER =="
L=docs/RUMOR_LEDGER.md
grep -q "Row 60" "$L" && ok "L1 row 60 present"        || bad "L1" "row 60 missing"
grep -q "Row 61" "$L" && ok "L2 row 61 present"        || bad "L2" "row 61 missing"
grep -q "Row 62" "$L" && ok "L3 row 62 present"        || bad "L3" "row 62 missing"
grep -q "PENDING" "$L" && grep -q "Row 54" "$L" && grep -q "Row 59" "$L" \
  && ok "L4 pending queue intact (54, 59)"             || bad "L4" "pending queue broken"
grep -q "RECONSTRUCTED-LOSSY" "$L" && ok "L5 lossy notice present" || bad "L5" "rows 1-58 lossy marker missing"

say "== REGISTRY INTEGRITY =="
python3 - "$ROOT" "$TMP" <<'PY'
import json, os, sys
root, out = sys.argv[1], sys.argv[2]
R = os.path.join(root, "data/registries")
res = []
def ok(i, m): res.append(("PASS", i, m))
def bad(i, m): res.append(("FAIL", i, m))
try:
    files = ["districts.json","amenities.json","species.json","vehicles.json",
             "services.json","bounties.json","crews.json"]
    data = {f: json.load(open(os.path.join(R, f))) for f in files}
    ok("R1", "all 7 registry JSON files parse")

    dis = {d["id"] for d in data["districts.json"]["districts"]}
    am  = data["amenities.json"]["amenities"]
    am_ids = {a["id"] for a in am}
    sp  = {s["id"] for s in data["species.json"]["species"]}

    bad_dis = sorted({a["district"] for a in am} - dis)
    ok("R2", "every amenity district exists") if not bad_dis else \
        bad("R2", f"unknown districts: {bad_dis}")

    ref_err = []
    for s in data["species.json"]["species"]:
        for a in s.get("consumed_by", []):
            if a not in am_ids: ref_err.append(f"{s['id']}→{a}")
    for v in data["vehicles.json"]["vehicles"]:
        for a in v.get("consumed_by", []):
            if a not in am_ids: ref_err.append(f"{v['id']}→{a}")
    for b in data["bounties.json"]["bounties"]:
        if b["board"] not in am_ids: ref_err.append(f"{b['id']}→board")
        if b["target_species"] not in sp: ref_err.append(f"{b['id']}→species")
    ok("R3", "species/vehicle/bounty cross-refs resolve") if not ref_err else \
        bad("R3", f"broken refs: {ref_err}")

    by_dis = {}
    for a in am: by_dis.setdefault(a["district"], []).append(a["id"])
    mism = [d["id"] for d in data["districts.json"]["districts"]
            if sorted(d["amenity_ids"]) != sorted(by_dis.get(d["id"], []))]
    ok("R4", "district↔amenity mapping bidirectional") if not mism else \
        bad("R4", f"mismatch: {mism}")

    ok("R5", "amenity count == 29 (27+2)") if len(am) == 29 else \
        bad("R5", f"count={len(am)}")
    ok("R6", "all amenity names unique") if len({a["name"] for a in am}) == len(am) else \
        bad("R6", "duplicate venue names")

    gate_bad = [a["id"] for a in am if a.get("gate") == "flight-streaming"
                and a["milestone"] != "M3.5"]
    ok("R7", "flight-gated rows sit at M3.5") if not gate_bad else bad("R7", f"{gate_bad}")
    miss_ev = [a["id"] for a in am if not a.get("evidence_source")]
    ok("A2", "every amenity carries evidence_source") if not miss_ev else bad("A2", f"{miss_ev}")
except Exception as e:
    bad("R1", f"registry parse/integrity: {e}")
with open(out, "w") as fh:
    for st, i, m in res: fh.write(f"{st}\t{i}\t{m}\n")
PY
while IFS=$'\t' read -r st i m; do [ "$st" = "PASS" ] && ok "$i" || bad "$i" "$m"; done < "$TMP"
: > "$TMP"

say "== POLICY =="
grep -rqiE --include='*.json' --exclude=repo_doctor.sh 'dual[- ]currenc|two[- ]currenc' data/ \
  && bad "P1" "dual-currency language in content" || ok "P1 single-currency guard"
if grep -rqiE --exclude=repo_doctor.sh 'loadlibrary|\.dll' data/ tools/ Source/ 2>/dev/null; then
  bad "P2" "DLL/loadlibrary term in content" ; else ok "P2 DLL ban"; fi
grep -q "bFlightEnabled=False" docs/ADR/ADR-0013-tier-budgets.md \
  && ok "P3 flight gate contract present"     || bad "P3" "flight gate contract missing"
grep -qiE "refus|no wasm" docs/ADR/ADR-0047-world-browser.md \
  && ok "P4 WASM refusal stands"              || bad "P4" "WASM decision missing"

say "== AMENITY ADR CROSS-REF =="
python3 - "$ROOT" "$TMP" <<'PY'
import json, os, sys
root, out = sys.argv[1], sys.argv[2]
am = json.load(open(os.path.join(root, "data/registries/amenities.json")))["amenities"]
adr = open(os.path.join(root, "docs/ADR/ADR-0049-amenity-parity.md")).read()
missing = [a["id"] for a in am if a["id"] not in adr]
with open(out, "w") as fh:
    if missing: fh.write(f"FAIL\tA3\tamenity ids missing from ADR 0049: {missing}\n")
    else:       fh.write("PASS\tA3\tall amenity ids appear in ADR 0049\n")
    if "branch" in adr.lower(): fh.write("PASS\tA4\tbranch venues documented in ADR\n")
    else:                       fh.write("FAIL\tA4\tbranch venues not documented\n")
PY
while IFS=$'\t' read -r st i m; do [ "$st" = "PASS" ] && ok "$i" || bad "$i" "$m"; done < "$TMP"

say "== SOURCE HONESTY =="
if find Source -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.Build.cs' -o -name '*.uproject' \) | grep -q .; then
  bad "S1" "fabricated code found in Source/ — not allowed in REBUILD" ; else ok "S1 no fabricated source"; fi
grep -q "NOT RECOVERED" Source/README.md && ok "S2 source marker honest" || bad "S2" "Source marker missing"

say "== AGENT CONTRACT =="
[ -f CLAUDE.md ] && grep -q "repo_doctor.sh" CLAUDE.md \
  && ok "C1 CLAUDE.md agent contract present" || bad "C1" "CLAUDE.md missing or broken"

say ""
say "Doctor summary: $PASS passed, $FAIL failed (of $((PASS+FAIL)) checks)."
if [ "$FAIL" -gt 0 ]; then
  say "FAILED:$FAILED_IDS"
fi
say "Doctor green ≠ code green: policy-checked, not linker-checked."
exit $([ "$FAIL" -gt 0 ] && echo 1 || echo 0)
