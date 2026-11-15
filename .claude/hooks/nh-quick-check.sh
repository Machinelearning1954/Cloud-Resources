#!/usr/bin/env bash
# PostToolUse hook for Edit and Write: after a file under neon-harbor/ is changed, run the quick
# project checks that apply to it and tell Claude at once if one fails.
#   - quiet and exit 0 when everything passes or the file is outside neon-harbor/
#   - exit 2 with the failing lines on stderr when a check fails (Claude sees stderr on exit 2)
# It only reads the tree and runs the project's own checkers. It changes nothing.
set -u
root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
nh="$root/neon-harbor"
[ -d "$nh" ] || exit 0
command -v python3 >/dev/null 2>&1 || exit 0

file="$(python3 -c 'import json,sys
try: print((json.load(sys.stdin).get("tool_input") or {}).get("file_path") or "")
except Exception: print("")' 2>/dev/null)"
case "$file" in "$nh"/*) ;; *) exit 0 ;; esac
rel="${file#"$nh"/}"
case "$rel" in prototypes/*/dev/out/*) exit 0 ;; esac          # scratch output

fail=""
run() { # run <label> <command...>: keep the last lines of output if it fails
  local label="$1" out; shift
  if ! out="$("$@" 2>&1)"; then fail="${fail}--- ${label} failed ---
$(printf '%s\n' "$out" | grep -E 'FAIL|Error|error|STALE|summary|passed' | tail -n 12)
"; fi
}

cd "$nh" || exit 0
run "policy doctor" bash tools/repo_doctor.sh
run "red-team scan" python3 tools/red_team.py --scan
case "$rel" in
  Config/*) run "engine config check" python3 tools/check_engine_config.py ;;
  data/registries/services.json) run "net budget check" python3 tools/check_net_budget.py ;;
  prototypes/kingston-3d/src/*|prototypes/kingston-3d/test/*)
    if command -v node >/dev/null 2>&1; then
      for t in sim save lock; do [ -f "prototypes/kingston-3d/test/$t.test.js" ] && run "3D prototype $t tests" node "prototypes/kingston-3d/test/$t.test.js"; done
    fi ;;
esac

[ -z "$fail" ] && exit 0
printf 'neon-harbor quick checks failed after editing %s\n%s' "$rel" "$fail" >&2
exit 2
