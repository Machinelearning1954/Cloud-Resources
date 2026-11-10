#!/usr/bin/env bash
# publish.sh — Neon Harbor: Kingston (NH-REBUILD-1)
# Publishes this tree to GitHub. Requires NEON_REMOTE (git remote URL) to be set:
#   NEON_REMOTE=git@github.com:<you>/neon-harbor.git bash tools/publish.sh
# Honest note: the ORIGINAL tree was lost before its publish.sh ever ran
# (docs/RECOVERY_LOG.md). This script publishing NH-REBUILD-1 does NOT restore it —
# it gives this rebuild a remote home so a recycle never bites again.
set -eu
cd "$(dirname "$0")/.."

REMOTE="${NEON_REMOTE:-}"
if [ -z "$REMOTE" ]; then
  echo "publish: NEON_REMOTE not set." >&2
  echo "  export NEON_REMOTE=git@github.com:<you>/neon-harbor.git" >&2
  exit 1
fi

bash tools/repo_doctor.sh   # never publish red

git init -q 2>/dev/null || true
git add -A
git -c user.name="neon-harbor" -c user.email="nh@neon-harbor.local" \
    commit -q -m "NH-REBUILD-1: docs-core reconstruction + ADR 0049 amenity parity (29 amenities)" || true
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
git push -u origin HEAD
echo "publish: pushed to $REMOTE"
