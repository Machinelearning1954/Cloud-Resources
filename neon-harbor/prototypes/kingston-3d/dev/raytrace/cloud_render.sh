#!/usr/bin/env bash
# Ray-trace the block on a rented or free cloud machine in one go: installs what it needs into its own
# folder, exports the scene from the built page, and renders it. Meant for a notebook service with a
# graphics card (the notebook next to this file runs it); also runs on a plain CPU.
#   bash dev/raytrace/cloud_render.sh <out-dir>
# Settings (environment): NH_DEVICE gpu|cpu (default gpu), NH_SIZE (1080x1920), NH_SAMPLES (256),
#   NH_VIEWS (hero,quay,aerial), NH_PYTHON (a Python 3.13 with bpy and playwright already installed: skips
#   the install), NH_CHROMIUM (a Chromium executable: skips the browser download).
# The Blender package needs exactly Python 3.13, so this does not rely on the machine's own Python.
set -euo pipefail
OUT="${1:?usage: cloud_render.sh <out-dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"; K3D="$(cd "$HERE/../.." && pwd)"
WORK="$(mktemp -d)"; mkdir -p "$OUT"
PY="${NH_PYTHON:-}"
if [ -z "$PY" ]; then
  python3 -m pip install -q uv
  python3 -m uv venv --python 3.13 "$WORK/venv"
  python3 -m uv pip install -q --python "$WORK/venv/bin/python" "bpy==5.3.0" playwright
  PY="$WORK/venv/bin/python"
fi
if [ -z "${NH_CHROMIUM:-}" ]; then "$PY" -m playwright install --with-deps chromium; fi
cd "$K3D"
"$PY" dev/raytrace/export_scene.py index.html "$WORK/scene-gold.json"  --dusk 0.42 --delivered 3 --at 1010,345
"$PY" dev/raytrace/export_scene.py index.html "$WORK/scene-night.json" --dusk 0.97 --delivered 4 --at 1010,345
DEV="${NH_DEVICE:-gpu}"; SIZE="${NH_SIZE:-1080x1920}"; SAMPLES="${NH_SAMPLES:-256}"; VIEWS="${NH_VIEWS:-hero,quay,aerial}"
"$PY" dev/raytrace/render.py "$WORK/scene-gold.json"  "$OUT/gold"  --device "$DEV" --samples "$SAMPLES" --size "$SIZE" --views "$VIEWS" --sun-from 238 --sun-height 24
"$PY" dev/raytrace/render.py "$WORK/scene-night.json" "$OUT/night" --device "$DEV" --samples "$SAMPLES" --size "$SIZE" --views hero,game
echo "renders are in $OUT"
