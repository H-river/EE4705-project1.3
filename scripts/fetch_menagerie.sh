#!/usr/bin/env bash
# Owner: backbone (ALL)
# Fetch the unitree_g1 and robotiq_2f85 directories of MuJoCo Menagerie at the pinned
# revision (assets/menagerie_revision.txt) into assets/menagerie/.
# Re-runnable; the checkout is git-ignored and never modified locally
# (all project adaptations live in assets/g1/, see assets/README.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/assets/menagerie"
REV="$(tr -d '[:space:]' < "$ROOT/assets/menagerie_revision.txt")"
URL="https://github.com/google-deepmind/mujoco_menagerie"
if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout "$URL" "$DEST"
fi
git -C "$DEST" sparse-checkout set unitree_g1 robotiq_2f85
git -C "$DEST" fetch -q origin "$REV"
git -C "$DEST" checkout -q "$REV"
echo "menagerie unitree_g1 at $(git -C "$DEST" rev-parse HEAD)"
ls "$DEST/unitree_g1"
