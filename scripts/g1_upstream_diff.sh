#!/usr/bin/env bash
# Owner: backbone (ALL)
# Regenerate assets/g1_upstream_diff.patch: every local change of the G1
# model relative to the pinned upstream Menagerie file.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
diff -u "$ROOT/assets/menagerie/unitree_g1/g1.xml" "$ROOT/assets/g1_2f85_ee4705.xml" \
  --label "upstream/unitree_g1/g1.xml@$(cat "$ROOT/assets/menagerie_revision.txt")" \
  --label "assets/g1_2f85_ee4705.xml" > "$ROOT/assets/g1_upstream_diff.patch" || { status=$?; test "$status" -eq 1 || exit "$status"; }
echo "wrote $ROOT/assets/g1_upstream_diff.patch ($(wc -l < "$ROOT/assets/g1_upstream_diff.patch") lines)"
