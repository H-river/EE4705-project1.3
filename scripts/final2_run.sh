#!/bin/bash
# Final2 e2e run on final50 (all trials, or a comma-separated subset).
# usage: scripts/final2_run.sh <name> [ids|ALL] [jobs=4]
# Cache off, thinking off, per-trial kill at 2700 s; appends the calls of
# every trial record on disk to runs/final2/CALLS.txt.
set -u
NAME=${1:?run name}
IDS=${2:-ALL}
JOBS=${3:-4}
REPO=/home/jiamo/EE4705/project1.3
cd "$REPO" || exit 2
source ~/.ee4705_env
export EE4705_QWEN_THINKING=${EE4705_QWEN_THINKING:-false}
OUT="$REPO/runs/final2/$NAME"
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT" >&2; exit 2; fi
ONLY=()
if [ "$IDS" != "ALL" ]; then ONLY=(--only "$IDS"); fi
"$REPO/.venv/bin/python" -m eval.runner --mode e2e --trials "$REPO/eval/trials/final50" \
    --jobs "$JOBS" --no-cache "${ONLY[@]}" --out "$OUT" > "$REPO/runs/final2/$NAME.log" 2>&1
"$REPO/.venv/bin/python" -m eval.final_table "$OUT" --trials "$REPO/eval/trials/final50" | tee -a "$REPO/runs/final2/$NAME.log"
"$REPO/.venv/bin/python" "$REPO/scripts/final2_tools.py" calls "$OUT" "$NAME" | tee -a "$REPO/runs/final2/$NAME.log"
