#!/bin/bash
# Final 50-trial e2e run.  usage: scripts/final_run.sh <N> [jobs=4]
# Cache disabled; per-trial kill at 2700 s (1.5 x the 1800 s trial cap).
# Extra env (e.g. EE4705_QWEN_THINKING=false) is inherited from the caller.
set -u
N=${1:?run number}
JOBS=${2:-4}
REPO=/home/jiamo/EE4705/project1.3
cd "$REPO" || exit 2
source ~/.ee4705_env
OUT="$REPO/runs/final/run_$N"
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT" >&2; exit 2; fi
"$REPO/.venv/bin/python" -m eval.runner --mode e2e --trials "$REPO/eval/trials/final50" \
    --jobs "$JOBS" --no-cache --out "$OUT" > "$REPO/runs/final/run_$N.log" 2>&1
"$REPO/.venv/bin/python" -m eval.final_table "$OUT" --trials "$REPO/eval/trials/final50" | tee -a "$REPO/runs/final/run_$N.log"
