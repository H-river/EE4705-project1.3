#!/usr/bin/env bash
# Stage 8 evaluation of one selected checkpoint: full executor on <cells_full> + skill level on <cells_skill>.
#   scripts/bonus/eval_stage8.sh <label> <kind> "<cells_full>" "<cells_skill>"
set -uo pipefail
cd "$(dirname "$0")/../.."
label=$1; kind=$2; full=$3; skill=$4
JOBS=${JOBS:-3} scripts/bonus/run_stage6.sh "$label" "$kind" "$PWD/runs/bonus/best/$label" "" "$full"
for c in $skill; do
  [ -f "runs/bonus/eval/${label}_$c.jsonl" ] && { echo "$label $c skill: done"; continue; }
  .venv/bin/python scripts/bonus/eval_grasp.py --policy "$kind" --ckpt "runs/bonus/best/$label" --cell "$c" --n 30 \
      --workers "${WORKERS:-3}" --out "runs/bonus/eval/${label}_$c.jsonl" 2>&1 | tail -1 | cut -c1-120
done
