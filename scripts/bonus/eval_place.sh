#!/usr/bin/env bash
# 8.3: full-executor C1 episodes with a learned PLACE carry (grasp stays scripted).
#   scripts/bonus/eval_place.sh <label> <kind> <ckpt> [cells]
set -uo pipefail
cd "$(dirname "$0")/../.."
label=$1; kind=$2; ckpt=$3; cells=${4:-C1}
for cell in $cells; do
  out=runs/bonus/manip/${label}_${cell}
  rm -rf "$out" "$out.place_calls.jsonl"
  EE4705_PLACE_LOG="$PWD/$out.place_calls.jsonl" EE4705_PLACE_POLICY=$kind EE4705_PLACE_CKPT=$PWD/$ckpt \
    .venv/bin/python -m eval.runner --mode manipulation --trials "eval/trials/bonus_${cell,,}" \
    --out "$out" --jobs "${JOBS:-3}" > "runs/bonus/logs/manip_${label}_${cell}.log" 2>&1
  echo "$label $cell: $(tail -1 runs/bonus/logs/manip_${label}_${cell}.log)"
done
