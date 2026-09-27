#!/usr/bin/env bash
# Bonus stage 6: full-executor evaluation (eval.runner --mode manipulation: GT perception + RulePlanner +
# StudentCExecutor), 30 episodes per cell, same trial files (= seeds) for every policy. 0 API calls.
#   scripts/bonus/run_stage6.sh <label> <scripted|act|diffusion|mlp> [ckpt] [kwargs-json] [cells]
# -> runs/bonus/manip/<label>_<cell>/ ; skips a cell whose run already has 30 trial records (R5).
set -uo pipefail
cd "$(dirname "$0")/../.."
label=$1; policy=$2; ckpt=${3:-}; kwargs=${4:-}; cells=${5:-C1 C2 C3 C4}
for cell in $cells; do
  out=runs/bonus/manip/${label}_${cell}
  n=$(ls "$out"/merged/*/trial_record.json "$out"/*_manipulation/*/trial_record.json 2>/dev/null | wc -l)
  if [ "$n" -ge 30 ]; then echo "$label $cell: already done ($n)"; continue; fi
  rm -rf "$out" "$out.grasp_calls.jsonl"
  EE4705_GRASP_LOG="$PWD/$out.grasp_calls.jsonl" EE4705_GRASP_POLICY=$policy EE4705_GRASP_CKPT=$ckpt EE4705_GRASP_KWARGS=$kwargs \
    .venv/bin/python -m eval.runner --mode manipulation --trials "eval/trials/bonus_${cell,,}" \
    --out "$out" --jobs "${JOBS:-3}" > "runs/bonus/logs/manip_${label}_${cell}.log" 2>&1
  echo "$label $cell: $(tail -1 runs/bonus/logs/manip_${label}_${cell}.log)"
done
