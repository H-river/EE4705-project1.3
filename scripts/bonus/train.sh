#!/usr/bin/env bash
# Bonus stages 3/4: train one lerobot policy (scripts/bonus/train_policy.py, fast data path)
# and evaluate each checkpoint with 20 closed-loop VAL episodes as it appears.
#   scripts/bonus/train.sh <name> <act|diffusion> <steps> <dataset_root> [extra train_policy.py args...]
# Output: runs/bonus/train/<name>/ (checkpoints; resumes from checkpoints/last), curve in
# runs/bonus/curves/<name>.csv, logs in runs/bonus/logs/. Seed 1000.
set -euo pipefail
cd "$(dirname "$0")/../.."
name=$1; kind=$2; steps=$3; root=$4; shift 4
out=runs/bonus/train/$name
mkdir -p runs/bonus/logs
.venv/bin/python scripts/bonus/train_policy.py --out "$out" --kind "$kind" --steps "$steps" --dataset "$root" \
    --workers "${TRAIN_WORKERS:-5}" "$@" >> "runs/bonus/logs/$name.train.log" 2>&1 &
pid=$!
.venv/bin/python scripts/bonus/watch_ckpts.py --run "$out" --policy "$kind" --name "$name" --last-step "$steps" \
    --train-pid "$pid" --workers "${EVAL_WORKERS:-3}" --device "${EVAL_DEVICE:-cpu}" >> "runs/bonus/logs/$name.watch.log" 2>&1 &
wpid=$!
set +e
wait "$pid"; echo "train exit $?" >> "runs/bonus/logs/$name.watch.log"
wait "$wpid"; echo "watch exit $?" >> "runs/bonus/logs/$name.watch.log"
