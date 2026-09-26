#!/usr/bin/env bash
# 8.6 (extra): ACT on 2k + 503 bottle + 1,000 ±20 cm demos; VAL (stone/cube), VAL2 (±20 cm), VAL3 (bottle) watchers.
cd "$(dirname "$0")/../.."
until grep -q "PLACE QUEUE DONE" runs/bonus/logs/queue_place.log 2>/dev/null; do sleep 60; done
until grep -q "^wrote" runs/bonus/convert_2k_bottle_wide.log 2>/dev/null; do sleep 60; done
TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train.sh act_wide act 50000 runs/bonus/lerobot/grasp_2k_bottle_wide &
tpid=$!
until pgrep -f "train_policy.py --out runs/bonus/train/act_wide" > /dev/null; do sleep 5; done
pid=$(pgrep -f "train_policy.py --out runs/bonus/train/act_wide" | head -1)
for c in VAL2 VAL3; do
  .venv/bin/python scripts/bonus/watch_ckpts.py --run runs/bonus/train/act_wide --policy act --name "act_wide_${c,,}" \
      --cell "$c" --last-step 50000 --train-pid "$pid" --workers 2 --device cpu >> "runs/bonus/logs/act_wide_${c,,}.watch.log" 2>&1 &
done
wait
echo WIDE QUEUE DONE
