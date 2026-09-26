#!/usr/bin/env bash
# 8.6 resume after the 10:15 OOM: one VAL watcher during training; VAL2/VAL3 afterwards (sequential, RAM).
cd "$(dirname "$0")/../.."
TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train.sh act_wide act 50000 runs/bonus/lerobot/grasp_2k_bottle_wide
for c in VAL2 VAL3; do
  .venv/bin/python scripts/bonus/watch_ckpts.py --run runs/bonus/train/act_wide --policy act --name "act_wide_${c,,}" \
      --cell "$c" --last-step 50000 --workers 3 --device cuda >> "runs/bonus/logs/act_wide_${c,,}.watch.log" 2>&1
done
echo WIDE2 DONE
