#!/usr/bin/env bash
# 8.3 queue: after the 8.1/8.2 grasp queue, train the learned PLACE carry (ACT, same recipe).
cd "$(dirname "$0")/../.."
until grep -q "QUEUE DONE" runs/bonus/logs/queue_8.log 2>/dev/null; do sleep 60; done
until grep -q "^wrote" runs/bonus/convert_place.log 2>/dev/null; do sleep 60; done
EVAL_TASK=place TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train_task.sh place_act act 50000 runs/bonus/lerobot/place_2k
echo PLACE QUEUE DONE
