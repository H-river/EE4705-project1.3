#!/usr/bin/env bash
# Stage 8 training queue (sequential GPU use). Resumable: train.sh resumes from checkpoints/last.
cd "$(dirname "$0")/../.."
wait_ckpt() { until [ -f "runs/bonus/train/$1/checkpoints/$(printf %06d $2)/pretrained_model/model.safetensors" ] && grep -q "^$2," "runs/bonus/curves/$1.csv" 2>/dev/null; do sleep 60; done; }
wait_ckpt act_5k 50000
TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train.sh act_1k act 50000 runs/bonus/lerobot/grasp_1k
TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train.sh act_500 act 50000 runs/bonus/lerobot/grasp_500
until grep -q "^wrote" runs/bonus/convert_2k_bottle.log 2>/dev/null; do sleep 60; done
TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train.sh act_2k_bottle act 50000 runs/bonus/lerobot/grasp_2k_bottle
echo QUEUE DONE
