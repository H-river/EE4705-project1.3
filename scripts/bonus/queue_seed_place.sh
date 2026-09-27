#!/usr/bin/env bash
# Seed-variance check (ACT 2k, seeds 1001, 1002) and 8.3b learned PLACE with ±20 cm place demos.
cd "$(dirname "$0")/../.."
.venv/bin/python scripts/bonus/to_lerobot.py --demos runs/bonus/demos_place_2k_wide --root runs/bonus/lerobot/place_2k_wide \
    > runs/bonus/convert_place_wide.log 2>&1 &
TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train.sh act_seed1001 act 50000 runs/bonus/lerobot/grasp_2k --seed 1001
wait
EVAL_TASK=place TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train_task.sh place_act_wide act 50000 runs/bonus/lerobot/place_2k_wide
TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train.sh act_seed1002 act 50000 runs/bonus/lerobot/grasp_2k --seed 1002
echo SEEDPLACE DONE
