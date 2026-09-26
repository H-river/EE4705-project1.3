#!/usr/bin/env bash
# 8.3c: learned PLACE on 2k + 1k ±20 cm + 1k diverse (bottle, any start heading) place demos.
cd "$(dirname "$0")/../.."
until [ -f runs/bonus/demos_place_div/manifest.json ]; do sleep 60; done
d=runs/bonus/demos_place_4k_div; mkdir -p $d
for f in runs/bonus/demos_place_2k_wide/ep_*.h5; do ln -sf "$(readlink -f $f)" $d/$(basename $f); done
ls runs/bonus/demos_place_div/ep_*.h5 | sort | head -1000 | while read f; do b=$(basename $f .h5); ln -sf "$(readlink -f $f)" $d/ep_2${b#ep_}.h5; done
.venv/bin/python scripts/bonus/to_lerobot.py --demos $d --root runs/bonus/lerobot/place_4k_div > runs/bonus/convert_place_div.log 2>&1
until grep -q "SEEDPLACE DONE" runs/bonus/logs/queue_seed_place.log 2>/dev/null; do sleep 60; done
EVAL_TASK=place TRAIN_WORKERS=5 EVAL_WORKERS=2 scripts/bonus/train_task.sh place_act_div act 50000 runs/bonus/lerobot/place_4k_div
echo PLACEDIV DONE
