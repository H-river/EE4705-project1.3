#!/usr/bin/env bash
# Bonus stages 3/4: train one lerobot policy + evaluate each checkpoint (20 VAL episodes).
#   scripts/bonus/train.sh <name> <act|diffusion> <steps> <dataset_root> [extra lerobot-train args...]
# Output: runs/bonus/train/<name>/ (checkpoints), runs/bonus/curves/<name>.csv, logs in runs/bonus/logs/.
# Training uses the 90 % train split from <dataset_root>/split.json; seed 1000.
set -euo pipefail
cd "$(dirname "$0")/../.."
name=$1; kind=$2; steps=$3; root=$4; shift 4
out=runs/bonus/train/$name
mkdir -p runs/bonus/logs
episodes=$(.venv/bin/python -c "import json,sys;print(json.dumps(json.load(open('$root/split.json'))['train']).replace(' ',''))")
common=(--dataset.repo_id=local/$(basename "$root") --dataset.root="$root" --dataset.episodes="$episodes"
        --policy.push_to_hub=false --policy.device=cuda --steps="$steps" --save_freq=5000 --log_freq=500
        --output_dir="$out" --seed=1000 --wandb.enable=false)
if [ "$kind" = act ]; then
  args=(--policy.type=act --policy.chunk_size=50 --policy.n_action_steps=25 --policy.vision_backbone=resnet18
        --batch_size=32 --num_workers=6)
else
  args=(--policy.type=diffusion --policy.horizon=32 --policy.n_action_steps=8 --policy.n_obs_steps=2
        --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 --batch_size=64 --num_workers=6)
fi
if [ -d "$out/checkpoints/last" ]; then
  resume=(--resume=true --config_path="$out/checkpoints/last/pretrained_model/train_config.json")
  .venv/bin/lerobot-train "${resume[@]}" > "runs/bonus/logs/$name.train.log" 2>&1 &
else
  .venv/bin/lerobot-train "${common[@]}" "${args[@]}" "$@" > "runs/bonus/logs/$name.train.log" 2>&1 &
fi
pid=$!
.venv/bin/python scripts/bonus/watch_ckpts.py --run "$out" --policy "$kind" --name "$name" --last-step "$steps" \
    --train-pid "$pid" --workers 3 > "runs/bonus/logs/$name.watch.log" 2>&1 &
wpid=$!
wait "$pid"; echo "train exit $?" >> "runs/bonus/logs/$name.watch.log"
wait "$wpid"
