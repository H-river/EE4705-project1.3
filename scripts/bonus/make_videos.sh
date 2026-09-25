#!/usr/bin/env bash
# Bonus stage 6 videos through the existing Recorder (eval.runner --video), 0 API calls.
#   scripts/bonus/make_videos.sh <policy> <ckpt|""> <kind>:<cell>:<trial> [...]
# e.g. make_videos.sh act runs/bonus/best/act success:c1:c1_00 failure:c2:c2_04 ood:c3:c3_00
# -> docs/bonus/videos/<policy>_<kind>_<trial>.mp4
set -euo pipefail
cd "$(dirname "$0")/../.."
policy=$1; ckpt=$2; shift 2
mkdir -p docs/bonus/videos
for spec in "$@"; do
  IFS=: read -r kind cell trial <<< "$spec"
  out=runs/bonus/videos/${policy}_${trial}
  rm -rf "$out"
  EE4705_GRASP_POLICY=$policy EE4705_GRASP_CKPT=${ckpt:+$PWD/$ckpt} .venv/bin/python -m eval.runner --mode manipulation \
      --trials "eval/trials/bonus_$cell" --only "$trial" --out "$out" --video > /dev/null 2>&1 || true
  mp4=$(find "$out" -name episode.mp4 | head -1)
  cp "$mp4" "docs/bonus/videos/${policy}_${kind}_${trial}.mp4"
  echo "${policy}_${kind}_${trial}.mp4 $(grep -ho '"outcome": "[A-Z_]*"' "$(dirname "$mp4")/trial_record.json" | head -1)"
done
