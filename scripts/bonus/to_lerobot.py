#!/usr/bin/env python
"""Stage 2 (bonus): HDF5 grasp demos -> LeRobotDataset.

Features (10 Hz):
  observation.image  (224, 224, 3) head RGB
  observation.state  (10,)  arm q (7) ⊕ target position in the base frame (3)
  action             (7,)   next-step arm q

The installed lerobot (0.6.1) writes codebase format v3.0 (the v2 layout is
no longer produced by lerobot >= 0.4); the feature schema is the one the
task asks for.  Split: 90/10 by episode with a seeded permutation, written
to <root>/split.json as LeRobot episode indices.

    python scripts/bonus/to_lerobot.py --demos runs/bonus/demos \
        --root runs/bonus/lerobot/grasp_2k [--max-episodes 2000] [--classes stone,cube]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FEATURES = {
    "observation.image": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (10,),
                          "names": [f"q{i}" for i in range(7)] + ["tx", "ty", "tz"]},
    "action": {"dtype": "float32", "shape": (7,), "names": [f"q{i}_next" for i in range(7)]},
}
TASK = "grasp the target object"


def episode_files(demos: list[pathlib.Path], classes: set[str] | None, limit: int | None):
    files = []
    for d in demos:
        for p in sorted(d.glob("ep_*.h5")):
            with h5py.File(p) as f:
                meta = json.loads(f.attrs["meta"])
            if classes and meta["class"] not in classes:
                continue
            files.append(p)
    return files[:limit] if limit else files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=pathlib.Path, nargs="+", default=[ROOT / "runs/bonus/demos"])
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--classes", default="", help="comma list; empty = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args(argv)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = episode_files(args.demos, set(filter(None, args.classes.split(","))), args.max_episodes)
    if args.root.exists():
        shutil.rmtree(args.root)
    ds = LeRobotDataset.create(repo_id=f"local/{args.root.name}", fps=10, features=FEATURES, root=args.root,
                               robot_type="unitree_g1_right_arm", use_videos=False, image_writer_threads=8)
    sources = []
    for k, p in enumerate(files):
        with h5py.File(p) as f:
            q, tb, act, img = f["q"][:], f["target_b"][:], f["action"][:], f["image"][:]
        state = np.concatenate([q, tb], 1).astype(np.float32)
        for t in range(len(q)):
            ds.add_frame({"observation.image": img[t], "observation.state": state[t],
                          "action": act[t].astype(np.float32), "task": TASK})
        ds.save_episode()
        sources.append(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p))
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(files)} episodes", flush=True)
    ds.finalize()
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(round(args.val_frac * len(files))))
    split = {"seed": args.seed, "val": sorted(int(i) for i in perm[:n_val]),
             "train": sorted(int(i) for i in perm[n_val:]), "sources": sources}
    (args.root / "split.json").write_text(json.dumps(split))
    print(f"wrote {len(files)} episodes ({len(split['train'])} train / {len(split['val'])} val) to {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
