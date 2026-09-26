#!/usr/bin/env python
"""Evaluate every lerobot-train checkpoint as it appears (bonus stages 3/4).

For <run>/checkpoints/<step>/pretrained_model: 20 closed-loop VAL episodes
(C1 scene distribution, seed 500 — disjoint from the C1 test seed 1001).
Appends to runs/bonus/curves/<name>.csv; exits after --last-step is done
(or when the training process is gone and nothing is left to evaluate).

    python scripts/bonus/watch_ckpts.py --run runs/bonus/train/act_base --policy act --last-step 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bonus"))


def evaluated(csv_path: pathlib.Path) -> set[int]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {int(r["step"]) for r in csv.DictReader(f)}


def place_val(args, ckpt: pathlib.Path, step: int, name: str):
    """Learned PLACE carry inside the full executor on the 20 VAL trials (grasp scripted)."""
    import os
    import place_metrics
    out = ROOT / "runs/bonus/manip/val" / f"{name}_{step:06d}"
    env = {**os.environ, "EE4705_PLACE_POLICY": args.policy, "EE4705_PLACE_CKPT": str(ckpt.resolve()),
           "EE4705_PLACE_LOG": str(out) + ".place_calls.jsonl",
           "EE4705_PLACE_KWARGS": '{"device": "%s"}' % args.device if args.device else ""}
    subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "eval.runner", "--mode", "manipulation", "--trials",
                    str(ROOT / "eval/trials/bonus_val"), "--out", str(out), "--jobs", str(args.workers)],
                   cwd=ROOT, env=env, capture_output=True)
    m = place_metrics.run(out)
    s = {"n": m["n"], "success": m["task_success"], "rate": m["task_success"] / max(1, m["n"]), "wrong_object": 0,
         "mean_time_to_attach_s": m["offset_cm_mean"]}  # for place, the last csv column holds the mean offset (cm)
    return s, [m]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cell", default="VAL", help="VAL (default) or VAL3 (bottle, 8.2)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--last-step", type=int, required=True)
    ap.add_argument("--train-pid", type=int, default=None)
    ap.add_argument("--device", default=None, help="cpu: keep eval off the GPU while training runs")
    ap.add_argument("--task", default="grasp", choices=("grasp", "place"),
                    help="place: full-executor episodes on eval/trials/bonus_val (20), score = task success")
    args = ap.parse_args(argv)
    import eval_grasp
    if args.seed is None:
        args.seed = eval_grasp.DEFAULT_SEED[args.cell]
    name = args.name or args.run.name
    out = ROOT / "runs/bonus/curves" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    detail = ROOT / "runs/bonus/eval/curves" / name
    detail.mkdir(parents=True, exist_ok=True)
    while True:
        done = evaluated(out)
        ckpts = sorted(p for p in (args.run / "checkpoints").glob("[0-9]*") if p.is_dir())
        todo = [p for p in ckpts if int(p.name) not in done and (p / "pretrained_model/model.safetensors").exists()]
        for p in todo:
            time.sleep(5)  # let the writer finish the directory
            step = int(p.name)
            if args.task == "place":
                s, rows = place_val(args, p / "pretrained_model", step, name)
            else:
                rows = eval_grasp.run(args.policy, str(p / "pretrained_model"), args.cell, args.n, args.seed, args.workers,
                                      {"device": args.device} if args.device else None)
                s = eval_grasp.summarize(rows)
            with (detail / f"{step:06d}.jsonl").open("w") as f:
                f.write(json.dumps({"summary": s}) + "\n")
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            new = not out.exists()
            with out.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["step", "success", "n", "rate", "wrong_object", "mean_time_to_attach_s"])
                w.writerow([step, s["success"], s["n"], f"{s['rate']:.3f}", s["wrong_object"],
                            "" if s["mean_time_to_attach_s"] is None else f"{s['mean_time_to_attach_s']:.2f}"])
            print(f"{name} step {step}: {s['success']}/{s['n']}", flush=True)
        if args.last_step in evaluated(out):
            return 0
        if args.train_pid is not None and not todo:
            alive = subprocess.run(["kill", "-0", str(args.train_pid)], capture_output=True).returncode == 0
            if not alive and not [p for p in (args.run / "checkpoints").glob("[0-9]*")
                                  if int(p.name) not in evaluated(out)]:
                return 1
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
