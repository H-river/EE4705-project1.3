#!/usr/bin/env python
"""Closed-loop grasp evaluation (bonus), 0 API calls.

Each episode: the Stage-1 post-APPROACH setup (GT target, tucked arm,
perturbed parking, open gripper), then ONE call of the selected grasp
primitive — ``skills.grasp`` (scripted) or ``LearnedGraspSkill`` — at the
GT target position.  Success = attached AND the attached body is the
intended one.  Scenes come from ``np.random.default_rng([seed, i])``, so
every policy sees the same scenes for the same (cell, seed, i).

Cells:
  C1 in-distribution: stone/cube ±10 cm, distractor 50 % (≥ 10 cm)
  C2 position shift: stone/cube ±20 cm
  C3 OOD object: bottle ±10 cm
  C4 near distractor: stone/cube ±10 cm, distractor ALWAYS, surface gap 0.3–3 cm
  VAL = C1 scenes with seed 500 (checkpoint selection; never used for the test table)

    python scripts/bonus/eval_grasp.py --policy act --ckpt <pretrained_model> \
        --cell C1 --n 30 --seed 1001 --workers 4 --out runs/bonus/eval/<name>.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/bonus"))

CELLS = {
    "C1": dict(classes=("stone", "cube"), pos_range=0.10),
    "VAL": dict(classes=("stone", "cube"), pos_range=0.10),
    "C2": dict(classes=("stone", "cube"), pos_range=0.20),
    "C3": dict(classes=("bottle",), pos_range=0.10),
    "C4": dict(classes=("stone", "cube"), pos_range=0.10, distractor_p=1.0, gap=(0.003, 0.03)),
}
DEFAULT_SEED = {"VAL": 500, "C1": 1001, "C2": 1002, "C3": 1003, "C4": 1004}

_W: dict = {}


def _init(policy: str, ckpt: str | None, kwargs: dict) -> None:
    # A failing initializer makes multiprocessing.Pool respawn workers forever (map() hangs),
    # so record the error and raise it from the first episode instead.
    try:
        from core.env import RobotEnv
        from core.world import SimWorld
        _W["world"] = SimWorld()
        _W["env"] = RobotEnv(_W["world"])
        if policy == "scripted":
            from core import skills
            _W["skill"] = skills.grasp
        else:
            if kwargs.get("device") == "cpu":
                import torch
                torch.set_num_threads(4)
            from executor.learned_grasp import LearnedGraspSkill
            _W["skill"] = LearnedGraspSkill(policy, ckpt, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _W["error"] = f"{type(exc).__name__}: {exc}"


def _episode(args) -> dict:
    import collect_grasp_demos as cg
    cell, seed, i = args
    if "error" in _W:
        raise RuntimeError(f"worker init failed: {_W['error']}")
    world, env, skill = _W["world"], _W["env"], _W["skill"]
    rng = np.random.default_rng([seed, i])
    cls, objs, robot = cg.sample_scene(rng, **CELLS[cell])
    target = cg.setup_post_approach(world, env, cls, objs, robot, rng)
    row = {"cell": cell, "seed": seed, "i": i, "class": cls, "objects": objs, "n_objects": len(objs)}
    if target is None:
        return {**row, "success": False, "reason": "approach_timeout", "wrong_object": False}
    pol = getattr(skill, "policy", None)
    if hasattr(pol, "set_episode_seed"):
        pol.set_episode_seed(seed * 10000 + i)  # same sampling noise for this episode on any worker
    t0, w0 = env.sim_time(), time.perf_counter()
    res = skill(env, target)
    body = world.attached_body_name()
    row.update(success=bool(env.is_attached() and body == cls), wrong_object=bool(body is not None and body != cls),
               reason=res.error_code.value, attached=body, time_to_attach=(env.sim_time() - t0) if body else None,
               wall_s=time.perf_counter() - w0,
               **{k: v for k, v in res.info.items() if k in ("stop", "ticks", "ee_error", "min_ee_error", "rollout_s")})
    return row


def run(policy, ckpt, cell, n, seed, workers, kwargs=None, start=0) -> list[dict]:
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(policy, ckpt, kwargs or {})) as pool:
        return pool.map(_episode, [(cell, seed, i) for i in range(start, start + n)], chunksize=1)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    tta = [r["time_to_attach"] for r in ok if r.get("time_to_attach") is not None]
    return {"n": n, "success": len(ok), "rate": len(ok) / max(1, n),
            "wrong_object": sum(r["wrong_object"] for r in rows),
            "mean_time_to_attach_s": float(np.mean(tta)) if tta else None,
            "reasons": {k: sum(r["reason"] == k for r in rows) for k in sorted({r["reason"] for r in rows})}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, choices=("scripted", "act", "diffusion", "mlp"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cell", default="C1", choices=sorted(CELLS))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--inference-steps", type=int, default=None)
    ap.add_argument("--scheduler", default=None, choices=(None, "DDIM", "DDPM"))
    ap.add_argument("--device", default=None, choices=(None, "cpu", "cuda"))
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    seed = DEFAULT_SEED[args.cell] if args.seed is None else args.seed
    kwargs = {"device": args.device} if args.device else {}
    if args.n_action_steps:
        kwargs["n_action_steps"] = args.n_action_steps
    if args.inference_steps or args.scheduler:
        kwargs.update(num_inference_steps=args.inference_steps, scheduler=args.scheduler)
    t0 = time.time()
    rows = run(args.policy, args.ckpt, args.cell, args.n, seed, args.workers, kwargs)
    s = summarize(rows)
    s.update(policy=args.policy, ckpt=args.ckpt, cell=args.cell, seed=seed, kwargs=kwargs, tag=args.tag,
             wall_s=time.time() - t0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(json.dumps({"summary": s}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
