#!/usr/bin/env python
"""Stage 1 (bonus): expert grasp demonstrations from the scripted skill.

One episode = the post-APPROACH pose (arm tucked, base parked by the
reference APPROACH with a ±3 cm / ±5° perturbation, gripper open) ->
``skills.grasp`` (REACH via-point + descend + ``try_attach_near_ee``) ->
0.5 s lift.  Ground-truth target positions (0 API calls).  Recorded at
10 Hz: head RGB 224x224, arm q (7), commanded arm setpoint (7), EE pos,
target position in the base frame (3), attach flag.  Only episodes in
which the INTENDED body ends up attached are kept.

Randomisation: target class in {stone, cube} (bottle is excluded: it is
the OOD object), position ±10 cm around the class's canonical smoke_1
pose (outside the red region), yaw uniform, robot start pose, base
perturbation, one distractor (stone2 or the other class) in 50 %.

    python scripts/bonus/collect_grasp_demos.py --n-keep 2000 --workers 12 \
        --out runs/bonus/demos --seed 0

Episode i uses ``np.random.default_rng([seed, i])``: re-running with the
same seed reproduces every episode; files that already exist are skipped
(resume after a restart).
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HZ = 10.0
IMG = 224
LIFT_S = 0.5
LIFT_M = 0.10
CANONICAL = {"stone": (0.40, -0.15, 0.88), "cube": (0.40, 0.15, 0.88), "bottle": (0.55, -0.35, 0.915)}
Z = {"stone": 0.88, "stone2": 0.88, "cube": 0.88, "bottle": 0.915}
RADIUS = {"stone": 0.030, "stone2": 0.030, "cube": 0.0354, "bottle": 0.028}  # xy bounding radius (scene_common.xml)
TABLE_X, TABLE_Y = (0.28, 0.62), (-0.46, 0.46)
REGION = ((0.40, 0.30), 0.08)  # centre, half size (scene_common.xml)
ARM_TUCK_OFFSET = np.array([0.16, -0.10, 1.12])  # executor/student_c.py _ARM_TUCK_OFFSET


def in_region(xy, margin=0.05) -> bool:
    (cx, cy), h = REGION
    return abs(xy[0] - cx) < h + margin and abs(xy[1] - cy) < h + margin


def sample_scene(rng: np.random.Generator, classes=("stone", "cube"), pos_range=0.10,
                 distractor_p=0.5, distractor_dist=(0.10, None), gap=None):
    """Returns (target_name, objects[list of (name,pos,yaw)], robot_init).
    ``distractor_dist`` bounds the centre distance (None = anywhere on the
    table); ``gap`` = (lo, hi) instead samples the SURFACE gap between the
    two bounding circles (the C4 "distractor < 5 cm" cell)."""
    cls = str(rng.choice(classes))
    base = np.array(CANONICAL[cls])
    while True:
        xy = base[:2] + rng.uniform(-pos_range, pos_range, size=2)
        if TABLE_X[0] <= xy[0] <= TABLE_X[1] and TABLE_Y[0] <= xy[1] <= TABLE_Y[1] and not in_region(xy):
            break
    objs = [(cls, (float(xy[0]), float(xy[1]), Z[cls]), float(rng.uniform(-math.pi, math.pi)))]
    if rng.random() < distractor_p:
        pool = [n for n in ("stone2", "stone", "cube") if n != cls]
        name = str(rng.choice(pool))
        dmin, dmax = distractor_dist
        if gap is not None:
            dmin = dmax = None
        for _ in range(1000):
            if gap is not None:
                r = RADIUS[cls] + RADIUS[name] + rng.uniform(*gap)
                a = rng.uniform(-math.pi, math.pi)
                dxy = xy + r * np.array([math.cos(a), math.sin(a)])
                dmin = dmax = float(np.linalg.norm(dxy - xy))
            elif dmax is None:
                dxy = np.array([rng.uniform(*TABLE_X), rng.uniform(*TABLE_Y)])
            else:
                r, a = rng.uniform(dmin, dmax), rng.uniform(-math.pi, math.pi)
                dxy = xy + r * np.array([math.cos(a), math.sin(a)])
            d = float(np.linalg.norm(dxy - xy))
            if (d >= dmin and (dmax is None or d <= dmax) and TABLE_X[0] <= dxy[0] <= TABLE_X[1]
                    and TABLE_Y[0] <= dxy[1] <= TABLE_Y[1] and not in_region(dxy, 0.02)):
                objs.append((name, (float(dxy[0]), float(dxy[1]), Z[name]), float(rng.uniform(-math.pi, math.pi))))
                break
    robot = {"x": float(rng.uniform(-0.18, 0.02)), "y": float(rng.uniform(-0.08, 0.08)),
             "yaw": float(rng.uniform(-0.2, 0.2))}
    return cls, objs, robot


def base_frame(p_world, base) -> np.ndarray:
    c, s = math.cos(base[2]), math.sin(base[2])
    d = np.asarray(p_world[:2]) - np.asarray(base[:2])
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], float(p_world[2])])


def head_rgb(env) -> np.ndarray:
    import cv2
    rgb = env.get_obs("head").rgb
    return cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)


def setup_post_approach(world, env, cls, objs, robot, rng, base_noise=(0.03, math.radians(5))):
    """Reset + tuck + perturbed APPROACH + open gripper (the state GRASP starts from)."""
    from core import skills
    from core.g1 import approach_base_pose
    from core.types import SceneConfig, SceneObjectSpec

    world.reset(SceneConfig(seed=0, objects=[SceneObjectSpec(n, p, y) for n, p, y in objs], robot_init=robot))
    env.step(int(round(0.5 / world.timestep)))
    b = env.get_base_pose()
    c, s = math.cos(b[2]), math.sin(b[2])
    lx, ly, tz = ARM_TUCK_OFFSET
    tuck = np.array([b[0] + c * lx - s * ly, b[1] + s * lx + c * ly, tz])
    env.set_arm_target(tuck)
    skills._step_until(env, lambda: np.linalg.norm(env.get_ee_pos() - tuck) < skills.EE_POS_TOL, 5.0)
    target = np.array(world.body_pos(cls))
    bx, by, yaw = approach_base_pose(target[:2], env.get_base_pose()[:2])
    bx += rng.uniform(-base_noise[0], base_noise[0])
    by += rng.uniform(-base_noise[0], base_noise[0])
    yaw += rng.uniform(-base_noise[1], base_noise[1])
    env.set_base_target(bx, by, yaw)

    def parked():
        p = env.get_base_pose()
        return (np.linalg.norm(p[:2] - [bx, by]) < skills.BASE_POS_TOL
                and abs(math.atan2(math.sin(p[2] - yaw), math.cos(p[2] - yaw))) < skills.BASE_YAW_TOL)
    if not skills._step_until(env, parked, 12.0):
        return None
    env.stop_motion()
    env.step(100)
    env.set_gripper("right", 1.0)
    skills._step_until(env, lambda: env.get_robot_state().gripper_opening["right"] >= 0.9, 2.0)
    return np.array(world.body_pos(cls))


class Recorder:
    """RobotEnv proxy: every step() is split so a sample is taken on the 10 Hz grid."""

    def __init__(self, env, world, target_world, with_image=True):
        self.env, self.world = env, world
        self.every = int(round(1.0 / HZ / world.timestep))
        self.target_world = np.asarray(target_world, float)
        self.with_image = with_image
        self.count = 0
        self.rows = {k: [] for k in ("t", "q", "q_des", "ctrl", "ee", "base", "target_b", "attached", "image")}
        self.sample()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def sample(self):
        r, w = self.rows, self.world
        base = self.env.get_base_pose()
        r["t"].append(w.sim_time)
        r["q"].append(w.robot.arm_q())
        r["q_des"].append(w.robot.arm_q_des())
        r["ctrl"].append(w.robot.arm_ctrl())
        r["ee"].append(self.env.get_ee_pos())
        r["base"].append(np.array(base))
        r["target_b"].append(base_frame(self.target_world, base))
        r["attached"].append(bool(self.env.is_attached()))
        if self.with_image:
            r["image"].append(head_rgb(self.env))

    def step(self, n=1):
        while n > 0:
            k = min(n, self.every - self.count)
            self.env.step(k)
            n -= k
            self.count += k
            if self.count >= self.every:
                self.count = 0
                self.sample()


def run_episode(world, env, i: int, seed: int, with_image=True, classes=("stone", "cube")) -> dict:
    from core import skills

    rng = np.random.default_rng([seed, i])
    cls, objs, robot = sample_scene(rng, classes=tuple(classes))
    t0 = time.perf_counter()
    target = setup_post_approach(world, env, cls, objs, robot, rng)
    meta = {"episode": i, "seed": [seed, i], "class": cls, "objects": objs, "robot_init": robot,
            "distractor": len(objs) > 1}
    if target is None:
        return {**meta, "success": False, "reason": "approach_timeout"}
    rec = Recorder(env, world, target, with_image)
    res = skills.grasp(rec, target)
    ok = env.is_attached() and world.attached_body_name() == cls
    if ok:
        for lift in (LIFT_M, LIFT_M / 2):  # the bottle's taller grasp point can make the 10 cm lift infeasible
            try:
                env.set_arm_target(env.get_ee_pos() + np.array([0.0, 0.0, lift]))
                break
            except ValueError:
                continue
        rec.step(int(round(LIFT_S / world.timestep)))
    meta.update(success=bool(ok), reason=res.error_code.value if not ok else "attached",
                wrong_object=bool(env.is_attached() and not ok),
                n_steps=len(rec.rows["t"]), wall_s=time.perf_counter() - t0)
    return {**meta, "rows": rec.rows}


def save_h5(path: pathlib.Path, ep: dict) -> None:
    import h5py
    rows = ep["rows"]
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for k in ("t", "q", "q_des", "ctrl", "ee", "base", "target_b", "attached"):
            f.create_dataset(k, data=np.asarray(rows[k]))
        if rows["image"]:
            f.create_dataset("image", data=np.asarray(rows["image"], np.uint8), chunks=(1, IMG, IMG, 3),
                             compression="gzip", compression_opts=4)
        # action = next-step arm q (last step repeats itself)
        q = np.asarray(rows["q"])
        f.create_dataset("action", data=np.concatenate([q[1:], q[-1:]], 0))
        f.attrs["meta"] = json.dumps({k: v for k, v in ep.items() if k != "rows"})
    tmp.rename(path)


_W = {}


def _worker(args):
    i, seed, out, with_image, classes = args
    if "world" not in _W:
        from core.env import RobotEnv
        from core.world import SimWorld
        _W["world"] = SimWorld()
        _W["env"] = RobotEnv(_W["world"])
    path = out / f"ep_{i:05d}.h5"
    try:
        ep = run_episode(_W["world"], _W["env"], i, seed, with_image, classes)
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the pool
        return {"episode": i, "seed": [seed, i], "class": "?", "distractor": False, "success": False,
                "reason": f"exception:{type(exc).__name__}", "n_steps": 0}
    if ep["success"]:
        save_h5(path, ep)
    return {k: v for k, v in ep.items() if k != "rows"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "runs/bonus/demos")
    ap.add_argument("--n-keep", type=int, default=2000)
    ap.add_argument("--start", type=int, default=0, help="first episode index")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--classes", default="stone,cube", help="target classes (8.2: bottle)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "episodes.jsonl"
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            d = json.loads(line)
            done[d["episode"]] = d
    kept = sum(1 for d in done.values() if d["success"] and (args.out / f"ep_{d['episode']:05d}.h5").exists())
    print(f"resume: {len(done)} logged, {kept} kept", flush=True)
    todo = (i for i in range(args.start, 10**7) if i not in done)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool, log_path.open("a") as log:
        pending = []
        while kept < args.n_keep:
            while len(pending) < args.workers * 2:
                i = next(todo)
                pending.append(pool.apply_async(_worker, ((i, args.seed, args.out, not args.no_image,
                                                                   tuple(args.classes.split(","))),)))
            r = pending.pop(0).get()
            log.write(json.dumps(r) + "\n")
            log.flush()
            done[r["episode"]] = r
            kept += int(r["success"])
            if len(done) % 50 == 0:
                print(f"{len(done)} tried, {kept} kept, {time.time() - t0:.0f}s", flush=True)
        for p in pending:  # finish in-flight episodes so the log stays complete
            r = p.get()
            log.write(json.dumps(r) + "\n")
            done[r["episode"]] = r
    write_manifest(args.out, args.seed)
    return 0


def write_manifest(out: pathlib.Path, seed: int) -> dict:
    eps = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    kept = [e for e in eps if e["success"] and (out / f"ep_{e['episode']:05d}.h5").exists()]
    lens = np.array([e["n_steps"] for e in kept]) if kept else np.zeros(1)
    reasons = {}
    for e in eps:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    m = {"seed": seed, "hz": HZ, "image": IMG, "tried": len(eps), "kept": len(kept),
         "expert_success_rate": len(kept) / max(1, len(eps)), "outcomes": reasons,
         "wrong_object": sum(e.get("wrong_object", False) for e in eps),
         "by_class": {c: sum(e["class"] == c for e in kept) for c in ("stone", "cube", "bottle")},
         "with_distractor": sum(e["distractor"] for e in kept),
         "length_steps": {"mean": float(lens.mean()), "std": float(lens.std()), "min": int(lens.min()),
                          "max": int(lens.max()), "p50": float(np.median(lens))}}
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    raise SystemExit(main())
