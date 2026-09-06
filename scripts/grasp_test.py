#!/usr/bin/env python
# Owner: backbone (ALL)
"""Physical grasp test (grasp_mode="physical") for the three scene objects.

For each object: N (default 10) grasp-lift-hold attempts from the object's
canonical smoke_1 trial pose with small seeded random offsets (±1 cm in
x/y, ±10° yaw), using the reference skills (approach the base, reach the
pre-grasp pose, descend, close-and-check through ``try_attach_near_ee``,
lift 0.15 m in 3 cm increments while checking the hold, hold 1 s).

Reported per object: success count, failure reasons (no_contact,
single_pad_contact, closed_on_nothing, object_not_between_pads,
slipped@<lift height>, unreachable), and the lift height at which slips
occurred.  Outputs under runs/grasp_test/<run_id>/: grasp_results.csv,
summary.json and right-wrist frames of representative failures.

Exit code 0 when every object reaches >= 7/10, 1 otherwise (experimental).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import skills  # noqa: E402
from core.env import RobotEnv  # noqa: E402
from core.oracle import EvalOracle  # noqa: E402
from core.types import SceneConfig, SceneObjectSpec  # noqa: E402
from core.world import SimWorld  # noqa: E402

CANONICAL = {  # smoke_1_standard layout
    "stone": (0.40, -0.15, 0.88),
    "cube": (0.40, 0.15, 0.88),
    "bottle": (0.55, -0.35, 0.915),
}
LIFT_HEIGHT = 0.15
LIFT_STEP = 0.03
HOLD_S = 1.0
REQUIRED = 7


def attempt(world: SimWorld, env: RobotEnv, oracle: EvalOracle, obj: str, rng: np.random.Generator,
            k: int, frames_dir: pathlib.Path) -> dict:
    base = np.array(CANONICAL[obj])
    dx, dy = rng.uniform(-0.01, 0.01, size=2)
    yaw = float(np.radians(rng.uniform(-10.0, 10.0)))
    pos = (float(base[0] + dx), float(base[1] + dy), float(base[2]))
    others = [SceneObjectSpec(n, CANONICAL[n]) for n in CANONICAL if n != obj]
    world.reset(SceneConfig(seed=k, objects=[SceneObjectSpec(obj, pos, yaw)] + others))
    world.step(int(round(1.0 / world.timestep)))
    row = {"object": obj, "attempt": k, "dx": dx, "dy": dy, "yaw_deg": np.degrees(yaw), "success": False,
           "reason": "", "slip_height_m": np.nan, "measured_lift_m": np.nan, "grasp_opening": np.nan, "pad_gap_m": np.nan}
    target = oracle.object_pos(obj)
    if not skills.approach(env, target).success:
        row["reason"] = "unreachable:approach"
        return row
    r = skills.reach(env, oracle.object_pos(obj) + np.array([0.0, 0.0, skills.GRASP_DESCEND_OFFSET]))
    if not r.success:
        row["reason"] = f"unreachable:{r.error_code.name}"
        return row
    handle = env.try_attach_near_ee()
    row["grasp_opening"] = world.gripper_opening()["right"]
    row["pad_gap_m"] = world.pad_gap("right")
    if handle is None:
        row["reason"] = world.last_attach_reason
        Image.fromarray(env.get_obs("right_wrist").rgb).save(frames_dir / f"{obj}_{k:02d}_{row['reason'].split(':')[0]}.png")
        return row
    world.step(int(round(0.3 / world.timestep)))
    z0 = oracle.object_pos(obj)[2]
    ee0 = env.get_ee_pos().copy()
    lifted = 0.0
    while lifted < LIFT_HEIGHT - 1e-9:
        lifted = min(LIFT_HEIGHT, lifted + LIFT_STEP)
        m = skills.move_to(env, ee0 + np.array([0.0, 0.0, lifted]))
        if not env.is_attached() or (oracle.object_pos(obj)[2] - z0) < lifted - 0.04:
            row["reason"] = "slipped"
            row["slip_height_m"] = lifted
            row["measured_lift_m"] = float(oracle.object_pos(obj)[2] - z0)
            Image.fromarray(env.get_obs("right_wrist").rgb).save(frames_dir / f"{obj}_{k:02d}_slipped.png")
            return row
        if not m.success:
            row["reason"] = f"lift_{m.error_code.name}"
            return row
    world.step(int(round(HOLD_S / world.timestep)))
    if env.is_attached() and (oracle.object_pos(obj)[2] - z0) > LIFT_HEIGHT - 0.04:
        row["success"] = True
        row["reason"] = "held"
    else:
        row["reason"] = "slipped"
        row["slip_height_m"] = LIFT_HEIGHT
        row["measured_lift_m"] = float(oracle.object_pos(obj)[2] - z0)
        Image.fromarray(env.get_obs("right_wrist").rgb).save(frames_dir / f"{obj}_{k:02d}_slipped_hold.png")
    env.detach()
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="physical grasp test")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/grasp_test"))
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--objects", nargs="*", default=list(CANONICAL))
    args = parser.parse_args(argv)
    if args.attempts <= 0 or not args.objects or any(o not in CANONICAL for o in args.objects):
        parser.error("positive attempts and objects from stone, cube, bottle are required")
    started = time.perf_counter()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out / run_id
    n = 1
    while out.exists():
        n += 1
        out = args.out / f"{run_id}_{n}"
    frames = out / "failure_frames"
    frames.mkdir(parents=True)

    world = SimWorld(grasp_mode="physical")
    env = RobotEnv(world)
    oracle = EvalOracle(world)
    rows = []
    try:
        for obj in args.objects:
            rng = np.random.default_rng(1000 + sum(map(ord, obj)))
            for k in range(args.attempts):
                row = attempt(world, env, oracle, obj, rng, k, frames)
                rows.append(row)
                # Save a frame for every failure, including approach/reach/lift failures.
                if not row["success"]:
                    Image.fromarray(env.get_obs("right_wrist").rgb).save(frames / f"{obj}_{k:02d}_failure.png")
                detail = (f" at commanded lift {row['slip_height_m']:.2f} m"
                          if np.isfinite(row["slip_height_m"]) else "")
                if np.isfinite(row["grasp_opening"]):
                    detail += f" (opening {row['grasp_opening']:.2f}, gap {row['pad_gap_m'] * 1000:.0f} mm)"
                print(f"  {obj} #{k}: {'OK' if row['success'] else 'FAIL'} {row['reason']}{detail}", flush=True)
    finally:
        world.close()

    with (out / "grasp_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {"run_id": run_id, "attempts_per_object": args.attempts, "objects": {}}
    all_ok = True
    for obj in args.objects:
        rs = [r for r in rows if r["object"] == obj]
        succ = sum(r["success"] for r in rs)
        reasons: dict[str, int] = {}
        for r in rs:
            if not r["success"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        slips = [r["slip_height_m"] for r in rs if np.isfinite(r["slip_height_m"])]
        summary["objects"][obj] = {"success": succ, "attempts": len(rs), "failure_reasons": reasons,
                                   "slip_heights_m": slips}
        all_ok &= succ / len(rs) >= REQUIRED / 10
        print(f"{obj}: {succ}/{len(rs)} held; failures {reasons}; slip heights {np.round(slips, 2).tolist()}")
    summary["wall_s"] = time.perf_counter() - started
    summary["slip_height_note"] = "commanded lift at first detected loss; measured_lift_m is object displacement then; detection at 3 cm lift increments"
    summary["all_objects_at_least_7_of_10"] = bool(all_ok)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Outputs: {out / 'grasp_results.csv'}, {out / 'summary.json'}, frames in {frames}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
