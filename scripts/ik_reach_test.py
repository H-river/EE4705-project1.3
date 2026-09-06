#!/usr/bin/env python
# Owner: backbone (ALL)
"""Right-arm reachability test for the Unitree G1 tabletop platform.

Evaluates a reproducible 5x5 grid over a DECLARED tabletop workspace and
the task-relevant approach / grasp / lift / place poses of the standard
smoke layout, reporting separately for every target:

* kinematic IK success (core.ik on a scratch state) with position and
  angular residuals, iterations and failure reason;
* whether the IK solution itself penetrates the table or the robot
  (a solution "inside the table" is NOT counted as usable);
* actual tracking success of the simulated arm (position < 0.01 m and
  orientation < 5 deg at the settled state), settling time, residuals and
  collision violations observed at the settled state.

Outputs (under runs/ik_reach/<run_id>/): reach_results.csv, reach_grid.png
(failed points marked), summary.json.  Exit code 0 when every task pose
was tracked collision-free, 1 otherwise (the grid is informational).

Usage:  python scripts/ik_reach_test.py [--out runs/ik_reach]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.g1 import APPROACH_TILT, FINGER_AZIMUTH, WAIST_REST, RIGHT_ARM_REST, approach_base_pose, approach_rotation  # noqa: E402
from core.rendering import mujoco  # noqa: E402
from core.skills import GRASP_DESCEND_OFFSET, PRE_REACH_HEIGHT  # noqa: E402
from core.types import SceneConfig, SceneObjectSpec  # noqa: E402
from core.world import SimWorld  # noqa: E402

# ----------------------------------------------------------------- declared workspace
TABLE_TOP_Z = 0.85  # m (assets/scene_common.xml)
GRID_X = np.linspace(0.28, 0.44, 5)  # world = base frame (base at origin, yaw 0)
GRID_Y = np.linspace(-0.32, -0.08, 5)
TARGET_Z = TABLE_TOP_Z + 0.05  # pinch TCP 5 cm above the table
BASE_POSE = (0.0, 0.0, 0.0)  # x, y, yaw
WAIST_POSE = (WAIST_REST["waist_yaw_joint"], WAIST_REST["waist_roll_joint"], WAIST_REST["waist_pitch_joint"])
INITIAL_ARM_Q = [RIGHT_ARM_REST[j] for j in (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")]
TARGET_ROT = approach_rotation(BASE_POSE[2])  # tilt 45 deg, finger azimuth 0 deg (core.g1)

POS_TOL = 0.01  # m
ROT_TOL = math.radians(5.0)
TRACK_TIMEOUT_S = 6.0
SETTLE_QVEL = 0.05  # rad/s

# task layout (smoke_1_standard)
STONE = np.array([0.40, -0.15, 0.88])
REGION = np.array([0.40, 0.30, TABLE_TOP_Z])


def contacts_str(c: list[tuple[str, str, float]]) -> str:
    return ";".join(f"{a}|{b}|{d:.4f}" for a, b, d in c)


def evaluate_target(world: SimWorld, name: str, pos: np.ndarray, rot: np.ndarray, allow_object_contact: bool) -> dict:
    """Reset-consistent evaluation of one target from the current base/waist pose."""
    row = {"name": name, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
    r = world.robot
    t0 = time.perf_counter()
    ik, policy = r.solve_arm_target(pos, rot, allow_position_only_fallback=False)
    row.update({
        "ik_success": ik.success, "ik_pos_err_mm": ik.pos_err * 1000.0, "ik_rot_err_deg": math.degrees(ik.rot_err),
        "ik_iters": ik.iters, "ik_reason": ik.reason, "ik_ms": (time.perf_counter() - t0) * 1000.0,
    })
    ik_contacts = world.robot_contacts_for_arm_q(ik.q) if ik.success else []
    if not allow_object_contact:
        ik_contacts = [c for c in ik_contacts]
    ik_contacts_bad = [c for c in ik_contacts if not (allow_object_contact and _touches_object(c))]
    row["ik_collision"] = contacts_str(ik_contacts_bad)
    row["ik_solution_usable"] = bool(ik.success and not ik_contacts_bad)
    # ---- tracking
    row.update({"track_success": False, "track_pos_err_mm": float("nan"), "track_rot_err_deg": float("nan"),
                "settle_time_s": float("nan"), "track_collision": "", "failure_reason": ""})
    if not ik.success:
        row["failure_reason"] = f"ik_{ik.reason}"
        return row
    try:
        world.set_arm_target(pos, rot)
    except ValueError as exc:
        row["failure_reason"] = f"command_rejected: {exc}"
        return row
    t_start = world.sim_time
    worst_contacts: dict[tuple[str, str], float] = {}
    settled = False
    while world.sim_time - t_start < TRACK_TIMEOUT_S:
        world.step(10)
        for a, b, d in world.robot_contacts():
            if allow_object_contact and _touches_object((a, b, d)):
                continue
            key = (a, b)
            worst_contacts[key] = min(worst_contacts.get(key, 0.0), d)
        pe, re = r.tracking_error()
        if pe < POS_TOL and re < ROT_TOL and float(np.max(np.abs(r.arm_qvel()))) < SETTLE_QVEL:
            settled = True
            break
    pe, re = r.tracking_error()
    row["track_pos_err_mm"] = pe * 1000.0
    row["track_rot_err_deg"] = math.degrees(re)
    row["settle_time_s"] = world.sim_time - t_start if settled else float("nan")
    final_contacts = [c for c in world.robot_contacts() if not (allow_object_contact and _touches_object(c))]
    row["track_collision"] = contacts_str(final_contacts)
    row["collision_during_motion"] = ";".join(f"{a}|{b}|{d:.4f}" for (a, b), d in worst_contacts.items())
    row["track_success"] = bool(settled and not final_contacts)
    if not settled:
        row["failure_reason"] = f"tracking_timeout(pos {pe * 1000:.1f} mm, rot {math.degrees(re):.1f} deg)"
    elif final_contacts:
        row["failure_reason"] = "collision_at_target"
    return row


def _touches_object(c: tuple[str, str, float]) -> bool:
    return c[0] in ("stone", "stone2", "cube", "bottle") or c[1] in ("stone", "stone2", "cube", "bottle")


def reset_for_grid(world: SimWorld) -> None:
    world.reset(SceneConfig(seed=0, objects=[], robot_init={"x": BASE_POSE[0], "y": BASE_POSE[1], "yaw": BASE_POSE[2]}))
    world.set_waist_target(*WAIST_POSE)
    world.step(int(round(0.5 / world.timestep)))  # settle in the rest posture


def run_grid(world: SimWorld) -> list[dict]:
    rows = []
    for x in GRID_X:
        for y in GRID_Y:
            reset_for_grid(world)
            row = evaluate_target(world, "grid", np.array([x, y, TARGET_Z]), TARGET_ROT, allow_object_contact=False)
            rows.append(row)
            print(f"  grid ({x:.2f},{y:.2f}): ik={'ok' if row['ik_success'] else 'FAIL'} "
                  f"({row['ik_pos_err_mm']:.1f} mm, {row['ik_rot_err_deg']:.1f} deg) usable={row['ik_solution_usable']} "
                  f"track={'ok' if row['track_success'] else 'FAIL'} settle={row['settle_time_s']:.2f}s "
                  f"{row['failure_reason']}")
    return rows


def run_task_poses(world: SimWorld) -> list[dict]:
    """Approach / grasp / lift / place poses of the standard smoke layout,
    from the base parking pose the reference skills would use."""
    rows = []
    scene = SceneConfig(seed=0, objects=[SceneObjectSpec("stone", tuple(STONE)),
                                         SceneObjectSpec("cube", (0.40, 0.15, 0.88)),
                                         SceneObjectSpec("bottle", (0.55, -0.35, 0.915))])
    world.reset(scene)
    world.step(int(round(1.0 / world.timestep)))
    stone = world.body_pos("stone")
    bx, by, byaw = approach_base_pose(stone[:2], world.base_pose()[:2])
    world.teleport_base(bx, by, byaw)  # privileged shortcut: park where APPROACH would
    world.step(int(round(0.5 / world.timestep)))
    rot = approach_rotation(byaw)
    grasp = stone + np.array([0.0, 0.0, GRASP_DESCEND_OFFSET])
    poses = [
        ("approach", grasp + np.array([0.0, 0.0, PRE_REACH_HEIGHT]), True),
        ("grasp", grasp, True),
        ("lift", grasp + np.array([0.0, 0.0, 0.15]), True),
    ]
    for name, p, allow in poses:
        row = evaluate_target(world, name, p, rot, allow_object_contact=allow)
        row.update({"base_x": bx, "base_y": by, "base_yaw": byaw})
        rows.append(row)
        print(f"  task {name}: ik={'ok' if row['ik_success'] else 'FAIL'} usable={row['ik_solution_usable']} "
              f"track={'ok' if row['track_success'] else 'FAIL'} settle={row['settle_time_s']:.2f}s {row['failure_reason']}")
    # placement: re-park for the region (as MOVE_TO does), place pose 5 cm above the support
    px, py, pyaw = approach_base_pose(REGION[:2], world.base_pose()[:2])
    world.teleport_base(px, py, pyaw)
    world.step(int(round(0.5 / world.timestep)))
    place = REGION + np.array([0.0, 0.0, 0.05])
    row = evaluate_target(world, "place", place, approach_rotation(pyaw), allow_object_contact=True)
    row.update({"base_x": px, "base_y": py, "base_yaw": pyaw})
    rows.append(row)
    print(f"  task place: ik={'ok' if row['ik_success'] else 'FAIL'} usable={row['ik_solution_usable']} "
          f"track={'ok' if row['track_success'] else 'FAIL'} settle={row['settle_time_s']:.2f}s {row['failure_reason']}")
    return rows


def plot(rows: list[dict], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = [r for r in rows if r["name"] == "grid"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ax = axes[0]
    for r in grid:
        if r["track_success"]:
            ax.plot(r["x"], r["y"], "o", color="tab:green", ms=14)
        elif r["ik_solution_usable"]:
            ax.plot(r["x"], r["y"], "s", color="tab:orange", ms=14)
        elif r["ik_success"]:
            ax.plot(r["x"], r["y"], "D", color="tab:purple", ms=14)
        else:
            ax.plot(r["x"], r["y"], "x", color="tab:red", ms=16, mew=3)
        ax.annotate(f"{r['track_pos_err_mm']:.0f}mm\n{r['track_rot_err_deg']:.1f}°" if r["ik_success"] else "IK×",
                    (r["x"], r["y"]), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=7)
    ax.set_xlabel("x [m] (base frame, forward)")
    ax.set_ylabel("y [m] (base frame, left)")
    ax.set_title(f"5x5 grid @ z={TARGET_Z:.2f} m, tilt {math.degrees(APPROACH_TILT):.0f}°, azimuth "
                 f"{math.degrees(FINGER_AZIMUTH):.0f}°\ngreen=tracked & collision-free, orange=IK ok / tracking fail,\n"
                 "purple=IK solution collides, red x=IK failed")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(GRID_X[0] - 0.03, GRID_X[-1] + 0.03)
    ax.set_ylim(GRID_Y[0] - 0.03, GRID_Y[-1] + 0.05)  # headroom for the top-row annotations
    ax = axes[1]
    xs = sorted({r["x"] for r in grid})
    ys = sorted({r["y"] for r in grid})
    mat = np.full((len(ys), len(xs)), np.nan)
    for r in grid:
        mat[ys.index(r["y"]), xs.index(r["x"])] = r["track_pos_err_mm"] if r["ik_success"] else np.nan
    im = ax.imshow(mat, origin="lower", extent=[xs[0] - 0.02, xs[-1] + 0.02, ys[0] - 0.03, ys[-1] + 0.03],
                   cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="settled position residual [mm]")
    ax.set_title("tracking residual (NaN = IK failed)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="G1 right-arm reachability test")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/ik_reach"))
    args = parser.parse_args(argv)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out / run_id
    n = 1
    while out.exists():
        n += 1
        out = args.out / f"{run_id}_{n}"
    out.mkdir(parents=True)

    print("Declared workspace:")
    print(f"  grid x = {np.round(GRID_X, 3).tolist()} m, y = {np.round(GRID_Y, 3).tolist()} m (world = base frame)")
    print(f"  target height z = {TARGET_Z:.3f} m (table top {TABLE_TOP_Z} + 0.05)")
    print(f"  orientation: tilted top-down approach, tilt {math.degrees(APPROACH_TILT):.0f} deg, finger azimuth "
          f"{math.degrees(FINGER_AZIMUTH):.0f} deg (R columns = finger, y, approach):\n{np.round(TARGET_ROT, 3)}")
    print(f"  base pose {BASE_POSE}, waist (yaw, roll, pitch) = {WAIST_POSE}, initial arm q = {np.round(INITIAL_ARM_Q, 3).tolist()}")
    print(f"  success: IK residual < 1 mm / 0.5 deg (solver); tracking < {POS_TOL * 1000:.0f} mm / {math.degrees(ROT_TOL):.0f} deg\n")

    world = SimWorld()
    t0 = time.time()
    try:
        print("Grid:")
        rows = run_grid(world)
        print("Task poses (smoke_1 layout):")
        rows += run_task_poses(world)
    finally:
        world.close()
    wall = time.time() - t0

    fields = ["name", "x", "y", "z", "ik_success", "ik_pos_err_mm", "ik_rot_err_deg", "ik_iters", "ik_reason", "ik_ms",
              "ik_collision", "ik_solution_usable", "track_success", "track_pos_err_mm", "track_rot_err_deg",
              "settle_time_s", "track_collision", "collision_during_motion", "failure_reason", "base_x", "base_y", "base_yaw"]
    with (out / "reach_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    plot(rows, out / "reach_grid.png")

    grid = [r for r in rows if r["name"] == "grid"]
    task = [r for r in rows if r["name"] != "grid"]
    ok_track = [r for r in grid if r["track_success"]]
    summary = {
        "run_id": run_id,
        "declared_workspace": {"grid_x": GRID_X.tolist(), "grid_y": GRID_Y.tolist(), "target_z": TARGET_Z,
                               "base_pose": BASE_POSE, "waist_pose": WAIST_POSE, "initial_arm_q": INITIAL_ARM_Q,
                               "target_rotation": TARGET_ROT.tolist()},
        "grid_points": len(grid),
        "grid_ik_success": sum(r["ik_success"] for r in grid),
        "grid_ik_solution_usable": sum(r["ik_solution_usable"] for r in grid),
        "grid_tracking_success": len(ok_track),
        "grid_settle_time_s": {"mean": float(np.mean([r["settle_time_s"] for r in ok_track])) if ok_track else None,
                               "max": float(np.max([r["settle_time_s"] for r in ok_track])) if ok_track else None},
        "grid_tracked_pos_residual_mm": {"mean": float(np.mean([r["track_pos_err_mm"] for r in ok_track])) if ok_track else None,
                                         "max": float(np.max([r["track_pos_err_mm"] for r in ok_track])) if ok_track else None},
        "grid_tracked_rot_residual_deg": {"mean": float(np.mean([r["track_rot_err_deg"] for r in ok_track])) if ok_track else None,
                                          "max": float(np.max([r["track_rot_err_deg"] for r in ok_track])) if ok_track else None},
        "grid_failures": [{"x": r["x"], "y": r["y"], "reason": r["failure_reason"] or r["ik_collision"]} for r in grid if not r["track_success"]],
        "task_poses": {r["name"]: {"ik_success": r["ik_success"], "usable": r["ik_solution_usable"],
                                   "track_success": r["track_success"], "settle_time_s": r["settle_time_s"],
                                   "pos_err_mm": r["track_pos_err_mm"], "rot_err_deg": r["track_rot_err_deg"],
                                   "collision": r["track_collision"], "failure_reason": r["failure_reason"]} for r in task},
        "wall_s": wall,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True))
    print(f"\nGrid: IK {summary['grid_ik_success']}/{len(grid)}, usable (collision-free IK) "
          f"{summary['grid_ik_solution_usable']}/{len(grid)}, tracked {len(ok_track)}/{len(grid)}")
    if ok_track:
        print(f"  tracked residuals: pos mean {summary['grid_tracked_pos_residual_mm']['mean']:.2f} mm "
              f"(max {summary['grid_tracked_pos_residual_mm']['max']:.2f}), rot mean "
              f"{summary['grid_tracked_rot_residual_deg']['mean']:.2f} deg (max {summary['grid_tracked_rot_residual_deg']['max']:.2f}); "
              f"settle mean {summary['grid_settle_time_s']['mean']:.2f} s (max {summary['grid_settle_time_s']['max']:.2f})")
    for f in summary["grid_failures"]:
        print(f"  FAILED grid point ({f['x']:.2f},{f['y']:.2f}): {f['reason']}")
    task_ok = all(r["track_success"] for r in task)
    print(f"Task poses: {'ALL tracked collision-free' if task_ok else 'FAILURES'}: "
          + ", ".join(f"{r['name']}={'ok' if r['track_success'] else 'FAIL'}" for r in task))
    print(f"Outputs: {out / 'reach_results.csv'}, {out / 'reach_grid.png'}, {out / 'summary.json'} (wall {wall:.1f} s)")
    return 0 if task_ok else 1


if __name__ == "__main__":
    sys.exit(main())
