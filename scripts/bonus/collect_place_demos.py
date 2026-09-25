#!/usr/bin/env python
"""Stage 8.3 (bonus): expert PLACE demonstrations from the scripted executor.

Each episode runs a full scripted pick-and-place (GT perception + RulePlanner +
StudentCExecutor, 0 API calls) on a Stage-1 scene (stone/cube ±10 cm, distractor
50 %).  Only the PLACE carry is recorded: from the post-MOVE_TO pose (holding the
object above the region) to the release pose, i.e. the ``skills.move_to(env,
release_pos)`` call inside StudentCExecutor._place, plus 0.3 s holding still.
Recorded at 10 Hz like the grasp demos: head RGB 224², arm q, the release pose
in the base frame (target_b), action = next q.  Kept only when the carry reached
the release pose AND the whole episode placed the target in the region (oracle).

    python scripts/bonus/collect_place_demos.py --n-keep 2000 --workers 6 --seed 2
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

import collect_grasp_demos as cg  # noqa: E402

HOLD_S = 0.3
INSTRUCTION = {"stone": "Move the gray stone to the red area.", "cube": "Move the blue cube to the red area."}
_W: dict = {}


def run_episode(i: int, seed: int) -> dict:
    from core import skills
    from core.mocks import GTPerception, RulePlanner, ScriptedClarifier
    from core.obs_store import ObservationStore
    from core.orchestrator import Orchestrator, OrchestratorConfig
    from core.types import SceneConfig, SceneObjectSpec
    import executor.student_c as sc

    world, env, oracle = _W["world"], _W["env"], _W["oracle"]
    rng = np.random.default_rng([seed, i])
    cls, objs, robot = cg.sample_scene(rng)
    world.reset(SceneConfig(seed=seed * 100000 + i, objects=[SceneObjectSpec(n, p, y) for n, p, y in objs],
                            robot_init=robot))
    world.step(int(round(1.0 / world.timestep)))
    recs = []

    def recording_move_to(env_, pos):
        rec = cg.Recorder(env_, world, pos)
        r = skills.move_to(rec, pos)
        env_.set_arm_joint_target(np.asarray(env_.get_arm_q()))  # hold still (the demo ends at rest)
        rec.step(int(round(HOLD_S / world.timestep)))
        recs.append((rec, r, float(np.linalg.norm(env_.get_ee_pos() - np.asarray(pos)))))
        return r

    sc.place_primitive = lambda: recording_move_to  # recording hook for this collector only
    store = ObservationStore()
    renv = type(env)(world, store=store)
    orch = Orchestrator(GTPerception(oracle), RulePlanner(), sc.StudentCExecutor(), renv, ScriptedClarifier([]),
                        config=OrchestratorConfig(), store=store)
    t0 = time.perf_counter()
    try:
        episode = orch.run(INSTRUCTION[cls])
        outcome = getattr(episode, "outcome", None)
    except Exception as exc:  # noqa: BLE001
        return {"episode": i, "seed": [seed, i], "class": cls, "success": False, "reason": f"exception:{exc!r}"[:200],
                "n_steps": 0, "distractor": len(objs) > 1}
    world.step(int(round(0.5 / world.timestep)))
    placed = bool(oracle.object_in_region(cls))
    meta = {"episode": i, "seed": [seed, i], "class": cls, "objects": objs, "robot_init": robot,
            "distractor": len(objs) > 1, "outcome": str(getattr(outcome, "value", outcome)),
            "wall_s": time.perf_counter() - t0, "n_place_calls": len(recs)}
    if len(recs) != 1:
        return {**meta, "success": False, "reason": f"place_calls={len(recs)}", "n_steps": 0}
    rec, r, err = recs[0]
    ok = bool(r.success and err < skills.EE_POS_TOL and placed)
    meta.update(success=ok, reason="placed" if ok else ("not_placed" if not placed else "carry_failed"),
                release_error_m=err, n_steps=len(rec.rows["t"]),
                object_offset_xy=(oracle.object_pos(cls)[:2] - np.asarray(oracle.region_bounds().center_xy)).tolist())
    return {**meta, "rows": rec.rows}


def _worker(args):
    i, seed, out = args
    if "world" not in _W:
        from core.env import RobotEnv
        from core.oracle import EvalOracle
        from core.world import SimWorld
        _W["world"] = SimWorld()
        _W["env"] = RobotEnv(_W["world"])
        _W["oracle"] = EvalOracle(_W["world"])
    ep = run_episode(i, seed)
    if ep["success"]:
        cg.save_h5(out / f"ep_{i:05d}.h5", ep)
    return {k: v for k, v in ep.items() if k != "rows"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "runs/bonus/demos_place")
    ap.add_argument("--n-keep", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "episodes.jsonl"
    done = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            d = json.loads(line)
            done[d["episode"]] = d
    kept = sum(1 for d in done.values() if d["success"] and (args.out / f"ep_{d['episode']:05d}.h5").exists())
    todo = (i for i in range(10**7) if i not in done)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool, log_path.open("a") as log:
        pending = []
        while kept < args.n_keep:
            while len(pending) < args.workers * 2:
                pending.append(pool.apply_async(_worker, ((next(todo), args.seed, args.out),)))
            r = pending.pop(0).get()
            log.write(json.dumps(r) + "\n")
            log.flush()
            done[r["episode"]] = r
            kept += int(r["success"])
            if len(done) % 50 == 0:
                print(f"{len(done)} tried, {kept} kept, {time.time() - t0:.0f}s", flush=True)
        for p in pending:
            r = p.get()
            log.write(json.dumps(r) + "\n")
    cg.write_manifest(args.out, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
