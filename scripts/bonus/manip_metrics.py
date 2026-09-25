#!/usr/bin/env python
"""Stage 6 (bonus): grasp metrics from eval.runner --mode manipulation runs.

Per trial (runs/bonus/manip/<name>/merged/<id>/trial_record.json):
  grasp_ok      first GRASP action succeeded AND the oracle's held object is the target
  grasp_any     some GRASP in the episode ended holding the target (after retries/replans)
  wrong_object  a successful GRASP held a non-target object
  undetected    GRASP reported success but the held object is wrong/none (post-condition miss)
  false_claim   the episode claimed success but the oracle says it failed
  grasp_s       sim duration of the first GRASP action (open + primitive + lift check + retreat)

    python scripts/bonus/manip_metrics.py runs/bonus/manip/scripted_C1 [...]
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml


def trial_metrics(rec: dict) -> dict:
    target = yaml.safe_load(pathlib.Path(rec["extra"]["trial_path"]).read_text())["expected"]["target"]
    ex = rec["extra"].get("execution_records", [])
    grasps = [e for e in ex if e["result"]["action"]["skill"] == "GRASP"]
    held = [g["held_gt_id"] for g in rec["extra"].get("grasp_records", [])]
    first = grasps[0] if grasps else None
    first_ok = bool(first and first["result"]["success"] and held and held[0] == target)
    reported = sum(1 for g in grasps if g["result"]["success"])
    return {
        "trial": rec["trial_id"], "target": target, "n_grasp_actions": len(grasps),
        "grasp_ok": first_ok, "grasp_any": target in held,
        "wrong_object": sum(h != target for h in held),
        "undetected": max(0, reported - sum(h == target for h in held)),
        "false_claim": bool(rec["claimed_success"] and not rec["actual_success"]),
        "task_success": bool(rec["actual_success"]), "outcome": rec["outcome"],
        "grasp_s": (first["end_sim_s"] - first["start_sim_s"]) if first else None,
        "grasp_error": first["result"]["error_code"] if first else None,
    }


def run_metrics(run: pathlib.Path) -> dict:
    rows = [trial_metrics(json.loads(p.read_text())) for p in sorted((run / "merged").glob("*/trial_record.json"))]
    n = len(rows)
    ok_t = [r["grasp_s"] for r in rows if r["grasp_ok"] and r["grasp_s"] is not None]
    return {"run": str(run), "n": n,
            "grasp_ok": sum(r["grasp_ok"] for r in rows), "grasp_any": sum(r["grasp_any"] for r in rows),
            "wrong_object": sum(r["wrong_object"] for r in rows), "undetected": sum(r["undetected"] for r in rows),
            "false_claims": sum(r["false_claim"] for r in rows), "task_success": sum(r["task_success"] for r in rows),
            "mean_grasp_s": sum(ok_t) / len(ok_t) if ok_t else None, "rows": rows}


def main(argv=None) -> int:
    for a in (argv or sys.argv[1:]):
        m = run_metrics(pathlib.Path(a))
        print(json.dumps({k: v for k, v in m.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
