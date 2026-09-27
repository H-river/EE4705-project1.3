#!/usr/bin/env python
"""Headline numbers for the report, from the 44/50 run (0 API calls).

Source run: runs/final2/full_final (tag final2-44). Reads the 50 trial_record.json files and rows.json and
prints JSON to stdout (saved as docs/submission/numbers.json).

    python scripts/submission_numbers.py > docs/submission/numbers.json
"""

from __future__ import annotations

import glob
import json
import pathlib
import statistics as st
import subprocess

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/final2/full_final"
GROUPS = {  # mutually exclusive, 50 in total (membership by trial id)
    "standard_c_layouts": [f"f{i:02d}" for i in range(6, 26)],
    "scene_variation": ["f02"] + [f"f{i:02d}" for i in range(26, 42)],
    "instruction_variation": ["f03"] + [f"f{i:02d}" for i in range(42, 47)],
    "search": ["f04"],
    "two_stone_colour": ["f47", "f48"],
    "clarification_or_infeasible": ["f05", "f49", "f50"],
    "smoke_standard": ["f01"],
}


def main() -> None:
    recs = {}
    for p in glob.glob(str(RUN / "job_*/runs/*/*/trial_record.json")):
        r = json.loads(pathlib.Path(p).read_text())
        recs[r["trial_id"]] = r
    rows = {r["trial"]: r for r in json.loads((RUN / "rows.json").read_text())["rows"]}
    score = json.loads((RUN / "rows.json").read_text())["score"]
    assert len(recs) == 50 and len(rows) == 50
    trial = {tid: yaml.safe_load(pathlib.Path(r["extra"]["trial_path"]).read_text()) for tid, r in recs.items()}

    conf = {"claimed_and_actual": 0, "claimed_not_actual": 0, "actual_not_claimed": 0, "neither": 0,
            "correct_refusal": 0, "clarification_then_success": 0}
    for tid, r in recs.items():
        c, a = bool(r["claimed_success"]), bool(r["actual_success"])
        if r["outcome"] == "REFUSED":
            conf["correct_refusal" if rows[tid]["correct"] else "neither"] += 1
            continue
        conf["claimed_and_actual" if c and a else "claimed_not_actual" if c else
             "actual_not_claimed" if a else "neither"] += 1
        if c and a and (trial[tid].get("expected") or {}).get("outcome") == "clarify":
            conf["clarification_then_success"] += 1

    ok = [tid for tid in recs if recs[tid]["claimed_success"] and recs[tid]["actual_success"]]
    sim_ok = [recs[t]["timings"]["sim_end_s"] for t in ok]
    wall = [recs[t]["timings"]["wall_s"] for t in recs]
    wall_ok = [recs[t]["timings"]["wall_s"] for t in ok]

    def tok(role, key):
        return sum((r.get("api_stats", {}).get(role) or {}).get(key, 0) for r in recs.values())

    calls_a = [rows[t]["calls_A"] for t in rows]
    calls_b = [rows[t]["calls_B"] for t in rows]
    groups = {g: {"n": len(ids), "correct": sum(rows[t]["correct"] for t in rows if t[:3] in ids),
                  "ids": ids} for g, ids in GROUPS.items()}
    per_object = {}
    for tid, t in trial.items():
        tgt = (t.get("expected") or {}).get("target")
        cls = "none (reject)" if tgt is None else "stone" if tgt.startswith("stone") else tgt
        d = per_object.setdefault(cls, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += bool(rows[tid]["correct"])
    failures = [{"trial": t, "outcome": rows[t]["outcome"], "module": rows[t]["module"], "cause": rows[t]["cause"],
                 "failed_actions": rows[t]["failed"]} for t in sorted(rows) if not rows[t]["correct"]]
    commit = subprocess.run(["git", "rev-parse", "--short", "final2-44"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    out = {
        "run": "runs/final2/full_final", "tag": "final2-44", "commit": commit,
        "score": score,
        "confusion": conf,
        "sim_task_time_success_s": {"n": len(sim_ok), "mean": st.mean(sim_ok), "median": st.median(sim_ok)},
        "wall_s_per_trial": {"mean": st.mean(wall), "median": st.median(wall),
                             "mean_successful": st.mean(wall_ok)},
        "calls_per_trial": {"A_mean": st.mean(calls_a), "B_mean": st.mean(calls_b),
                            "total_mean": st.mean(a + b for a, b in zip(calls_a, calls_b)),
                            "A_total": sum(calls_a), "B_total": sum(calls_b)},
        "tokens_final_run": {"A_prompt": tok("A", "prompt_tokens"), "A_completion": tok("A", "completion_tokens"),
                             "B_prompt": tok("B", "prompt_tokens"), "B_completion": tok("B", "completion_tokens"),
                             "A_latency_s": tok("A", "total_latency_s"), "B_latency_s": tok("B", "total_latency_s")},
        "groups": groups,
        "per_object": per_object,
        "failures": failures,
    }
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
