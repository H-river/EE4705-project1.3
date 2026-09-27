#!/usr/bin/env python
"""8.3: placement metrics from full-executor runs.
place_ok = the PLACE action succeeded (release + unchanged vision verification); task = oracle success;
offset = |xy| of the object's final offset from the region centre as measured by the final placement check
(GT perception), in cm.   python scripts/bonus/place_metrics.py runs/bonus/manip/<run> [...]"""
import json, pathlib, re, sys
import numpy as np


def run(path: pathlib.Path) -> dict:
    recs = sorted((path / "merged").glob("*/trial_record.json")) or sorted(path.glob("*_manipulation/*/trial_record.json"))
    n = task = place_ok = places = 0
    offs = []
    for p in recs:
        r = json.loads(p.read_text())
        n += 1
        task += bool(r["actual_success"])
        pl = [e for e in r["extra"].get("execution_records", []) if e["result"]["action"]["skill"] == "PLACE"]
        places += len(pl)
        place_ok += any(e["result"]["success"] for e in pl)
        m = re.search(r"offset xyz=\[([-\d.e]+), ([-\d.e]+)", (r["extra"].get("verification") or {}).get("detail", ""))
        if m and r["actual_success"]:
            offs.append(100 * float(np.hypot(float(m.group(1)), float(m.group(2)))))
    out = {"run": str(path), "n": n, "task_success": task, "place_ok": place_ok, "place_actions": places,
           "offset_cm_mean": float(np.mean(offs)) if offs else None, "offset_cm_p90": float(np.percentile(offs, 90)) if offs else None}
    side = pathlib.Path(str(path) + ".place_calls.jsonl")
    if side.exists():
        rows = [json.loads(l) for l in side.read_text().splitlines()]
        out.update(learned_calls=len(rows), release_err_cm_mean=100 * float(np.mean([x["ee_error"] for x in rows])),
                   reached=sum(x["ee_error"] < 0.012 for x in rows))
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(json.dumps(run(pathlib.Path(a))))
