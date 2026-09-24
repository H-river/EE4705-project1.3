# Owner: backbone (ALL)
"""Final-run scoring: merge lane records, write E2E_TABLE.md, print the score.

Correct outcome per trial (expected.outcome in the trial YAML):
  success  claimed_success and actual_success
  reject   outcome REFUSED
  clarify  a clarification was asked and its scripted answer consumed, and the
           task was then claimed and achieved
False claim: claimed_success and actual_success is not True.

The attribution column is a first-pass heuristic from the recorded events
(rules of the earlier rounds); the run report may override it after reading
the trial.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import re

import yaml


def expected_outcome(trial: dict) -> str:
    e = trial.get("expected", {}) or {}
    if "outcome" in e:
        return e["outcome"]
    if not e.get("feasible", True):
        return "reject"
    return "clarify" if e.get("clarification_required") else "success"


def is_correct(record: dict, expected: str) -> bool:
    claimed_achieved = bool(record.get("claimed_success")) and record.get("actual_success") is True
    if expected == "reject":
        return record.get("outcome") == "REFUSED"
    if expected == "clarify":
        asked = any(c.get("response") is not None for c in record.get("clarifications", []))
        return asked and claimed_achieved
    return claimed_achieved


def is_false_claim(record: dict) -> bool:
    return bool(record.get("claimed_success")) and record.get("actual_success") is not True


def _failed_codes(record):
    return [e.get("error") for e in record.get("events", [])
            if e.get("type") == "action" and not e.get("success")]


def _failed_skills(record):
    return [f"{e.get('skill')}:{e.get('error')}" for e in record.get("events", [])
            if e.get("type") == "action" and not e.get("success")]


def attribute(record: dict, expected: str) -> tuple[str, str]:
    """(module, short cause) for an incorrect trial; ('-', '') when correct."""
    if is_correct(record, expected):
        return "-", ""
    outcome = record.get("outcome")
    error = str(record.get("error") or "")
    events = record.get("events", [])
    plans = [e for e in events if e.get("type") == "plan"]
    rejected = sum(e.get("type") == "plan_rejected" for e in events)
    codes = _failed_codes(record)
    execs = record.get("extra", {}).get("execution_records") or []
    contact = "contact" in json.dumps(execs)[:200000].lower()
    if outcome == "TIMEOUT":
        return "unknown", "killed at the wall-clock limit"
    if outcome == "ERROR" and re.search(r"HTTP 5\d\d|HTTP 429|name resolution|ConnectionError|content filter|timed out",
                                        error, re.I):
        return "provider", "provider error: " + error.strip().splitlines()[-1][:80]
    if outcome == "ERROR":
        return "unknown", "ERROR: " + (error.strip().splitlines()[-1][:80] if error.strip() else "?")
    if expected == "reject":
        return "B", f"expected refusal, got {outcome}"
    if outcome == "REFUSED":
        return "B", "refused a feasible task"
    if outcome == "CLARIFICATION_EXHAUSTED":
        return "B", "clarification loop"
    if expected == "clarify" and not record.get("clarifications"):
        return "B", "did not ask for clarification"
    if is_false_claim(record):
        return "A", "false claim: vision verification passed, oracle disagrees"
    if record.get("actual_success") is True and not record.get("claimed_success"):
        return "A", "placed correctly but verification never passed"
    if record.get("extra", {}).get("wrong_object"):
        return "B", "wrong object manipulated"
    if rejected >= 3:
        return "B", f"{rejected} plans rejected by the contract"
    if outcome == "SEARCH_EXHAUSTED":
        return "A/C", "search exhausted (" + ",".join(dict.fromkeys(codes)) + ")"
    if contact or any(c in ("UNREACHABLE", "TIMEOUT", "GRASP_MISSED", "NOT_HOLDING", "PLACE_FAILED")
                      for c in codes):
        main = collections.Counter(c for c in codes if c).most_common(1)
        return "C", "execution failures " + (main[0][0] if main else "") + (" +contact" if contact else "")
    if "TARGET_LOST" in codes or "TARGET_UNRESOLVABLE" in codes:
        return "A", "target lost / not localized"
    if len(plans) >= 8:
        return "B", "planning loop"
    return "unknown", f"{outcome} codes={','.join(codes[-4:])}"


def _link(src: pathlib.Path, dst: pathlib.Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def merge_and_table(root: pathlib.Path, trials_dir: pathlib.Path, ids: list[str]) -> dict:
    root = pathlib.Path(root)
    trials = {}
    for p in sorted(pathlib.Path(trials_dir).glob("*.yaml")):
        spec = yaml.safe_load(p.read_text())
        trials[str(spec.get("id", p.stem))] = spec
    merged = root / "merged"
    merged.mkdir(exist_ok=True)
    rows = []
    for tid in ids:
        paths = sorted(root.glob(f"job_*/runs/*/{tid}/trial_record.json"), key=lambda p: p.stat().st_mtime)
        record = json.loads(paths[-1].read_text()) if paths else {"trial_id": tid, "outcome": "MISSING"}
        if paths:
            _link(paths[-1].parent, merged / tid)
        exp = expected_outcome(trials.get(tid, {}))
        module, cause = attribute(record, exp)
        stats = record.get("api_stats", {}) or {}
        calls = sum(int((stats.get(r) or {}).get("attempts", 0)) for r in ("A", "B"))
        events = record.get("events", [])
        rows.append({
            "trial": tid, "expected": exp, "outcome": record.get("outcome"),
            "claimed": record.get("claimed_success"), "actual": record.get("actual_success"),
            "correct": is_correct(record, exp), "false_claim": is_false_claim(record),
            "plans": sum(e.get("type") == "plan" for e in events),
            "actions": sum(e.get("type") == "action" for e in events),
            "wall_s": round(float((record.get("timings") or {}).get("wall_s", 0.0)), 1),
            "calls": calls, "calls_A": int((stats.get("A") or {}).get("attempts", 0)),
            "calls_B": int((stats.get("B") or {}).get("attempts", 0)),
            "failed": ",".join(_failed_skills(record)[-5:]), "module": module, "cause": cause,
            "attempts_on_disk": len(paths),
        })
    manip = [r for r in rows if r["expected"] == "success"]
    other = [r for r in rows if r["expected"] != "success"]
    score = {
        "n": len(rows), "correct": sum(r["correct"] for r in rows),
        "false_claims": sum(r["false_claim"] for r in rows),
        "manipulation_correct": sum(r["correct"] for r in manip), "manipulation_n": len(manip),
        "reject_clarify_correct": sum(r["correct"] for r in other), "reject_clarify_n": len(other),
        "calls": sum(r["calls"] for r in rows),
    }
    score["line"] = f"SCORE {score['correct']}/{score['n']} false_claims={score['false_claims']}"
    hist = collections.Counter(f"{r['module']}: {r['cause'].split('(')[0].strip()}" for r in rows if not r["correct"])
    lines = [f"# E2E table — {root.name}", "",
             f"**{score['line']}** · manipulation claimed∧achieved {score['manipulation_correct']}/"
             f"{score['manipulation_n']} · reject/clarify correct {score['reject_clarify_correct']}/"
             f"{score['reject_clarify_n']} · live calls (A+B attempts) {score['calls']}", "",
             "| # | trial | expected | outcome | claimed | actual | correct | plans | actions | wall s | calls A/B "
             "| last failed actions | module | cause |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['trial']} | {r['expected']} | {r['outcome']} | {r['claimed']} | {r['actual']} | "
                     f"{'✔' if r['correct'] else ('✘ FALSE CLAIM' if r['false_claim'] else '✘')} | {r['plans']} | "
                     f"{r['actions']} | {r['wall_s']} | {r['calls_A']}/{r['calls_B']} | {r['failed']} | "
                     f"{r['module']} | {r['cause']} |")
    lines += ["", "## Cause histogram (incorrect trials)", "", "| module: cause | trials |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in hist.most_common()]
    by_module = collections.Counter(r["module"] for r in rows if not r["correct"])
    lines += ["", "By module: " + ", ".join(f"{k} {v}" for k, v in by_module.most_common())]
    (root / "E2E_TABLE.md").write_text("\n".join(lines) + "\n")
    (root / "rows.json").write_text(json.dumps({"score": score, "rows": rows}, indent=2))
    return score


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Re-merge a --jobs run root and print its score")
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--trials", type=pathlib.Path, default=pathlib.Path("eval/trials/final50"))
    a = ap.parse_args()
    meta = json.loads((a.root / "parallel_meta.json").read_text())
    ids = [t for lane in meta["lanes"] for t in lane]
    order = sorted(ids)
    print(merge_and_table(a.root, a.trials, order)["line"])
