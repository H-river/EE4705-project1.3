"""Final2 6.6: how e2e episodes ended, per orchestrator budget (0 live calls).

    python -m eval.orchestrator_budgets [--runs runs/final runs/final2]

Every trial_record.json under the given roots (the newest record per run
root and trial) is classified by how it ended:
  success          CLAIMED_SUCCESS
  max_search       SEARCH_EXHAUSTED (max_search_attempts)
  max_replans      FAILED / LIMIT_EXCEEDED after the replan budget (no 'limit' event)
  max_total_plans  LIMIT_EXCEEDED with limit=max_total_plans
  max_total_actions LIMIT_EXCEEDED with limit=max_total_actions
  max_clarify      CLARIFICATION_EXHAUSTED
  refused / error / other
It also reports, per ending, how many episodes had the object in the region
by the oracle (actual_success) and the plans / actions used, which is what a
new limit would trade against.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import defaultdict


def ending(record: dict) -> str:
    out = record.get("outcome")
    events = record.get("events") or []
    limits = [e.get("which") for e in events if e.get("type") == "limit"]
    if out == "CLAIMED_SUCCESS":
        return "success"
    if out == "SEARCH_EXHAUSTED":
        return "max_search"
    if out == "CLARIFICATION_EXHAUSTED":
        return "max_clarify"
    if out == "LIMIT_EXCEEDED" and limits:
        return limits[-1]
    if out in ("FAILED", "LIMIT_EXCEEDED"):
        return "max_replans"
    if out == "REFUSED":
        return "refused"
    return "error" if out in ("ERROR", "TIMEOUT", "INTERRUPTED") else "other"


def collect(roots):
    latest = {}
    for root in roots:
        for path in pathlib.Path(root).rglob("trial_record.json"):
            run = next((p for p in path.parents if p.parent.name in ("final", "final2")), path.parent)
            key = (str(run), path.parent.name)
            if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
                latest[key] = path
    return [json.loads(p.read_text()) for p in latest.values()]


def table(records):
    rows = defaultdict(list)
    for r in records:
        ev = r.get("events") or []
        rows[ending(r)].append((bool(r.get("actual_success")),
                                sum(e.get("type") == "plan" for e in ev),
                                sum(e.get("type") == "action" for e in ev)))
    order = ["success", "max_search", "max_replans", "max_total_plans", "max_total_actions",
             "max_clarify", "refused", "error", "other"]
    lines = ["| ending | episodes | object placed (oracle) | plans median / max | actions median / max |",
             "|---|---|---|---|---|"]
    for k in order:
        if k not in rows:
            continue
        v = rows[k]
        plans, actions = [x[1] for x in v], [x[2] for x in v]
        lines.append(f"| {k} | {len(v)} | {sum(x[0] for x in v)} | {statistics.median(plans):g} / {max(plans)} "
                     f"| {statistics.median(actions):g} / {max(actions)} |")
    return "\n".join(lines), rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", default=["runs/final", "runs/final2"])
    a = ap.parse_args(argv)
    text, _ = table(collect(a.runs))
    print(text)


if __name__ == "__main__":
    main()
