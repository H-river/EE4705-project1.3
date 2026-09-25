#!/usr/bin/env python
"""Bonus stage 6: build docs/bonus/RESULTS.md tables from the evaluation outputs.

Sources:
  runs/bonus/manip/<label>_<cell>/       full executor (eval.runner --mode manipulation), 30 trials
  runs/bonus/manip/<label>_<cell>.grasp_calls.jsonl   learned-skill sidecar (stop reason, rollout time)
  runs/bonus/eval/<label>_<cell>.jsonl   skill level (scripts/bonus/eval_grasp.py), 30 episodes
Prints the markdown blocks; the prose around them lives in docs/bonus/RESULTS.md.

    python scripts/bonus/make_results.py > runs/bonus/results_tables.md
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bonus"))

import manip_metrics  # noqa: E402

CELLS = ["C1", "C2", "C3", "C4"]
CELL_NAMES = {"C1": "C1 in-dist.", "C2": "C2 ±20 cm", "C3": "C3 bottle (OOD)", "C4": "C4 near distractor"}


def manip(label: str, cell: str):
    run = ROOT / "runs/bonus/manip" / f"{label}_{cell}"
    if not run.exists():
        return None
    return manip_metrics.run_metrics(run)


def skill(label: str, cell: str):
    p = ROOT / "runs/bonus/eval" / f"{label}_{cell}.jsonl"
    if not p.exists():
        return None
    return json.loads(p.read_text().splitlines()[0])["summary"]


def fmt_t(v):
    return "—" if v is None else f"{v:.2f}"


def main_table(policies: list[tuple[str, str]]) -> str:
    out = ["| Policy | " + " | ".join(CELL_NAMES[c] for c in CELLS) + " | WRONG_OBJECT | undetected | false claims |",
           "|---|" + "---|" * (len(CELLS) + 3)]
    for label, name in policies:
        cells, wo, und, fc = [], 0, 0, 0
        for c in CELLS:
            m = manip(label, c)
            if m is None:
                cells.append("—")
                continue
            cells.append(f"{m['grasp_ok']}/{m['n']}")
            wo += m["wrong_object"]
            und += m["undetected"]
            fc += m["false_claims"]
        out.append(f"| {name} | " + " | ".join(cells) + f" | {wo} | {und} | {fc} |")
    return "\n".join(out)


def task_table(policies) -> str:
    out = ["| Policy | " + " | ".join(CELL_NAMES[c] for c in CELLS) + " |", "|---|" + "---|" * len(CELLS)]
    for label, name in policies:
        row = []
        for c in CELLS:
            m = manip(label, c)
            row.append("—" if m is None else f"{m['task_success']}/{m['n']}")
        out.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(out)


def skill_table(policies) -> str:
    out = ["| Policy | " + " | ".join(f"{CELL_NAMES[c]} (t̄ s)" for c in CELLS) + " | WRONG_OBJECT |",
           "|---|" + "---|" * (len(CELLS) + 1)]
    for label, name in policies:
        row, wo = [], 0
        for c in CELLS:
            s = skill(label, c)
            if s is None:
                row.append("—")
                continue
            row.append(f"{s['success']}/{s['n']} ({fmt_t(s['mean_time_to_attach_s'])})")
            wo += s["wrong_object"]
        out.append(f"| {name} | " + " | ".join(row) + f" | {wo} |")
    return "\n".join(out)


def detection_table(policies) -> str:
    """Every learned-skill miss inside the full executor, and what the executor made of it."""
    out = ["| Policy | learned-skill calls | misses (no attach) | GRASP actions reported success / failed "
           "(≤ 2 skill calls each) | episodes recovered after a miss | undetected (reported success, wrong/no object) "
           "| false task claims |", "|---|---|---|---|---|---|---|"]
    for label, name in policies:
        if label == "scripted":
            continue
        calls = misses = failed = okd = recovered = und = fc = 0
        for c in CELLS:
            side = ROOT / "runs/bonus/manip" / f"{label}_{c}.grasp_calls.jsonl"
            m = manip(label, c)
            if m is None or not side.exists():
                continue
            rows = [json.loads(line) for line in side.read_text().splitlines()]
            calls += len(rows)
            misses += sum(not r["attached"] for r in rows)
            for p in sorted((ROOT / "runs/bonus/manip" / f"{label}_{c}").glob("**/trial_record.json")):
                if p.parent.parent.name == "merged" or "_manipulation" in p.parent.parent.name:
                    rec = json.loads(p.read_text())
                    g = [e for e in rec["extra"].get("execution_records", []) if e["result"]["action"]["skill"] == "GRASP"]
                    failed += sum(not e["result"]["success"] for e in g)
                    okd += sum(bool(e["result"]["success"]) for e in g)
            recovered += sum(1 for r in m["rows"] if r["grasp_any"] and not r["grasp_ok"])
            und += m["undetected"]
            fc += m["false_claims"]
        out.append(f"| {name} | {calls} | {misses} | {okd} / {failed} | {recovered} | {und} | {fc} |")
    return "\n".join(out)


def ablation_table(rows: list[tuple[str, str]]) -> str:
    out = ["| Variant | full executor C1 (grasp) | skill level C1 | mean time-to-attach (skill, s) | undetected |",
           "|---|---|---|---|---|"]
    for label, name in rows:
        m, s = manip(label, "C1"), skill(label, "C1")
        out.append(f"| {name} | {'—' if m is None else str(m['grasp_ok']) + '/' + str(m['n'])} | "
                   f"{'—' if s is None else str(s['success']) + '/' + str(s['n'])} | "
                   f"{'—' if s is None else fmt_t(s['mean_time_to_attach_s'])} | "
                   f"{'—' if m is None else m['undetected']} |")
    return "\n".join(out)


def render() -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_tables()
    return buf.getvalue()


def main() -> int:
    """No args: print the tables. --write: replace the block between the TABLES markers in docs/bonus/RESULTS.md."""
    if "--write" in sys.argv:
        doc = ROOT / "docs/bonus/RESULTS.md"
        text = doc.read_text()
        a, b = "<!-- TABLES:BEGIN (scripts/bonus/make_results.py --write) -->", "<!-- TABLES:END -->"
        i, j = text.index(a) + len(a), text.index(b)
        doc.write_text(text[:i] + "\n" + render() + "\n" + text[j:])
        print(f"updated {doc}")
        return 0
    print(render())
    return 0


def _print_tables() -> None:
    policies = [("scripted", "Scripted skill"), ("act", "ACT"), ("diffusion", "Diffusion Policy"),
                ("mlp", "BC-MLP")]
    policies = [p for p in policies if any(manip(p[0], c) or skill(p[0], c) for c in CELLS)]
    print("### Full executor: grasp success (first GRASP holds the intended object), 30 episodes/cell\n")
    print(main_table(policies))
    print("\n### Full executor: task success (object placed in the region, oracle)\n")
    print(task_table(policies))
    print("\n### Full executor: learned-grasp failures and the unchanged post-condition\n")
    print(detection_table(policies))
    print("\n### Skill level: grasp success and mean time-to-attach, 30 episodes/cell\n")
    print(skill_table(policies))
    abl = [("act_nas10", "ACT n_action_steps 10"), ("act", "ACT n_action_steps 25 (trained)"),
           ("act_nas50", "ACT n_action_steps 50"),
           ("diffusion_inf5", "DP DDIM 5 steps"), ("diffusion", "DP DDIM 10 steps (trained)"),
           ("diffusion_inf50", "DP DDIM 50 steps"), ("diffusion_ddpm50", "DP DDPM 50 steps"),
           ("act_state", "ACT state-only"), ("diffusion_state", "DP state-only")]
    abl = [a for a in abl if manip(a[0], "C1") or skill(a[0], "C1")]
    print("\n### Ablations on C1 (30 episodes each)\n")
    print(ablation_table(abl))


if __name__ == "__main__":
    raise SystemExit(main())
