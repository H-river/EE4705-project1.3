"""Aggregate Student B planner audits into docs/validation/B_METRICS.md.

    python -m eval.b_metrics [--runs runs/night] [--out docs/validation/B_METRICS.md]
                             [--variation runs/night/planning_variation]

Reads every B audit JSON (files carrying ``system_prompt`` and ``responses``)
under ``--runs``. Only responses whose source is ``live`` or ``cache`` count
(fixtures are ignored). Call types come from the planner input:
post-clarification (``clarification`` set), held-object replan
(``execution_context.held_instance_id`` set), other replan
(``history_total`` > 0) and initial plan.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from collections import Counter, defaultdict


def load_audits(root: pathlib.Path) -> list[dict]:
    audits, seen = [], set()
    for path in sorted(root.rglob("*.json")):
        if "cache" in path.parts:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(data, dict) and "system_prompt" in data and "responses" in data):
            continue
        key = data.get("audit_path") or str(path)
        if key in seen:
            continue
        seen.add(key)
        if data.get("response_source") not in ("live", "cache"):
            continue
        data["_file"] = str(path)
        audits.append(data)
    return audits


def call_type(audit: dict) -> str:
    inp = audit.get("input", {})
    if inp.get("clarification"):
        return "post-clarification"
    if (inp.get("execution_context") or {}).get("held_instance_id"):
        return "held-object replan"
    if inp.get("history_total", 0) > 0:
        return "other replan"
    return "initial plan"


def _rate(n: int, d: int) -> str:
    return f"{n}/{d} = {n / d:.1%}" if d else "n/a (0)"


def _stats(values: list[float]) -> str:
    if not values:
        return "n/a"
    return f"mean {statistics.mean(values):.1f}, median {statistics.median(values):.1f}"


def summarise(audits: list[dict]) -> str:
    n = len(audits)
    first_valid = sum(bool(a.get("first_pass_valid")) for a in audits)
    repaired = [a for a in audits if a.get("repair_count", 0) > 0]
    repaired_ok = sum(bool(a.get("accepted")) for a in repaired)
    accepted = sum(bool(a.get("accepted")) for a in audits)
    norm_total = sum(a.get("normalization_count", 0) or 0 for a in audits)
    norm_kinds: Counter = Counter()
    rejected_first: Counter = Counter()
    per_type = defaultdict(lambda: {"n": 0, "pt": [], "ct": [], "lat": [], "cached": 0})
    for a in audits:
        responses = a.get("responses") or []
        for r in responses:
            for item in r.get("normalizations") or []:
                norm_kinds[str(item.get("kind", item) if isinstance(item, dict) else item)[:60]] += 1
        if responses and responses[0].get("validation_error"):
            rejected_first[str(responses[0]["validation_error"])[:80]] += 1
        t = per_type[call_type(a)]
        t["n"] += 1
        for r in responses:
            resp = r.get("response") or {}
            if not resp:
                continue
            if resp.get("cached"):
                t["cached"] += 1
                continue
            t["pt"].append(resp.get("prompt_tokens", 0))
            t["ct"].append(resp.get("completion_tokens", 0))
            t["lat"].append(resp.get("latency_s", 0.0))
    lines = [
        f"- planner calls (audits): {n}",
        f"- first-pass-valid rate: {_rate(first_valid, n)}",
        f"- repair rate (calls that needed a repair): {_rate(len(repaired), n)}",
        f"- repair success rate (accepted after repair): {_rate(repaired_ok, len(repaired))}",
        f"- accepted overall: {_rate(accepted, n)}",
        f"- normalisations applied: {norm_total} in total",
        "",
        "**Validator ablation**: first responses the contract rejected before repair."
        " Without the validator these plans would have gone to the robot as they were:"
        f" **{sum(rejected_first.values())}/{n}**.",
        "",
        "| first-response rejection reason | count |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in rejected_first.most_common()] or ["| (none) | 0 |"]
    if norm_kinds:
        lines += ["", "| normalisation | count |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in norm_kinds.most_common()]
    lines += ["", "| call type | calls | live responses | cached | prompt tokens | completion tokens | latency s |",
              "|---|---|---|---|---|---|---|"]
    for name in ("initial plan", "held-object replan", "post-clarification", "other replan"):
        t = per_type.get(name)
        if not t:
            continue
        lines.append(f"| {name} | {t['n']} | {len(t['lat'])} | {t['cached']} | {_stats(t['pt'])} | "
                     f"{_stats(t['ct'])} | {_stats(t['lat'])} |")
    return "\n".join(lines) + "\n"


def variation_table(run_dir: pathlib.Path) -> str:
    """Per-category goal/status accuracy from an eval.b_benchmark output folder."""
    summary = json.loads((run_dir / "summary.json").read_text())
    per = defaultdict(lambda: [0, 0, 0])  # n, goal_ok, status_ok
    for case in summary.get("cases", []):
        c = per[case.get("category", "?")]
        c[0] += 1
        errors = [str(e) for e in case.get("errors") or []]
        planned = bool(case.get("stages"))  # no stages = planner raised, nothing to score
        goal_errors = [e for e in errors if e.startswith(("object_id", "region_id", "Ambiguous"))]
        c[1] += planned and not goal_errors
        c[2] += planned and not any(e.startswith("status") for e in errors)
    lines = ["| category | cases | goal accuracy | status accuracy |", "|---|---|---|---|"]
    for cat, (n, g, s) in sorted(per.items()):
        lines.append(f"| {cat} | {n} | {_rate(g, n)} | {_rate(s, n)} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Student B planner audit metrics")
    parser.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs/night"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/validation/B_METRICS.md"))
    parser.add_argument("--variation", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)
    audits = load_audits(args.runs)
    text = [f"# Student B metrics\n\nSource: every B audit under `{args.runs}` (live or cached responses only).\n",
            summarise(audits)]
    if args.variation is not None:
        text += ["\n## Variation suite (`eval/trials/student_b_variation`)\n", variation_table(args.variation)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(text))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
