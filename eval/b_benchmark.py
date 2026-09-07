"""Labelled Student B planning evaluation. Expected answers never enter model input."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from core.types import Action, ErrorCode, ExecutionContext, ExecutionResult, PlanStatus, Skill
from planner.config import QwenPlannerConfig
from planner.contract import COMPILER_VERSION, compile_plan
from planner.prompts import PROMPT_VERSION, json_value
from planner.run import load_scene
from planner.student_b import StudentBPlanner


def score_plan(plan, goal, expected, context):
    """Compare against external labels and action dependencies, not one exact plan string."""
    errors = []
    def check(ok, detail):
        if not ok:
            errors.append(detail)
    status = expected["status"]
    check(plan.status.value == status, f"status: expected {status}, got {plan.status.value}")
    for role in ("object", "region"):
        key = role + "_id"
        if key in expected:
            check(goal.get(key) == expected[key], f"{key}: expected {expected[key]}, got {goal.get(key)}")
    for role in expected.get("unbound_roles", []):
        check(not goal.get(role + "_id"), f"Ambiguous {role} must not be bound to a guessed ID")
    if plan.status is PlanStatus.NEEDS_CLARIFICATION:
        check(bool(plan.clarification_question) and not plan.actions, "Clarification needs a question and no actions")
    elif plan.status is PlanStatus.INFEASIBLE:
        check(bool(plan.reason) and not plan.actions, "Refusal needs a reason and no actions")
    elif plan.status is PlanStatus.NEEDS_SEARCH:
        allowed = expected.get("search_targets", [])
        check(bool(plan.actions), "Search plan is empty")
        for a in plan.actions:
            check(a.skill is Skill.SEARCH and a.target in allowed, f"Unexpected search: {a.skill.value} {a.target}")
    elif plan.status is PlanStatus.READY and status == "READY":
        obj, region = expected["object_id"], expected["region_id"]
        held = context.held_instance_id
        released = not held and context.last_release_instance_id == obj
        approached = transported = verified = False
        for i, a in enumerate(plan.actions):
            check(not verified or a.skill is Skill.STOP, f"action {i}: motion after final verification")
            if a.skill in (Skill.APPROACH, Skill.REACH):
                check(a.target == obj and held is None, f"action {i}: wrong approach target/state")
                approached = True
            elif a.skill is Skill.GRASP:
                check(a.target == obj and held is None and approached, f"action {i}: wrong grasp target/order")
                held, released, approached, transported = a.target, False, False, False
            elif a.skill is Skill.MOVE_TO:
                check(a.target == region and held == obj, f"action {i}: wrong transport target/state")
                transported = True
            elif a.skill is Skill.PLACE:
                check(a.target == region and a.params.get("object") == obj and held == obj and transported,
                      f"action {i}: wrong placement target/order")
                held, released, transported = None, True, False
            elif a.skill is Skill.VERIFY:
                if a.params.get("condition") == "object_in_region":
                    check(released and a.params.get("object") == obj and a.params.get("region") == region,
                          f"action {i}: wrong placement verification")
                    verified = True
                else:
                    check(a.target == obj, f"action {i}: verification targets another object")
            elif a.skill is Skill.STOP:
                check(verified and i == len(plan.actions)-1, f"action {i}: premature STOP")
            else:
                check(False, f"action {i}: unexpected skill {a.skill.value}")
        check(verified and bool(plan.actions) and plan.actions[-1].skill is Skill.STOP,
              "Ready plan must end with goal verification and STOP")
    return errors


def _read_scene_dict(raw, path):
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return load_scene(path)


def run_case(case, out, config, planner_factory=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "case.json").write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n")
    local_config = replace(config, audit_dir=str(out / "audit"))
    planner = planner_factory(local_config) if planner_factory else StudentBPlanner(local_config)
    result = {"id": case["id"], "category": case["category"], "instruction": case["instruction"],
              "correct": False, "first_response_correct": False, "stages": [], "errors": [],
              "response_source": planner.client.response_source}
    start = time.monotonic()
    try:
        initial = case.get("initial")
        if initial:
            scene0 = _read_scene_dict(initial["scene"], out / "initial_scene.json")
            plan0 = planner.plan(case["instruction"], scene0)
            errors0 = score_plan(plan0, planner.goal or {}, initial["expected"], ExecutionContext(scene0))
            result["stages"].append(_stage_result("initial", plan0, planner, errors0, initial["expected"],
                                                   ExecutionContext(scene0), scene0))
            if errors0:
                result["errors"].extend("initial: " + e for e in errors0)
                return _finish(result, planner, start, out)
        scene = _read_scene_dict(case["scene"], out / "scene.json")
        state = case.get("context", {})
        context = ExecutionContext(scene, state.get("held_instance_id"), state.get("last_release_instance_id"))
        if initial:
            history = [ExecutionResult(Action(Skill(r["skill"]), r.get("target"), r.get("params", {})),
                                        r["success"], ErrorCode(r.get("error_code", "NONE")))
                       for r in case.get("history", [])]
            plan = planner.replan(case["instruction"], scene, history, context, case.get("clarification"))
        else:
            plan = planner.plan(case["instruction"], scene)
        errors = score_plan(plan, planner.goal or {}, case["expected"], context)
        result["stages"].append(_stage_result("final", plan, planner, errors, case["expected"],
                                               context, scene0 if initial else scene))
        result["errors"].extend(errors)
        result["correct"] = not result["errors"]
        result["first_response_correct"] = all(s["first_response_correct"] for s in result["stages"])
    except Exception as exc:
        # An API/format failure stays in the denominator. Do not use a rule fallback.
        detail = str(exc)
        if config.llm.api_key:
            detail = detail.replace(config.llm.api_key, "[REDACTED]")
        result["errors"].append(type(exc).__name__ + ": " + detail)
        if planner.last_diagnostics:
            result["failed_audit_path"] = planner.last_diagnostics["audit_path"]
    return _finish(result, planner, start, out)


def _stage_result(name, plan, planner, errors, expected, context, initial_scene):
    audit = planner.last_diagnostics
    first_errors = ["No initial model response"]
    response = audit["responses"][0]["response"] if audit["responses"] else None
    if response and response.get("parsed"):
        try:
            known = {g.instance_id: g for g in initial_scene.objects + initial_scene.regions}
            first_plan, first_goal = compile_plan(response["parsed"], context, audit["input"]["original_goal"], known)
            first_errors = score_plan(first_plan, first_goal, expected, context)
        except Exception as exc:
            first_errors = [str(exc)]
    return {"name": name, "plan": json_value(plan), "goal": planner.goal, "errors": errors,
            "first_response_correct": not first_errors, "first_response_errors": first_errors,
            "repair_count": audit["repair_count"], "normalization_count": audit.get("normalization_count", 0),
            "audit_path": audit["audit_path"],
            "cached_responses": sum(bool(r["response"] and r["response"]["cached"]) for r in audit["responses"])}


def _finish(result, planner, start, out):
    result["latency_s"] = time.monotonic() - start
    result["api_stats"] = planner.client.stats.as_dict()
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def write_report(out, manifest, results):
    results = sorted(results, key=lambda r: r["id"])
    n = len(results)
    summary = {"schema": "student-b-benchmark-v1", "manifest": manifest,
               "complete": n == manifest["n_cases_expected"],
               "n_cases_expected": manifest["n_cases_expected"],
               "n_cases": n, "n_correct": sum(r["correct"] for r in results),
               "accuracy": sum(r["correct"] for r in results) / n if n else None,
               "first_response_accuracy": sum(r["first_response_correct"] for r in results) / n if n else None,
               "categories": {}, "cases": results}
    for category in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == category]
        summary["categories"][category] = {"correct": sum(r["correct"] for r in rows), "total": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    rows = []
    for r in results:
        ident = html.escape(r["id"])
        rows.append(f'<tr><td>{"PASS" if r["correct"] else "FAIL"}</td><td><a href="{ident}/result.json">{ident}</a></td>'
                    f'<td>{html.escape(r["instruction"])}</td><td>{r["latency_s"]:.2f}s</td>'
                    f'<td>{html.escape("; ".join(r["errors"]))}</td><td><a href="{ident}/case.json">Input + gold</a></td></tr>')
    accuracy = f'{100*summary["accuracy"]:.1f}%' if n else 'pending'
    report = '<!doctype html><meta charset="utf-8"><title>Qwen B evaluation</title><style>body{font:16px system-ui;margin:32px;background:#101827;color:#e5eaf1}a{color:#7fd8ff}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #334155;text-align:left}h1{color:#8ee5c1}</style>'
    report += f'<h1>Student B planning: {summary["n_correct"]}/{n} ({accuracy})</h1>'
    if not summary["complete"]:
        report += f'<p>RUNNING: {n}/{manifest["n_cases_expected"]} cases complete. Rates above describe completed cases only.</p>'
    report += '<p>Fixed labelled scene inputs. This measures planning, not physical robot success. All errors stay in the denominator.</p>'
    report += f'<p>Split: {html.escape(manifest["split"])} | Model: {html.escape(manifest["settings"]["model"])} | <a href="summary.json">Full results</a> | <a href="suite.json">Frozen labels</a></p>'
    report += '<table><tr><th>Result</th><th>Case</th><th>Instruction</th><th>Time</th><th>Failure</th><th>Ground truth</th></tr>' + ''.join(rows) + '</table>'
    (out / "index.html").write_text(report)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args(argv)
    config = QwenPlannerConfig.from_env()
    if config.llm.cache_only:
        parser.error("A live benchmark requires CACHE_ONLY=false")
    # Fresh requests make the denominator and response provenance straightforward.
    config.llm = replace(config.llm, cache_dir=None)
    raw = args.suite.read_bytes()
    suite = json.loads(raw)
    ids = [c["id"] for c in suite["cases"]]
    if len(ids) != len(set(ids)) or not ids or any(not i.replace('_', '').isalnum() for i in ids):
        parser.error("Case IDs must be unique and contain only letters, digits and underscores")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("Output directory must be empty")
    manifest = {"split": suite["split"], "suite_sha256": hashlib.sha256(raw).hexdigest(),
                "n_cases_expected": len(ids),
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "settings": config.public_settings(), "prompt_version": PROMPT_VERSION,
                "compiler_version": COMPILER_VERSION, "workers": args.workers,
                "denominator_policy": "all cases, including API and output errors",
                "physical_execution": False}
    source_root = Path(__file__).resolve().parents[1]
    manifest["source_sha256"] = {}
    for name in ("planner/contract.py", "planner/prompts.py", "planner/student_b.py", "planner/config.py",
                 "core/llm_client.py", "eval/b_benchmark.py"):
        content = (source_root / name).read_bytes()
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
        destination = args.out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (args.out / "suite.json").write_bytes(raw)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, c, args.out / c["id"], config) for c in suite["cases"]]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_report(args.out, manifest, results)
            print(f'{"PASS" if result["correct"] else "FAIL"} {result["id"]}: ' + '; '.join(result["errors"]), flush=True)
    summary = write_report(args.out, manifest, results)
    print(f'{summary["n_correct"]}/{summary["n_cases"]} = {100*summary["accuracy"]:.1f}%; report: {args.out.resolve()}', flush=True)
    return 0 if summary["accuracy"] >= .9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
