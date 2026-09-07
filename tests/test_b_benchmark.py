"""Check scoring against external labels without an API call."""
import json
from pathlib import Path

from core.types import Action, ExecutionContext, Plan, Skill
from eval.b_benchmark import run_case, score_plan
from planner.config import QwenPlannerConfig
from planner.contract import compile_plan
from planner.fixtures import example_response, example_scene, fixture_planner
from planner.prompts import json_value


def test_semantically_wrong_but_internally_valid_plan_fails_gold_labels():
    wire = example_response()
    wire["goal"].update(object_id="p1", object_name="cube")
    for item in wire["actions"]:
        if item["target"] == "p0":
            item["target"] = "p1"
        if item["object"] == "p0":
            item["object"] = "p1"
    context = ExecutionContext(example_scene())
    plan, goal = compile_plan(wire, context)
    errors = score_plan(plan, goal, {"status":"READY", "object_id":"p0", "region_id":"p2"}, context)
    assert any("object_id" in e for e in errors)
    assert any("grasp target" in e for e in errors)


def test_correct_goal_with_missing_dependencies_is_not_a_correct_plan():
    expected = {"status":"READY", "object_id":"p0", "region_id":"p2"}
    plan = Plan([Action(Skill.PLACE, "p2", {"object":"p0"}),
                 Action(Skill.VERIFY, params={"condition":"object_in_region","object":"p0","region":"p2"}),
                 Action(Skill.STOP)])
    assert score_plan(plan, expected, expected, ExecutionContext(example_scene()))


def test_case_hides_gold_and_writes_failure_without_dropping_it(tmp_path):
    case = {"id":"offline_score", "category":"standard", "instruction":"Move the stone to the red area.",
            "scene":json_value(example_scene()),
            "expected":{"status":"READY", "object_id":"p0", "region_id":"p2", "private_marker":"DO_NOT_SEND_GOLD"}}
    captured = []
    def factory(config):
        planner = fixture_planner([example_response()], config.audit_dir)
        captured.append(planner)
        return planner
    config = QwenPlannerConfig(fixture_planner([], tmp_path / "unused").client.config)
    result = run_case(case, tmp_path / "ok", config, factory)
    assert result["correct"] and result["first_response_correct"]
    assert "DO_NOT_SEND_GOLD" not in json.dumps(captured[0].client.transport.requests)
    def failing(config):
        return fixture_planner(["broken", "broken"], config.audit_dir)
    result = run_case(case, tmp_path / "fail", config, failing)
    assert not result["correct"] and result["errors"]
    assert (tmp_path / "fail/result.json").exists()


def test_suites_are_separate_and_have_valid_independent_labels():
    root = Path(__file__).resolve().parents[1] / "eval/trials/student_b"
    dev, evaluation = [json.loads((root / f"{name}.json").read_text()) for name in ("development", "evaluation")]
    assert len(dev["cases"]) == 13 and len(evaluation["cases"]) == 32
    assert not ({c["instruction"] for c in dev["cases"]} & {c["instruction"] for c in evaluation["cases"]})
    for suite in (dev, evaluation):
        for case in suite["cases"]:
            expected = case["expected"]
            sources = [case["scene"]] + ([case["initial"]["scene"]] if "initial" in case else [])
            known = {g["instance_id"]:g for s in sources for g in s["objects"]+s["regions"]}
            for role in ("object", "region"):
                ident = expected.get(role+"_id")
                if ident:
                    assert ident in known and known[ident]["kind"] == role
