"""Offline contract tests. Canned output does NOT measure Qwen language accuracy."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.llm_client import RequestsTransport
from core.types import Action, ErrorCode, ExecutionContext, ExecutionResult, GroundStatus, PlanStatus, Skill
from core.validation import validate_plan
from planner.config import DEFAULT_MODEL, PlannerConfigError, QwenPlannerConfig
from planner.contract import PlanContractError, compile_plan
from planner.fixtures import INSTRUCTION, example_response, example_scene, fixture_planner, wire_action
from planner.run import load_scene, main
from planner.student_b import PlannerError


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A live network transport was called in an offline test")
    monkeypatch.setattr(RequestsTransport, "post", forbidden)


def test_compiles_only_a_coordinates_and_records_provenance(tmp_path):
    planner = fixture_planner([example_response()], tmp_path)
    scene = example_scene()
    plan = planner.plan(INSTRUCTION, scene)
    assert validate_plan(plan, ExecutionContext(scene)) == []
    assert plan.actions[0].params["pos"] == list(scene.objects[0].pos_world)
    assert plan.actions[2].params["pos"] == pytest.approx([.4, .3, 1.033])
    assert plan.actions[3].params["pos"] == [.4, .3, .853]
    assert plan.actions[1].target == "p0"
    audit = json.loads(Path(planner.last_diagnostics["audit_path"]).read_text())
    assert audit["response_source"] == "fixture" and audit["first_pass_valid"]
    assert audit["responses"][0]["response"]["prompt_tokens"] == 10
    assert "fixture-only-placeholder" not in json.dumps(audit)
    assert "gt_id" not in json.dumps(audit["input"])
    payload = planner.client.transport.requests[0]["payload"]
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "enable_thinking" not in payload


@pytest.mark.parametrize("damage", ["unknown_id", "coordinates", "order", "wrong_verify", "missing_stop",
                                    "wrong_color", "wrong_kind", "extra_action_field", "missing_field"])
def test_rejects_bad_output_then_repairs_once(tmp_path, damage):
    bad = example_response()
    if damage == "unknown_id":
        bad["goal"]["object_id"] = "invented"
    elif damage == "coordinates":
        bad["actions"][0]["params"] = {"pos": [0, 0, 1]}
    elif damage == "order":
        bad["actions"][1], bad["actions"][3] = bad["actions"][3], bad["actions"][1]
    elif damage == "wrong_verify":
        bad["actions"][-2]["object"] = "p1"
    elif damage == "missing_stop":
        bad["actions"].pop()
    elif damage == "wrong_color":
        bad["goal"]["object_color"] = "blue"
    elif damage == "wrong_kind":
        bad["goal"]["region_id"] = "p1"
    elif damage == "extra_action_field":
        bad["actions"][0]["object"] = "p0"
    else:
        del bad["status"]
    planner = fixture_planner([bad, example_response()], tmp_path)
    assert planner.plan(INSTRUCTION, example_scene()).status is PlanStatus.READY
    assert planner.last_diagnostics["repair_count"] == 1
    repair = json.loads(planner.client.transport.requests[1]["payload"]["messages"][-1]["content"])["repair"]
    assert repair["validation_error"] and repair["previous_output"]


def test_invalid_json_is_bounded_and_retains_usage(tmp_path):
    planner = fixture_planner(["not JSON", "still not JSON", example_response()], tmp_path)
    with pytest.raises(PlannerError):
        planner.plan(INSTRUCTION, example_scene())
    assert len(planner.client.transport.requests) == 2
    assert planner.client.stats.prompt_tokens == 20
    assert planner.last_diagnostics["accepted"] is False
    assert len(planner.last_diagnostics["responses"]) == 2
    assert planner.goal is None


def test_service_error_is_not_task_refusal_and_does_not_use_rule_fallback(tmp_path):
    planner = fixture_planner([], tmp_path)
    planner.client.transport.responses = [(401, {"error": {"message": "bad fixture-only-placeholder"}})]
    with pytest.raises(PlannerError, match="401") as caught:
        planner.plan(INSTRUCTION, example_scene())
    assert "fixture-only-placeholder" not in str(caught.value)
    assert len(planner.client.transport.requests) == 1
    assert planner.last_diagnostics["accepted"] is False
    assert planner.goal is None


def test_replan_keeps_goal_uses_new_scene_and_excludes_private_info(tmp_path):
    planner = fixture_planner([example_response(), example_response()], tmp_path)
    scene = example_scene()
    first = planner.plan(INSTRUCTION, scene)
    scene2 = copy.deepcopy(scene)
    scene2.objects[0].pos_world = (.45, -.12, .88)
    scene2.frame_id = 9
    history = [ExecutionResult(first.actions[1], False, ErrorCode.GRASP_MISSED,
                               post_frame_id=9, info={"held_gt_id": "do-not-send"})]
    plan = planner.replan(INSTRUCTION, scene2, history, ExecutionContext(scene))
    assert plan.actions[1].params["pos"] == [.45, -.12, .88]
    data = planner.last_diagnostics["input"]
    assert data["scene"]["frame_id"] == 9
    assert data["history_tail"][0]["error_code"] == "GRASP_MISSED"
    assert "do-not-send" not in json.dumps(data)
    assert data["original_goal"]["object_id"] == "p0"


@pytest.mark.parametrize("switch_goal", [False, True])
def test_wrong_held_object_cannot_become_goal(tmp_path, switch_goal):
    bad = example_response()
    bad["actions"] = bad["actions"][2:]
    if switch_goal:
        bad["goal"].update(object_id="p1", object_name="cube")
        bad["actions"][1]["object"] = bad["actions"][2]["object"] = "p1"
    planner = fixture_planner([example_response(), bad, bad], tmp_path)
    scene = example_scene()
    planner.plan(INSTRUCTION, scene)
    with pytest.raises(PlannerError, match="Original goal|Held object"):
        planner.replan(INSTRUCTION, scene, [], ExecutionContext(scene, held_instance_id="p1"))
    assert planner.goal["object_id"] == "p0"


@pytest.mark.parametrize("released", [False, True])
def test_occluded_held_or_released_identity_allows_future_verification(tmp_path, released):
    remaining = example_response()
    remaining["actions"] = remaining["actions"][4 if released else 2:]
    planner = fixture_planner([example_response(), remaining], tmp_path)
    scene = example_scene()
    planner.plan(INSTRUCTION, scene)
    occluded = replace(scene, objects=[])
    context = ExecutionContext(occluded, held_instance_id=None if released else "p0",
                               last_release_instance_id="p0" if released else None)
    plan = planner.replan(INSTRUCTION, occluded, [], context)
    assert validate_plan(plan, context) == []
    assert all(a.skill is not Skill.GRASP for a in plan.actions)
    assert occluded.objects == []  # no fabricated perception or position
    # An untracked missing identity still fails the shared validator.
    assert validate_plan(plan, ExecutionContext(occluded))


@pytest.mark.parametrize("status", ["NEEDS_SEARCH", "NEEDS_CLARIFICATION", "INFEASIBLE"])
def test_nonready_status_contract(tmp_path, status):
    wire = example_response()
    wire.update(status=status, actions=[])
    wire["goal"]["object_id"] = ""
    if status == "NEEDS_SEARCH":
        wire["actions"] = [wire_action("SEARCH", "stone")]
    elif status == "NEEDS_CLARIFICATION":
        wire["clarification_question"] = "Which stone do you mean?"
    planner = fixture_planner([wire], tmp_path)
    assert planner.plan(INSTRUCTION, example_scene()).status.value == status


def test_search_then_clarification_binds_unresolved_goal(tmp_path):
    search = example_response()
    search.update(status="NEEDS_SEARCH", actions=[wire_action("SEARCH", "stone")])
    search["goal"]["object_id"] = ""
    ready = example_response()
    ready["goal"]["object_color"] = "gray"
    planner = fixture_planner([search, ready, ready], tmp_path)
    scene = example_scene()
    planner.plan(INSTRUCTION, replace(scene, objects=[]))
    assert planner.goal["object_id"] == ""
    planner.replan(INSTRUCTION, scene, [], ExecutionContext(scene), "the gray one")
    planner.replan(INSTRUCTION, scene, [], ExecutionContext(scene))
    assert planner.goal["object_id"] == "p0" and planner.goal["object_color"] == "gray"
    assert planner.last_diagnostics["input"]["clarification"] == "the gray one"


def test_unlocalized_or_ambiguous_target_is_never_used_for_motion():
    for status in (GroundStatus.UNLOCALIZED, GroundStatus.AMBIGUOUS):
        scene = example_scene()
        scene.objects[0] = replace(scene.objects[0], status=status, pos_world=None)
        with pytest.raises(PlanContractError, match="current, unambiguous"):
            compile_plan(example_response(), ExecutionContext(scene))


def test_each_regrasp_requires_a_new_approach_and_transport():
    wire = example_response()
    first_cycle = wire["actions"][:4]
    tail = wire["actions"][4:]
    wire["actions"] = first_cycle + [wire_action("GRASP", "p0")] + first_cycle[2:] + tail
    with pytest.raises(PlanContractError, match="APPROACH or REACH"):
        compile_plan(wire, ExecutionContext(example_scene()))
    wire["actions"] = first_cycle + first_cycle[:2] + [first_cycle[3]] + tail
    with pytest.raises(PlanContractError, match="preceding MOVE_TO"):
        compile_plan(wire, ExecutionContext(example_scene()))


def test_reset_and_context_preconditions(tmp_path):
    planner = fixture_planner([example_response()], tmp_path)
    scene = example_scene()
    with pytest.raises(PlannerError, match="before replan"):
        planner.replan(INSTRUCTION, scene, [], ExecutionContext(scene))
    planner.plan(INSTRUCTION, scene)
    with pytest.raises(PlannerError, match="Instruction changed"):
        planner.replan("new task", scene, [], ExecutionContext(scene))
    planner.reset()
    assert planner.goal is None and planner.last_diagnostics is None
    with pytest.raises(PlannerError, match="unique"):
        planner.plan(INSTRUCTION, replace(scene, objects=[scene.objects[0]] * 2))


def test_configuration_requires_explicit_endpoint_and_never_shows_key():
    with pytest.raises(PlannerConfigError, match="BASE_URL"):
        QwenPlannerConfig.from_env({})
    env = {"EE4705_QWEN_BASE_URL": "https://workspace.example/compatible-mode/v1", "DASHSCOPE_API_KEY": "test-secret"}
    config = QwenPlannerConfig.from_env(env)
    assert config.llm.model == DEFAULT_MODEL
    assert "test-secret" not in repr(config) + json.dumps(config.public_settings())
    config = QwenPlannerConfig.from_env({**env, "DASHSCOPE_API_KEY": "", "EE4705_QWEN_CACHE_ONLY": "true"})
    assert config.llm.cache_only and not config.llm.api_key
    for url in ("https://{WorkspaceId}.example/v1", "http://remote.example/v1", "https://key@example/v1",
                "https://example/v1?key=bad", "https://example/v1/chat/completions"):
        with pytest.raises(PlannerConfigError):
            QwenPlannerConfig.from_env({**env, "EE4705_QWEN_BASE_URL": url})
    for key, value in (("EE4705_QWEN_THINKING", "maybe"), ("EE4705_QWEN_TIMEOUT_S", "nan"),
                       ("EE4705_QWEN_MAX_RETRIES", "100"), ("EE4705_QWEN_OUTPUT_MODE", "auto")):
        with pytest.raises(PlannerConfigError):
            QwenPlannerConfig.from_env({**env, key: value})


def test_offline_cli_writes_reusable_inputs_and_cannot_overwrite(tmp_path):
    assert main(["--offline-demo", "--out", str(tmp_path)]) == 0
    assert load_scene(tmp_path / "scene.json").objects[0].instance_id == "p0"
    assert json.loads((tmp_path / "diagnostics.json").read_text())["response_source"] == "fixture"
    with pytest.raises(SystemExit):
        main(["--offline-demo", "--out", str(tmp_path)])


def test_fixture_student_b_connects_to_real_simulated_demo_a_and_c(tmp_path, monkeypatch):
    from demo.run import run_episode
    # Fixed response for this fixed scene: demo A observes stone p0 and region p2.
    planner = fixture_planner([example_response()], tmp_path / "audit")
    import planner.student_b as module
    class OfflineStudentB:
        IMPLEMENTED = True
        def __new__(cls):
            return planner
    monkeypatch.setattr(module, "StudentBPlanner", OfflineStudentB)
    result = run_episode(tmp_path / "episode", students=["B"], video=False)
    assert result["claimed_success"] and result["actual_success"], result["episode"]
    assert result["actual"]["checks"]["stable"]
    assert result["grasp_records"][0]["held_gt_id"] == "stone"
    assert planner.last_diagnostics["response_source"] == "fixture"
