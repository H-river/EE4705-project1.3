"""Final2 stage 2 (B): 2.1 supported-class vocabulary, 2.2 object memory for
APPROACH only, 2.3 relational reference binding, 2.4 history-aware replanning."""
import json
from pathlib import Path

import pytest

from core.types import (Action, ErrorCode, ExecutionContext, ExecutionResult, GroundedObject,
                        GroundStatus, PlanStatus, SceneDescription, Skill)
from core.validation import validate_plan
from planner.contract import WIRE_VERSION, PlanContractError, compile_plan, supported_classes
from planner.fixtures import example_response, fixture_planner, wire_action
from planner.prompts import SYSTEM_PROMPT, planning_input
from planner.relations import bind_reference, resolve
from tests.test_b_memory import INSTRUCTION, STONE_POS, region, scene, search_response, stone

F45 = json.loads((Path(__file__).parent / "fixtures/f45_refusal.json").read_text())


def f45_scene():
    items = [GroundedObject(g["instance_id"], g["name"], GroundStatus(g["status"]),
                            pos_world=tuple(g["pos_world"]) if g["pos_world"] else None, kind=g["kind"],
                            frame_id=g["frame_id"], attributes=dict(g["attributes"]),
                            region_half_extents_xy=g["region_half_extents_xy"])
             for g in F45["scene"]["instances"]]
    return SceneDescription([g for g in items if g.kind == "object"], [g for g in items if g.kind == "region"],
                            F45["scene"]["caption"], frame_id=F45["scene"]["frame_id"])


# ------------------------------------------------------------- 2.1 f45 ----

def test_every_supported_class_is_listed_for_the_model():
    classes = supported_classes()
    assert classes == {"objects": ["bottle", "cube", "stone"], "regions": ["red_region"]}
    data = planning_input(F45["instruction"], f45_scene(), [], ExecutionContext(f45_scene()), None, None)
    assert data["supported_classes"] == classes
    assert "closest supported class before refusing" in SYSTEM_PROMPT


def test_f45_refusal_of_a_supported_class_goes_back_for_repair():
    with pytest.raises(PlanContractError, match="'bottle' and 'red_region' are supported"):
        compile_plan(F45["reply"], ExecutionContext(f45_scene()))


def test_f45_repair_turns_into_a_search(tmp_path):
    search = {**F45["reply"], "status": "NEEDS_SEARCH", "reason": "bottle not localized",
              "actions": [wire_action("SEARCH", "bottle")]}
    planner = fixture_planner([F45["reply"], search], tmp_path)
    plan = planner.plan(F45["instruction"], f45_scene())
    assert plan.status is PlanStatus.NEEDS_SEARCH and plan.actions[0].target == "green bottle"
    audit = planner.last_diagnostics
    assert audit["repair_count"] == 1 and "supported classes" in audit["responses"][0]["validation_error"]


@pytest.mark.parametrize("reason", ["Stacking objects on other objects is not supported.",
                                    "Pouring is not a supported operation.",
                                    "Throwing the stone is not supported."])
def test_operation_refusals_still_compile(reason):
    wire = {"schema_version": WIRE_VERSION, "status": "INFEASIBLE",
            "goal": {"object_id": "", "object_name": "stone", "object_color": "", "region_id": "a1",
                     "region_name": "red_region"}, "actions": [], "reason": reason, "clarification_question": ""}
    plan, _ = compile_plan(wire, ExecutionContext(f45_scene()))
    assert plan.status is PlanStatus.INFEASIBLE


# ------------------------------------------------------ 2.2 APPROACH memory ----

def test_search_for_a_vanished_goal_object_starts_at_its_last_position(tmp_path):
    planner = fixture_planner([example_response("s1", "r1"), search_response("stone")], tmp_path, use_memory=True)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([], [region(frame=4)], 4)
    plan = planner.replan(INSTRUCTION, s, [], ExecutionContext(s))
    assert plan.status is PlanStatus.NEEDS_SEARCH
    assert [a.skill for a in plan.actions] == [Skill.APPROACH, Skill.SEARCH]
    assert plan.actions[0].params["pos"] == list(STONE_POS)
    assert planner.last_diagnostics["used_memory_for_approach"]["instance_id"] == "s1"
    assert not validate_plan(plan, ExecutionContext(s))


def test_unlocalized_goal_object_also_uses_memory_for_approach_only(tmp_path):
    planner = fixture_planner([example_response("s1", "r1"), search_response("stone")], tmp_path, use_memory=True)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([stone(GroundStatus.UNLOCALIZED, 4)], [region(frame=4)], 4)
    plan = planner.replan(INSTRUCTION, s, [], ExecutionContext(s))
    assert [a.skill for a in plan.actions] == [Skill.APPROACH, Skill.SEARCH]
    assert not any(a.skill is Skill.GRASP for a in plan.actions)


def test_released_object_is_never_recalled_for_approach(tmp_path):
    planner = fixture_planner([example_response("s1", "r1"), search_response("stone")], tmp_path, use_memory=True)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([], [region(frame=4)], 4)
    plan = planner.replan(INSTRUCTION, s, [], ExecutionContext(s, None, "s1"))
    assert [a.skill for a in plan.actions] == [Skill.SEARCH]


def test_grasp_positions_never_come_from_memory(tmp_path):
    # A READY plan needs the object LOCALIZED in the current scene; memory
    # never makes a vanished object graspable.
    planner = fixture_planner([example_response("s1", "r1"), example_response("s1", "r1"),
                               example_response("s1", "r1")], tmp_path, use_memory=True)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([], [region(frame=4)], 4)
    plan = planner.replan(INSTRUCTION, s, [], ExecutionContext(s))
    assert plan.status is PlanStatus.REJECTED


def test_memory_off_keeps_the_plain_search(tmp_path):
    planner = fixture_planner([example_response("s1", "r1"), search_response("stone")], tmp_path, use_memory=False)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([], [region(frame=4)], 4)
    assert [a.skill for a in planner.replan(INSTRUCTION, s, [], ExecutionContext(s)).actions] == [Skill.SEARCH]


# ------------------------------------------------------- 2.3 relations ----

def obj(ident, name, pos, colour):
    return GroundedObject(ident, name, GroundStatus.LOCALIZED, pos_world=pos, attributes={"color": colour},
                          frame_id=1)


def two_stones():
    # robot at the origin looking along +x: +y is its left
    return SceneDescription([obj("s1", "stone", (.40, -.20, .875), "gray"),
                             obj("s2", "stone", (.40, .10, .875), "gray"),
                             obj("c1", "cube", (.40, .20, .875), "blue")],
                            [GroundedObject("r1", "red_region", GroundStatus.LOCALIZED, kind="region",
                                            pos_world=(.4, .3, .853), frame_id=1)], frame_id=1)


@pytest.mark.parametrize("relation,expected", [("next_to", "s2"), ("closest_to", "s2"), ("farthest_from", "s1"),
                                               ("right_of", "s2"), ("left_of", None)])
def test_relations_resolve_from_positions(relation, expected):
    chosen, _ = resolve(two_stones(), "stone", "", {"relation": relation, "anchor_class": "cube"})
    assert chosen == expected


def test_left_right_follow_the_robot_frame():
    # base turned 180 deg: the robot's left is world -y
    chosen, _ = resolve(two_stones(), "stone", "", {"relation": "left_of", "anchor_class": "cube"}, base_yaw=3.14159)
    assert chosen == "s2"


def test_ties_and_missing_or_duplicate_anchors_are_not_decisive():
    s = two_stones()
    s.objects[1] = obj("s2", "stone", (.40, .59, .875), "gray")  # 0.39 m vs s1's 0.40 m from the cube
    assert resolve(s, "stone", "", {"relation": "next_to", "anchor_class": "cube"})[0] is None
    assert resolve(two_stones(), "stone", "", {"relation": "next_to", "anchor_class": "bottle"})[0] is None
    s = two_stones()
    s.objects.append(obj("c2", "cube", (.40, -.30, .875), "blue"))
    assert resolve(s, "stone", "", {"relation": "next_to", "anchor_class": "cube"})[0] is None


def test_binding_rewrites_the_model_choice_everywhere(tmp_path):
    reply = example_response("s1", "r1")
    reply["reference"] = {"relation": "next_to", "anchor_class": "cube", "anchor_color": "blue"}
    planner = fixture_planner([reply], tmp_path)
    plan = planner.plan("Move the stone next to the blue cube to the red area.", two_stones())
    assert plan.status is PlanStatus.READY and planner.goal["object_id"] == "s2"
    targets = [a.target for a in plan.actions if a.skill in (Skill.APPROACH, Skill.GRASP)]
    assert targets == ["s2", "s2"] and plan.actions[3].params["object"] == "s2"
    norm = planner.last_diagnostics["responses"][0]["normalizations"]
    assert any(n.get("change") == "relational binding" for n in norm)


def test_no_reference_or_locked_goal_leaves_the_reply_alone():
    reply = example_response("s1", "r1")
    assert bind_reference(reply, two_stones()) is reply
    reply["reference"] = {"relation": "next_to", "anchor_class": "cube", "anchor_color": ""}
    assert bind_reference(reply, two_stones(), locked_goal={"object_id": "s1"}) is reply


