"""Stage M1: B's episode memory (planner/memory.py).

A reports the current frame only; B remembers where each instance was last
LOCALIZED and, when the goal region alone is out of view, plans from that
position instead of searching again (the object/region search ping-pong).
"""
from core.types import (ExecutionContext, GroundedObject, GroundStatus, PlanStatus,
                        SceneDescription, Skill)
from core.validation import validate_plan
from planner.fixtures import example_response, fixture_planner, wire_action
from planner.contract import WIRE_VERSION
from planner.memory import EpisodeMemory

INSTRUCTION = "Move the stone to the red area."
REGION_POS = (0.40, 0.30, 0.853)
STONE_POS = (0.40, -0.15, 0.875)


def stone(status=GroundStatus.LOCALIZED, frame=1):
    return GroundedObject("s1", "stone", status, bbox_xyxy=(10, 10, 30, 30),
                          pos_world=STONE_POS if status is GroundStatus.LOCALIZED else None,
                          attributes={"color": "gray"}, confidence=.9, frame_id=frame)


def region(status=GroundStatus.LOCALIZED, frame=1):
    return GroundedObject("r1", "red_region", status, kind="region",
                          bbox_xyxy=(100, 100, 200, 160) if status is not GroundStatus.NOT_FOUND else None,
                          pos_world=REGION_POS if status is GroundStatus.LOCALIZED else None,
                          region_half_extents_xy=(.08, .08), confidence=.9, frame_id=frame)


def scene(objects, regions, frame):
    return SceneDescription(objects=objects, regions=regions, frame_id=frame)


def search_response(target):
    return {"schema_version": WIRE_VERSION, "status": "NEEDS_SEARCH",
            "goal": {"object_id": "s1", "object_name": "stone", "object_color": "",
                     "region_id": "r1", "region_name": "red_region"},
            "actions": [wire_action("SEARCH", target)], "reason": "not in view", "clarification_question": ""}


# ---------------------------------------------------------------- unit ----

def test_recall_returns_last_localized_position():
    m = EpisodeMemory()
    m.update(scene([stone()], [region()], 1))
    m.update(scene([stone(frame=5)], [region(GroundStatus.UNLOCALIZED, 5)], 5))
    assert m.recall("r1") == REGION_POS
    assert m.age("r1") == 4
    assert m.recall("nope") is None


def test_age_expiry():
    m = EpisodeMemory()
    m.update(scene([], [region(frame=1)], 1))
    m.update(scene([], [], 41))
    assert m.recall("r1") == REGION_POS
    m.update(scene([], [], 42))
    assert m.recall("r1") is None
    assert m.recall("r1", max_age_frames=100) == REGION_POS


def test_base_move_expiry():
    m = EpisodeMemory()
    m.update(scene([], [region()], 1), base_pose=(0.0, 0.0, 0.0))
    assert m.recall("r1", base_pose=(0.3, 0.3, 1.0)) == REGION_POS  # 0.42 m
    assert m.recall("r1", base_pose=(0.4, 0.4, 0.0)) is None  # 0.57 m
    assert m.recall("r1", base_pose=None) == REGION_POS  # no pose: age only


def test_held_and_released_objects_are_never_recalled():
    m = EpisodeMemory()
    m.update(scene([stone()], [region()], 1))
    m.update(scene([], [region(frame=2)], 2), held_instance_id="s1")
    assert m.recall("s1") is None
    m2 = EpisodeMemory()
    m2.update(scene([stone()], [region()], 1))
    m2.update(scene([], [region(frame=2)], 2), last_release_instance_id="s1")
    assert m2.recall("s1") is None
    assert m2.recall("r1") == REGION_POS


# ---------------------------------------------------------- planner use ----

def test_alternating_visibility_plans_ready_from_memory(tmp_path):
    """Region seen, stone not -> SEARCH stone; then stone seen, region out of
    view -> READY from the remembered region instead of SEARCH red_region."""
    first = search_response("stone")
    first["goal"]["object_id"] = ""
    planner = fixture_planner([first, example_response("s1", "r1")], tmp_path, use_memory=True)
    s1 = scene([], [region(frame=3)], 3)
    plan = planner.plan(INSTRUCTION, s1)
    assert plan.status is PlanStatus.NEEDS_SEARCH
    s2 = scene([stone(frame=9)], [], 9)
    context = ExecutionContext(s2)
    plan = planner.replan(INSTRUCTION, s2, [], context)
    assert plan.status is PlanStatus.READY
    place = next(a for a in plan.actions if a.skill is Skill.PLACE)
    assert tuple(place.params["pos"]) == REGION_POS
    audit = planner.last_diagnostics
    assert audit["used_memory_for"] == "r1" and audit["memory_age_frames"] == 6
    # The orchestrator validates against A's scene, where r1 is absent.
    assert validate_plan(plan, context) == []


def test_memory_not_used_when_region_too_old(tmp_path):
    planner = fixture_planner([example_response("s1", "r1"), search_response("red_region")],
                              tmp_path, use_memory=True)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([stone(frame=60)], [], 60)
    plan = planner.replan(INSTRUCTION, s, [], ExecutionContext(s))
    assert plan.status is PlanStatus.NEEDS_SEARCH
    assert planner.last_diagnostics["used_memory_for"] is None


def test_memory_not_used_for_the_object(tmp_path):
    """GRASP needs fresh 3D evidence: a vanished stone is searched for."""
    planner = fixture_planner([example_response("s1", "r1"), search_response("stone")],
                              tmp_path, use_memory=True)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([], [region(frame=4)], 4)
    plan = planner.replan(INSTRUCTION, s, [], ExecutionContext(s))
    assert plan.status is PlanStatus.NEEDS_SEARCH
    assert planner.last_diagnostics["used_memory_for"] is None


def test_memory_while_holding_the_goal(tmp_path):
    carry = example_response("s1", "r1")
    carry["actions"] = carry["actions"][2:]
    planner = fixture_planner([example_response("s1", "r1"), carry], tmp_path, use_memory=True)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([], [region(GroundStatus.NOT_FOUND, 8)], 8)
    context = ExecutionContext(s, held_instance_id="s1")
    plan = planner.replan(INSTRUCTION, s, [], context)
    assert plan.status is PlanStatus.READY
    assert [a.skill for a in plan.actions][:2] == [Skill.MOVE_TO, Skill.PLACE]
    assert validate_plan(plan, context) == []


def test_memory_can_be_disabled(tmp_path):
    planner = fixture_planner([example_response("s1", "r1"), search_response("red_region")], tmp_path)
    planner.plan(INSTRUCTION, scene([stone()], [region()], 1))
    s = scene([stone(frame=2)], [], 2)
    assert planner.replan(INSTRUCTION, s, [], ExecutionContext(s)).status is PlanStatus.NEEDS_SEARCH


def test_place_without_pos_still_needs_region_in_view():
    from core.types import Action, Plan
    s = scene([], [], 5)
    plan = Plan([Action(Skill.MOVE_TO, "r1", {"pos": [0.4, 0.3, 1.0]}),
                 Action(Skill.PLACE, "r1", {"object": "s1"}),
                 Action(Skill.STOP)])
    codes = [e.code for e in validate_plan(plan, ExecutionContext(s, held_instance_id="s1"))]
    assert "MISSING_REFERENCE" in codes
