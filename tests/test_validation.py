# Owner: backbone (ALL)
"""Focused tests for core.validation.validate_plan."""

from __future__ import annotations

from core.types import (
    Action,
    ExecutionContext,
    GroundedObject,
    GroundStatus,
    Plan,
    PlanStatus,
    SceneDescription,
    Skill,
)
from core.validation import validate_plan


def make_scene() -> SceneDescription:
    return SceneDescription(
        objects=[
            GroundedObject("p0", "stone", GroundStatus.LOCALIZED, bbox_xyxy=(10, 10, 40, 40),
                           pos_world=(0.4, 0.1, 0.45), confidence=0.9, frame_id=1),
            GroundedObject("p1", "cube", GroundStatus.UNLOCALIZED, bbox_xyxy=(100, 50, 130, 80),
                           confidence=0.7, frame_id=1),
            GroundedObject("p2", "stone", GroundStatus.AMBIGUOUS, bbox_xyxy=(200, 50, 230, 80),
                           confidence=0.4, frame_id=1),
        ],
        regions=[
            GroundedObject("r0", "red_region", GroundStatus.LOCALIZED, bbox_xyxy=(300, 200, 400, 260),
                           pos_world=(0.6, 0.4, 0.4), confidence=0.95, kind="region", frame_id=1),
        ],
        frame_id=1,
    )


def ctx(held: str | None = None) -> ExecutionContext:
    return ExecutionContext(scene=make_scene(), held_instance_id=held)


def codes(errors):
    return [e.code for e in errors]


def test_held_object_replan_move_place_valid():
    """A replan while already holding p0 that contains MOVE_TO and PLACE
    (no fresh GRASP) must be valid."""
    plan = Plan(
        actions=[
            Action(Skill.MOVE_TO, target="r0"),
            Action(Skill.PLACE, target="r0", params={"object": "p0"}),
            Action(Skill.VERIFY, params={"condition": "object_in_region", "object": "p0", "region": "r0"}),
        ],
        status=PlanStatus.READY,
    )
    assert validate_plan(plan, ctx(held="p0")) == []


def test_search_in_ready_plan_invalid():
    plan = Plan(actions=[Action(Skill.SEARCH, target="stone")], status=PlanStatus.READY)
    assert "SEARCH_IN_READY" in codes(validate_plan(plan, ctx()))


def test_search_in_needs_search_plan_valid():
    plan = Plan(actions=[Action(Skill.SEARCH, target="stone")], status=PlanStatus.NEEDS_SEARCH)
    assert validate_plan(plan, ctx()) == []


def test_place_wrong_object_invalid():
    plan = Plan(
        actions=[Action(Skill.PLACE, target="r0", params={"object": "p1"})],
        status=PlanStatus.READY,
    )
    errs = codes(validate_plan(plan, ctx(held="p0")))
    assert "WRONG_OBJECT" in errs


def test_place_without_holding_invalid():
    plan = Plan(actions=[Action(Skill.PLACE, target="r0")], status=PlanStatus.READY)
    assert "NOT_HOLDING" in codes(validate_plan(plan, ctx()))


def test_verify_visible_allows_unlocalized_instance():
    """VERIFY(object_visible) may reference p1, which has a bbox but no 3D
    position."""
    plan = Plan(
        actions=[Action(Skill.VERIFY, target="p1", params={"condition": "object_visible"})],
        status=PlanStatus.READY,
    )
    assert validate_plan(plan, ctx()) == []


def test_reach_rejects_unlocated_instance():
    plan = Plan(actions=[Action(Skill.REACH, target="p1")], status=PlanStatus.READY)
    assert "UNLOCATED_REFERENCE" in codes(validate_plan(plan, ctx()))


def test_missing_reference():
    plan = Plan(actions=[Action(Skill.REACH, target="p99")], status=PlanStatus.READY)
    assert "MISSING_REFERENCE" in codes(validate_plan(plan, ctx()))


def test_uncertain_reference_rejected_for_grasp():
    plan = Plan(actions=[Action(Skill.GRASP, target="p2")], status=PlanStatus.READY)
    assert "UNCERTAIN_REFERENCE" in codes(validate_plan(plan, ctx()))


def test_param_types_and_bounds_checked():
    # Wrong type
    plan = Plan(actions=[Action(Skill.MOVE_TO, params={"pos": "over there"})], status=PlanStatus.READY)
    assert "BAD_PARAM" in codes(validate_plan(plan, ctx(held="p0")))
    # Non-finite
    plan = Plan(actions=[Action(Skill.MOVE_TO, params={"pos": [0.1, float("nan"), 0.2]})], status=PlanStatus.READY)
    assert "BAD_PARAM" in codes(validate_plan(plan, ctx(held="p0")))
    # Out of workspace bounds
    plan = Plan(actions=[Action(Skill.MOVE_TO, params={"pos": [50.0, 0.0, 0.2]})], status=PlanStatus.READY)
    assert "BAD_PARAM" in codes(validate_plan(plan, ctx(held="p0")))
    # Valid position passes
    plan = Plan(actions=[Action(Skill.MOVE_TO, params={"pos": [0.5, 0.2, 0.5]})], status=PlanStatus.READY)
    assert validate_plan(plan, ctx(held="p0")) == []


def test_grasp_updates_simulated_state():
    """Second GRASP without release is ALREADY_HOLDING; GRASP then PLACE
    then GRASP again is fine."""
    plan = Plan(
        actions=[Action(Skill.GRASP, target="p0"), Action(Skill.GRASP, target="p0")],
        status=PlanStatus.READY,
    )
    assert "ALREADY_HOLDING" in codes(validate_plan(plan, ctx()))
    plan = Plan(
        actions=[
            Action(Skill.GRASP, target="p0"),
            Action(Skill.PLACE, target="r0"),
            Action(Skill.GRASP, target="p0"),
        ],
        status=PlanStatus.READY,
    )
    assert validate_plan(plan, ctx()) == []


def test_actions_after_stop_invalid():
    plan = Plan(
        actions=[Action(Skill.STOP), Action(Skill.REACH, target="p0")],
        status=PlanStatus.READY,
    )
    assert "BAD_TERMINATION" in codes(validate_plan(plan, ctx()))


def test_clarification_plan_shape():
    plan = Plan(status=PlanStatus.NEEDS_CLARIFICATION)  # missing question
    assert "BAD_STATUS" in codes(validate_plan(plan, ctx()))
    plan = Plan(status=PlanStatus.NEEDS_CLARIFICATION, clarification_question="which stone?")
    assert validate_plan(plan, ctx()) == []


def test_valid_partial_plan_is_not_success_permission():
    """Validity of a partial plan (no VERIFY) is a distinct notion from
    permission to claim task success: validate_plan accepts it, and nothing
    in the result carries any success claim."""
    plan = Plan(actions=[Action(Skill.MOVE_TO, target="r0")], status=PlanStatus.READY)
    errs = validate_plan(plan, ctx(held="p0"))
    assert errs == []
    # validate_plan returns only PlanErrors; task success is gated solely by
    # the orchestrator's final verification (tested in test_contract.py).
    assert not hasattr(plan, "success")
