"""Round 7 step 2: B never raises on its own contract failures.

Fixture: round 6, c_2_10_bottle. B's replan answered READY although red_region
a1 was no longer in view; the repair answered READY starting with SEARCH red_region,
the contract raised "READY cannot SEARCH", and the episode ended in ERROR.
Now a READY plan that contains SEARCH becomes NEEDS_SEARCH with that SEARCH
as its only action, and any other contract failure after repair is returned
as Plan(status=REJECTED, reason=...) (contract v4), which the orchestrator
replans instead of ending in ERROR.
"""
import json
from pathlib import Path

import pytest

from core.env import RobotEnv
from core.interfaces import Planner
from core.mocks import GTPerception, ScriptedClarifier, TeleportExecutor
from core.oracle import EvalOracle
from core.orchestrator import Orchestrator
from core.types import (ExecutionContext, GroundedObject, GroundStatus, Plan, PlanStatus,
                        SceneDescription, Skill, TrialOutcome)
from core.validation import validate_plan
from planner.fixtures import example_response, fixture_planner
from tests.conftest import standard_scene

FIXTURE = json.loads((Path(__file__).parent / "fixtures/b_ready_with_search_c_2_10.json").read_text())


def fixture_scene():
    raw = FIXTURE["scene"]
    items = [GroundedObject(g["instance_id"], g["name"], GroundStatus(g["status"]),
                            bbox_xyxy=tuple(g["bbox_xyxy"]) if g["bbox_xyxy"] else None,
                            pos_world=tuple(g["pos_world"]) if g["pos_world"] else None,
                            confidence=g["confidence"], kind=g["kind"], frame_id=g["frame_id"],
                            attributes=g["attributes"],
                            region_half_extents_xy=tuple(g["region_half_extents_xy"]) if g["region_half_extents_xy"] else None)
             for g in raw["instances"]]
    return SceneDescription(objects=[g for g in items if g.kind == "object"],
                            regions=[g for g in items if g.kind == "region"],
                            caption=raw.get("caption", ""), frame_id=raw.get("frame_id", -1))


def planner_after_first_plan(tmp_path, replies):
    """B as it was at the recorded replan: goal a4 -> a1 locked by an earlier
    READY plan made while both were visible (a1 has since left the view)."""
    goal = FIXTURE["original_goal"]
    bottle = next(g for g in fixture_scene().objects if g.instance_id == goal["object_id"])
    region = GroundedObject(goal["region_id"], "red_region", GroundStatus.LOCALIZED, kind="region",
                            pos_world=(0.40, 0.30, 0.853), region_half_extents_xy=(0.08, 0.08), frame_id=12)
    earlier = SceneDescription(objects=[bottle], regions=[region], frame_id=12)
    first = example_response(goal["object_id"], goal["region_id"])
    first["goal"].update(object_name="bottle", object_color="green")
    planner = fixture_planner([first] + list(replies), tmp_path)
    assert planner.plan(FIXTURE["instruction"], earlier).status is PlanStatus.READY
    return planner


def replan(planner):
    scene = fixture_scene()
    return planner.replan(FIXTURE["instruction"], scene, [], ExecutionContext(scene)), scene


def test_fixture_is_the_recorded_failure():
    assert FIXTURE["validation_errors"] == ["Action 2: target needs a current, unambiguous 3D position",
                                            "READY cannot SEARCH"]


def test_ready_with_search_becomes_needs_search(tmp_path):
    planner = planner_after_first_plan(tmp_path, FIXTURE["responses"])
    plan, scene = replan(planner)
    assert plan.status is PlanStatus.NEEDS_SEARCH
    assert [(a.skill, a.target) for a in plan.actions] == [(Skill.SEARCH, "red_region")]
    assert validate_plan(plan, ExecutionContext(scene)) == []
    audit = planner.last_diagnostics
    assert audit["accepted"] and audit["repair_count"] == 1
    assert audit["responses"][0]["validation_error"] == FIXTURE["validation_errors"][0]
    assert audit["responses"][-1]["normalizations"][-1]["change"] == "READY with SEARCH -> NEEDS_SEARCH"


def test_other_contract_failures_return_rejected(tmp_path):
    first = FIXTURE["responses"][0]  # READY while the region is not in view, twice
    planner = planner_after_first_plan(tmp_path, [first, first])
    plan, scene = replan(planner)
    assert plan.status is PlanStatus.REJECTED and plan.actions == []
    assert "3D position" in plan.reason
    assert validate_plan(plan, ExecutionContext(scene)) == []


class ScriptedPlanner(Planner):
    def __init__(self, plans):
        self.plans = list(plans)

    def plan(self, instruction, scene):
        return self.plans.pop(0)

    def replan(self, instruction, scene, history, context, clarification=None):
        return self.plans.pop(0)


def run(world, plans):
    world.reset(standard_scene())
    world.step(200)
    oracle = EvalOracle(world)
    orch = Orchestrator(GTPerception(oracle), ScriptedPlanner(plans), TeleportExecutor(world),
                        RobotEnv(world), ScriptedClarifier([]))
    return orch.run("Move the stone to the red area.")


def test_orchestrator_replans_after_rejected_instead_of_error(world):
    rejected = Plan(status=PlanStatus.REJECTED, reason="Planner output rejected by the contract: x")
    result = run(world, [rejected, Plan(status=PlanStatus.INFEASIBLE, reason="stop here")])
    assert result.outcome is TrialOutcome.REFUSED  # reached the second plan
    assert [e["type"] for e in result.events].count("plan_rejected") == 1


def test_repeated_rejections_are_bounded(world):
    rejected = Plan(status=PlanStatus.REJECTED, reason="Planner output rejected by the contract: x")
    result = run(world, [rejected] * 20)
    assert result.outcome is TrialOutcome.LIMIT_EXCEEDED
