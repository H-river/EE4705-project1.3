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


