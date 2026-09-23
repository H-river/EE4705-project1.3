"""An INFEASIBLE goal may name the class it is refusing.

Both fixtures are live qwen3.7-plus replies from the night run:
- evaluation test_26 "Place the gray stone inside the drawer." (runs/night/planning/test_26)
- variation var_15 "Move the purple sphere to the red area." (runs/night/planning_variation/var_15)
In both, the repaired reply was a correct INFEASIBLE that the contract rejected
with "Unknown region/object class", so the case was scored wrong.
"""
import pytest

from core.types import ExecutionContext, PlanStatus
from planner.contract import WIRE_VERSION, PlanContractError, compile_plan
from planner.fixtures import example_scene, example_response


def refusal(object_name='stone', object_id='p0', region_name='red_region', region_id='p2',
            reason='The destination is not a supported region.'):
    return {"schema_version": WIRE_VERSION, "status": "INFEASIBLE",
            "goal": {"object_id": object_id, "object_name": object_name, "object_color": "",
                     "region_id": region_id, "region_name": region_name},
            "actions": [], "reason": reason, "clarification_question": ""}


def context():
    return ExecutionContext(example_scene())


def test_infeasible_may_name_an_unsupported_region_class_test_26():
    wire = refusal(region_name='drawer', region_id='')
    plan, goal = compile_plan(wire, context())
    assert plan.status is PlanStatus.INFEASIBLE and not plan.actions
    assert goal['region_name'] == 'drawer'


def test_infeasible_may_name_an_unsupported_object_class_var_15():
    wire = refusal(object_name='sphere', object_id='', region_id='',
                   reason="A purple sphere is not one of the supported objects.")
    plan, _ = compile_plan(wire, context())
    assert plan.status is PlanStatus.INFEASIBLE


def test_infeasible_still_needs_a_reason_and_no_actions():
    wire = refusal(region_name='drawer', region_id='')
    wire['reason'] = ''
    wire['actions'] = example_response()['actions']
    with pytest.raises(PlanContractError):
        compile_plan(wire, context())


@pytest.mark.parametrize('status', ['READY', 'NEEDS_SEARCH', 'NEEDS_CLARIFICATION'])
def test_unsupported_class_still_rejected_for_every_other_status(status):
    wire = refusal(region_name='drawer', region_id='')
    wire['status'] = status
    if status == 'NEEDS_SEARCH':
        wire['actions'] = [{"skill": "SEARCH", "target": "drawer", "object": "",
                            "region": "", "condition": ""}]
    with pytest.raises(PlanContractError, match="Unknown region class"):
        compile_plan(wire, context())


def test_infeasible_with_a_known_class_keeps_the_id_checks():
    wire = refusal(region_id='does_not_exist')
    with pytest.raises(PlanContractError, match="Unknown perceived region ID"):
        compile_plan(wire, context())
