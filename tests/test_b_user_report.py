"""Final2 6.1: the one-sentence outcome report (no model call, no oracle)."""
from planner.report import outcome_sentence

GOAL = {"object_id": "a2", "object_name": "stone", "object_color": "gray", "region_id": "a1",
        "region_name": "red_region"}
OK = {"passed": True, "detail": "offset xyz=[-0.0087, -0.0215, 0.0125], half extents=[0.08, 0.08]; released and stable"}


def test_success_names_object_region_and_offset():
    assert outcome_sentence("CLAIMED_SUCCESS", GOAL, OK) == \
        "Placed the grey stone on the red area (2 cm from its centre), confirmed visually."


def test_unconfirmed_out_of_view():
    v = {"passed": False, "detail": "exact object or region instance is not visible"}
    assert outcome_sentence("FAILED", GOAL, v) == "Could not confirm - the grey stone is out of view."


def test_unconfirmed_without_3d():
    v = {"passed": False, "detail": "reliable 3D grounding is required"}
    assert "cannot tell exactly where" in outcome_sentence("LIMIT_EXCEEDED", GOAL, v)


def test_refusal_uses_the_planner_reason():
    events = [{"type": "plan", "status": "INFEASIBLE", "reason": "Stacking objects on other objects is not supported."}]
    assert outcome_sentence("REFUSED", {}, None, events) == \
        "Did not start: Stacking objects on other objects is not supported."


def test_clarification_and_search_and_failures():
    clar = [{"question": "Which stone should I move?", "response": None}]
    assert "Which stone should I move?" in outcome_sentence("CLARIFICATION_EXHAUSTED", GOAL, None, (), clar)
    bottle = {**GOAL, "object_name": "bottle", "object_color": "green"}
    assert outcome_sentence("SEARCH_EXHAUSTED", bottle) == "Could not find the green bottle after searching the table."
    events = [{"type": "action", "skill": "GRASP", "success": False, "error": "TARGET_LOST"}]
    assert outcome_sentence("LIMIT_EXCEEDED", bottle, None, events) == \
        "Gave up after repeated failures; the last was GRASP (target lost)."
    assert outcome_sentence("ERROR") == "Stopped on an internal error; nothing is confirmed."


def test_no_goal_is_fine():
    assert outcome_sentence("CLAIMED_SUCCESS", None, OK).startswith("Placed the object on the target area")
