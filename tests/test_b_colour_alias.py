"""Round 9: instruction colour words match a perceived colour that is the
same or a shade of it (red <-> dark_red, grey <-> gray).

Fixture: A's frame-0 scenes of the two "Move the red stone to the red area."
trials of final RUN 1 (f19 = c_2_04: dark_red stone a3 LOCALIZED, B asked
which stone; f20 = c_2_05: a3 UNLOCALIZED, SEARCH("red stone") never matched
A's 'dark_red' and the search was exhausted)."""
import json
from pathlib import Path

import pytest

from core.types import GroundedObject, GroundStatus, PlanStatus, SceneDescription, Skill
from planner.contract import WIRE_VERSION, colour_matches, palette_colour
from planner.fixtures import example_response, fixture_planner, wire_action
from planner.prompts import color_words, planning_input
from core.types import ExecutionContext

FIX = json.loads((Path(__file__).parent / "fixtures/red_stone_scenes_f19_f20.json").read_text())


def scene_of(tid):
    raw = FIX[tid]["scene"]
    items = [GroundedObject(g["instance_id"], g["name"], GroundStatus(g["status"]),
                            bbox_xyxy=tuple(g["bbox_xyxy"]) if g["bbox_xyxy"] else None,
                            pos_world=tuple(g["pos_world"]) if g["pos_world"] else None,
                            confidence=g["confidence"], kind=g["kind"], frame_id=g["frame_id"],
                            attributes={"color": g["attributes"]["color"]},
                            region_half_extents_xy=tuple(g["region_half_extents_xy"]) if g["region_half_extents_xy"] else None)
             for g in raw["objects"] + raw["regions"]]
    return SceneDescription([g for g in items if g.kind == "object"], [g for g in items if g.kind == "region"],
                            frame_id=raw["frame_id"])


@pytest.mark.parametrize("word,attr,ok", [("red", "dark_red", True), ("red", "red", True), ("dark red", "dark_red", True),
                                          ("grey", "gray", True), ("gray", "grey", True), ("dark red", "red", False),
                                          ("red", "gray", False), ("blue", "dark_red", False), ("", "red", False)])
def test_colour_matches(word, attr, ok):
    assert colour_matches(word, attr) is ok


def test_palette_colour_uses_the_vocabulary_shades():
    assert palette_colour("red", "stone") == "dark_red"
    assert palette_colour("grey", "stone") == "gray"
    assert palette_colour("gray", "stone") == "gray"
    assert palette_colour("dark red", "stone") == "dark_red"
    assert palette_colour("blue", "stone") == "blue"  # no shade: unchanged, the checks reject it


def test_f19_red_stone_binds_the_dark_red_stone(tmp_path):
    reply = example_response("a3", "a2")
    reply["goal"]["object_color"] = "red"
    planner = fixture_planner([reply], tmp_path)
    plan = planner.plan(FIX["f19_c_2_04_stone"]["instruction"], scene_of("f19_c_2_04_stone"))
    assert plan.status is PlanStatus.READY and plan.actions[0].target == "a3"
    assert planner.goal["object_color"] == "dark_red"
    norm = planner.last_diagnostics["responses"][0]["normalizations"]
    assert {"field": "goal.object_color", "change": "colour alias", "from": "red", "to": "dark_red"} in norm


def test_f20_search_for_the_red_stone_asks_a_for_dark_red(tmp_path):
    reply = {"schema_version": WIRE_VERSION, "status": "NEEDS_SEARCH",
             "goal": {"object_id": "", "object_name": "stone", "object_color": "red",
                      "region_id": "a2", "region_name": "red_region"},
             "actions": [wire_action("SEARCH", "red stone")], "reason": "a3 has no 3D position",
             "clarification_question": ""}
    planner = fixture_planner([reply], tmp_path)
    plan = planner.plan(FIX["f20_c_2_05_stone"]["instruction"], scene_of("f20_c_2_05_stone"))
    assert plan.status is PlanStatus.NEEDS_SEARCH
    assert [(a.skill, a.target) for a in plan.actions] == [(Skill.SEARCH, "dark_red stone")]


def test_a_colour_that_is_not_a_shade_is_still_rejected(tmp_path):
    reply = example_response("a3", "a2")
    reply["goal"]["object_color"] = "blue"
    planner = fixture_planner([reply, reply], tmp_path)
    plan = planner.plan("Move the blue stone to the red area.", scene_of("f19_c_2_04_stone"))
    assert plan.status is PlanStatus.REJECTED and "color mismatch" in plan.reason


def test_replan_keeps_the_aliased_goal(tmp_path):
    reply = example_response("a3", "a2")
    reply["goal"]["object_color"] = "red"
    planner = fixture_planner([reply, reply], tmp_path)
    scene = scene_of("f19_c_2_04_stone")
    planner.plan(FIX["f19_c_2_04_stone"]["instruction"], scene)
    plan = planner.replan(FIX["f19_c_2_04_stone"]["instruction"], scene, [], ExecutionContext(scene))
    assert plan.status is PlanStatus.READY and planner.goal["object_color"] == "dark_red"


def test_planning_input_lists_colour_words():
    assert color_words("dark_red") == ["dark_red", "dark red", "red"]
    assert color_words("gray") == ["gray", "grey"]
    data = planning_input("x", scene_of("f19_c_2_04_stone"), [], ExecutionContext(scene_of("f19_c_2_04_stone")),
                          None, None)
    a3 = next(i for i in data["scene"]["instances"] if i["instance_id"] == "a3")
    assert "red" in a3["color_words"]
