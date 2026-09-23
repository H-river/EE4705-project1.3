"""Round 7 step 3: SEARCH carries the goal colour from B through C to A.

c_2_05 (rounds 5-6): SEARCH("stone") kept grounding the gray stone, or
returned AMBIGUOUS when both stones were in view, while the goal was the
dark_red stone. Now B's SEARCH target is "<colour> <class>", C passes it to
perception.ground() unchanged, and A matches it on both class and colour.
"""
import json
import pathlib

import numpy as np
import pytest
import yaml

from core.env import RobotEnv
from core.llm_client import FakeTransport, LLMClient, LLMConfig
from core.types import (Action, ExecutionContext, GroundedObject, GroundStatus, Observation,
                        SceneDescription, Skill)
from eval.runner import scene_config_from_spec
from executor.student_c import StudentCExecutor
from perception.config import VisionConfig
from perception.student_a import StudentAPerception, colour_class_query
from planner.contract import PlanContractError, compile_plan
from planner.fixtures import wire_action

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "two_stones_c_2_05"


def search_wire(target, colour="dark_red"):
    return {"schema_version": "student-b-plan-v1", "status": "NEEDS_SEARCH",
            "goal": {"object_id": "", "object_name": "stone", "object_color": colour,
                     "region_id": "", "region_name": "red_region"},
            "actions": [wire_action("SEARCH", target)], "reason": "not visible", "clarification_question": ""}


# ---------------------------------------------------------------- B

@pytest.mark.parametrize("model_target,expected", [
    ("stone", "dark_red stone"), ("dark_red stone", "dark_red stone"), ("dark red stone", "dark_red stone"),
    ("red_region", "red_region"),
])
def test_b_search_target_carries_the_goal_colour(model_target, expected):
    plan, _ = compile_plan(search_wire(model_target), ExecutionContext(SceneDescription()))
    assert [(a.skill, a.target) for a in plan.actions] == [(Skill.SEARCH, expected)]


def test_b_search_without_a_goal_colour_stays_the_class():
    plan, _ = compile_plan(search_wire("stone", colour=""), ExecutionContext(SceneDescription()))
    assert plan.actions[0].target == "stone"


def test_b_rejects_a_search_colour_that_contradicts_the_goal():
    with pytest.raises(PlanContractError, match="colour"):
        compile_plan(search_wire("gray stone"), ExecutionContext(SceneDescription()))


# ---------------------------------------------------------------- C

class Recording:
    def __init__(self):
        self.targets = []

    def ground(self, obs, target):
        self.targets.append(target)
        return GroundedObject("a3", "stone", GroundStatus.LOCALIZED, bbox_xyxy=(300, 200, 340, 240),
                              pos_world=(0.6, 0.2, 0.875), frame_id=obs.frame_id)

    def describe(self, obs, query=None):
        raise AssertionError("SEARCH must use ground()")

    def reset(self):
        pass


def test_c_forwards_the_search_target_unchanged(world):
    spec = yaml.safe_load((ROOT / "eval/trials/student_c_v2/c_2_05_stone.yaml").read_text())
    world.reset(scene_config_from_spec(spec["scene"]))
    world.step(200)
    perception = Recording()
    c = StudentCExecutor()
    c.reset()
    result = c.execute(Action(Skill.SEARCH, target="dark_red stone"), RobotEnv(world), perception)
    assert result.success and perception.targets == ["dark_red stone"]


# ---------------------------------------------------------------- A

@pytest.fixture(scope="module")
def two_stones():
    from PIL import Image
    meta = json.loads((FIXTURE / "meta.json").read_text())
    obs = Observation(meta["frame_id"], np.array(Image.open(FIXTURE / "rgb.png")),
                      np.load(FIXTURE / "depth.npz")["depth"], np.array(meta["intrinsics"]),
                      np.array(meta["t_world_camera"]), meta["sim_time"], meta["camera_name"])
    return obs, meta["wire"], meta["scene_stones"]


def make_a(tmp_path, wire):
    transport = FakeTransport([FakeTransport.completion(json.dumps(wire))])
    config = VisionConfig(LLMConfig("test-vlm", "https://fixture.invalid/v1", api_key="test-secret",
                                    max_retries=0, retry_backoff_s=0), str(tmp_path / "audit"))
    return StudentAPerception(config, LLMClient(config.llm, transport=transport))


def test_query_parser():
    assert colour_class_query("dark_red stone") == ("stone", "dark_red")
    assert colour_class_query("dark red stone") == ("stone", "dark_red")
    assert colour_class_query("stone") is None and colour_class_query("a3") is None
    assert colour_class_query("stone beside the cube") is None


def test_a_grounds_the_dark_red_stone_on_the_two_stone_frame(two_stones, tmp_path):
    obs, wire, recorded = two_stones
    assert wire["selected"] == [0, 1]  # the recorded reply selected BOTH stones
    found = make_a(tmp_path, wire).ground(obs, "dark_red stone")
    assert found is not None and found.status is GroundStatus.LOCALIZED
    assert found.name == "stone" and found.attributes["color"] == "dark_red"
    assert np.linalg.norm(np.asarray(found.pos_world) - recorded["a3"][2]) < 0.01


def test_a_bare_class_query_is_still_ambiguous(two_stones, tmp_path):
    obs, wire, _ = two_stones
    assert make_a(tmp_path, wire).ground(obs, "stone").status is GroundStatus.AMBIGUOUS


def test_a_matches_both_fields_even_when_the_model_selects_nothing(two_stones, tmp_path):
    obs, wire, _ = two_stones
    found = make_a(tmp_path, {**wire, "selected": []}).ground(obs, "dark_red stone")
    assert found is not None and found.attributes["color"] == "dark_red"
    assert make_a(tmp_path, {**wire, "selected": []}).ground(obs, "green bottle") is None
