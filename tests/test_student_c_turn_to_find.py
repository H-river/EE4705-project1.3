"""Final2 3.1: after parking, a target that is not in view is looked for by
turning the base toward it (<= 20 deg steps, at most 3); never grasped blind.

RUN 5 f25: after APPROACH the green bottle was cut off at the right image
edge (frame 17), A did not detect it, and GRASP failed TARGET_LOST four times."""
import math

import numpy as np
import pytest

from core import skills
from core.types import Action, ErrorCode, GroundedObject, GroundStatus, SceneDescription, Skill, SkillResult
from executor.student_c import StudentCExecutor
from tests.test_student_c import SensorEnv, move_success

PLANNED = [0.35, -0.35, 0.91]


class TurnEnv(SensorEnv):
    def __init__(self):
        super().__init__()
        self.yaws = []

    def set_arm_target(self, pos):
        self.ee = np.asarray(pos, dtype=float).copy()

    def set_base_target(self, *values):
        super().set_base_target(*values)
        self.yaws.append(values[2])


class VisibleAfterTurn:
    """The bottle is seen once the base has turned at least ``needed`` rad in
    ``sign`` direction (world-frame positions)."""

    def __init__(self, env, needed, sign=-1.0, fresh=(0.36, -0.34, 0.91)):
        self.env, self.needed, self.sign, self.fresh = env, needed, sign, fresh
        self.frames = []

    def describe(self, obs):
        self.frames.append(obs.frame_id)
        turned = self.sign * self.env.base[2] >= self.needed - 1e-9
        objects = [GroundedObject("a4", "bottle", GroundStatus.LOCALIZED, pos_world=self.fresh,
                                  frame_id=obs.frame_id)] if turned else []
        return SceneDescription(objects, [], frame_id=obs.frame_id, sim_time=obs.sim_time)


@pytest.fixture()
def motion(monkeypatch):
    grasps = []

    def grasp(env, pos):
        grasps.append(np.asarray(pos, dtype=float))
        env.attached = True
        return SkillResult(True)

    monkeypatch.setattr(skills, "approach", lambda env, pos: SkillResult(True))
    monkeypatch.setattr(skills, "grasp", grasp)
    monkeypatch.setattr(skills, "move_to", move_success)
    return grasps


def approach_then_grasp(env, c, perception, planned=PLANNED):
    a = c.execute(Action(Skill.APPROACH, "a4", {"pos": planned}), env, perception)
    g = c.execute(Action(Skill.GRASP, "a4", {"pos": planned}), env, perception)
    return a, g


def test_f25_turns_right_until_the_bottle_is_seen_then_grasps_the_fresh_position(motion):
    env, c = TurnEnv(), StudentCExecutor()
    perception = VisibleAfterTurn(env, math.radians(40))
    a, g = approach_then_grasp(env, c, perception)
    turn = a.info["turned_to_find"]
    assert a.success and turn["found"] and turn["steps"] == 2 and turn["direction"] == "right"
    assert g.success, g.info
    assert np.allclose(motion[-1], perception.fresh)  # not the planned position


def test_turns_toward_a_target_on_the_left(motion):
    env, c = TurnEnv(), StudentCExecutor()
    perception = VisibleAfterTurn(env, math.radians(20), sign=1.0)
    a, g = approach_then_grasp(env, c, perception, planned=[0.35, 0.35, 0.91])
    assert a.info["turned_to_find"]["direction"] == "left" and g.success


def test_at_most_three_20_degree_steps_and_never_grasped_blind(motion):
    env, c = TurnEnv(), StudentCExecutor()
    perception = VisibleAfterTurn(env, math.radians(90))  # never visible
    a, g = approach_then_grasp(env, c, perception)
    turn = a.info["turned_to_find"]
    assert a.success and not turn["found"] and turn["steps"] == 3
    turns = env.yaws[:3]
    assert all(abs(b - a_) <= math.radians(20) + 1e-9 for a_, b in zip([0.0] + turns, turns))
    assert not g.success and g.error_code is ErrorCode.TARGET_LOST and motion == []


def test_visible_target_does_not_turn_and_keeps_the_planned_grasp(motion):
    env, c = TurnEnv(), StudentCExecutor()
    perception = VisibleAfterTurn(env, 0.0)
    a, g = approach_then_grasp(env, c, perception)
    assert "turned_to_find" not in a.info and len(perception.frames) == 2  # parked view + GRASP
    assert g.success and np.allclose(motion[-1], PLANNED)
