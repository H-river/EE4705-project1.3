"""Final2 3.5: a grasp that is out of reach from the parking pose is retried
once from a parking pose rotated 90 deg around the target, chosen to stay
clear of the table; APPROACH honours B's standoff_rotation_deg (2.4).

s31_a f38 (bottle at (0.573, -0.429), 7 cm from the table's side edge): the
front parking pose is 2.7 cm from the table, APPROACH hit it and backed off,
and GRASP was UNREACHABLE ('best pos_err 179.8 mm') every time."""
import math

import numpy as np
import pytest

from core import skills
from core.types import Action, ErrorCode, GroundedObject, GroundStatus, SceneDescription, Skill, SkillResult
from executor.student_c import StudentCExecutor
from tests.test_student_c_turn_to_find import TurnEnv

F38 = (0.573, -0.429, 0.91)


class Visible:
    def __init__(self, pos=F38):
        self.pos, self.frames = pos, []

    def describe(self, obs):
        self.frames.append(obs.frame_id)
        g = GroundedObject("a0", "bottle", GroundStatus.LOCALIZED, pos_world=self.pos, frame_id=obs.frame_id)
        return SceneDescription([g], [], frame_id=obs.frame_id, sim_time=obs.sim_time)


def test_f38_front_parking_is_blocked_and_the_side_is_clear():
    c = StudentCExecutor
    front = c._rotated_parking(F38, (0.05, -0.30, 0.0), 0.0)
    side = c._rotated_parking(F38, (0.05, -0.30, 0.0), 90.0)
    assert c._table_clearance(front[:2]) < c._PARK_CLEAR_M
    assert c._table_clearance(side[:2]) >= 0.25
    legs = c._base_legs((0.05, -0.30, 0.0), side)
    assert legs is not None and len(legs) == 2  # via a waypoint: the straight line grazes the corner
    start = (0.05, -0.30)
    for leg in legs:
        assert c._path_clearance(start, leg[:2]) >= c._PATH_CLEAR_M
        start = leg[:2]


@pytest.fixture()
def env(monkeypatch):
    e = TurnEnv()
    e.base = np.array([0.05, -0.30, 0.0])
    e.base_target = e.base.copy()
    return e


def test_unreachable_grasp_is_retried_from_the_rotated_pose(env, monkeypatch):
    grasps = []

    def grasp(env_, pos):
        grasps.append((np.asarray(pos, dtype=float), env_.base.copy()))
        if len(grasps) == 1:
            return SkillResult(False, ErrorCode.UNREACHABLE, {"detail": "best pos_err 179.8 mm"})
        env_.attached = True
        return SkillResult(True)

    monkeypatch.setattr(skills, "grasp", grasp)
    monkeypatch.setattr(skills, "move_to", lambda env_, pos: SkillResult(True))
    perception = Visible((0.575, -0.43, 0.91))
    result = StudentCExecutor().execute(Action(Skill.GRASP, "a0", {"pos": list(F38)}), env, perception)
    assert result.success, result.info
    repark = result.info["grasp_attempts"][0]["repark"]
    assert repark["success"] and repark["rotation_deg"] == 90.0
    first_base, second_base = grasps[0][1], grasps[1][1]
    assert StudentCExecutor._table_clearance(second_base[:2]) >= 0.25 > StudentCExecutor._table_clearance(first_base[:2])
    assert np.allclose(grasps[1][0], perception.pos)  # re-observed: the fresh position
    assert len(perception.frames) == 2


def test_no_clear_rotation_means_no_repark(env, monkeypatch):
    monkeypatch.setattr(StudentCExecutor, "_PARK_CLEAR_M", 5.0)
    monkeypatch.setattr(skills, "grasp", lambda env_, pos: SkillResult(False, ErrorCode.UNREACHABLE))
    result = StudentCExecutor().execute(Action(Skill.GRASP, "a0", {"pos": list(F38)}), env, Visible())
    assert not result.success and result.error_code is ErrorCode.UNREACHABLE
    assert not result.info["grasp_attempts"][0]["repark"]["success"]


def test_grasp_missed_keeps_the_same_place_retry(env, monkeypatch):
    calls = []

    def grasp(env_, pos):
        calls.append(env_.base.copy())
        if len(calls) == 1:
            return SkillResult(False, ErrorCode.GRASP_MISSED)
        env_.attached = True
        return SkillResult(True)

    monkeypatch.setattr(skills, "grasp", grasp)
    monkeypatch.setattr(skills, "move_to", lambda env_, pos: SkillResult(True))
    result = StudentCExecutor().execute(Action(Skill.GRASP, "a0", {"pos": list(F38)}), env, Visible())
    assert result.success and np.allclose(calls[0], calls[1])


def test_approach_honours_bs_standoff_rotation(env, monkeypatch):
    def forbidden(*a):
        raise AssertionError("the default parking must not be used")

    monkeypatch.setattr(skills, "approach", forbidden)
    result = StudentCExecutor().execute(
        Action(Skill.APPROACH, "a0", {"pos": list(F38), "standoff_rotation_deg": 90.0}), env, Visible())
    assert result.success and result.info["standoff_rotation_deg"] == 90.0
    assert StudentCExecutor._table_clearance(env.base[:2]) >= 0.25
