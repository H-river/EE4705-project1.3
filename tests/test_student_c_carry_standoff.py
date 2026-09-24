"""Final run F1: carry parking keeps >= 0.25 m from the table box, and a
table contact during transport gets one retry parked 5 cm farther away
(run 1 f23: torso-table contact on both MOVE_TO attempts)."""
import math

import numpy as np

from core import skills
from core.g1 import ARM_WORKSPACE_OFFSET
from core.types import Action, ErrorCode, Skill, SkillResult
from executor.student_c import StudentCExecutor, _BodyTableContact
from tests.test_student_c import ScenePerception, SensorEnv

REGION_CARRY = np.array([0.40, 0.30, 0.853 + 0.18])


def test_parking_keeps_the_table_standoff_and_the_arm_offset():
    c = StudentCExecutor()
    for base in [(-0.05, -0.2), (0.0, 0.0), (0.15, -0.01)]:
        pose = c._carry_parking(REGION_CARRY, base, 0.25)
        assert c._table_distance(pose[:2]) >= 0.25 - 1e-9
        # Target sits at ARM_WORKSPACE_OFFSET in the base frame, as in skills.approach.
        yaw = pose[2]
        rel = REGION_CARRY[:2] - np.array(pose[:2])
        local = np.array([math.cos(yaw) * rel[0] + math.sin(yaw) * rel[1],
                          -math.sin(yaw) * rel[0] + math.cos(yaw) * rel[1]])
        assert np.allclose(local, ARM_WORKSPACE_OFFSET, atol=1e-9)


def test_parking_keeps_the_default_heading_when_it_is_already_clear():
    c = StudentCExecutor()
    from core.g1 import approach_base_pose
    base = (-0.6, 0.3)
    assert np.allclose(c._carry_parking(REGION_CARRY, base, 0.25), approach_base_pose(REGION_CARRY[:2], np.array(base)))


def test_contact_during_carry_backs_off_and_retries_farther(monkeypatch):
    env, c = SensorEnv(), StudentCExecutor()
    env.attached, c._held_id = True, "p0"
    standoffs = []

    def carry(e, pos, standoff):
        standoffs.append(standoff)
        if len(standoffs) == 1:
            raise _BodyTableContact([("torso_link", "table", -1e-4)])
        env.ee = np.asarray(pos, dtype=float).copy()
        return SkillResult(True)
    monkeypatch.setattr(c, "_carry", carry)
    result = c.execute(Action(Skill.MOVE_TO, "p2"), env, ScenePerception())
    assert result.success and result.recovery_attempted
    assert standoffs == [0.25, 0.30]
    assert result.info["contact_retry"]["contacts"][0][0] == "torso_link"


def test_second_contact_fails_with_the_held_object_kept(monkeypatch):
    env, c = SensorEnv(), StudentCExecutor()
    env.attached, c._held_id = True, "p0"

    def carry(e, pos, standoff):
        raise _BodyTableContact([("torso_link", "table", -1e-4)])
    monkeypatch.setattr(c, "_carry", carry)
    result = c.execute(Action(Skill.MOVE_TO, "p2"), env, ScenePerception())
    assert not result.success and result.error_code is ErrorCode.TIMEOUT and env.attached
