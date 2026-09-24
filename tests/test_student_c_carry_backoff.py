"""Final2 3.4: before carrying toward the region, C backs the base 10 cm
straight away from the table when it stands within 45 cm of it.

RUN 5 f23 (robot start x = -0.05, table front edge x = 0.25): every MOVE_TO
ended in torso-table contact during transport."""
import numpy as np
from core import skills
from core.types import Action, ErrorCode, GroundedObject, GroundStatus, SceneDescription, Skill, SkillResult
from executor.student_c import StudentCExecutor
from tests.test_student_c import ScenePerception, SensorEnv


def carry(env, monkeypatch):
    starts = []
    def move_to(env_, pos):
        starts.append(env_.base.copy())
        env_.ee = np.asarray(pos, dtype=float).copy()
        return SkillResult(True)
    monkeypatch.setattr(skills, "move_to", move_to)
    env.attached = True
    c = StudentCExecutor(); c._held_id = "p0"
    return c.execute(Action(Skill.MOVE_TO, "p2", {"pos": [0.4, 0.3, 1.033]}), env, ScenePerception()), starts


def test_f23_start_pose_backs_off_10_cm_from_the_table(monkeypatch):
    env = SensorEnv(); env.base = np.array([-0.05, 0.0, 0.0]); env.base_target = env.base.copy()
    result, starts = carry(env, monkeypatch)
    assert result.success and result.info["carry_backoff"]["success"]
    assert np.allclose(starts[0][:2], [-0.15, 0.0])  # straight back, heading kept
    assert starts[0][2] == 0.0


def test_far_from_the_table_nothing_changes(monkeypatch):
    env = SensorEnv(); env.base = np.array([-0.30, 0.0, 0.0]); env.base_target = env.base.copy()
    result, starts = carry(env, monkeypatch)
    assert result.success and "carry_backoff" not in result.info
    assert np.allclose(starts[0], [-0.30, 0.0, 0.0])


def test_side_of_the_table_backs_off_sideways(monkeypatch):
    env = SensorEnv(); env.base = np.array([0.40, -0.80, 1.5]); env.base_target = env.base.copy()
    result, starts = carry(env, monkeypatch)
    assert np.allclose(starts[0][:2], [0.40, -0.90])
