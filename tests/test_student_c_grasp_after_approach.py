"""Final run F: GRASP right after APPROACH when parking took the target out of
the head camera's view (run 1: f25/f38/f41/f45, bottle at the table's near
right edge). The fresh frame shows no bottle at all, so GRASP used to raise
TARGET_LOST on every attempt."""
import numpy as np

from core import skills
from core.types import Action, ErrorCode, ExecutionResult, GroundStatus, Skill, SkillResult
from executor.student_c import StudentCExecutor
from tests.test_student_c import ScenePerception, SensorEnv

PLANNED = [.35, -.35, .91]


def grasp_ok(monkeypatch, env, seen):
    def grasp(e, pos):
        seen.append(np.asarray(pos).tolist())
        env.attached = True
        return SkillResult(True)
    monkeypatch.setattr(skills, "grasp", grasp)
    monkeypatch.setattr(skills, "move_to", lambda e, p: SkillResult(True))
    # APPROACH itself (tuck + parking) is covered elsewhere; only its success matters here.
    monkeypatch.setattr(StudentCExecutor, "_approach", lambda self, a, e, p: ExecutionResult(a, True))


def approached(c, env, target="p0"):
    result = c.execute(Action(Skill.APPROACH, target, {"pos": PLANNED}), env, ScenePerception())
    assert result.success


def test_grasp_uses_the_planned_position_when_the_target_left_the_view(monkeypatch):
    env, seen = SensorEnv(), []
    grasp_ok(monkeypatch, env, seen)
    c = StudentCExecutor()
    approached(c, env)
    result = c.execute(Action(Skill.GRASP, "p0", {"pos": PLANNED}), env, ScenePerception(missing=True))
    assert result.success and env.attached
    assert seen == [PLANNED]
    assert result.info["grasp_attempts"][0]["out_of_view_after_approach"] is True


def test_not_after_an_approach_to_another_target_or_another_action(monkeypatch):
    env, seen = SensorEnv(), []
    grasp_ok(monkeypatch, env, seen)
    c = StudentCExecutor()
    approached(c, env, target="p9")
    result = c.execute(Action(Skill.GRASP, "p0", {"pos": PLANNED}), env, ScenePerception(missing=True))
    assert not result.success and result.error_code is ErrorCode.TARGET_LOST and not seen
    approached(c, env)
    c.execute(Action(Skill.VERIFY, "p0", {"condition": "object_visible"}), env, ScenePerception())
    result = c.execute(Action(Skill.GRASP, "p0", {"pos": PLANNED}), env, ScenePerception(missing=True))
    assert result.error_code is ErrorCode.TARGET_LOST and not seen


def test_not_when_the_base_moved_or_the_target_is_seen_unlocalized(monkeypatch):
    env, seen = SensorEnv(), []
    grasp_ok(monkeypatch, env, seen)
    c = StudentCExecutor()
    approached(c, env)
    env.base = env.base + np.array([0.05, 0.0, 0.0])
    result = c.execute(Action(Skill.GRASP, "p0", {"pos": PLANNED}), env, ScenePerception(missing=True))
    assert result.error_code is ErrorCode.TARGET_LOST and not seen
    env.base = env.base - np.array([0.05, 0.0, 0.0])
    approached(c, env)
    result = c.execute(Action(Skill.GRASP, "p0", {"pos": PLANNED}), env,
                       ScenePerception(status=GroundStatus.UNLOCALIZED))
    assert result.error_code is ErrorCode.TARGET_LOST and not seen


def test_not_without_a_planned_position(monkeypatch):
    env, seen = SensorEnv(), []
    grasp_ok(monkeypatch, env, seen)
    c = StudentCExecutor()
    approached(c, env)
    result = c.execute(Action(Skill.GRASP, "p0"), env, ScenePerception(missing=True))
    assert result.error_code is ErrorCode.TARGET_LOST and not seen
