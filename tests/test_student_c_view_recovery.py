"""Round fix: PLACE retries the view when the placement is visible but unlocalized.

Reproduces c_2_01/c_2_02/c_2_08 (runs/night/e2e_v2): the object was placed in the
region, but from the post-place pose the region box touched the image edge, so
verify_placement returned "reliable 3D grounding is required" and C reported
PLACE_FAILED without trying another viewpoint.
"""
import numpy as np

from core import skills
from core.types import Action, ErrorCode, Skill, SkillResult, VerificationResult
from executor.student_c import StudentCExecutor
from tests.test_student_c import ScenePerception, SensorEnv, move_success


def _setup(monkeypatch, details):
    import executor.closed_loop as module
    env, c = SensorEnv(), StudentCExecutor()
    env.attached = True
    c._held_id = 'p0'
    monkeypatch.setattr(skills, 'move_to', move_success)
    monkeypatch.setattr(skills, 'reach', move_success)

    def release(env):
        env.attached = False
        return SkillResult(True)
    monkeypatch.setattr(skills, 'place', release)
    checks = []
    targets = []
    original = c._restore_view

    def restore(env, yaw_offset=0.):
        targets.append(yaw_offset)
        return original(env, yaw_offset=yaw_offset)
    monkeypatch.setattr(c, '_restore_view', restore)

    def verify(*args):
        detail = details[min(len(checks), len(details) - 1)]
        checks.append(detail)
        return VerificationResult(detail == 'ok', 'object_in_region', detail, len(checks))
    monkeypatch.setattr(module, 'verify_placement', verify)
    return env, c, checks, targets


def test_unlocalized_placement_returns_to_view_pose_first(monkeypatch):
    env, c, checks, targets = _setup(monkeypatch, ['reliable 3D grounding is required', 'ok'])
    result = c.execute(Action(Skill.PLACE, 'p2', {'object': 'p0'}), env, ScenePerception())
    assert result.success and result.error_code is ErrorCode.NONE
    assert result.recovery_attempted and result.info['view_recovery_attempts'] == 1
    assert targets == [0.0]


def test_unlocalized_placement_gives_up_after_three_views(monkeypatch):
    env, c, checks, targets = _setup(monkeypatch, ['reliable 3D grounding is required'])
    result = c.execute(Action(Skill.PLACE, 'p2', {'object': 'p0'}), env, ScenePerception())
    assert not result.success and result.error_code is ErrorCode.PLACE_FAILED
    assert len(targets) == 3 and targets[0] == 0.0 and abs(abs(targets[1]) - 0.2) < 1e-9
    assert len(checks) == 4


def test_bad_geometry_after_view_retry_is_not_retried_again(monkeypatch):
    env, c, checks, targets = _setup(monkeypatch, ['reliable 3D grounding is required', 'outside region'])
    result = c.execute(Action(Skill.PLACE, 'p2', {'object': 'p0'}), env, ScenePerception())
    assert not result.success and result.info['detail'] == 'outside region'
    assert targets == [0.0]
