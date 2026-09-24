"""Final2 6.7: when the two final-verification frames disagree, a third frame
from a view turned 15 deg decides; otherwise UNDETERMINED. Off by default."""
import numpy as np

from core.orchestrator import OrchestratorConfig
from core.types import GroundedObject, GroundStatus, SceneDescription
from core.verification import verify_placement
from tests.test_student_c import SensorEnv

REGION = (0.40, 0.30, 0.853)


class Frames:
    """Stone offsets (x) per capture: inside (0.0) or missing (None)."""

    def __init__(self, offsets):
        self.offsets, self.n = list(offsets), 0

    def describe(self, obs):
        dx = self.offsets[self.n]
        self.n += 1
        objects = [] if dx is None else [GroundedObject("a3", "stone", GroundStatus.LOCALIZED,
                                                        pos_world=(0.40 + dx, 0.30, 0.871), frame_id=obs.frame_id)]
        region = GroundedObject("a2", "red_region", GroundStatus.LOCALIZED, kind="region", pos_world=REGION,
                                region_half_extents_xy=(0.08, 0.08), frame_id=obs.frame_id)
        return SceneDescription(objects, [region], frame_id=obs.frame_id, sim_time=obs.sim_time)


def run(offsets, arbitrate=True):
    env, perception = SensorEnv(), Frames(offsets)
    return verify_placement(env, perception, "a3", "a2", arbitrate=arbitrate), perception, env


def test_agreeing_frames_need_no_third():
    result, p, _ = run([0.0, 0.005])
    assert result.passed and p.n == 2 and "stable" in result.detail
    result, p, _ = run([None, None])
    assert result.passed is False and p.n == 2


def test_disagreement_resolved_by_a_third_view_and_the_base_returns():
    result, p, env = run([0.0, None, 0.004])
    assert result.passed and p.n == 3 and "two of three" in result.detail
    assert abs(env.base[2]) < 1e-9  # turned 15 deg for the third frame, then back


def test_disagreement_not_resolved_is_undetermined():
    result, p, _ = run([None, 0.0, None])
    assert result.passed is None and result.detail.startswith("UNDETERMINED") and p.n == 3


def test_third_frame_must_agree_on_position():
    result, _, _ = run([0.0, None, 0.05])  # passes, but 5 cm from the first frame
    assert result.passed is None


def test_off_by_default():
    assert OrchestratorConfig().verification_arbitration is False
    result, p, _ = run([None, 0.0], arbitrate=False)
    assert result.passed is False and p.n == 1  # old behaviour: first failing frame decides
