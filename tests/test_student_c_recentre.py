"""Round 6 step 4: SEARCH turns toward a visible-but-unlocalized target once.

A view where perception returns the target with a box but no 3D position
(typically clipped by the image edge) is not "not found": the base turns
toward the box centre by min(20°, its bearing) and takes one extra view,
which counts as a search step. At most one re-centre per planned view.
"""
import math

import pytest
import yaml

from core.env import RobotEnv
from core.skills import BASE_YAW_TOL
from core.types import Action, ErrorCode, GroundedObject, GroundStatus, Skill
from eval.runner import scene_config_from_spec
from executor.student_c import StudentCExecutor
from tests.test_student_c_table_search import ROOT

EDGE_BOX = (600, 200, 640, 260)     # clipped by the right image edge
NEAR_BOX = (360, 200, 400, 260)     # a little right of centre
CENTRE_BOX = (300, 200, 340, 260)


class Scripted:
    """ground() answers from a script; records the base yaw at each call."""

    def __init__(self, env, answers):
        self.env, self.answers, self.calls = env, list(answers), []

    def ground(self, obs, target):
        self.calls.append((obs, float(self.env.get_base_pose()[2])))
        box, localized = self.answers.pop(0) if self.answers else (EDGE_BOX, False)
        if box is None:
            return None
        return GroundedObject("a1", target, GroundStatus.LOCALIZED if localized else GroundStatus.UNLOCALIZED,
                              bbox_xyxy=box, pos_world=(0.4, -0.15, 0.875) if localized else None,
                              frame_id=obs.frame_id)

    def describe(self, obs, query=None):
        raise AssertionError("SEARCH must use ground()")

    def reset(self):
        pass


@pytest.fixture
def env(world):
    spec = yaml.safe_load((ROOT / "eval/trials/smoke/smoke_4_search.yaml").read_text())
    world.reset(scene_config_from_spec(spec["scene"]))
    world.step(500)
    return RobotEnv(world)


def search(env, perception):
    c = StudentCExecutor()
    c.reset()
    return c.execute(Action(Skill.SEARCH, target="stone"), env, perception)


def turned(calls, i):
    d = calls[i][1] - calls[i - 1][1]
    return math.atan2(math.sin(d), math.cos(d))


@pytest.mark.parametrize("box", [EDGE_BOX, NEAR_BOX])
def test_edge_box_then_centred_box_is_found_after_one_recentre(env, box):
    p = Scripted(env, [(box, False), (CENTRE_BOX, True)])
    result = search(env, p)
    assert result.success and result.error_code is ErrorCode.NONE
    assert result.info["views"] == 2 and result.info["recentred"] == 1
    bearing = StudentCExecutor._bearing_to_box(p.calls[0][0], box)
    expected = math.copysign(min(math.radians(20), abs(bearing)), bearing)
    assert bearing < 0  # a box right of centre means turning right (yaw decreases)
    assert abs(turned(p.calls, 1) - expected) < BASE_YAW_TOL + 0.02
    if box == EDGE_BOX:
        assert abs(bearing) > math.radians(20)  # the turn was capped at 20°
    else:
        assert abs(bearing) < math.radians(20)  # turned by the bearing itself


def test_at_most_one_recentre_per_view_and_it_counts_as_a_step(env):
    p = Scripted(env, [])  # always visible at the edge, never localized
    result = search(env, p)
    assert result.error_code is ErrorCode.SEARCH_NOT_FOUND
    views, recentred = result.info["views"], result.info["recentred"]
    assert views == len(p.calls) <= 13
    assert recentred >= 1 and recentred <= views - recentred  # each re-centre follows its own planned view


def test_not_visible_or_centred_views_are_not_recentred(env):
    p = Scripted(env, [(None, False), ((310, 200, 330, 260), False)] + [(None, False)] * 20)
    result = search(env, p)
    assert result.error_code is ErrorCode.SEARCH_NOT_FOUND
    assert result.info["recentred"] == 0  # nothing seen, or the box is already centred (< 2°)
