"""Final2 3.3: a placed object that A detects on the region but cannot fit in
3D is projected onto the region plane (pos_basis='plane_projected'), for
verification only.

Fixture: final RUN 5 f48 (dark red stone), frame 92 = the first sample of the
final verification. The stone box touches the right image edge; its fit
failed ('clipped object centre is below the table top'), so the final
verification failed with 'reliable 3D grounding is required' although the
oracle found the stone in the region, stable (placed-but-unverified). One
frame earlier (91) the same stone was LOCALIZED; that position is the check."""
from dataclasses import replace
import json
import pathlib

import numpy as np
import pytest

from core.types import (Action, GroundedObject, GroundStatus, ReleasedHint, SceneDescription, Skill,
                        TrackingHint)
from core.verification import check_placement
from tests.test_student_a_partial_object import make_a

FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'placed_unlocalized_stone_f48'


@pytest.fixture(scope='module')
def frame():
    from PIL import Image
    from core.types import Observation
    meta = json.loads((FIXTURE / 'meta.json').read_text())
    obs = Observation(meta['frame_id'], np.array(Image.open(FIXTURE / 'rgb.png')),
                      np.load(FIXTURE / 'depth.npz')['depth'], np.array(meta['intrinsics']),
                      np.array(meta['t_world_camera']), meta['sim_time'], meta['camera_name'])
    return obs, meta


def stone(scene):
    return next(g for g in scene.objects if g.name == 'stone')


def test_f48_placed_stone_is_projected_and_verifies(frame, tmp_path):
    obs, meta = frame
    scene = make_a(tmp_path, [meta['wire']]).describe(obs)  # no fresh hint: a verification capture
    s = stone(scene)
    assert s.status is GroundStatus.LOCALIZED and s.attributes['pos_basis'] == 'plane_projected'
    assert np.linalg.norm(np.array(s.pos_world[:2]) - np.array(meta['stone_pos_frame_91'][:2])) < 0.03
    region = next(g for g in scene.regions)
    assert check_placement(scene, s.instance_id, region.instance_id, attached=False).passed


def test_not_in_planning_scenes(frame, tmp_path):
    obs, meta = frame
    hint = TrackingHint(released=ReleasedHint('a9', (0.4, 0.3, 1.0)))
    s = stone(make_a(tmp_path, [meta['wire']]).describe(obs, hint=hint))  # the orchestrator's perceive
    assert s.status is GroundStatus.UNLOCALIZED


def test_never_for_the_held_instance(frame, tmp_path):
    obs, meta = frame
    a = make_a(tmp_path, [meta['wire'], meta['wire']])
    first = stone(a.describe(obs))
    a._held_id = first.instance_id
    again = stone(a.describe(replace(obs, frame_id=93, sim_time=obs.sim_time + 1)))
    assert again.status is GroundStatus.UNLOCALIZED


def test_needs_three_centimetres_inside_the_footprint(frame, tmp_path):
    obs, meta = frame
    a = make_a(tmp_path, [meta['wire']])
    a.PLANE_PROJECTED_INSET_M = 0.07  # the projection lands 3.3 cm from the centre: not 7 cm inside
    assert stone(a.describe(obs)).status is GroundStatus.UNLOCALIZED


def test_c_never_grasps_a_plane_projected_object():
    from core.types import ErrorCode
    from executor.student_c import StudentCExecutor
    from tests.test_student_c import SensorEnv

    class Projected:
        def describe(self, obs):
            g = GroundedObject('a1', 'stone', GroundStatus.LOCALIZED, pos_world=(.39, .29, .88), frame_id=obs.frame_id,
                               attributes={'pos_basis': 'plane_projected'})
            return SceneDescription([g], [], frame_id=obs.frame_id, sim_time=obs.sim_time)

    env = SensorEnv()
    result = StudentCExecutor().execute(Action(Skill.GRASP, 'a1', {'pos': [.39, .29, .88]}), env, Projected())
    assert not result.success and result.error_code is ErrorCode.TARGET_LOST and env.steps == 0
